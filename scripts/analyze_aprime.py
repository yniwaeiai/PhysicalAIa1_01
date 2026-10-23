"""観察研究 A' の trace（168 本）を解析する（計画 v2 §3・§7）。CPU のみ。
使い方: nice -n 10 /path/to/work/.venv/bin/python kadai/scripts/analyze_aprime.py
出力: kadai/state/aprime_windows_20261014.csv, kadai/state/aprime_summary_20261014.md, kadai/figs/aprime/*.png

定義（固定）:
  prog(t) = max(clip((q-q0)/(thr-q0)), clip(1-tgt_dist/tgt_dist0))   thr: drawer -0.14 / K4 -0.005 / K6 -0.005
  窓の検出: t>=10 で、累積最大 prog の増分が 30 step にわたり < 0.02 となる最初の s（s+30 <= T-1）。窓 = [s, s+40)
  指令 1.0 = 7.5 mm/step。
"""
import csv, glob, json, os, collections
from pathlib import Path
import numpy as np

K = Path(__file__).resolve().parents[1]
THR = {"drawer": -0.14, "k4": -0.005, "k6": -0.005}
MM = 7.5
GROUP = {}
for s in [1045]: GROUP[("drawer", s)] = "unseen"
for s in [1113, 1200, 1337, 1341, 1352, 1390, 1413]: GROUP[("k6", s)] = "unseen"
for s in [1027, 1030, 1037, 1039]: GROUP[("drawer", s)] = "luck"
for s in [1000, 1024, 1043]: GROUP[("k4", s)] = "luck"
for s in [1139, 1177, 1180, 1193, 1300, 1417, 1428, 1461, 1488, 1492]: GROUP[("k6", s)] = "luck"
for s in [1014, 1016]: GROUP[("drawer", s)] = "mid"
for s in [1014, 1253, 1442]: GROUP[("k6", s)] = "mid"
for t in ["drawer", "k4", "k6"]:
    for s in range(3100, 3108): GROUP[(t, s)] = "hold"


def group_of(t, s):
    return GROUP.get((t, s), "other")


def prog_of(d):
    t = str(d["task"]); q = d["q"]; q0 = q[0]
    pa = np.clip((q - q0) / (THR[t] - q0), 0, 1)
    td = d["tgt_dist"]; pb = np.clip(1 - td / td[0], 0, 1)
    return np.maximum(pa, pb)


def detect(prog, T, t0=10, L=30, eps=0.02):
    cm = np.maximum.accumulate(prog)
    for s in range(t0, T - L):
        if cm[s + L] - cm[s] < eps:
            return s
    return None


def load_all():
    eps = []
    for f in sorted(glob.glob(str(K / "state/trace_aprime/*.npz"))):
        d = np.load(f, allow_pickle=True)
        eps.append({k: d[k] for k in d.files} | {"file": f})
    return eps


def jl(path):
    return [json.loads(l) for l in open(path) if l.strip()]


def pos_norm(a): return np.linalg.norm(a[:, :3], axis=1)


def window_axes(e, s, base):
    T = len(e["act_exec"]); w = np.arange(s, min(s + 40, T))
    a = e["act_exec"]; ap = pos_norm(a)
    ee = e["ee"]
    disp = np.r_[0, np.linalg.norm(np.diff(ee, axis=0), axis=1)]  # step t の変位 = ee[t]-ee[t-1]（m）
    prog = e["prog"]
    b = base[str(e["task"])]
    msk = np.abs(b["prog"] - prog[s]) <= 0.1
    bpos = np.median(b["ap"][msk]) if msk.sum() >= 5 else np.nan
    r1 = np.median(ap[w]) / bpos if bpos > 0 else np.nan
    ar = np.median(np.linalg.norm(a[w, 3:6], axis=1))
    ok = ap[w] >= 0.05
    fol = np.median(disp[w][ok] / (ap[w][ok] * MM / 1000)) if ok.sum() >= 3 else np.nan
    # 反転率（|a_pos|>=0.1 の対）
    ca = set(int(x) for x in e["chunk_at"])
    nb = rb = ni = ri = 0
    for t in w:
        if t < 1: continue
        if ap[t] >= 0.1 and ap[t - 1] >= 0.1:
            c = float(a[t, :3] @ a[t - 1, :3] / (ap[t] * ap[t - 1]))
            if t in ca: nb += 1; rb += c < 0
            else: ni += 1; ri += c < 0
    jm = float(e["jmargin"][w].min())
    # 接触
    cnt = collections.Counter(); fs = collections.defaultdict(list); cab_max = []
    for t in w:
        cs = json.loads(str(e["contact_geom"][t]))
        cm_ = 0.0
        for c in cs:
            cnt[(c[1], c[2])] += 1; fs[(c[1], c[2])].append(c[3])
            if c[1].startswith("wooden_cabinet"): cm_ = max(cm_, c[3])
        cab_max.append(cm_)
    if cnt:
        top = cnt.most_common(1)[0][0]; topf = float(np.median(fs[top])); tops = f"{top[0]}|{top[1]}"
    else:
        tops, topf = "", 0.0
    iv = float(np.mean(e["in_view"][w]))
    spd = float(np.median(disp[w])) * 1000
    q5 = bool(0.3 <= np.median(ap[w]) <= 0.6 and np.median(cab_max) > 5 and spd < 0.5)
    return dict(win_len=len(w), a_pos_med=float(np.median(ap[w])), base_pos=bpos, ratio1=r1, a_rot_med=float(ar), follow=fol,
                rev_b=(rb / nb if nb else np.nan), n_b=nb, rev_i=(ri / ni if ni else np.nan), n_i=ni,
                rev_all=((rb + ri) / (nb + ni) if nb + ni else np.nan), n_pairs=nb + ni,
                jmargin_min=jm, jm_lt05=int(jm < 0.05), contact_top=tops, contact_force_med=topf, in_view=iv,
                ee_speed_mm=spd, q5=int(q5))


def main():
    eps = load_all()
    runs = {}
    for f in glob.glob(str(K / "state/eval_runs_20261014/aprime_*.jsonl")):
        for r in jl(f):
            hold = "aprime_hold_" in f
            runs[(r["task"], r["seed"], r["rep"], hold)] = r
    for e in eps:
        e["prog"] = prog_of(e); e["task"] = str(e["task"]); e["seed"] = int(e["seed"]); e["rep"] = int(e["rep"])
        e["hold"] = e["seed"] >= 3100
        e["success"] = bool(e["success"])
        r = runs.get((e["task"], e["seed"], e["rep"], e["hold"]))
        e["clean"] = bool(r["clean"]) if r else None
        e["disp_mm"] = r["disp_mm"] if r else None
        e["top"] = r["top"] if r else None
    # 局面を揃える基準: 同タスクの成功 ep の (prog, |a_pos|) を全 step 集める
    base = {}
    for t in ["drawer", "k4", "k6"]:
        P, A = [], []
        for e in eps:
            if e["task"] == t and e["success"]:
                P.append(e["prog"]); A.append(pos_norm(e["act_exec"]))
        base[t] = dict(prog=np.concatenate(P), ap=np.concatenate(A))
    out = []; log = []
    for e in eps:
        T = len(e["act_exec"]); s = detect(e["prog"], T)
        e["s"] = s
        if s is not None:
            ax = window_axes(e, s, base)
            ax["prog_at_s"] = float(e["prog"][s])
            ax["follow_bad"] = int(ax["ratio1"] >= 0.5 and ax["follow"] < 0.2) if not np.isnan(ax["follow"]) else 0
            ax["cmd_small"] = int(ax["ratio1"] < 0.25)
            ax["out_of_view"] = int(ax["in_view"] < 0.5)
            ax["unstable"] = int(ax["n_pairs"] >= 5 and ax["rev_all"] >= 0.2)
            e["ax"] = ax
            out.append(dict(task=e["task"], seed=e["seed"], rep=e["rep"], group=group_of(e["task"], e["seed"]), success=int(e["success"]),
                            clean=int(bool(e["clean"])), win_start=s, steps=T, **ax))
    # ---- CSV
    csvp = K / "state/aprime_windows_20261014.csv"
    cols = list(out[0].keys())
    with open(csvp, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols); w.writeheader()
        for r in out: w.writerow({k: (round(v, 4) if isinstance(v, float) else v) for k, v in r.items()})
    L = []
    P = L.append
    # ---- Q0
    P("## Q0: 同一 seed の反復で ref_pos0 / ref_quat0 が一致するか")
    from scipy.spatial.transform import Rotation as R
    by = collections.defaultdict(list)
    for e in eps: by[(e["task"], e["seed"])].append(e)
    mism = []
    for k, v in by.items():
        pos = np.array([x["ref_pos0"] for x in v]); qs = np.array([x["ref_quat0"] for x in v])
        dp = np.linalg.norm(pos[:, None] - pos[None], axis=2).max() * 1000
        rr = [R.from_quat(q) for q in qs]
        da = max(np.degrees((rr[i].inv() * rr[j]).magnitude()) for i in range(len(rr)) for j in range(len(rr)))
        mism.append((k, len(v), dp, da))
    nmis = sum(1 for m in mism if m[2] > 1e-6 or m[3] > 1e-4)
    P(f"- seed 数 {len(mism)}、そのうち反復間で不一致 {nmis}")
    if nmis:
        arr = np.array([[m[2], m[3]] for m in mism])
        P(f"- seed 内の最大広がり: 位置 中央値 {np.median(arr[:,0]):.2f} mm / 最大 {arr[:,0].max():.2f} mm、角度 中央値 {np.median(arr[:,1]):.2f}° / 最大 {arr[:,1].max():.2f}°")
        for t in ["drawer", "k4", "k6"]:
            a = np.array([[m[2], m[3]] for m in mism if m[0][0] == t])
            P(f"  - {t}: 位置 med {np.median(a[:,0]):.2f} / max {a[:,0].max():.2f} mm、角度 med {np.median(a[:,1]):.2f} / max {a[:,1].max():.2f}°")
        # 成否と位置: seed 内で成功 ep と失敗 ep の位置の差（平均位置間距離）と、各群内の広がりとの比
        diffs, within, n_mixed = [], [], 0
        for k, v in by.items():
            su = [x for x in v if x["success"]]; fa = [x for x in v if not x["success"]]
            if su and fa:
                n_mixed += 1
                ps = np.mean([x["ref_pos0"] for x in su], 0); pf = np.mean([x["ref_pos0"] for x in fa], 0)
                diffs.append(np.linalg.norm(ps - pf) * 1000)
                allp = np.array([x["ref_pos0"] for x in v]); within.append(np.linalg.norm(allp - allp.mean(0), axis=1).mean() * 1000)
        if diffs:
            P(f"- 成否が混在する seed {n_mixed} 本: 成功群と失敗群の平均位置の距離 中央値 {np.median(diffs):.2f} mm（seed 内の位置の平均ばらつき 中央値 {np.median(within):.2f} mm）")
        # 単純な向き: 位置のどの軸が成否と相関するか（混在 seed の中で z スコアの符号検定）
        sg = []
        for k, v in by.items():
            su = [x for x in v if x["success"]]; fa = [x for x in v if not x["success"]]
            if su and fa:
                ps = np.mean([x["ref_pos0"] for x in su], 0) - np.mean([x["ref_pos0"] for x in fa], 0); sg.append(ps * 1000)
        if sg:
            sg = np.array(sg); P(f"- 混在 seed での (成功 − 失敗) 位置差の平均 xyz [mm]: {np.round(sg.mean(0),2).tolist()}、符号が正の seed 数 {(sg>0).sum(0).tolist()} / {len(sg)}")
    # ---- 成否表
    P("\n## 成否の表（成功 / 衝突なし。失敗の内訳）")
    t0 = []
    for f in glob.glob(str(K / "state/eval_runs_20261013/t2_T0_*.jsonl")):
        t0 += jl(f)
    tab = collections.defaultdict(lambda: [0, 0, 0])   # (set,task,seed) -> [n, succ, clean]
    for e in eps:
        k = ("保留(strong)" if e["hold"] else "A'標的", e["task"], e["seed"]); tab[k][0] += 1; tab[k][1] += e["success"]; tab[k][2] += int(bool(e["clean"]))
    t0tab = collections.defaultdict(lambda: [0, 0, 0])
    for r in t0:
        if r.get("error"): continue
        k = (r["task"], r["seed"]); t0tab[k][0] += 1; t0tab[k][1] += int(r["success"]); t0tab[k][2] += int(r["clean"])
    P("| 集合 | タスク | 群 | seed | A' 成功 | A' 衝突なし | T0 成功 | T0 衝突なし | 12 反復 成功 | 12 反復 衝突なし |")
    P("|---|---|---|---|---|---|---|---|---|---|")
    agg = collections.defaultdict(lambda: np.zeros(8))
    for (st, t, s), (n, su, cl) in sorted(tab.items()):
        t0v = t0tab.get((t, s), [0, 0, 0]) if st == "A'標的" else [0, 0, 0]
        g = group_of(t, s)
        P(f"| {st} | {t} | {g} | {s} | {su}/{n} | {cl}/{n} | {t0v[1]}/{t0v[0]} | {t0v[2]}/{t0v[0]} | {su+t0v[1]}/{n+t0v[0]} | {cl+t0v[2]}/{n+t0v[0]} |")
        a = agg[(st, t)]; a += [n, su, cl, t0v[0], t0v[1], t0v[2], 0, 0]
    P("\n集合×タスク合計")
    P("| 集合 | タスク | A' 成功 | A' 衝突なし | T0 成功 | T0 衝突なし | 12 反復 成功 | 12 反復 衝突なし |")
    P("|---|---|---|---|---|---|---|---|")
    for (st, t), a in sorted(agg.items()):
        P(f"| {st} | {t} | {int(a[1])}/{int(a[0])} | {int(a[2])}/{int(a[0])} | {int(a[4])}/{int(a[3])} | {int(a[5])}/{int(a[3])} | {int(a[1]+a[4])}/{int(a[0]+a[3])} | {int(a[2]+a[5])}/{int(a[0]+a[3])} |")
    # 失敗の内訳
    P("\n失敗（衝突なしでない ep）の内訳（A' の 168 本）: 停留あり = 未完了で窓が検出 / 停留なし = 未完了で窓なし / 衝突のみ = done だが対象外物体 > 1 mm")
    P("| 集合 | タスク | 衝突なし | 停留あり | 停留なし | 衝突のみ | 計 |")
    P("|---|---|---|---|---|---|---|")
    cat = collections.defaultdict(lambda: collections.Counter())
    for e in eps:
        st = "保留(strong)" if e["hold"] else "A'標的"
        if e["success"] and e["clean"]: c = "ok"
        elif e["success"]: c = "col"
        elif e["s"] is not None: c = "stall"
        else: c = "nostall"
        cat[(st, e["task"])][c] += 1
    for k, c in sorted(cat.items()):
        P(f"| {k[0]} | {k[1]} | {c['ok']} | {c['stall']} | {c['nostall']} | {c['col']} | {sum(c.values())} |")
    # ---- 偽陽性率
    P("\n## 検出器の偽陽性率（成功 ep のうち窓が検出される割合。成功 = done）")
    P("| タスク | 成功 ep | 窓あり | 偽陽性率 | 未完了 ep | 窓あり | 検出率 |")
    P("|---|---|---|---|---|---|---|")
    fp = {}
    for t in ["drawer", "k4", "k6"]:
        su = [e for e in eps if e["task"] == t and e["success"]]; fa = [e for e in eps if e["task"] == t and not e["success"]]
        a = sum(e["s"] is not None for e in su); b = sum(e["s"] is not None for e in fa)
        fp[t] = a / max(len(su), 1)
        P(f"| {t} | {len(su)} | {a} | {a/max(len(su),1):.2f} | {len(fa)} | {b} | {b/max(len(fa),1):.2f} |")
    # ---- Q1〜Q4（群の単位 seed、未完了 ep の窓のみ。保留集合は除く）
    def seed_mean(task_set, grp, key, only_fail=True):
        per = collections.defaultdict(list)
        for e in eps:
            if e["task"] in task_set and group_of(e["task"], e["seed"]) == grp and e["s"] is not None and (not e["success"] or not only_fail):
                v = e["ax"][key]
                if not (isinstance(v, float) and np.isnan(v)): per[e["seed"] if len(task_set) == 1 else (e["task"], e["seed"])].append(v)
        vals = [np.mean(v) for v in per.values()]
        return (np.mean(vals) if vals else np.nan), len(vals)
    def verdict(diff, ge=0.3):
        if np.isnan(diff): return "判定不能"
        return "支持" if diff > ge else ("反証" if diff < 0 else "保留")
    P("\n## Q1〜Q3（群 = seed 単位平均。未完了 ep の窓のみ。差 = 前者 − 後者。支持 > +0.3 / 反証 < 0）")
    P("| 検定 | 対象タスク | 成功未観測群（値, n seed） | 運群（値, n seed） | 差 | 判定 |")
    P("|---|---|---|---|---|---|")
    qres = {}
    for nm, key, hi, lo, desc in [("Q1 追従不良", "follow_bad", "unseen", "luck", "unseen − luck"), ("Q3 画角外", "out_of_view", "unseen", "luck", "unseen − luck")]:
        for ts, tl in [(("drawer", "k6"), "drawer+k6"), (("drawer",), "drawer"), (("k6",), "k6")]:
            a, na = seed_mean(set(ts), hi, key); b, nb = seed_mean(set(ts), lo, key)
            d = a - b
            P(f"| {nm} | {tl} | {a:.2f} ({na}) | {b:.2f} ({nb}) | {d:+.2f} | {verdict(d) if tl=='drawer+k6' else '(参考)'+verdict(d)} |")
            qres[(nm, tl)] = d
    # Q2: 運群の (cmd_small or unstable) が unseen より高い
    for e in eps:
        if e["s"] is not None: e["ax"]["small_or_unstable"] = int(e["ax"]["cmd_small"] or e["ax"]["unstable"])
    for nm, key in [("Q2 指令小 or 方向不安定", "small_or_unstable"), ("Q2 (内訳) 指令小のみ", "cmd_small"), ("Q2 (内訳) 方向不安定のみ", "unstable")]:
        for ts, tl in [(("drawer", "k6"), "drawer+k6"), (("drawer",), "drawer"), (("k6",), "k6")]:
            a, na = seed_mean(set(ts), "luck", key); b, nb = seed_mean(set(ts), "unseen", key)
            d = a - b
            main = (nm.startswith("Q2 指令小 or") and tl == "drawer+k6")
            P(f"| {nm} (luck − unseen) | {tl} | luck {a:.2f} ({na}) | unseen {b:.2f} ({nb}) | {d:+.2f} | {verdict(d) if main else '(参考)'+verdict(d)} |")
    # Q4
    P("\n## Q4 境界対の反転率 / 内部対の反転率（停留窓、未完了 ep）")
    P("| 範囲 | 窓数 | 境界対 反転率 | 内部対 反転率 | 比 | 判定（≥2 支持 / ≤1 反証） |")
    P("|---|---|---|---|---|---|")
    def q4(sel, lab):
        per = collections.defaultdict(lambda: [0, 0, 0, 0])
        for e in sel:
            ax = e["ax"]
            nb, ni = ax["n_b"], ax["n_i"]
            rb_ = 0 if np.isnan(ax["rev_b"]) else ax["rev_b"] * nb; ri_ = 0 if np.isnan(ax["rev_i"]) else ax["rev_i"] * ni
            p = per[(e["task"], e["seed"])]; p[0] += rb_; p[1] += nb; p[2] += ri_; p[3] += ni
        # seed 単位に集約（seed 内で対をプール）→ seed 平均
        rbs = [p[0] / p[1] for p in per.values() if p[1] > 0]; ris = [p[2] / p[3] for p in per.values() if p[3] > 0]
        tb, ti = sum(p[1] for p in per.values()), sum(p[3] for p in per.values())
        pb = sum(p[0] for p in per.values()) / max(tb, 1); pi = sum(p[2] for p in per.values()) / max(ti, 1)
        mb, mi = (np.mean(rbs) if rbs else np.nan), (np.mean(ris) if ris else np.nan)
        ratio = mb / mi if mi and mi > 0 else np.nan
        v = "判定不能" if np.isnan(mb) or np.isnan(mi) else ("支持(内部=0)" if mi == 0 and mb > 0 else ("判定不能" if mi == 0 else ("支持" if ratio >= 2 else ("反証" if ratio <= 1 else "保留"))))
        P(f"| {lab} | {len(sel)} | seed平均 {mb:.3f}（プール {pb:.3f}, 対 {tb}） | seed平均 {mi:.3f}（プール {pi:.3f}, 対 {ti}） | {ratio:.2f}（プール比 {pb/pi if pi>0 else float('nan'):.2f}） | {v} |")
        return ratio
    fl = [e for e in eps if e["s"] is not None and not e["success"]]
    r4 = q4(fl, "全 未完了 ep の窓")
    q4([e for e in fl if group_of(e["task"], e["seed"]) in ("unseen", "luck", "mid")], "標的 seed のみ")
    for t in ["drawer", "k6", "k4"]: q4([e for e in fl if e["task"] == t], t)
    q4([e for e in eps if e["s"] is not None and e["success"]], "（対照）成功 ep の偽陽性窓")
    # ---- Q5
    P("\n## Q5（drawer のみ）: 窓が「|a_pos| 中央値 ∈ [0.3, 0.6] かつ 棚(wooden_cabinet*) geom に法線力 > 5 N（窓内の最大力の中央値）かつ 手先 < 0.5 mm/step」を満たす割合")
    P("| 範囲 | 窓数 | 該当 | 割合 | 判定（≥0.6 支持 / ≤0.3 反証） |")
    P("|---|---|---|---|---|")
    def q5(sel, lab):
        n = len(sel); k = sum(e["ax"]["q5"] for e in sel); r = k / n if n else np.nan
        v = "判定不能" if n == 0 else ("支持" if r >= 0.6 else ("反証" if r <= 0.3 else "保留"))
        P(f"| {lab} | {n} | {k} | {r:.2f} | {v} |"); return r
    dfl = [e for e in fl if e["task"] == "drawer"]
    q5(dfl, "drawer 未完了 ep の窓（全部）")
    q5([e for e in dfl if group_of("drawer", e["seed"]) != "hold"], "うち標的 seed")
    q5([e for e in dfl if group_of("drawer", e["seed"]) == "hold"], "うち保留集合")
    q5([e for e in eps if e["task"] == "drawer" and e["success"] and e["s"] is not None], "（対照）成功 ep の偽陽性窓")
    # ---- 窓の概況
    P("\n## 窓の概況（未完了 ep の窓。中央値）")
    P("| タスク | 群 | 窓数 | 窓開始 s | prog@s | 比① | |a_rot| | 追従比 | in_view | 最小余裕 | 余裕<0.05 | Q5該当 | 最頻接触（件数上位） |")
    P("|---|---|---|---|---|---|---|---|---|---|---|---|---|")
    for t in ["drawer", "k4", "k6"]:
        for g in ["unseen", "luck", "mid", "hold", "other"]:
            sel = [e for e in fl if e["task"] == t and group_of(t, e["seed"]) == g]
            if not sel: continue
            m = lambda k: np.nanmedian([e["ax"][k] for e in sel])
            cc = collections.Counter(e["ax"]["contact_top"] for e in sel).most_common(2)
            P(f"| {t} | {g} | {len(sel)} | {np.median([e['s'] for e in sel]):.0f} | {m('prog_at_s'):.2f} | {m('ratio1'):.2f} | {m('a_rot_med'):.2f} | {m('follow'):.2f} | {m('in_view'):.2f} | {m('jmargin_min'):.3f} | {sum(e['ax']['jm_lt05'] for e in sel)} | {sum(e['ax']['q5'] for e in sel)} | {'; '.join(f'{a}({b})' for a,b in cc)} |")
    # 保存して返す
    (K / "state/aprime_summary_body.tmp").write_text("\n".join(L))
    # ---- 図: 代表例
    make_figs(eps)
    print("\n".join(L))


def make_figs(eps):
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    figd = K / "figs/aprime"; figd.mkdir(parents=True, exist_ok=True)
    for t in ["drawer", "k6", "k4"]:
        for g in (["luck", "unseen"] if t != "k4" else ["luck"]):
            sel = [e for e in eps if e["task"] == t and group_of(t, e["seed"]) == g and e["s"] is not None and not e["success"]]
            if not sel: continue
            # 窓が最も長く保たれた（prog が低く、窓内に推論が 4 回以上入る）ものを優先、決定的に seed 順
            sel = [e for e in sel if len([c for c in e["chunk_at"] if e["s"] <= c < e["s"] + 40]) >= 2] or sel
            e = sorted(sel, key=lambda x: (x["seed"], x["rep"]))[len(sel) // 2]
            ca = [int(c) for c in e["chunk_at"]]
            idx = [i for i, c in enumerate(ca) if e["s"] <= c < e["s"] + 40]
            # 窓内の推論から 4 コマを等間隔に。足りなければ窓の直前から補う
            if len(idx) < 4:
                j0 = max(0, (idx[0] if idx else int(np.searchsorted(ca, e["s"]))) - (4 - len(idx)))
                idx = list(range(j0, min(j0 + 4, len(ca))))
            pick = [idx[int(round(x))] for x in np.linspace(0, len(idx) - 1, 4)] if len(idx) > 4 else idx[:4]
            fig, axs = plt.subplots(1, 4, figsize=(12, 3.6))
            for a, i in zip(axs, pick):
                a.imshow(e["img_wrist"][i]); a.axis("off")
                a.set_title(f"step {ca[i]}  q={e['q'][min(ca[i], len(e['q'])-1)]:.3f}", fontsize=9)
            fig.suptitle(f"{t} seed {e['seed']} rep {e['rep']} group={g}  window start s={e['s']}  prog@s={e['prog'][e['s']]:.2f}  (img_wrist at inference)", fontsize=10)
            fig.tight_layout(); fig.savefig(figd / f"{t}_{g}_s{e['seed']}_r{e['rep']}.png", dpi=110); plt.close(fig)


if __name__ == "__main__":
    main()
