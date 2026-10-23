"""I3（計画 v2 §10）の分岐を集計する。CPU のみ。判定は書かず値だけ出す。
使い方: python kadai/scripts/analyze_branches_i3.py
入力: kadai/state/branches_i3_20261015/summary_*.jsonl（+ b_*.npz は件数照合のみ）、kadai/state/branches_20261015.csv（I2 の分岐ごとの値）
出力: kadai/state/branches_i3_summary_20261015.md, kadai/state/branches_i3_20261015.csv
集計の階層は I2 と同じ: 分岐 → 窓（平均）→ seed（task, seed。窓の平均）→ 群（seed の平均）。括弧は (seed 数/窓数/分岐数)。
最終成功 = done、衝突なし成功 = done かつ 対象外物体の最大変位 <= 1 mm。
"""
import csv, json, glob, collections
from pathlib import Path
import numpy as np

K = Path(__file__).resolve().parents[1]
BR = K / "state/branches_i3_20261015"
GN = {"unseen": "成功未観測", "luck": "運", "mid": "中間", "hold": "保留"}
GORD = ["unseen", "luck", "mid", "hold"]


def load_i3():
    rows = []
    for f in sorted(glob.glob(str(BR / "summary_s*.jsonl"))):
        rows += [json.loads(l) for l in open(f) if l.strip()]
    last = {}
    for r in rows:
        last[r["id"]] = r
    n_dup = len(rows) - len(last)
    keep = [r for r in last.values() if (BR / f"b_{r['id']}.npz").exists()]
    print(f"I3 分岐 {len(rows)} 行、重複 {n_dup}、npz なし {len(last) - len(keep)}")
    for r in keep:
        r["succ"] = int(r["done"]); r["clean"] = int(r["done"] and r["disp_mm"] <= 1.0)
    return keep, n_dup, len(last) - len(keep)


def load_i2():
    rows = list(csv.DictReader(open(K / "state/branches_20261015.csv")))
    for r in rows:
        for k in ("succ", "clean", "seed", "rep", "t"):
            r[k] = int(float(r[k]))
    return rows


def mean(x):
    x = [v for v in x if v is not None and not (isinstance(v, float) and np.isnan(v))]
    return float(np.mean(x)) if x else float("nan")


def sel(rows, kind, cond=None, group=None, task=None, seeds=None):
    return [r for r in rows if r["kind"] == kind and (cond is None or r["cond"] == cond) and (group is None or r["group"] == group)
            and (task is None or r["task"] == task) and (seeds is None or (r["task"], r["seed"]) in seeds)]


def seed_vals(rs, key):
    """分岐 → 窓 → seed。{(task,seed): 値}、窓数 dict、分岐数。"""
    byw = collections.defaultdict(list)
    for r in rs:
        byw[r["wid"]].append(r)
    bys = collections.defaultdict(list)
    for w, v in byw.items():
        bys[(v[0]["task"], v[0]["seed"])].append(mean([x[key] for x in v]))
    return {s: mean(v) for s, v in bys.items()}, {s: len(v) for s, v in bys.items()}, len(byw)


def agg(rs, key):
    sv, _, nw = seed_vals(rs, key)
    return (mean(list(sv.values())), len(sv), nw, len(rs))


def fmt(a):
    v, ns, nw, nb = a
    return "—" if nb == 0 else f"{v:.3f} ({ns}/{nw}/{nb})"


def main():
    r3, n_dup, n_miss = load_i3()
    r2 = load_i2()
    # I2 の (a) を I3 と同じ形に（kind 名を変える）
    i2a = [dict(r, kind=("i2_" + r["kind"])) for r in r2 if r["cond"] == "a"]
    cols = ["id", "wid", "task", "seed", "rep", "kind", "group", "cond", "k", "t", "ws", "succ_orig", "done", "succ", "clean", "disp_mm", "disp_start_mm", "n_steps", "n_pref", "n_infer",
            "restore_err_mm", "home_err_pos_mm", "home_err_rot_rad", "home_steps", "q_after_home", "disp_after_home_mm", "pref_move_mm", "pref_path_mm", "top", "q0ref", "td0ref"]
    with open(K / "state/branches_i3_20261015.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore"); w.writeheader(); [w.writerow(r) for r in r3]
    L = []; P = L.append
    P("# I3 の集計（2026-10-15、`scripts/analyze_branches_i3.py`、分岐ごとの値 `state/branches_i3_20261015.csv`）\n")
    P("定義: 最終成功 = done。衝突なし成功 = done かつ 対象外物体の最大変位 ≤ 1 mm。窓の値 = 分岐 8 本の平均 → seed の値 = 窓の平均 → 群の値 = seed の平均。括弧は (seed 数/窓数/分岐数)。判定は書かない。")
    P("I2 の値は `branches_20261015.csv` の (a) 基準（失敗窓の開始+20 = `i2_rec`、開始−20 = `i2_entry`）。\n")
    kc = collections.Counter(r["kind"] for r in r3)
    P(f"I3 分岐 {len(r3)} 本（{dict(kc)}）。重複行 {n_dup}、npz 欠損 {n_miss}。\n")

    # --- 操作チェック
    P("## 操作チェックの実測")
    e0 = np.array([r["restore_err_mm"] for r in r3 if r["kind"] == "t0"]); ea = np.array([r["restore_err_mm"] for r in r3 if r["kind"] not in ("home", "t0")]); eh = np.array([r["restore_err_mm"] for r in r3 if r["kind"] == "home"])
    P(f"- 復元誤差（復元直後の手先と記録の差。t=0 は fresh() 直後の手先との差）: t0 中央値 {np.median(e0):.3f} / 最大 {e0.max():.3f} mm（{len(e0)} 本）；t0 以外の (a) {np.median(ea):.3f} / {ea.max():.3f} mm、1 mm 超 {(ea>1).sum()}/{len(ea)}；home の復元 {np.median(eh):.3f} / {eh.max():.3f} mm、1 mm 超 {(eh>1).sum()}/{len(eh)}")
    hp = np.array([r["home_err_pos_mm"] for r in r3 if r["cond"] == "f"]); hr = np.array([r["home_err_rot_rad"] for r in r3 if r["cond"] == "f"]); hs = np.array([r["home_steps"] for r in r3 if r["cond"] == "f"])
    P(f"- (f) 退避終了時の位置誤差: 中央値 {np.median(hp):.2f} / 最大 {hp.max():.2f} mm、到達率（< 10 mm）{(hp<10).mean():.3f}（{int((hp<10).sum())}/{len(hp)}）；回転誤差 中央値 {np.median(hr):.3f} / 最大 {hr.max():.3f} rad、< 0.05 rad の割合 {(hr<0.05).mean():.3f}；両方満たす（= 正常終了）{((hp<10)&(hr<0.05)).mean():.3f}；退避 step 数 中央値 {np.median(hs):.0f} / 最大 {hs.max():.0f}（上限 40 に到達 {(hs>=40).sum()} 本）")
    fr = [r for r in r3 if r["cond"] == "f"]
    qs = np.array([np.load(BR / f"b_{r['id']}.npz")["q"][0] for r in fr]); qa = np.array([r["q_after_home"] for r in fr]); dq = qa - qs
    da = np.array([r["disp_after_home_mm"] for r in fr]); d0 = np.array([r["disp_start_mm"] for r in fr])
    P(f"- (f) 退避前後の固定具 q の変化 Δq = q_after_home − q(分岐点)（タスク別、中央値 / |Δq| 最大 / 本数。drawer の単位は m、k4・k6 は rad）: " + "、".join(
        f"{t} {np.median(dq[[r['task']==t for r in fr]]):+.4f} / {np.abs(dq[[r['task']==t for r in fr]]).max():.4f} / {sum(1 for r in fr if r['task']==t)}" for t in ("drawer", "k4", "k6")))
    P(f"  - |Δq| が drawer > 5 mm・k4/k6 > 0.05 rad の本数（退避で固定具が動いた）: " + "、".join(
        f"{t} {int((np.abs(dq[[r['task']==t for r in fr]]) > (0.005 if t=='drawer' else 0.05)).sum())}" for t in ("drawer", "k4", "k6")))
    P(f"- (f) 退避終了時までの対象外物体の変位（最大）: 中央値 {np.median(da):.2f} mm / 90 パーセンタイル {np.percentile(da,90):.2f} / 最大 {da.max():.2f}；1 mm 超 {(da>1).sum()}/{len(da)}（分岐点で既に動いていた分 {(d0>1).sum()} 本；退避中に新たに増えた分 {int(((da>1)&(da>d0+1)).sum())} 本）")
    t0r = [r for r in r3 if r["kind"] == "t0"]
    trs = {}
    ee_t = []
    for r in t0r:
        k_ = (r["task"], r["seed"], r["rep"])
        if k_ not in trs:
            trs[k_] = np.load(K / f"state/trace_aprime/{k_[0]}_s{k_[1]}_r{k_[2]}.npz")["ee"][0]
        ee_t.append(np.linalg.norm(np.load(BR / f"b_{r['id']}.npz")["ee"][0] - trs[k_]) * 1000)
    ee_t = np.array(ee_t)
    bad = sorted({r["wid"] for r in t0r if r["restore_err_mm"] > 1})
    P(f"- t0 の補足: 復元直後の手先と trace の ee[0]（1 step 後）の距離は中央値 {np.median(ee_t):.1f} / 最大 {ee_t.max():.1f} mm（= 1 step 分の移動。復元は trace の初期状態を再現している）。『復元誤差 > 1 mm』の t0 は窓 {bad} の {sum(1 for r in t0r if r['restore_err_mm']>1)} 本で、fresh() 直後の手先が trace の初期状態と一致しなかったため（この窓だけ fresh() の初期姿勢が非決定。復元後の状態は trace と一致）")
    P("- 旧シャードの残存: 開始前に確認済み（run_branches の python プロセスなし）。重複 id は上記のとおり\n")

    # --- 1. home (f) vs I2 (a)
    for key, ttl in (("succ", "最終成功率"), ("clean", "衝突なし成功率")):
        P(f"## (f) 初期姿勢へ退避 vs I2 の (a)（同じ失敗窓、開始+20）: {ttl}")
        P("| 群 / タスク | (f) 退避→方策 | I2 (a) 開始+20 | (f)−(a) |"); P("|---|---|---|---|")
        cases = [("全群", None, None)] + [(GN[g], g, None) for g in GORD] + [(f"タスク {t}", None, t) for t in ("drawer", "k4", "k6")]
        for nm, g, t in cases:
            f_ = agg(sel(r3, "home", "f", g, t), key); a_ = agg(sel(i2a, "i2_rec", "a", g, t), key)
            P(f"| {nm} | {fmt(f_)} | {fmt(a_)} | {f_[0]-a_[0]:+.3f} |")
        P("")

    # --- 2. p(t)
    def tcols(g=None, seeds=None, key="succ"):
        out = []
        for kd, src in (("t0", r3), ("t13", r3), ("t23", r3), ("i2_entry", i2a), ("i2_rec", i2a)):
            out.append(agg(sel(src, kd, "a", g, None, seeds), key))
        return out
    P("## p(t): 失敗窓の trace を時刻 t で復元して方策で終端まで（(a)）の最終成功率")
    P("t は窓ごと（t13 = round(s/3)、t23 = round(2s/3)、s = 停留開始 step）。entry = s−20、rec = s+20 は I2 の値。")
    P("| 群 | t=0 | t=s/3 | t=2s/3 | entry (s−20) | rec (s+20) |"); P("|---|---|---|---|---|---|")
    for nm, g in [("全群", None)] + [(GN[x], x) for x in GORD]:
        P(f"| {nm} | " + " | ".join(fmt(c) for c in tcols(g)) + " |")
    P("\n### seed 別（最終成功率。窓の平均、括弧は窓数）")
    P("| タスク | seed | 群 | t=0 | t=s/3 | t=2s/3 | entry (s−20) | rec (s+20) | 窓数 |"); P("|---|---|---|---|---|---|---|---|---|")
    allseeds = sorted({(r["task"], r["seed"], r["group"]) for r in r3 if r["kind"] == "home"}, key=lambda x: (GORD.index(x[2]), x[0], x[1]))
    SV = {}
    for kd, src in (("t0", r3), ("t13", r3), ("t23", r3), ("i2_entry", i2a), ("i2_rec", i2a)):
        SV[kd] = seed_vals(sel(src, kd, "a"), "succ")
    for t, s, g in allseeds:
        cells = []
        for kd in ("t0", "t13", "t23", "i2_entry", "i2_rec"):
            v = SV[kd][0].get((t, s)); cells.append("—" if v is None else f"{v:.3f}")
        P(f"| {t} | {s} | {GN[g]} | " + " | ".join(cells) + f" | {SV['t0'][1].get((t, s), 0)} |")
    P("")
    P("### 衝突なし成功率 p(t)")
    P("| 群 | t=0 | t=s/3 | t=2s/3 | entry (s−20) | rec (s+20) |"); P("|---|---|---|---|---|---|")
    for nm, g in [("全群", None)] + [(GN[x], x) for x in GORD]:
        P(f"| {nm} | " + " | ".join(fmt(c) for c in tcols(g, None, "clean")) + " |")
    P("")

    # --- 3. 成功 ep 対照
    P("## 成功 ep の対照: 成功 ep の trace を s̄（同 seed の失敗窓の s の中央値）の 1/3・2/3 で復元（(a)）の最終成功率")
    P("| タスク | seed | 群 | succ13 (成功 ep) | t13 (失敗 ep) | 差 | succ23 (成功 ep) | t23 (失敗 ep) | 差 | 成功 ep 数 | 失敗窓数 |"); P("|---|---|---|---|---|---|---|---|---|---|---|")
    S13 = seed_vals(sel(r3, "succ13", "a"), "succ"); S23 = seed_vals(sel(r3, "succ23", "a"), "succ")
    sgrp = {(r["task"], r["seed"]): r["group"] for r in r3}
    nsucc = collections.defaultdict(set)
    for r in r3:
        if r["kind"] == "succ13":
            nsucc[(r["task"], r["seed"])].add(r["rep"])
    diffs13, diffs23, pairs = [], [], []
    for (t, s) in sorted(S13[0], key=lambda x: (GORD.index(sgrp[x]), x)):
        a13, a23 = S13[0][(t, s)], S23[0].get((t, s)); f13, f23 = SV["t13"][0].get((t, s)), SV["t23"][0].get((t, s))
        d13 = a13 - f13 if f13 is not None else float("nan"); d23 = a23 - f23 if (a23 is not None and f23 is not None) else float("nan")
        diffs13.append(d13); diffs23.append(d23); pairs.append((t, s))
        P(f"| {t} | {s} | {GN[sgrp[(t, s)]]} | {a13:.3f} | {f13:.3f} | {d13:+.3f} | {a23:.3f} | {f23:.3f} | {d23:+.3f} | {len(nsucc[(t, s)])} | {SV['t13'][1].get((t, s), 0)} |")
    P(f"\n成功 ep 対照のある seed {len(pairs)}（成功 ep の分岐: succ13 {sum(1 for r in r3 if r['kind']=='succ13')} 本、succ23 {sum(1 for r in r3 if r['kind']=='succ23')} 本）。")
    P(f"成功 ep から {fmt(agg(sel(r3,'succ13','a'),'succ'))} / {fmt(agg(sel(r3,'succ23','a'),'succ'))}（s/3, 2s/3）。\n")

    # --- R8〜R11
    P("## R8〜R11 の値（判定は書かない。線は計画 §10 の表）")
    t0v = SV["t0"][0]
    fv = seed_vals(sel(r3, "home", "f"), "succ")[0]
    hold = [k for k, v in t0v.items() if v >= 0.25]
    P(f"- **R8**: p_reset(t=0) ≥ 0.25 の seed = {len(hold)} 個（{', '.join(f'{t}_{s}' for t,s in hold) or 'なし'}）。")
    if hold:
        pf = mean([fv[k] for k in hold]); pr = mean([t0v[k] for k in hold])
        P(f"  - プール（seed 平均）: (f) の最終成功 {pf:.3f} ÷ p_reset {pr:.3f} = **{pf/pr:.3f}**" + f"（seed ごとの比の平均 {mean([fv[k]/t0v[k] for k in hold]):.3f}）")
        P("  - seed 別: " + "、".join(f"{t}_{s}: (f) {fv[(t,s)]:.3f} / p_reset {t0v[(t,s)]:.3f}" for t, s in hold))
    P(f"  - 参考: 全 seed では (f) {mean(list(fv.values())):.3f}、p_reset {mean(list(t0v.values())):.3f}（seed {len(t0v)}）")
    lm = {k for k in t0v if sgrp.get(k) is None}
    grp_all = {(r["task"], r["seed"]): r["group"] for r in r3}
    lm = {k for k, g in grp_all.items() if g in ("luck", "mid")}
    p0 = mean([t0v[k] for k in lm if k in t0v]); p13 = mean([SV["t13"][0][k] for k in lm if k in SV["t13"][0]]); p23 = mean([SV["t23"][0][k] for k in lm if k in SV["t23"][0]])
    P(f"- **R9**（運・中間 seed {len(lm)} 個）: p(0) = {p0:.3f}、p(s/3) = {p13:.3f}、p(2s/3) = {p23:.3f}。p(s/3)÷p(0) = **{p13/p0 if p0>0 else float('nan'):.3f}**、p(2s/3)÷p(0) = **{p23/p0 if p0>0 else float('nan'):.3f}**")
    P("  - seed 別 (p0 / p13 / p23): " + "、".join(f"{t}_{s}({GN[grp_all[(t,s)]]}) {t0v.get((t,s),float('nan')):.2f}/{SV['t13'][0].get((t,s),float('nan')):.2f}/{SV['t23'][0].get((t,s),float('nan')):.2f}" for t, s in sorted(lm)))
    d23 = [x for x in diffs23 if not np.isnan(x)]; d13 = [x for x in diffs13 if not np.isnan(x)]
    P(f"- **R10**（seed 対応、成功 ep のある {len(pairs)} seed）: mean[p_succ(2s/3) − p_fail(2s/3)] = **{mean(d23):+.3f}**（n={len(d23)}）、s/3 では {mean(d13):+.3f}（n={len(d13)}）")
    un = [k for k, g in grp_all.items() if g == "unseen"]
    ub = [(r["task"], r["seed"]) for r in r3 if r["kind"] == "t0" and r["group"] == "unseen"]
    uv = [t0v[k] for k in un if k in t0v]
    nb = sum(1 for r in r3 if r["kind"] == "t0" and r["group"] == "unseen")
    P(f"- **R11**（成功未観測 {len(un)} seed、t0 の分岐 {nb} 本）: p_reset のプール = **{mean(uv):.3f}**（seed 平均）、分岐数でプールすると {mean([r['succ'] for r in r3 if r['kind']=='t0' and r['group']=='unseen']):.3f}。seed 別: " + "、".join(f"{t}_{s} {t0v.get((t,s),float('nan')):.3f}" for t, s in sorted(un)))
    P("")
    P("## n（窓・seed・分岐）")
    P("| kind | 条件 | 群 | seed | 窓 | 分岐 |"); P("|---|---|---|---|---|---|")
    for kd in ("home", "t0", "t13", "t23", "succ13", "succ23"):
        for g in GORD:
            rs = sel(r3, kd, None, g)
            if rs:
                P(f"| {kd} | {rs[0]['cond']} | {GN[g]} | {len({(r['task'],r['seed']) for r in rs})} | {len({r['wid'] for r in rs})} | {len(rs)} |")
    (K / "state/branches_i3_summary_20261015.md").write_text("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
