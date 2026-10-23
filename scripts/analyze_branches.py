"""I1・I2 の分岐（run_branches.py の出力）を集計し、計画 v2 §8 の R1〜R7 を判定する。CPU のみ。
使い方: nice -n 10 /path/to/work/.venv/bin/python kadai/scripts/analyze_branches.py
入力: kadai/state/branches_20261015/{summary_*.jsonl, b_*.npz, i1_*.npz}、kadai/state/trace_aprime/*.npz（基準の |a_pos| 用）、aprime_windows_20261014.csv
出力: kadai/state/branches_summary_20261015.md, kadai/state/branches_20261015.csv（1 分岐 1 行）

定義:
  prog = max(clip((q-q0)/(thr-q0),0,1), clip(1-tgt_dist/tgt_dist0,0,1))（analyze_aprime.py と同一。q0・tgt_dist0 は元 trace の step 0 の値）
  Δprog60 = 分岐点（復元直後）から 60 step 後の prog − 分岐点の prog。前置き（toward 8 step / retreat 最大 35 step）も 60 step に数える（同じ step 予算）。
            感度分析として「前置き終了後 60 step」版も出す。done で終わった分岐は以後 prog = 1 とする。60 step 未満で終端した分岐は最後の値を延長。
  回復 = Δprog60 >= 0.05。成功 = done。衝突なし成功 = done かつ 対象外物体の最大移動 <= 1 mm（元の評価の clean と同じ。初期配置との比較）。
  窓の値 = 窓内の分岐の平均。seed の値 = その seed の窓の平均。群の値 = 群内 seed の平均（seed 単位）。
"""
import csv, json, glob, collections
from pathlib import Path
import numpy as np

K = Path(__file__).resolve().parents[1]
BR = K / "state/branches_20261015"
THR = {"drawer": -0.14, "k4": -0.005, "k6": -0.005}
GN = {"unseen": "成功未観測", "luck": "運", "mid": "中間", "hold": "保留"}
GORD = ["unseen", "luck", "mid", "hold"]
KORD = ["rec", "fp", "entry"]
KN = {"rec": "失敗窓(開始+20)", "fp": "偽陽性窓(開始+20)", "entry": "失敗窓の進入(開始-20)"}
CORD = ["a", "b", "d", "e"]
CN = {"a": "(a)基準", "b": "(b)replan1", "d": "(d)toward", "e": "(e)retreat"}
THR_REC = 0.05


def prog_arr(q, td, q0, td0, task, done):
    pa = np.clip((q - q0) / (THR[task] - q0), 0, 1); pb = np.clip(1 - td / td0, 0, 1)
    p = np.maximum(pa, pb)
    if done:
        p = p.copy(); p[-1] = 1.0
    return p


def at(p, k, done):
    if k >= len(p):
        return 1.0 if done else float(p[-1])
    return float(p[k])


def load_branches():
    rows = []
    for f in sorted(glob.glob(str(BR / "summary_*.jsonl"))):
        if "check" in f:
            continue
        rows += [json.loads(l) for l in open(f) if l.strip()]
    # 重複 id（シャード数変更時に同じ分岐が 2 度走った）は最後の行を採る。npz と行の整合（step 数・done）を確認し、合わないものは捨てる
    last = {}
    for r in rows:
        last[r["id"]] = r
    n_dup = len(rows) - len(last); rows = list(last.values()); keep = []
    for r in rows:
        if not (BR / f"b_{r['id']}.npz").exists():
            continue
        z = np.load(BR / f"b_{r['id']}.npz")
        if len(z["ee"]) - 1 != r["n_steps"] or len(z["q"]) - 1 != r["n_steps"]:
            continue
        keep.append(r)
    print(f"重複 {n_dup} 本を統合、npz と不整合で除外 {len(rows) - len(keep)} 本")
    rows = keep
    for r in rows:
        z = np.load(BR / f"b_{r['id']}.npz")
        p = prog_arr(z["q"], z["td"], r["q0ref"], r["td0ref"], r["task"], r["done"])
        n0 = int(r["n_pref"])
        r["prog0"] = float(p[0]); r["dprog60"] = at(p, 60, r["done"]) - p[0]
        r["dprog60_post"] = at(p, n0 + 60, r["done"]) - p[n0]
        r["prog_end"] = 1.0 if r["done"] else float(p[-1])
        r["rec"] = int(r["dprog60"] >= THR_REC); r["rec_post"] = int(r["dprog60_post"] >= THR_REC)
        r["succ"] = int(r["done"]); r["clean"] = int(r["done"] and r["disp_mm"] <= 1.0)
        r["moved_nontarget"] = int(r["disp_mm"] > 1.0)
    return rows


def mean(x):
    x = [v for v in x if v is not None and not (isinstance(v, float) and np.isnan(v))]
    return float(np.mean(x)) if x else float("nan")


def agg(rows, key, kind, cond, group=None, task=None):
    """窓 → seed → 群の順の平均。返り値 (値, seed 数, 窓 数, 分岐数)。"""
    sel = [r for r in rows if r["kind"] == kind and r["cond"] == cond and (group is None or r["group"] == group) and (task is None or r["task"] == task)]
    byw = collections.defaultdict(list)
    for r in sel:
        byw[r["wid"]].append(r)
    wv = {w: (mean([x[key] for x in v]), v[0]["task"], v[0]["seed"]) for w, v in byw.items()}
    bys = collections.defaultdict(list)
    for w, (v, t, s) in wv.items():
        bys[(t, s)].append(v)
    sv = [mean(v) for v in bys.values()]
    return (mean(sv), len(bys), len(wv), len(sel))


def fmt(a):
    v, ns, nw, nb = a
    return "—" if nb == 0 else f"{v:.3f} (seed {ns}/窓 {nw}/分岐 {nb})"


def verdict(v, sup, ref):
    return "支持" if sup(v) else ("反証" if ref(v) else "保留")


def base_apos():
    base = {}
    for t in THR:
        P, A = [], []
        for f in sorted(glob.glob(str(K / "state/trace_aprime/*.npz"))):
            z = np.load(f)
            if str(z["task"]) != t or not bool(z["success"]):
                continue
            q = z["q"]; q0 = q[0]; td = z["tgt_dist"]
            P.append(np.maximum(np.clip((q - q0) / (THR[t] - q0), 0, 1), np.clip(1 - td / td[0], 0, 1))); A.append(np.linalg.norm(z["act_exec"][:, :3], axis=1))
        base[t] = (np.concatenate(P), np.concatenate(A))
    return base


def i1_stats(base):
    wins = list(csv.DictReader(open(K / "state/aprime_windows_20261014.csv")))
    wd = {(w["task"], int(w["seed"]), int(w["rep"])): w for w in wins}
    out = []
    for f in sorted(glob.glob(str(BR / "i1_*.npz"))):
        if "check" in f:
            continue
        z = np.load(f); task, seed, rep, kind, t = str(z["task"]), int(z["seed"]), int(z["rep"]), str(z["kind"]), int(z["t"])
        w = wd[(task, seed, rep)]
        ch = z["chunks"][:, :5, :]   # 実行される先頭 5 step（replan 5）
        d = ch[:, :, :3].sum(axis=1); nrm = np.linalg.norm(d, axis=1)
        ok = nrm > 1e-6; d = d[ok] / nrm[ok, None]
        C = d @ d.T; iu = np.triu_indices(len(d), 1); cos = float(C[iu].mean()) if len(d) > 1 else float("nan")
        apos = float(np.median(np.linalg.norm(ch[:, :, :3], axis=2)))
        tr = np.load(K / f"state/trace_aprime/{task}_s{seed}_r{rep}.npz")
        q = tr["q"]; q0 = q[0]; td = tr["tgt_dist"]
        pr = float(max(np.clip((q[t - 1] - q0) / (THR[task] - q0), 0, 1), np.clip(1 - td[t - 1] / td[0], 0, 1)))
        bp, ba = base[task]; m = np.abs(bp - pr) <= 0.1
        b = float(np.median(ba[m])) if m.sum() >= 5 else float("nan")
        # 先頭チャンク全 10 step 版の方向（感度）
        d10 = z["chunks"][:, :, :3].sum(axis=1); n10 = np.linalg.norm(d10, axis=1); d10 = d10[n10 > 1e-6] / n10[n10 > 1e-6, None]
        cos10 = float((d10 @ d10.T)[np.triu_indices(len(d10), 1)].mean())
        out.append(dict(wid=f"{task}_s{seed}_r{rep}_{kind}", task=task, seed=seed, rep=rep, kind=kind, group=w["group"], cos=cos, cos10=cos10, apos=apos, base=b, ratio=apos / b if b > 0 else float("nan"), prog=pr))
    return out


def main():
    rows = load_branches()
    base = base_apos()
    i1 = i1_stats(base)
    cols = ["id", "wid", "task", "seed", "rep", "kind", "group", "cond", "k", "t", "prog0", "dprog60", "dprog60_post", "rec", "rec_post", "succ", "clean", "disp_mm", "disp_start_mm", "top", "n_infer", "n_steps", "n_pref", "pref_move_mm", "pref_path_mm", "restore_err_mm", "prog_end"]
    with open(K / "state/branches_20261015.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore"); w.writeheader(); [w.writerow(r) for r in rows]
    L = []
    P = L.append
    P("# I1・I2 の集計（2026-10-15、`scripts/analyze_branches.py`、分岐ごとの値 `state/branches_20261015.csv`）\n")
    P("定義: Δprog60 = 分岐点から 60 step 後の prog − 分岐点の prog（前置きも 60 step に数える）。回復 = Δprog60 ≥ 0.05。窓の値 = 分岐 8（(b) は 4）本の平均 → seed の値 = 窓の平均 → 群の値 = seed の平均。括弧は (seed 数 / 窓数 / 分岐数)。\n")
    P(f"分岐 {len(rows)} 本、I1 窓 {len(i1)} 本。\n")
    # 操作チェック（本番全分岐の実測）
    P("## 操作チェック（本番全分岐の実測）")
    err = np.array([r["restore_err_mm"] for r in rows])
    P(f"- 復元直後の手先と記録の差: 最大 {err.max():.3f} mm / 中央値 {np.median(err):.3f} mm / 1 mm 超 {int((err>1).sum())} 本（全 {len(err)} 本）")
    for c in ["d", "e"]:
        v = np.array([r["pref_move_mm"] for r in rows if r["cond"] == c])
        P(f"- ({c}) 前置き中の手先移動（開始点からの最大距離）: 最小 {v.min():.1f} / 中央値 {np.median(v):.1f} mm、5 mm 未満 {int((v<5).sum())}/{len(v)} 本")
    ia = np.mean([r["n_infer"] / max(r["n_steps"], 1) for r in rows if r["cond"] == "a"]); ib = np.mean([r["n_infer"] / max(r["n_steps"], 1) for r in rows if r["cond"] == "b"])
    ra = collections.defaultdict(list)
    for r in rows:
        ra[(r["wid"], r["cond"])].append(r["n_infer"] / max(r["n_steps"], 1))
    ratios = [np.mean(ra[(w, "b")]) / np.mean(ra[(w, "a")]) for w in {k[0] for k in ra} if (w, "b") in ra and (w, "a") in ra]
    P(f"- 推論回数 / 実行 step: (a) {ia:.3f}、(b) {ib:.3f}（窓ごとの比 (b)/(a) 中央値 {np.median(ratios):.2f}）\n")
    # 窓の種類 × 群 × 条件
    for key, ttl in [("rec", "回復率（Δprog60 ≥ 0.05）"), ("rec_post", "回復率（前置き終了後 60 step。感度）"), ("succ", "最終成功率"), ("clean", "衝突なし成功率（done かつ 対象外物体 ≤ 1 mm）"), ("dprog60", "Δprog60 の平均")]:
        P(f"## {ttl}")
        P("| 窓の種類 | 群 | " + " | ".join(CN[c] for c in CORD) + " |"); P("|---|---|" + "---|" * len(CORD))
        for kd in KORD:
            for g in [None] + GORD:
                cells = [fmt(agg(rows, key, kd, c, g)) for c in CORD]
                if all(x == "—" for x in cells):
                    continue
                P(f"| {KN[kd]} | {'全群' if g is None else GN[g]} | " + " | ".join(cells) + " |")
        P("")
    P("## タスク別（回復率）")
    P("| 窓の種類 | タスク | " + " | ".join(CN[c] for c in CORD) + " |"); P("|---|---|" + "---|" * len(CORD))
    for kd in KORD:
        for t in THR:
            cells = [fmt(agg(rows, "rec", kd, c, None, t)) for c in CORD]
            if all(x == "—" for x in cells):
                continue
            P(f"| {KN[kd]} | {t} | " + " | ".join(cells) + " |")
    P("")
    # R1〜R7
    P("## R1〜R7（計画 v2 §8 の線）")
    A = lambda kd, c, g=None, key="rec": agg(rows, key, kd, c, g)
    r1 = A("rec", "a"); r2 = A("fp", "a"); ea = A("rec", "e"); da = A("rec", "d"); ba_ = A("rec", "b")
    r3 = ea[0] - r1[0]; r4 = ea[0] - da[0]; r5 = ba_[0] - r1[0]
    r7 = A("entry", "a", "unseen")
    P("| # | 値 | 判定 | 備考 |"); P("|---|---|---|---|")
    P(f"| R1 失敗窓 (a) 回復率 | {fmt(r1)} | {verdict(r1[0], lambda v: v < 0.3, lambda v: v > 0.5)} | 群別: " + "、".join(f"{GN[g]} {fmt(A('rec','a',g))}" for g in GORD) + " |")
    P(f"| R2 偽陽性窓 (a) 回復率 | {fmt(r2)} | {verdict(r2[0], lambda v: v > 0.7, lambda v: v < 0.5)} | |")
    P(f"| R3 (e)−(a)（失敗窓） | {r3:+.3f}（(e) {fmt(ea)}） | {verdict(r3, lambda v: v > 0.3, lambda v: v < 0.1)} | 群別 (e)−(a): " + "、".join(f"{GN[g]} {A('rec','e',g)[0]-A('rec','a',g)[0]:+.3f}" for g in GORD) + " |")
    P(f"| R4 (e)−(d)（失敗窓） | {r4:+.3f}（(d) {fmt(da)}） | " + ("支持" if r4 > 0.2 else ("反証" if r4 <= 0 else "保留")) + " | 線: 支持 > +0.2、反証 (d) ≥ (e) |")
    P(f"| R5 (b)−(a)（失敗窓） | {r5:+.3f}（(b) {fmt(ba_)}） | " + ("支持" if abs(r5) < 0.15 else ("反証" if abs(r5) > 0.3 else "保留")) + " | |")
    # R6
    fi = [x for x in i1 if x["kind"] == "rec"]
    def seedmean(lst, k):
        bys = collections.defaultdict(list)
        for x in lst:
            bys[(x["task"], x["seed"])].append(x[k])
        return mean([mean(v) for v in bys.values()]), len(bys)
    cs, ns = seedmean(fi, "cos"); rs, _ = seedmean(fi, "ratio")
    both = mean([float(x["cos"] > 0.7 and x["ratio"] > 0.5) for x in fi])
    r6 = "支持" if (cs > 0.7 and rs > 0.5) else ("反証" if cs < 0.4 else "保留")
    P(f"| R6 I1 失敗窓 | 平均対コサイン {cs:.3f}、|a_pos|/基準 {rs:.3f}、両方満たす窓 {both:.2f}（seed {ns}/窓 {len(fi)}） | {r6} | |")
    P(f"| R7 進入(−20)からの (a)・成功未観測 | {fmt(r7)} | {verdict(r7[0], lambda v: v < 0.3, lambda v: v > 0.5)} | 全群: {fmt(A('entry','a'))} |")
    P("")
    P("## I1: 再サンプルの分散（窓ごと 16 チャンク。方向 = 先頭 5 step の位置指令の和、対コサインの平均。基準 = 同タスクの成功 ep で prog が ±0.1 の step の |a_pos| 中央値）")
    P("| 窓の種類 | 群 | 平均対コサイン | (全 10 step 版) | |a_pos| 中央値 | 基準 | 比 | n(seed/窓) |"); P("|---|---|---|---|---|---|---|---|")
    for kd in KORD:
        for g in [None] + GORD:
            lst = [x for x in i1 if x["kind"] == kd and (g is None or x["group"] == g)]
            if not lst:
                continue
            c1, ns1 = seedmean(lst, "cos"); c2, _ = seedmean(lst, "cos10"); a1, _ = seedmean(lst, "apos"); b1, _ = seedmean(lst, "base"); r1_, _ = seedmean(lst, "ratio")
            P(f"| {KN[kd]} | {'全群' if g is None else GN[g]} | {c1:.3f} | {c2:.3f} | {a1:.3f} | {b1:.3f} | {r1_:.3f} | {ns1}/{len(lst)} |")
    P("\n## n（窓・seed）")
    P("| 窓の種類 | 群 | タスク | seed 数 | 窓数 |"); P("|---|---|---|---|---|")
    cnt = collections.defaultdict(set); cw = collections.defaultdict(set)
    for r in rows:
        cnt[(r["kind"], r["group"], r["task"])].add((r["task"], r["seed"])); cw[(r["kind"], r["group"], r["task"])].add(r["wid"])
    for k_ in sorted(cnt, key=lambda x: (KORD.index(x[0]), GORD.index(x[1]), x[2])):
        P(f"| {KN[k_[0]]} | {GN[k_[1]]} | {k_[2]} | {len(cnt[k_])} | {len(cw[k_])} |")
    (K / "state/branches_summary_20261015.md").write_text("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
