"""I4（計画 v2 §14、腕の状態の入れ替え）の分岐を集計する。CPU のみ。判定は書かず値だけ出す。
使い方: python kadai/scripts/analyze_branches_i4.py
入力: kadai/state/branches_i4_20261015/summary_s*.jsonl + b_*.npz、kadai/state/branches_i3_20261015.csv（(a) = t13/t23、p_succ = succ13/succ23）
出力: kadai/state/branches_i4_summary_20261015.md, kadai/state/branches_i4_20261015.csv
窓 = (失敗窓 f, 成功 ep srep, t ∈ {t13, t23})（t は失敗窓のもの）。窓の値 = 分岐 8 本の平均 → seed の値 = 窓の平均 → 群/タスクの値 = seed の平均。括弧は (seed 数/窓数/分岐数)。
(a) は窓ごとに「その失敗窓の I3 の t13/t23 の最終成功」、p_succ は「その成功 ep の I3 の succ13/succ23」を窓に対応づけて平均する（差は窓対応の平均）。
最終成功 = done。衝突なし成功 = done かつ 対象外物体の最大変位 <= 1 mm。貫入 = 入れ替え直後の接触の法線力 f0 > 20 N。
"""
import csv, json, glob, collections
from pathlib import Path
import numpy as np

K = Path(__file__).resolve().parents[1]
BR = K / "state/branches_i4_20261015"
THR = {"drawer": -0.14, "k4": -0.005, "k6": -0.005}
GN = {"luck": "運", "mid": "中間", "hold": "保留"}
FTH = 20.0


def mean(x):
    x = [v for v in x if v is not None and not (isinstance(v, float) and np.isnan(v))]
    return float(np.mean(x)) if x else float("nan")


def prog_arr(q, td, q0, td0, task, done):
    p = np.maximum(np.clip((q - q0) / (THR[task] - q0), 0, 1), np.clip(1 - td / td0, 0, 1))
    if done:
        p = p.copy(); p[-1] = 1.0
    return p


def load():
    rows = []
    for f in sorted(glob.glob(str(BR / "summary_s*.jsonl"))):
        rows += [json.loads(l) for l in open(f) if l.strip()]
    last = {r["id"]: r for r in rows}
    n_dup = len(rows) - len(last)
    keep = [r for r in last.values() if (BR / f"b_{r['id']}.npz").exists()]
    for r in keep:
        z = np.load(BR / f"b_{r['id']}.npz")
        p = prog_arr(z["q"], z["td"], r["q0ref"], r["td0ref"], r["task"], r["done"])
        r["dprog60"] = (1.0 if (r["done"] and len(p) <= 60) else float(p[min(60, len(p) - 1)])) - float(p[0])
        r["succ"] = int(r["done"]); r["clean"] = int(r["done"] and r["disp_mm"] <= 1.0)
        r["pen"] = int(r["f0"] > FTH)
    return keep, n_dup, len(last) - len(keep)


def load_i3():
    rows = list(csv.DictReader(open(K / "state/branches_i3_20261015.csv")))
    a = collections.defaultdict(list); s = collections.defaultdict(list)
    for r in rows:
        if r["cond"] != "a":
            continue
        v = (int(float(r["succ"])), int(float(r["clean"])))
        if r["kind"] in ("t13", "t23"):
            a[(r["wid"], "succ")].append(v[0]); a[(r["wid"], "clean")].append(v[1])
        elif r["kind"] in ("succ13", "succ23"):
            kd = "t13" if r["kind"] == "succ13" else "t23"
            s[(r["task"], int(r["seed"]), int(r["rep"]), kd, "succ")].append(v[0]); s[(r["task"], int(r["seed"]), int(r["rep"]), kd, "clean")].append(v[1])
    return {k: mean(v) for k, v in a.items()}, {k: mean(v) for k, v in s.items()}


def windows(rows, key, excl):
    """窓ごとの値: {wid: dict(task, seed, group, kind, s, r, a, ps, n_s, n_r)}。excl=True なら f0 > 20 N の分岐を除く。"""
    A, S = I3
    by = collections.defaultdict(lambda: {"s": [], "r": []})
    meta = {}
    for r in rows:
        if excl and r["pen"]:
            continue
        by[r["wid"]][r["cond"]].append(r[key]); meta[r["wid"]] = r
    out = {}
    for w, d in by.items():
        m = meta[w]
        out[w] = dict(task=m["task"], seed=m["seed"], group=m["group"], kind=m["kind"], s=mean(d["s"]), r=mean(d["r"]), n_s=len(d["s"]), n_r=len(d["r"]),
                      a=A.get((m["fwid"], key)), ps=S.get((m["task"], m["seed"], m["srep"], m["kind"], key)))
    return out


def agg(W, kind, field, task=None, group=None, mask=None):
    """seed 平均。返り値 (値, seed 数, 窓数, 分岐数)。mask: 窓の絞り込み（field の窓値が nan の窓は落とす）。"""
    bys = collections.defaultdict(list); nb = 0; nw = 0
    for w, v in W.items():
        if v["kind"] != kind or (task and v["task"] != task) or (group and v["group"] != group):
            continue
        x = v[field]
        if x is None or np.isnan(x):
            continue
        bys[(v["task"], v["seed"])].append(x); nw += 1
        nb += v["n_s"] if field in ("s", "a") else v["n_r"] if field in ("r", "ps") else 0
    sv = {s: mean(x) for s, x in bys.items()}
    return mean(list(sv.values())), len(sv), nw, nb, sv


def diff(W, kind, f1, f2, task=None, group=None):
    """窓対応の差の seed 平均。"""
    bys = collections.defaultdict(list)
    for w, v in W.items():
        if v["kind"] != kind or (task and v["task"] != task) or (group and v["group"] != group):
            continue
        if v[f1] is None or v[f2] is None or np.isnan(v[f1]) or np.isnan(v[f2]):
            continue
        bys[(v["task"], v["seed"])].append(v[f1] - v[f2])
    sv = [mean(x) for x in bys.values()]
    return mean(sv), len(sv)


def fmt(a, nan="—"):
    v, ns, nw, nb = a[:4]
    return nan if nw == 0 else f"{v:.3f} ({ns}/{nw}/{nb})"


def main():
    global I3
    rows, n_dup, n_miss = load(); I3 = load_i3()
    cols = ["id", "wid", "fwid", "task", "seed", "rep", "srep", "kind", "group", "cond", "k", "t", "ws", "done", "succ", "clean", "disp_mm", "disp_start_mm", "n_steps", "n_infer", "dprog60",
            "ee_robot_err_mm", "ee_base_gap_mm", "q_base_err", "jq_robot_err", "swap_jerr_q", "swap_jerr_v", "swap_oerr_q", "f0", "f1", "pen", "top"]
    with open(K / "state/branches_i4_20261015.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore"); w.writeheader(); [w.writerow(r) for r in rows]
    L = []; P = L.append
    P("# I4 の集計（2026-10-15、`scripts/analyze_branches_i4.py`、分岐ごとの値 `state/branches_i4_20261015.csv`）\n")
    P("定義: 条件 (s) = base が失敗 ep の完全状態・ロボット関節（7 + 指 2 の qpos/qvel）だけ同 t の成功 ep の値。(r) = base が成功 ep・ロボットが失敗 ep の値。"
      "(a) = I3 の t13/t23（同じ失敗窓）の最終成功。p_succ(t) = I3 の succ13/succ23（同じ seed・同じ成功 ep）。窓の値 = 分岐 8 本の平均 → seed → タスク/群（seed 平均）。括弧は (seed 数/窓数/分岐数)。判定は書かない。")
    P("差は窓対応（(s) − その失敗窓の (a)、(r) − その成功 ep の p_succ）の seed 平均。『貫入除外』= 入れ替え直後の接触法線力 f0 > 20 N の分岐を除いた版（(a)・p_succ は除外しない）。\n")
    kc = collections.Counter((r["kind"], r["cond"]) for r in rows)
    P(f"I4 分岐 {len(rows)} 本（{dict(kc)}）。重複行 {n_dup}、npz 欠損 {n_miss}。窓 {len({r['wid'] for r in rows})}。\n")

    # --- 操作チェック
    P("## 操作チェックの実測")
    def stat(key, cond=None, kind=None):
        x = np.array([r[key] for r in rows if (cond is None or r["cond"] == cond) and (kind is None or r["kind"] == kind)], float)
        return x
    for cond in ("s", "r"):
        e = stat("ee_robot_err_mm", cond); g = stat("ee_base_gap_mm", cond); qe = np.abs(stat("q_base_err", cond)); je = stat("jq_robot_err", cond)
        P(f"- ({cond}) 入れ替え直後の手先位置 − robot 側 ep の trace の ee[t−1]: 中央値 {np.median(e):.4f} / 最大 {e.max():.4f} mm、> 1 mm {int((e>1).sum())}/{len(e)}。"
          f"（参考: 手先位置 − base 側の ee[t−1] = 入れ替えで動いた量 中央値 {np.median(g):.2f} / 90% {np.percentile(g,90):.2f} / 最大 {g.max():.2f} mm）")
        P(f"  - 固定具 q − base 側 q[t−1]: |差| 中央値 {np.median(qe):.2e} / 最大 {qe.max():.2e}（drawer は m、k4・k6 は rad）。関節角 − robot 側 jq[t−1]: 最大 {np.nanmax(je):.2e} rad")
        for tk in ("drawer", "k4", "k6"):
            qk = np.abs(np.array([r["q_base_err"] for r in rows if r["cond"] == cond and r["task"] == tk], float))
            if len(qk):
                P(f"    - {tk}: |Δq| 最大 {qk.max():.2e}")
    P(f"- flattened 状態の検算（ロボット関節の qpos / qvel が robot 側と一致、それ以外の qpos が base 側と一致）: |差| 最大 {stat('swap_jerr_q').max():.2e} / {stat('swap_jerr_v').max():.2e} / {stat('swap_oerr_q').max():.2e}")
    for kd in ("t13", "t23"):
        f0 = stat("f0", None, kd); f1 = stat("f1", None, kd)
        P(f"- 入れ替え直後の接触法線力 f0（{kd}、{len(f0)} 本）: 0 N が {int((f0==0).sum())} 本、中央値 {np.median(f0):.2f} / 90% {np.percentile(f0,90):.1f} / 最大 {f0.max():.1f} N、> 20 N {int((f0>FTH).sum())} 本"
          f"（(s) {int(((stat('f0','s',kd))>FTH).sum())}、(r) {int(((stat('f0','r',kd))>FTH).sum())}）；1 step 後 f1 の最大 {np.nanmax(f1):.1f} N、> 20 N {int((f1>FTH).sum())} 本")
    nstep = stat("n_steps")
    P(f"- 到達（終端まで走った/done）: done {int(sum(r['done'] for r in rows))}/{len(rows)}、250 step 打ち切り {int(((nstep + np.array([r['t'] for r in rows])) >= 250).sum())} 本、n_infer 中央値 {np.median(stat('n_infer')):.0f}\n")

    # --- 本表
    for key, ttl in (("succ", "最終成功率"), ("clean", "衝突なし成功率")):
        for excl in (False, True):
            W = windows(rows, key, excl)
            sfx = "（貫入除外 f0 > 20 N）" if excl else ""
            P(f"## {ttl}{sfx}: タスク × t")
            P("| タスク | t | (a) 失敗 ep 基準 | (s) 腕を成功側に | (s)−(a) | p_succ 成功 ep 基準 | (r) 腕を失敗側に | (r)−p_succ |"); P("|---|---|---|---|---|---|---|---|")
            for tk in (None, "k6", "drawer", "k4"):
                for kd in ("t13", "t23"):
                    ds = diff(W, kd, "s", "a", tk); dr = diff(W, kd, "r", "ps", tk)
                    P(f"| {tk or '全タスク'} | {'s/3' if kd=='t13' else '2s/3'} | {fmt(agg(W,kd,'a',tk))} | {fmt(agg(W,kd,'s',tk))} | {ds[0]:+.3f} (n={ds[1]}) | {fmt(agg(W,kd,'ps',tk))} | {fmt(agg(W,kd,'r',tk))} | {dr[0]:+.3f} (n={dr[1]}) |")
            P("")
            if key == "succ":
                P(f"### 群別{sfx}")
                P("| 群 | t | (a) | (s) | (s)−(a) | p_succ | (r) | (r)−p_succ |"); P("|---|---|---|---|---|---|---|---|")
                for g in ("luck", "mid", "hold"):
                    for kd in ("t13", "t23"):
                        ds = diff(W, kd, "s", "a", None, g); dr = diff(W, kd, "r", "ps", None, g)
                        P(f"| {GN[g]} | {'s/3' if kd=='t13' else '2s/3'} | {fmt(agg(W,kd,'a',None,g))} | {fmt(agg(W,kd,'s',None,g))} | {ds[0]:+.3f} | {fmt(agg(W,kd,'ps',None,g))} | {fmt(agg(W,kd,'r',None,g))} | {dr[0]:+.3f} |")
                P("")
    # 除外本数
    P("## 貫入として除外された分岐数（f0 > 20 N）")
    P("| タスク | t | 条件 | 除外 / 全分岐 | 全分岐が除外された窓 |"); P("|---|---|---|---|---|")
    for tk in ("k6", "drawer", "k4"):
        for kd in ("t13", "t23"):
            for c in ("s", "r"):
                rs = [r for r in rows if r["task"] == tk and r["kind"] == kd and r["cond"] == c]
                ws = collections.defaultdict(list)
                for r in rs:
                    ws[r["wid"]].append(r["pen"])
                P(f"| {tk} | {kd} | {c} | {sum(r['pen'] for r in rs)} / {len(rs)} | {sum(1 for v in ws.values() if all(v))} / {len(ws)} |")
    P("")
    # seed 別
    W = windows(rows, "succ", False); Wx = windows(rows, "succ", True)
    P("## seed 別（最終成功率、除外なし。窓対応の平均。括弧は窓数）")
    P("| タスク | seed | 群 | t | (a) | (s) | p_succ | (r) | 窓 | 2s/3 貫入除外後 (s) | (r) |"); P("|---|---|---|---|---|---|---|---|---|---|---|")
    seeds = sorted({(v["task"], v["seed"], v["group"]) for v in W.values()}, key=lambda x: (x[0], x[1]))
    for tk, sd, g in seeds:
        for kd in ("t13", "t23"):
            ws = [v for v in W.values() if v["task"] == tk and v["seed"] == sd and v["kind"] == kd]
            if not ws:
                continue
            wx = [v for v in Wx.values() if v["task"] == tk and v["seed"] == sd and v["kind"] == kd]
            f = lambda xs, k_: mean([v[k_] for v in xs])
            P(f"| {tk} | {sd} | {GN[g]} | {'s/3' if kd=='t13' else '2s/3'} | {f(ws,'a'):.3f} | {f(ws,'s'):.3f} | {f(ws,'ps'):.3f} | {f(ws,'r'):.3f} | {len(ws)} | "
              + (f"{f(wx,'s'):.3f} | {f(wx,'r'):.3f}" if kd == "t23" and wx else "—|—") + " |")
    P("")
    # R14-R16
    P("## R14〜R16 の値（線は書かない。計画 §14 の表）")
    for nm, tk, f1, f2 in (("R14 = k6 s/3 の (s)−(a)", "k6", "s", "a"), ("R15 = k6 s/3 の (r)−p_succ(s/3)", "k6", "r", "ps"), ("R16 = drawer s/3 の (s)−(a)", "drawer", "s", "a")):
        d, n = diff(W, "t13", f1, f2, tk)
        bys = collections.defaultdict(list)
        for v in W.values():
            if v["task"] == tk and v["kind"] == "t13" and v[f1] is not None and v[f2] is not None and not np.isnan(v[f1]) and not np.isnan(v[f2]):
                bys[v["seed"]].append(v[f1] - v[f2])
        P(f"- **{nm}** = **{d:+.3f}**（seed {n}、窓 {sum(len(x) for x in bys.values())}）。seed 別: " + "、".join(f"{s} {mean(x):+.2f}" for s, x in sorted(bys.items())))
    (K / "state/branches_i4_summary_20261015.md").write_text("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
