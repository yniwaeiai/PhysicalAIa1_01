"""再試行ラッパ W の集計（計画 v2 §11）。CPU のみ。
使い方: /path/to/work/.venv/bin/python kadai/scripts/analyze_wrapper.py
入力: kadai/state/wrapper_eval_20261015/*.jsonl（基準 arm=base: D は影で記録のみ / ラッパ arm=wrap）
出力: kadai/state/wrapper_eval_summary_20261015.md
指標: s250 = done が 250 step 以内、s400 = done（max_steps 400 以内）、c250/c400 = それぞれ かつ 対象外物体の最大変位 <= 1 mm。
集計: エピソード → seed 平均 → 群（または全体）の seed 平均。括弧は (seed 数 / ep 数)。差は 同じ seed の対応（ラッパ − 基準）の seed 平均、95% 区間は seed の bootstrap（5,000 回）。
"""
import glob, json, collections
from pathlib import Path
import numpy as np

K = Path(__file__).resolve().parents[1]
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
M = ["s250", "s400", "c250", "c400"]
rng = np.random.default_rng(0)


def load():
    d = {}
    for f in sorted(glob.glob(str(K / "state/wrapper_eval_20261015/*.jsonl"))):
        for l in open(f):
            r = json.loads(l)
            if r.get("error"):
                continue
            d[(r["task"], r["seed"], r["rep"], r["arm"])] = r
    return d


def vals(r):
    return dict(s250=float(r["success250"]), s400=float(r["success_end"]), c250=float(r["clean250"]), c400=float(r["clean"]))


def seed_table(d):
    """(task, seed, arm) -> {指標: 平均, n}"""
    g = collections.defaultdict(list)
    for (t, s, rep, arm), r in d.items():
        g[(t, s, arm)].append(vals(r))
    return {k: dict(n=len(v), **{m: float(np.mean([x[m] for x in v])) for m in M}) for k, v in g.items()}


def pooled(st, keys, arm, m):
    xs = [st[(t, s, arm)][m] for (t, s) in keys if (t, s, arm) in st]
    return (float(np.mean(xs)) if xs else float("nan")), len(xs), sum(st[(t, s, arm)]["n"] for (t, s) in keys if (t, s, arm) in st)


def diff(st, keys, m):
    ks = [(t, s) for (t, s) in keys if (t, s, "base") in st and (t, s, "wrap") in st]
    if not ks:
        return float("nan"), (float("nan"), float("nan")), 0
    x = np.array([st[(t, s, "wrap")][m] - st[(t, s, "base")][m] for t, s in ks])
    bs = [rng.choice(x, len(x)).mean() for _ in range(5000)]
    return float(x.mean()), (float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))), len(x)


def f(x):
    return "nan" if x != x else f"{x:.3f}"


def main():
    d = load(); st = seed_table(d); L = []; P = L.append
    seeds = sorted({(t, s) for (t, s, a) in st})
    P("# 再試行ラッパ W の集計（2026-10-15）\n")
    P(f"エピソード {len(d)}（base {sum(1 for k in d if k[3]=='base')} / wrap {sum(1 for k in d if k[3]=='wrap')}）、seed {len(seeds)}。指標は s250 / s400（成功）、c250 / c400（衝突なし成功）。\n")
    P("## 1. 群別・タスク別の成功率（seed 平均。括弧は seed 数 / ep 数）\n")
    P("| 群 | タスク | 指標 | 基準 | ラッパ | 差（ラッパ − 基準） | 差の 95% 区間 |"); P("|---|---|---|---|---|---|---|")
    stoch = lambda t=None: [(tt, s) for (tt, s) in seeds if GROUP.get((tt, s)) in ("luck", "mid", "hold") and (t is None or tt == t)]
    sets = []
    for gname in ["unseen", "luck", "mid", "hold"]:
        sets.append((gname, "全", [(t, s) for (t, s) in seeds if GROUP.get((t, s)) == gname]))
        for t in ["drawer", "k4", "k6"]:
            ks = [(tt, s) for (tt, s) in seeds if GROUP.get((tt, s)) == gname and tt == t]
            if ks: sets.append((gname, t, ks))
    sets.append(("確率的群（luck+mid+hold）", "全", stoch()))
    for t in ["drawer", "k4", "k6"]: sets.append(("確率的群（luck+mid+hold）", t, stoch(t)))
    sets.append(("全 seed", "全", seeds))
    for t in ["drawer", "k4", "k6"]: sets.append(("全 seed", t, [(tt, s) for (tt, s) in seeds if tt == t]))
    for gname, t, ks in sets:
        for m in M:
            b, nb, eb = pooled(st, ks, "base", m); w, nw, ew = pooled(st, ks, "wrap", m); dm, ci, n = diff(st, ks, m)
            P(f"| {gname} | {t} | {m} | {f(b)} ({nb}/{eb}) | {f(w)} ({nw}/{ew}) | {dm:+.3f} | [{ci[0]:+.3f}, {ci[1]:+.3f}] |")
    # 発火
    P("\n## 2. 発火・退避\n")
    wr = [r for k, r in d.items() if k[3] == "wrap"]; bs = [r for k, r in d.items() if k[3] == "base"]
    P("| 群 | タスク | ラッパ ep | 発火率 | 発火 step 中央値 [四分位] | 退避の到達率（位置 1.5 cm・回転 0.05 rad） | 退避 step 数 中央値 | 位置誤差 mm 中央値 | 発火 ep のうち再開後に成功（s400） | 発火 ep のうち再開後に成功（c400） | 基準（影の D）の発火率 | 基準: 発火 ep の成功率（s400） | 基準: 非発火 ep の成功率（s400） |")
    P("|---|---|---|---|---|---|---|---|---|---|---|---|---|")
    def row(gname, t, sel):
        w = [r for r in wr if sel(r)]; b = [r for r in bs if sel(r)]
        if not w: return
        fw = [r for r in w if r["fire_step"] is not None]; fb = [r for r in b if r["fire_step"] is not None]; nb = [r for r in b if r["fire_step"] is None]
        fs = np.array([r["fire_step"] for r in fw]) if fw else np.array([np.nan])
        rt = [r["retreat"] for r in fw]
        P(f"| {gname} | {t} | {len(w)} | {len(fw)/len(w):.3f} | {np.median(fs):.0f} [{np.percentile(fs,25):.0f}, {np.percentile(fs,75):.0f}] | {np.mean([x['reached'] for x in rt]) if rt else float('nan'):.3f} | "
          f"{np.median([x['up_steps']+x['home_steps'] for x in rt]) if rt else float('nan'):.0f} | {np.median([x['err_pos_mm'] for x in rt]) if rt else float('nan'):.1f} | "
          f"{np.mean([r['success_end'] for r in fw]) if fw else float('nan'):.3f} ({sum(r['success_end'] for r in fw)}/{len(fw)}) | {np.mean([r['clean'] for r in fw]) if fw else float('nan'):.3f} | "
          f"{len(fb)/max(len(b),1):.3f} | {np.mean([r['success_end'] for r in fb]) if fb else float('nan'):.3f} ({sum(r['success_end'] for r in fb)}/{len(fb)}) | {np.mean([r['success_end'] for r in nb]) if nb else float('nan'):.3f} ({sum(r['success_end'] for r in nb)}/{len(nb)}) |")
    for gname in ["unseen", "luck", "mid", "hold"]:
        row(gname, "全", lambda r, g=gname: GROUP.get((r["task"], r["seed"])) == g)
        for t in ["drawer", "k4", "k6"]:
            row(gname, t, lambda r, g=gname, t=t: GROUP.get((r["task"], r["seed"])) == g and r["task"] == t)
    row("確率的群", "全", lambda r: GROUP.get((r["task"], r["seed"])) in ("luck", "mid", "hold"))
    row("全 seed", "全", lambda r: True)
    for t in ["drawer", "k4", "k6"]: row("全 seed", t, lambda r, t=t: r["task"] == t)
    # 発火 step の分布（ラッパ）
    P("\n## 3. 発火 step の分布（ラッパ ep、タスク別）\n"); P("| タスク | N | 発火 ep | min | P10 | 中央値 | P90 | max | 発火後の残り step（400 − 再開 step）中央値 |"); P("|---|---|---|---|---|---|---|---|---|")
    for t in ["drawer", "k4", "k6"]:
        fw = [r for r in wr if r["task"] == t and r["fire_step"] is not None]
        if not fw: continue
        fs = np.array([r["fire_step"] for r in fw]); rem = np.array([400 - r["retreat"]["resume_T"] for r in fw])
        P(f"| {t} | {fw[0]['det_n']} | {len(fw)} | {fs.min()} | {np.percentile(fs,10):.0f} | {np.median(fs):.0f} | {np.percentile(fs,90):.0f} | {fs.max()} | {np.median(rem):.0f} |")
    # 操作チェック
    P("\n## 4. 操作チェック\n")
    nfire_wrap = sum(r["fire_step"] is not None for r in wr)
    P(f"- ラッパ ep で発火したもの {nfire_wrap}/{len(wr)}。退避中に手先が 5 mm 以上動いた割合（moved_mm）: {np.mean([r['retreat']['moved_mm'] >= 5 for r in wr if r['retreat']]):.3f}。")
    P(f"- 再開後に推論した割合（n_infer_after_resume ≥ 1、done で終わった ep を除く）: {np.mean([(r['n_infer_after_resume'] or 0) >= 1 for r in wr if r['retreat'] and not r['success_end']]) if any(r['retreat'] and not r['success_end'] for r in wr) else float('nan'):.3f}。")
    P(f"- 基準 ep が行動を変えていないこと: base の retry フラグ = {set(r['retry'] for r in bs)}、base の retreat 記録 = {sum(r['retreat'] is not None for r in bs)} 本。")
    P(f"- 退避中に done になった ep（退避の終端で done）: {sum(1 for r in wr if r['retreat'] and r['success_end'] and r['done_step'] is not None and r['done_step'] <= r['retreat']['resume_T'])}。")
    P(f"- ep 数の対応: seed×rep で base/wrap が揃わない組 {len({(k[0],k[1],k[2]) for k in d}) * 2 - len(d)}。エラー ep は除外。")
    # W1-W5
    P("\n## 5. W1〜W5 の値（線は書かない。判定は親）\n")
    stk = stoch(); unk = [(t, s) for (t, s) in seeds if GROUP.get((t, s)) == "unseen"]
    for lab, ks, m in [("W2: 確率的群の s250 差", stk, "s250"), ("W3: 確率的群の s400 差", stk, "s400"), ("W4: 全 seed の c400 差", seeds, "c400"), ("W4': 全 seed の c250 差", seeds, "c250"),
                       ("W5: 成功未観測の s250 差", unk, "s250"), ("W5': 成功未観測の s400 差", unk, "s400")]:
        dm, ci, n = diff(st, ks, m); P(f"- {lab}: {dm:+.3f}（95% 区間 [{ci[0]:+.3f}, {ci[1]:+.3f}]、seed {n}）")
    P("- W1（D のオフライン性能）は `kadai/state/stall_detector_20261015.md`。オンラインの影の D（基準 ep）の発火率は上の表 2。")
    (K / "state/wrapper_eval_summary_20261015.md").write_text("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
