"""R-a 収集の集計。入力 kadai/state/rec_collect_20261016/{summary_s*.jsonl, selected.jsonl, rollout_*.jsonl} → kadai/state/rec_collect_summary_20261016.md"""
import glob, json, collections
from pathlib import Path
import numpy as np
K = Path(__file__).resolve().parents[1]; D = K / "state/rec_collect_20261016"
rd = lambda pat: [json.loads(l) for f in sorted(glob.glob(str(D / pat))) for l in open(f) if l.strip()]
S = rd("summary_s*.jsonl"); SEL = rd("selected.jsonl"); R = rd("rollout_*.jsonl")
TASKS = ["drawer", "k4", "k6"]; TS = [5, 10, 15, 20, 25, 30, 35, 40]
o = []; P = o.append
P("# R-a 回復データ収集の集計（2026-10-16）\n")
P("## 方策 rollout（v1、seed 5000-5099、250 step、replan 5、--ref-pose、OSMesa）\n\n| タスク | 本数 | done | clean | 平均 step |\n|---|---|---|---|---|")
for t in TASKS:
    r = [x for x in R if x["task"] == t]
    if r: P(f"| {t} | {len(r)} | {sum(x['success'] for x in r)} ({np.mean([x['success'] for x in r]):.3f}) | {sum(x['clean'] for x in r)} ({np.mean([x['clean'] for x in r]):.3f}) | {np.mean([x['steps'] for x in r]):.0f} |")
P("\n## 分岐の処理（理由別、タスク × t）\n")
reasons = collections.Counter((r["task"], r.get("reason", "error")) for r in S)
rs = sorted({k[1] for k in reasons})
P("| タスク | " + " | ".join(rs) + " |\n|---|" + "---|" * len(rs))
for t in TASKS: P(f"| {t} | " + " | ".join(str(reasons[(t, x)]) for x in rs) + " |")
P("\n## 層別: 試行数（接触履歴なしで expert を走らせた数）/ clean 受理（上限前）/ 受理率 — タスク × t × 由来（由来 = 方策 rollout の最終成否 S 成功 / F 失敗）\n")
for t in TASKS:
    P(f"\n### {t}\n\n| t | 由来 | 試行 | 受理(上限前) | 受理率 | 選択(上限・rollout4本後) | 選択フレーム |\n|---|---|---|---|---|---|---|")
    for tt in TS:
        for og, nm in ((True, "S"), (False, "F")):
            ex = [r for r in S if r["task"] == t and r["t"] == tt and r.get("origin_success") == og and r.get("reason") in ("accepted", "prefix_displaced", "expert_collided", "expert_not_done")]
            ac = [r for r in ex if r["reason"] == "accepted"]
            sl = [r for r in SEL if r["task"] == t and r["t"] == tt and r["origin_success"] == og]
            P(f"| {tt} | {nm} | {len(ex)} | {len(ac)} | {len(ac)/max(len(ex),1):.2f} | {len(sl)} | {sum(r['frames'] for r in sl)} |")
P("\n## 選択 demo の合計（タスク × 由来）\n\n| タスク | 由来 | demo 数 | 総フレーム |\n|---|---|---|---|")
for t in TASKS:
    for og, nm in ((True, "S"), (False, "F")):
        sl = [r for r in SEL if r["task"] == t and r["origin_success"] == og]
        P(f"| {t} | {nm} | {len(sl)} | {sum(r['frames'] for r in sl)} |")
ran = [r for r in S if "restore_err_mm" in r and "reason" in r and r["reason"] != "contact_history"]
P("\n## 操作チェック（expert を走らせた全分岐）\n")
for key, nm in (("restore_err_mm", "復元誤差 手先 (mm)"), ("state0_vs_trace_ee_mm", "demo 先頭 state vs trace ee (mm)"), ("state0_vs_sim_ee_mm", "demo 先頭 state vs 復元 sim ee (mm)"), ("img_diff_agent", "復元画像 vs trace 画像 agent の平均絶対差(0-255)"), ("img_diff_wrist", "同 wrist")):
    v = np.array([r[key] for r in ran if key in r]); P(f"- {nm}: n={len(v)} 中央値 {np.median(v):.3f} / 最大 {v.max():.3f}" + (f" / <1.5mm の割合 {np.mean(v<1.5):.4f}" if "mm" in nm else ""))
P("\n局面転換 step（先頭局面の完了 step = pre_step、t 起点の相対）の中央値 [25%, 75%]、タスク別:")
for t in TASKS:
    v = np.array([r["pre_step"] for r in ran if r["task"] == t and r.get("pre_step") is not None])
    if len(v): P(f"- {t}: {np.median(v):.0f} [{np.percentile(v,25):.0f}, {np.percentile(v,75):.0f}] (n={len(v)})")
Path(K / "state/rec_collect_summary_20261016.md").write_text("\n".join(o) + "\n")
print("\n".join(o))
