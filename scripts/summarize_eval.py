"""並列シャードの評価結果（*.jsonl）を束ねて、タスク別の成功・衝突なし・ステップ数・傾きを表にする。
使い方: python summarize_eval.py <名前の接頭辞 例: data/eval_runs/base_t3> [...]
"""
import glob, json, sys
import numpy as np

for prefix in sys.argv[1:]:
    rows = []
    for p in sorted(glob.glob(prefix + "*.jsonl")):
        rows += [json.loads(l) for l in open(p) if l.strip()]
    print(f"== {prefix}  (episodes {len(rows)}, shards {len(glob.glob(prefix + '*.jsonl'))})")
    for task in ("drawer", "k4", "k6"):
        rs = [r for r in rows if r["task"] == task and not r.get("error")]
        if not rs:
            continue
        ok = [r for r in rs if r["success"]]
        f = lambda xs: (f"{np.median(xs):.0f}" if xs else "-")
        print(f"  {task:7s} n={len(rs):3d}  success {sum(r['success'] for r in rs):3d} ({100*np.mean([r['success'] for r in rs]):.0f}%)  clean {sum(r['clean'] for r in rs):3d} ({100*np.mean([r['clean'] for r in rs]):.0f}%)  "
              f"steps(成功) med {f([r['steps'] for r in ok])}  tilt_med(全) {np.median([r['tilt_med'] for r in rs]):.1f}  "
              f"tilt_med(成功) {f([r['tilt_med'] for r in ok])}  tilt_med(失敗) {f([r['tilt_med'] for r in rs if not r['success']])}  path(成功) {np.median([r['path_m'] for r in ok]) if ok else float('nan'):.2f}")
        errs = [r for r in rows if r["task"] == task and r.get("error")]
        if errs:
            print(f"    (環境エラーで失敗扱い {len(errs)} 件)")
