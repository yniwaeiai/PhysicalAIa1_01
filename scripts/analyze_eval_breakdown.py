"""評価 jsonl の内訳: 摂動水準別の成功・衝突なし、K4 の対象外物体の移動量の分布、失敗の経路長・終端 q。
使い方: python analyze_eval_breakdown.py <jsonl を束ねたファイル>
"""
import collections, json, sys
import numpy as np
rs = [json.loads(l) for l in open(sys.argv[1]) if l.strip()]
lv = lambda r: {0.0: "none", 0.1: "mid", 0.2: "strong"}[r["level"][0]]
print("== 摂動水準別")
for t in ("drawer", "k4", "k6"):
    for L in ("none", "mid", "strong"):
        xs = [r for r in rs if r["task"] == t and lv(r) == L]
        if xs:
            print(f"  {t:6s} {L:6s} n={len(xs):2d} success {sum(r['success'] for r in xs):2d} clean {sum(r['clean'] for r in xs):2d}")
print("== K4: 衝突ありのエピソードの対象外物体の移動量（mm）")
k4 = [r for r in rs if r["task"] == "k4" and not r["clean"]]
print("  n", len(k4), sorted(round(r["disp_mm"]) for r in k4))
print("  100 mm 超（掴んで運んだ疑い）", sum(r["disp_mm"] > 100 for r in k4), " 以下（かすり傷）", sum(r["disp_mm"] <= 100 for r in k4), " 物体", dict(collections.Counter(r["top"] for r in k4)))
print("== 失敗の経路長・終端 q")
for t in ("drawer", "k4", "k6"):
    f = [r for r in rs if r["task"] == t and not r["success"]]
    if f:
        print(f"  {t}: 失敗 {len(f)}  path<0.5m {sum(r['path_m'] < 0.5 for r in f)}  path 中央値 {np.median([r['path_m'] for r in f]):.2f}  q_end 中央値 {np.median([r['q_end'] for r in f]):.3f}  q_end {sorted(round(r['q_end'], 3) for r in f)}")
