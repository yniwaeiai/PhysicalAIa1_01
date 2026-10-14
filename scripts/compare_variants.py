"""摂動条件の比較（結果の前に固定した設計 §14 の分析）。
各条件: タスク別の成功率・衝突なし率（Wilson 95% 区間）、成功時ステップ数の中央値、操作チェック（act_dev・n_kicks）。
基準 N0（無摂動）に対する、同一 seed の対応つき比較: 成功の勝ち（条件だけ成功）・負け（N0 だけ成功）と、符号検定（両側、厳密二項）の p。
N0 と N0b（同じ設定の再実行）の差が、ノイズ床。使い方: python compare_variants.py <N0 jsonl> <N0b prefix> <A05 prefix> <A15 prefix> <K prefix>
"""
import glob, json, math, sys
import numpy as np


def load(path_or_prefix):
    fs = [path_or_prefix] if path_or_prefix.endswith(".jsonl") else sorted(glob.glob(path_or_prefix + "*.jsonl"))
    return [json.loads(l) for f in fs for l in open(f) if l.strip()]


def wilson(k, n, z=1.96):
    if n == 0:
        return (float("nan"),) * 2
    p = k / n; d = 1 + z * z / n; c = p + z * z / (2 * n); h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return ((c - h) / d, (c + h) / d)


def sign_p(b, c):
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    return min(1.0, 2 * sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n)


names = ["N0", "N0b", "A05", "A15", "K"]
runs = dict(zip(names, [load(a) for a in sys.argv[1:6]]))
by = {n: {(r["task"], r["seed"]): r for r in rs if not r.get("error")} for n, rs in runs.items()}
for task in ("drawer", "k4", "k6"):
    print(f"\n=== {task}")
    print(f"  {'cond':4s} {'n':>3s} {'success (95%CI)':>24s} {'clean (95%CI)':>24s} {'steps med':>9s} {'act_dev':>8s} {'kicked eps':>10s}")
    for n in names:
        rs = [r for r in runs[n] if r["task"] == task and not r.get("error")]
        k, kc = sum(r["success"] for r in rs), sum(r["clean"] for r in rs)
        lo, hi = wilson(k, len(rs)); lo2, hi2 = wilson(kc, len(rs))
        st = np.median([r["steps"] for r in rs if r["success"]]) if k else float("nan")
        dev = np.mean([r.get("act_dev", 0.0) for r in rs]); kk = sum(r.get("n_kicks", 0) > 0 for r in rs)
        print(f"  {n:4s} {len(rs):3d} {k:3d} ({100*k/len(rs):3.0f}%) [{100*lo:3.0f},{100*hi:3.0f}] {kc:3d} ({100*kc/len(rs):3.0f}%) [{100*lo2:3.0f},{100*hi2:3.0f}] {st:9.0f} {dev:8.4f} {kk:10d}")
    for n in names[1:]:
        for metric in ("success", "clean"):
            seeds = [s for (t, s) in by["N0"] if t == task and (task, s) in by[n]]
            b = sum(by[n][(task, s)][metric] and not by["N0"][(task, s)][metric] for s in seeds)
            c = sum(by["N0"][(task, s)][metric] and not by[n][(task, s)][metric] for s in seeds)
            print(f"    {n} vs N0 [{metric:7s}] paired n={len(seeds)}  条件のみ成功 {b:2d} / N0 のみ成功 {c:2d}  差 {b-c:+d} ({100*(b-c)/max(len(seeds),1):+.0f} pt)  符号検定 p={sign_p(b,c):.2f}")
# K 条件で実際に kick が発動したエピソードに限った比較
rs = [r for r in runs["K"] if r.get("n_kicks", 0) > 0 and not r.get("error")]
print(f"\n=== K 条件で kick が発動したエピソード {len(rs)} 本（タスク別）")
for task in ("drawer", "k4", "k6"):
    xs = [r for r in rs if r["task"] == task]
    if xs:
        n0 = [by["N0"][(task, r["seed"])] for r in xs if (task, r["seed"]) in by["N0"]]
        print(f"  {task}: n={len(xs)}  K の成功 {sum(r['success'] for r in xs)}  同じ seed の N0 の成功 {sum(r['success'] for r in n0)}  kick 回数の中央値 {np.median([r['n_kicks'] for r in xs]):.0f}")
