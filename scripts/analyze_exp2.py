"""第 2 弾（初期ノイズのスケーリング・q 停留→再サンプリング）の分析。事前に固定した設計は作業履歴 §17.3。
(A) 全体評価: N0（無摂動、第 1 弾の最終評価）・N0b（再実行、ノイズ床）・F10・F15・S25 を、タスク別に成功・衝突なし・ステップ数、同一 seed の対応つき比較（vs N0）。
(B) 標的評価: 既知の失敗 seed × 8 反復。seed ごとの成功確率 p（T0 / TF15 / TS25）、p の分布、対応のある比較（seed 単位の符号検定）。
使い方: python analyze_exp2.py <N0 jsonl> <データのディレクトリ data/eval_runs>
"""
import glob, json, math, sys
import numpy as np


def load(pat):
    return [json.loads(l) for f in sorted(glob.glob(pat)) for l in open(f) if l.strip()]


def wilson(k, n, z=1.96):
    if n == 0:
        return (float("nan"),) * 2
    p = k / n; d = 1 + z * z / n; c = p + z * z / (2 * n); h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return ((c - h) / d, (c + h) / d)


def sign_p(b, c):
    n = b + c
    if n == 0:
        return 1.0
    return min(1.0, 2 * sum(math.comb(n, i) for i in range(min(b, c) + 1)) / 2 ** n)


n0_path, d = sys.argv[1], sys.argv[2].rstrip("/")
runs = {"N0": load(n0_path), "N0b": load(f"{d}/v1f_N0b_*.jsonl"), "F10": load(f"{d}/v2_F10_*.jsonl"), "F15": load(f"{d}/v2_F15_*.jsonl"), "S25": load(f"{d}/v2_S25_*.jsonl")}
by = {n: {(r["task"], r["seed"]): r for r in rs if not r.get("error")} for n, rs in runs.items()}
print("(A) 全体評価（各タスク 48 本、seed 1000–1047）")
for task in ("drawer", "k4", "k6"):
    print(f"\n=== {task}")
    for n in runs:
        rs = [r for r in runs[n] if r["task"] == task and not r.get("error")]
        k, kc = sum(r["success"] for r in rs), sum(r["clean"] for r in rs)
        lo, hi = wilson(k, len(rs)); lo2, hi2 = wilson(kc, len(rs))
        st = np.median([r["steps"] for r in rs if r["success"]]) if k else float("nan")
        tau = [r["tau_mean"] for r in rs if r.get("tau_mean")]; nr = sum(r.get("n_resample", 0) > 0 for r in rs)
        print(f"  {n:4s} n={len(rs):3d} success {k:3d} ({100*k/len(rs):3.0f}%) [{100*lo:3.0f},{100*hi:3.0f}]  clean {kc:3d} ({100*kc/len(rs):3.0f}%) [{100*lo2:3.0f},{100*hi2:3.0f}]  steps {st:4.0f}  tau_mean {np.mean(tau) if tau else float('nan'):.2f} (記録 {len(tau)} 本)  q停留で再サンプリングした本数 {nr}")
    for n in ("N0b", "F10", "F15", "S25"):
        for metric in ("success", "clean"):
            seeds = [s for (t, s) in by["N0"] if t == task and (task, s) in by[n]]
            b = sum(by[n][(task, s)][metric] and not by["N0"][(task, s)][metric] for s in seeds)
            c = sum(by["N0"][(task, s)][metric] and not by[n][(task, s)][metric] for s in seeds)
            print(f"    {n} vs N0 [{metric:7s}] 条件のみ成功 {b:2d} / N0 のみ成功 {c:2d}  差 {b-c:+d} ({100*(b-c)/max(len(seeds),1):+.0f} pt)  符号検定 p={sign_p(b,c):.2f}")
print("\n(B) 標的評価（既知の失敗 seed × 8 反復、条件間で同じ初期条件・摂動）")
tgt = {"T0": load(f"{d}/t2_T0_*.jsonl"), "TF15": load(f"{d}/t2_TF15_*.jsonl"), "TS25": load(f"{d}/t2_TS25_*.jsonl")}
for task in ("drawer", "k4", "k6"):
    P = {}
    for n, rs in tgt.items():
        g = {}
        for r in rs:
            if r["task"] == task and not r.get("error"):
                g.setdefault(r["seed"], []).append(r["success"])
        P[n] = {s: (sum(v) / len(v), len(v)) for s, v in g.items()}
    seeds = sorted(set.intersection(*[set(p) for p in P.values()])) if all(P.values()) else []
    if not seeds:
        continue
    print(f"\n=== {task}（seed {len(seeds)} 本）")
    for n in tgt:
        ps = np.array([P[n][s][0] for s in seeds]); tot = sum(P[n][s][1] for s in seeds); suc = sum(round(P[n][s][0] * P[n][s][1]) for s in seeds)
        print(f"  {n:5s} 反復全体の成功 {suc}/{tot} ({100*suc/tot:.0f}%)  seed ごとの成功確率 p: 中央値 {np.median(ps):.2f}  p=0 の seed {int((ps==0).sum())}  p≥0.5 の seed {int((ps>=0.5).sum())}  p=1 の seed {int((ps==1).sum())}")
    print("  seed 別 p (T0 / TF15 / TS25): " + "  ".join(f"{s}:{P['T0'][s][0]:.2f}/{P['TF15'][s][0]:.2f}/{P['TS25'][s][0]:.2f}" for s in seeds))
    for n in ("TF15", "TS25"):
        diff = [P[n][s][0] - P["T0"][s][0] for s in seeds]
        b, c = sum(x > 0 for x in diff), sum(x < 0 for x in diff)
        print(f"    {n} vs T0: seed の p が上がった {b} / 下がった {c} / 同じ {len(diff)-b-c}  平均差 {100*np.mean(diff):+.1f} pt  符号検定 p={sign_p(b,c):.2f}")
