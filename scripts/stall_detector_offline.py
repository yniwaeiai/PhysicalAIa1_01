"""停留検出 D のオフライン設計（計画 v2 §11 W。A' の trace 168 本）。CPU のみ。
使い方: /path/to/work/.venv/bin/python kadai/scripts/stall_detector_offline.py
D(N, T) = 実行済み step T > N で、直近 N step の 手先速度の平均 < 1.5 mm/step かつ 直近 N step の位置指令 ‖act_raw[:3]‖ の中央値 >= 0.3。
N はタスク別: 成功 ep で N=1 の条件を満たす最長連続長の 95 パーセンタイル + 5（結果を見て変えない）。
trace の添字: ee[k], act_raw[k] は step T=k+1 の実行後・実行した指令（初期の手先位置は trace に無いので T > N を要求）。
"""
import csv, glob
from pathlib import Path
import numpy as np

K = Path(__file__).resolve().parents[1]
SPEED_MM, CMD_MIN = 1.5, 0.3


def load():
    win = {}
    for r in csv.DictReader(open(K / "state/aprime_windows_20261014.csv")):
        win[(r["task"], int(r["seed"]), int(r["rep"]))] = (int(r["win_start"]) if r["win_start"] else None)
    eps = []
    for f in sorted(glob.glob(str(K / "state/trace_aprime/*.npz"))):
        d = np.load(f, allow_pickle=True)
        t, s, r = str(d["task"]), int(d["seed"]), int(d["rep"])
        ee = np.asarray(d["ee"], float); cmd = np.linalg.norm(np.asarray(d["act_raw"], float)[:, :3], axis=1)
        eps.append(dict(task=t, seed=s, rep=r, success=bool(d["success"]), ee=ee, cmd=cmd, s=win.get((t, s, r))))
    return eps


def step_speed(ee):   # sp[k] = |ee[k]-ee[k-1]| (mm)、k>=1。sp[0] は不明（初期位置が trace に無い）
    sp = np.full(len(ee), np.nan); sp[1:] = np.linalg.norm(np.diff(ee, axis=0), axis=1) * 1000
    return sp


def max_run_n1(ep):
    sp = step_speed(ep["ee"]); c = (sp < SPEED_MM) & (ep["cmd"] >= CMD_MIN)   # nan < x は False
    best = cur = 0
    for v in c:
        cur = cur + 1 if v else 0; best = max(best, cur)
    return best


def first_fire(ep, N):
    """最初に発火する step T（1 始まり）。なければ None。"""
    ee, cmd = ep["ee"], ep["cmd"]; n = len(ee)
    for T in range(N + 1, n + 1):
        k = T - 1   # ee[k] = step T 後
        sp = np.linalg.norm(np.diff(ee[k - N:k + 1], axis=0), axis=1).mean() * 1000
        if sp < SPEED_MM and np.median(cmd[k - N + 1:k + 1]) >= CMD_MIN:
            return T
    return None


def main():
    eps = load(); lines = []; P = lambda s="": (print(s), lines.append(s))
    P("# 停留検出 D のオフライン設計（A' trace 168 本、2026-10-15）\n")
    P(f"D: 直近 N step の手先速度の平均 < {SPEED_MM} mm/step かつ 位置指令 ‖act_raw[:3]‖ の中央値 >= {CMD_MIN}。N = (成功 ep で N=1 の条件を満たす最長連続長の 95 パーセンタイル) + 5。\n")
    Ns = {}
    P("## N の決定（成功 ep の最長連続長）\n"); P("| タスク | 成功 ep | 最長連続長 中央値 | 最大 | P95 | N = P95 + 5 |"); P("|---|---|---|---|---|---|")
    for t in ["drawer", "k4", "k6"]:
        run = [max_run_n1(e) for e in eps if e["task"] == t and e["success"]]
        p95 = float(np.percentile(run, 95)); Ns[t] = int(np.ceil(p95)) + 5
        P(f"| {t} | {len(run)} | {np.median(run):.0f} | {max(run)} | {p95:.2f} | **{Ns[t]}** |")
    P(f"\nN = {Ns}（P95 は切り上げてから +5）\n")
    P("## 性能（上の N で）\n")
    P("| タスク | N | 成功 ep の誤発火率 (n) | 失敗 ep の検出率（停留窓あり、ep のどこかで発火）(n) | 検出のうち発火 step >= s の割合 | 発火 step − s の中央値 [四分位] | 発火 step の中央値 | s + N + 5 の中央値 | 発火 − s ≤ N+5 の割合（検出のうち） |")
    P("|---|---|---|---|---|---|---|---|---|")
    for t in ["drawer", "k4", "k6"]:
        N = Ns[t]; S = [e for e in eps if e["task"] == t and e["success"]]; F = [e for e in eps if e["task"] == t and not e["success"] and e["s"] is not None]
        fs = [first_fire(e, N) for e in S]; ff = [first_fire(e, N) for e in F]
        det = [(f, e["s"]) for f, e in zip(ff, F) if f is not None]
        fa = np.mean([f is not None for f in fs]) if S else float("nan"); dr = len(det) / len(F) if F else float("nan")
        d = np.array([f - s for f, s in det]) if det else np.array([np.nan])
        ge = np.mean([f >= s for f, s in det]) if det else float("nan")
        P(f"| {t} | {N} | {fa:.3f} ({sum(f is not None for f in fs)}/{len(S)}) | {dr:.3f} ({len(det)}/{len(F)}) | {ge:.2f} | {np.median(d):.0f} [{np.percentile(d,25):.0f}, {np.percentile(d,75):.0f}] | {np.median([f for f,_ in det]) if det else float('nan'):.0f} | {np.median([s+N+5 for _,s in det]) if det else float('nan'):.0f} | {np.mean(d <= N + 5):.2f} |")
    P("\n## 失敗 ep のうち未検出の内訳\n"); P("| タスク | 未検出 ep（task seed rep, s） |"); P("|---|---|")
    for t in ["drawer", "k4", "k6"]:
        miss = [f"{e['seed']}r{e['rep']}(s={e['s']})" for e in eps if e["task"] == t and not e["success"] and e["s"] is not None and first_fire(e, Ns[t]) is None]
        P(f"| {t} | {', '.join(miss) if miss else '-'} |")
    P("\n## 参考: 成功 ep の誤発火の位置（発火 step の中央値）\n")
    for t in ["drawer", "k4", "k6"]:
        fs = [(first_fire(e, Ns[t]), e["seed"], e["rep"]) for e in eps if e["task"] == t and e["success"]]
        fs = [x for x in fs if x[0] is not None]; P(f"- {t}: {fs}")
    (K / "state/stall_detector_20261015.md").write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
