"""L1 追加: 窓平均からの距離 r と成否。state/lanes_20261016.md の末尾に追記する（既存節は変えない）。"""
import csv, collections
from pathlib import Path
import numpy as np
from scipy.stats import spearmanr
K = Path(__file__).resolve().parents[1]; BR = K / "state/branches_i3_20261015"; F = np.load(K / "state/lanes_frames_20261016.npz")
STEPS = [8, 13, 18, 23, 30]; TASKS = ["drawer", "k4", "k6"]; rng = np.random.default_rng(1)
W = collections.OrderedDict()
for r in csv.DictReader(open(K / "state/branches_i3_20261015.csv")):
    if r["kind"] != "t0": continue
    f = F[r["wid"]]; ee = np.load(BR / f"b_{r['id']}.npz")["ee"][STEPS]
    w = W.setdefault(r["wid"], dict(task=r["task"], P=[], s=[])); w["P"].append((ee - f[:3]) @ f[3:].reshape(3, 3)); w["s"].append(int(r["succ"]))
for w in W.values():
    w["P"] = np.array(w["P"]) * 1000; w["s"] = np.array(w["s"])
    w["R"] = np.linalg.norm(w["P"] - w["P"].mean(0), axis=2)   # (8, nstep)
def auroc(sc, y):   # y=1 が正（失敗）
    p, n = sc[y == 1], sc[y == 0]; d = p[:, None] - n[None, :]
    return ((d > 0).sum() + .5 * (d == 0).sum()) / d.size
def q(x): return f"{np.median(x):.1f} [{np.percentile(x,25):.1f}, {np.percentile(x,75):.1f}]"
L = ["", "## 追加: 距離の大きさ（窓平均からの 3 次元距離 r、mm、固定具座標）",
     "混合窓（成功も失敗もある窓）だけプール。AUROC は失敗を正、r を予測子（> 0.5 = r が大きいほど失敗）。perm = 窓内でラベルをシャッフルした AUROC（100 回、平均±SD）。r 中央値 [Q1,Q3] は失敗 / 成功。",
     "| 対象 | 混合窓 | 本数 | step | AUROC(r→失敗) | perm | r 失敗 | r 成功 |", "|---|---|---|---|---|---|---|---|"]
for t in TASKS + ["全タスク"]:
    ws = [w for w in W.values() if (t == "全タスク" or w["task"] == t) and 0 < w["s"].sum() < 8]
    for si, s in enumerate(STEPS):
        sc = np.concatenate([w["R"][:, si] for w in ws]); y = np.concatenate([1 - w["s"] for w in ws])
        nl = []
        for _ in range(100):
            yp = np.concatenate([rng.permutation(1 - w["s"]) for w in ws]); nl.append(auroc(sc, yp))
        L.append(f"| {t} | {len(ws)} | {len(sc)} | {s} | {auroc(sc,y):.3f} | {np.mean(nl):.3f}±{np.std(nl):.3f} | {q(sc[y==1])} | {q(sc[y==0])} |")
L += ["", "窓ごとの 成功率 と 窓内 SD（合成 RMS = √ΣVar、ddof=1）の Spearman 相関（全 56 窓。負 = 揺らぎの大きい窓ほど成功率が低い）", "| 対象 | 窓数 | step | Spearman ρ | p |", "|---|---|---|---|---|"]
for t in TASKS + ["全タスク"]:
    ws = [w for w in W.values() if t == "全タスク" or w["task"] == t]
    for si, s in enumerate(STEPS):
        sd = [np.sqrt((w["P"][:, si].var(0, ddof=1)).sum()) for w in ws]; sr = [w["s"].mean() for w in ws]
        rho, p = spearmanr(sd, sr); L.append(f"| {t} | {len(ws)} | {s} | {rho:+.3f} | {p:.3f} |")
txt = "\n".join(L) + "\n"
with open(K / "state/lanes_20261016.md", "a") as f: f.write(txt)
print(txt)
