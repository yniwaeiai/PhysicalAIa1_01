"""F の追加（事後の探索）: 関節配置の特徴。CPU のみ・環境不要。state_features_20261015.csv に列を足し、md に節を追記する。
特徴: jq1..jq7（分岐点 t の trace 添字 t-1 の jq）、jmargin_min（trace の jmargin の 7 関節最小）。jq4・jq2 は肘の高さの代理。
再実行は analyze_state_features.py の後に行う（そちらが csv/md を作り直す）。"""
import csv, collections
from pathlib import Path
import numpy as np
from scipy.stats import rankdata
K = Path(__file__).resolve().parents[1]
rng = np.random.default_rng(20261016)
rows = list(csv.DictReader(open(K / "state/state_features_20261015.csv")))
BASE = ["x_mm","y_mm","z_mm","tilt_deg","yaw_deg","tgtdist_mm","dir_x","dir_y","dir_z","grip_open","fix_q","t"]
JF = [f"jq{i}" for i in range(1, 8)] + ["jmargin_min"]
LIM = np.array([[-2.8973,2.8973],[-1.7628,1.7628],[-2.8973,2.8973],[-3.0718,-0.0698],[-2.8973,2.8973],[-0.0175,3.7525],[-2.8973,2.8973]])
mx = 0
for r in rows:
    tr = np.load(K / f"state/trace_aprime/{r['task']}_s{r['seed']}_r{r['rep']}.npz", allow_pickle=True)
    i = int(float(r["t"])) - 1
    jq = tr["jq"][i]; jm = tr["jmargin"][i]
    mx = max(mx, float(np.abs(jm - np.minimum(jq - LIM[:, 0], LIM[:, 1] - jq)).max()))
    for k in range(7): r[f"jq{k+1}"] = float(jq[k])
    r["jmargin_min"] = float(jm.min())
print("jmargin vs Panda 限界からの再計算の最大差(rad):", mx)
cols = list(rows[0].keys())
with open(K / "state/state_features_20261015.csv", "w", newline="") as fh:
    wr = csv.DictWriter(fh, fieldnames=cols); wr.writeheader(); wr.writerows(rows)
for r in rows:
    for c in BASE + JF + ["p"]: r[c] = float(r[c])
    r["is_succ"] = int(r["is_succ"]); r["seed"] = int(r["seed"])
def sel(task, tt): return [r for r in rows if r["task"] == task and r["tt"] == tt]
def sp(x, y):
    if np.ptp(x) == 0 or np.ptp(y) == 0: return np.nan
    return float(np.corrcoef(rankdata(x), rankdata(y))[0, 1])
def perm(x, y, n=1000):
    r0 = sp(x, y)
    if np.isnan(r0): return np.nan
    rx = rankdata(x); c = sum(abs(np.corrcoef(rx, rankdata(rng.permutation(y)))[0, 1]) >= abs(r0) - 1e-12 for _ in range(n))
    return (c + 1) / (n + 1)
def auroc(h, l):
    a = np.asarray(h)[:, None]; b = np.asarray(l)[None, :]
    return float(((a > b).sum() + .5 * (a == b).sum()) / (a.size * b.size)) if a.size and b.size else np.nan
md = ["\n## 追加（事後）: 関節配置の特徴（事後の探索。計画 §11 の事前登録にない。多重比較: 8 特徴 × t 2 × タスク 2 を上の 12 特徴に追加、補正なし）",
      "jq4 と jq2 は肘の高さの代理。jmargin_min = trace の jmargin（関節限界までの余裕、rad）の 7 関節最小。Spearman / 順列 p（1000 回）/ AUROC（p≥0.5 を高）は上と同じ定義。k4 は値のみ。"]
for task in ("drawer", "k6", "k4"):
    for tt in ("13", "23"):
        R = sel(task, tt); p = np.array([r["p"] for r in R]); hi = p >= .5; lo = p < .25
        md.append(f"\n### {task} t={'s/3' if tt=='13' else '2s/3'} n={len(R)} p≥0.5: {int(hi.sum())} p<0.25: {int(lo.sum())}\n| 特徴 | Spearman | 順列 p | AUROC | 平均 p≥0.5 | 平均 p<0.25 |\n|---|---|---|---|---|---|")
        for ft in JF:
            x = np.array([r[ft] for r in R]); pp = perm(x, p) if task != "k4" else np.nan
            md.append(f"| {ft} | {sp(x,p):+.3f} | {'' if np.isnan(pp) else f'{pp:.3f}'} | {auroc(x[hi],x[lo]):.3f} | {x[hi].mean() if hi.any() else np.nan:.3g} | {x[lo].mean() if lo.any() else np.nan:.3g} |")
FT = BASE + JF
for tt, lab in (("13", "s/3"), ("23", "2s/3")):
    md.append(f"\n## 補助（事後）: 同 seed の成功対照 − 失敗窓（{lab}、seed 内平均の差、既存 12 特徴 + 関節）")
    for task in ("drawer", "k6", "k4"):
        R = sel(task, tt); seeds = sorted({r["seed"] for r in R if r["is_succ"]}); dd = []
        md.append(f"\n{task}\n| seed | 群 | 失敗/対照 | " + " | ".join(FT) + " |\n|" + "---|" * (3 + len(FT)))
        for sd in seeds:
            F = [r for r in R if r["seed"] == sd and not r["is_succ"]]; S = [r for r in R if r["seed"] == sd and r["is_succ"]]
            if not F: continue
            d = [np.mean([r[f] for r in S]) - np.mean([r[f] for r in F]) for f in FT]; dd.append(d)
            md.append(f"| {sd} | {F[0]['group']} | {len(F)}/{len(S)} | " + " | ".join(f"{v:+.3g}" for v in d) + " |")
        if dd:
            dd = np.array(dd)
            md.append("| 平均 | | | " + " | ".join(f"{v:+.3g}" for v in dd.mean(0)) + " |")
            md.append("| 正の seed 数 | | | " + " | ".join(f"{int((dd[:, j] > 0).sum())}/{len(dd)}" for j in range(len(FT))) + " |")
md.append("\n## 群別の jmargin_min（s/3、失敗窓。窓の値の最小 / 中央値 / 窓数）と p≥0.5 の窓（成功対照を含む全群）")
md.append("| タスク | unseen | luck | mid | hold | p≥0.5 の窓 |\n|---|---|---|---|---|---|")
for task in ("drawer", "k6", "k4"):
    R = sel(task, "13"); cells = []
    for g in ("unseen", "luck", "mid", "hold"):
        v = [r["jmargin_min"] for r in R if r["group"] == g and not r["is_succ"]]
        cells.append(f"{min(v):.3f} / {np.median(v):.3f} / {len(v)}" if v else "-")
    v = [r["jmargin_min"] for r in R if r["p"] >= .5]
    cells.append(f"{min(v):.3f} / {np.median(v):.3f} / {len(v)}" if v else "-")
    md.append(f"| {task} | " + " | ".join(cells) + " |")
s = open(K / "state/state_features_20261015.md").read()
s = s.split("\n## 追加（事後）")[0]
open(K / "state/state_features_20261015.md", "w").write(s + "\n".join(md) + "\n")
