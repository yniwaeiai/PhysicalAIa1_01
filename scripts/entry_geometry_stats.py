"""entry_geometry_20261015.csv から G1-G3 の統計と図。"""
import csv, json
from pathlib import Path
import numpy as np
from scipy.stats import spearmanr, rankdata
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
K = Path(__file__).resolve().parents[1]
R = list(csv.DictReader(open(K / "state/entry_geometry_20261015.csv")))
# ★trace の tgt_pos は numpy view の別名参照で全 step が「最終状態」の値（eval_kadai_closedloop.target_info の np.asarray）。対象相対列(h*)は無効なので CSV から落とす
_cols = [c for c in R[0].keys() if not c.startswith(("hx", "hy", "hz"))]
with open(K / "state/entry_geometry_20261015.csv", "w", newline="") as _f:
    _w = csv.DictWriter(_f, fieldnames=_cols); _w.writeheader(); _w.writerows([{c: r[c] for c in _cols} for r in R])
R = [{c: r[c] for c in _cols} for r in R]
def f(r, k): return float(r[k])
TS = [-10, 0, 20]
def auroc(s, y):
    r = rankdata(s); n1 = y.sum(); n0 = len(y) - n1
    return (r[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)
def loo_centroid(X, y):
    sc = []
    for i in range(len(y)):
        m = np.ones(len(y), bool); m[i] = False
        cf = X[m & (y == 1)].mean(0); cp = X[m & (y == 0)].mean(0)
        sc.append(np.linalg.norm(X[i] - cp) - np.linalg.norm(X[i] - cf))   # 大きい = 失敗重心に近い
    return np.array(sc)
res = {}; lines = []
for task in ["drawer", "k6"]:
    rs = [r for r in R if r["task"] == task]; y = np.array([1 if r["kind"] == "fail" else 0 for r in rs])
    res[task] = dict(n_fail=int(y.sum()), n_fp=int((1 - y).sum()))
    for t in TS:
        X = np.array([[f(r, f"x{t}"), f(r, f"y{t}"), f(r, f"z{t}")] for r in rs]) * 1000
        a = auroc(loo_centroid(X, y), y)
        md = np.median(X[y == 1], 0) - np.median(X[y == 0], 0)
        # 姿勢
        tl = np.array([f(r, f"tilt{t}") for r in rs]); yw = np.array([f(r, f"yaw{t}") for r in rs])
        res[task][f"t{t}"] = dict(auroc_pos=a, med_diff_mm=md.tolist(), 
                                  tilt_med_fail=float(np.median(tl[y == 1])), tilt_med_fp=float(np.median(tl[y == 0])), yaw_med_fail=float(np.median(yw[y == 1])), yaw_med_fp=float(np.median(yw[y == 0])),
                                  auroc_tilt=auroc(tl, y), auroc_yaw=auroc(yw, y))
        # G2
        cfp = X[y == 0].mean(0); dev = np.linalg.norm(X[y == 1] - cfp, axis=1); fr = [r for r, yy in zip(rs, y) if yy]
        g2 = {}
        for k in ["cy", "cs", "rn"]:
            v = np.array([f(r, k) for r in fr]); g2[k] = float(spearmanr(dev, v)[0]) if len(set(v)) > 1 else None
        for k in ["yaw_deg", "shift_mm"]:
            v = np.abs(np.array([f(r, k) for r in fr])); g2["abs_" + k] = float(spearmanr(dev, v)[0]) if len(set(v)) > 1 else None
        res[task][f"t{t}"]["G2_spearman"] = g2
        if task == "k6":
            ed = np.array([f(r, f"edge_dz{t}") for r in rs]); dn = np.array([f(r, f"door_n{t}") for r in rs])
            res[task][f"t{t}"]["G3"] = dict(below_edge_fail=float((ed[y == 1] < 0).mean()), below_edge_fp=float((ed[y == 0] < 0).mean()),
                                            behind_fail=float((dn[y == 1] < 0).mean()), behind_fp=float((dn[y == 0] < 0).mean()),
                                            edge_dz_med_fail_mm=float(np.median(ed[y == 1]) * 1000), edge_dz_med_fp_mm=float(np.median(ed[y == 0]) * 1000),
                                            door_n_med_fail_mm=float(np.median(dn[y == 1]) * 1000), door_n_med_fp_mm=float(np.median(dn[y == 0]) * 1000))
# k4
rs = [r for r in R if r["task"] == "k4"]
res["k4"] = [dict(seed=r["seed"], kind=r["kind"], **{f"{a}{t}": round(f(r, f"{a}{t}") * (1000 if a in "xyz" else 1), 1) for t in TS for a in ["x", "y", "z"]}) for r in rs]
json.dump(res, open(K / "state/entry_geometry_stats_20261015.json", "w"), indent=1)
# 図
fig, ax = plt.subplots(2, 3, figsize=(13, 8))
for i, task in enumerate(["drawer", "k6"]):
    rs = [r for r in R if r["task"] == task]
    for j, t in enumerate(TS):
        a = ax[i, j]
        for kind, c, mk in [("fp", "tab:blue", "o"), ("fail", "tab:red", "x")]:
            x = [f(r, f"x{t}") * 1000 for r in rs if r["kind"] == kind]; y_ = [f(r, f"y{t}") * 1000 for r in rs if r["kind"] == kind]
            z = [f(r, f"z{t}") * 1000 for r in rs if r["kind"] == kind]
            sc = a.scatter(x, y_, c=c, marker=mk, s=40, label=f"{'fail' if kind=='fail' else 'false-pos (success ep)'} n={len(x)}", alpha=.8)
        a.set_title(f"{task}  win start {t:+d} step (x-y, fixture frame, mm)"); a.set_xlabel("x"); a.set_ylabel("y"); a.set_aspect("equal", "datalim")
        if j == 0: a.legend(fontsize=8)
plt.rcParams["font.family"] = "sans-serif"
plt.tight_layout(); plt.savefig(K / "figs/entry_geometry_20261015.png", dpi=110)
print(json.dumps(res, indent=1, ensure_ascii=False)[:9000])
