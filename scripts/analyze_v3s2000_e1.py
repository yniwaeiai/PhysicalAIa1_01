#!/usr/bin/env python3
"""再開の損傷の切り分け: E1 のみ。v1r / v3 / v3s2000 の done・clean をタスク別に、seed 対応差を seed 単位 bootstrap(B=10000, seed 固定)で。
使い方: python analyze_v3s2000_e1.py [--dir kadai/state/v3_eval_20261016] [--out kadai/state/v3s2000_e1_check_20261016.md]"""
import argparse, glob, json, os, collections
import numpy as np
ap = argparse.ArgumentParser()
ap.add_argument("--dir", default="kadai/state/v3_eval_20261016")
ap.add_argument("--out", default="kadai/state/v3s2000_e1_check_20261016.md")
a = ap.parse_args()
M = ["v1r", "v3", "v3s2000"]; T = ["drawer", "k4", "k6"]
D = {}; dup = 0
for m in M:
    for t in T:
        for p in sorted(glob.glob(f"{a.dir}/{m}_E1_{t}_s*.json.jsonl")):
            for l in open(p):
                if l.strip():
                    r = json.loads(l); k = (m, t, int(r["seed"]))
                    dup += k in D; D[k] = r
def vec(m, t, key):
    return np.array([float(D[(m, t, s)][key]) for s in range(3000, 3048)])
rng = np.random.default_rng(20261016)
def boot(x):
    n = len(x); idx = rng.integers(0, n, (10000, n)); b = x[idx].mean(1)
    return x.mean(), np.percentile(b, 2.5), np.percentile(b, 97.5)
o = ["# 再開の損傷の切り分け（E1 のみ、seed 3000〜3047、強い摂動）", ""]
o += ["## 操作チェック", "", "| モデル | タスク | 本数 | seed 範囲 | 欠け |", "|---|---|---|---|---|"]
ok = True
for m in M:
    for t in T:
        ss = sorted(s for (mm, tt, s) in D if mm == m and tt == t)
        miss = sorted(set(range(3000, 3048)) - set(ss)); ok &= not miss
        lv = {tuple(D[(m, t, s)]["level"]) for s in ss}
        o.append(f"| {m} | {t} | {len(ss)} | {ss[0] if ss else '-'}〜{ss[-1] if ss else '-'} | {miss or 'なし'} (水準 {sorted(lv)}) |")
o += ["", f"重複 {dup}。全 144 本 x 3 モデル揃い: {ok}", ""]
if not ok:
    open(a.out, "w").write("\n".join(o)); raise SystemExit("欠けあり")
o += ["## タスク別 done / clean", "", "| タスク | v1r done | v3 done | v3s2000 done | v1r clean | v3 clean | v3s2000 clean |", "|---|---|---|---|---|---|---|"]
tab = {}
for t in T + ["all"]:
    row = []
    for key in ("success", "clean"):
        for m in M:
            x = np.concatenate([vec(m, tt, key) for tt in T]) if t == "all" else vec(m, t, key)
            row.append(f"{x.mean():.3f}")
    o.append(f"| {t} | " + " | ".join(row) + " |")
o += ["", "## seed 対応差（95% 区間は seed 単位 bootstrap, B=10000）", "", "| 比較 | タスク | 指標 | 差 | 95% 区間 |", "|---|---|---|---|---|"]
for new, base in (("v3s2000", "v3"), ("v3s2000", "v1r"), ("v3", "v1r")):
    for t in T + ["drawer+k6", "all"]:
        for key, kn in (("clean", "clean"), ("success", "done")):
            ts = {"drawer+k6": ["drawer", "k6"], "all": T}.get(t, [t])
            x = np.concatenate([vec(new, tt, key) - vec(base, tt, key) for tt in ts])
            mu, lo, hi = boot(x)
            o.append(f"| {new} − {base} | {t} | {kn} | {mu:+.3f} | [{lo:+.3f}, {hi:+.3f}] |")
open(a.out, "w").write("\n".join(o) + "\n"); print("\n".join(o[-60:]))
