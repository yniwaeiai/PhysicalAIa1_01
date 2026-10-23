#!/usr/bin/env python3
"""P1b: drawer の E2(1000〜1047)・E3m(6000〜6047) を v1r / v3 / v3s2000 で。水準別、none 水準(E2+E3m 合算)の clean を主指標。seed 単位 bootstrap B=10000。"""
import glob, json, numpy as np, collections
D = "kadai/state/v3_eval_20261016"; M = ["v1r", "v3", "v3s2000"]
C = {"E2": 1000, "E3m": 6000}
def lvl(seed):
    r = np.random.default_rng((seed, 777)).random()
    return "none" if r < 0.25 else ("mid" if r < 0.75 else "strong")
R = {}; dup = 0
for m in M:
    for c in C:
        for p in glob.glob(f"{D}/{m}_{c}_drawer_s*.json.jsonl"):
            for l in open(p):
                if l.strip():
                    r = json.loads(l); k = (m, c, int(r["seed"])); dup += k in R; R[k] = r
units = [(c, s) for c, o in C.items() for s in range(o, o + 48)]
o = ["# P1b: drawer の E2・E3m（無摂動水準を含む）v3@2000 の直接確認", "", "## 操作チェック", "",
     "| モデル | 条件 | 本数 | seed 範囲 | 欠け | 水準の内訳（行の level）| 水準が pick_level と一致 |", "|---|---|---|---|---|---|---|"]
LV = {(0.0, 0.0, 0.0): "none", (0.1, 0.03, 5.0): "mid", (0.2, 0.06, 10.0): "strong"}
ok = True
for m in M:
    for c, off in C.items():
        ss = sorted(s for (mm, cc, s) in R if mm == m and cc == c)
        miss = sorted(set(range(off, off + 48)) - set(ss)); ok &= not miss
        cnt = collections.Counter(LV.get(tuple(R[(m, c, s)]["level"]), "?") for s in ss)
        match = all(LV.get(tuple(R[(m, c, s)]["level"])) == lvl(s) for s in ss)
        o.append(f"| {m} | {c} | {len(ss)} | {ss[0] if ss else '-'}〜{ss[-1] if ss else '-'} | {miss or 'なし'} | {dict(cnt)} | {match} |")
o += ["", f"重複 {dup}。揃い: {ok}", ""]
if not ok:
    open("kadai/state/v3s2000_e2e3m_drawer_check_20261016.md", "w").write("\n".join(o)); raise SystemExit("欠け")
def vec(m, key, us): return np.array([float(R[(m, c, s)][key]) for c, s in us])
sel = {"none": [u for u in units if lvl(u[1]) == "none"], "mid": [u for u in units if lvl(u[1]) == "mid"],
       "strong": [u for u in units if lvl(u[1]) == "strong"], "全水準": units}
rng = np.random.default_rng(20261016)
def boot(x):
    idx = rng.integers(0, len(x), (10000, len(x))); b = x[idx].mean(1); return x.mean(), np.percentile(b, 2.5), np.percentile(b, 97.5)
o += ["## 水準別 done / clean（E2+E3m 合算、drawer）", "", "| 水準 | 単位数 | v1r done | v3 done | v3s2000 done | v1r clean | v3 clean | v3s2000 clean |", "|---|---|---|---|---|---|---|---|"]
for k, us in sel.items():
    o.append(f"| {k} | {len(us)} | " + " | ".join(f"{vec(m, key, us).mean():.3f}" for key in ("success", "clean") for m in M) + " |")
o += ["", "条件別（参考）:", "", "| 条件 | 水準 | 単位数 | v1r clean | v3 clean | v3s2000 clean | v1r done | v3 done | v3s2000 done |", "|---|---|---|---|---|---|---|---|---|"]
for c in C:
    for k in ("none", "mid", "strong"):
        us = [u for u in sel[k] if u[0] == c]
        o.append(f"| {c} | {k} | {len(us)} | " + " | ".join(f"{vec(m, key, us).mean():.3f}" for key in ("clean", "success") for m in M) + " |")
o += ["", "## seed 対応差（95% 区間は seed 単位 bootstrap, B=10000）。★主指標 = none 水準の clean", "", "| 比較 | 水準 | 指標 | 差 | 95% 区間 |", "|---|---|---|---|---|"]
for new, base in (("v3s2000", "v3"), ("v3s2000", "v1r"), ("v3", "v1r")):
    for k, us in sel.items():
        for key, kn in (("clean", "clean"), ("success", "done")):
            mu, lo, hi = boot(vec(new, key, us) - vec(base, key, us))
            star = " ★" if (k == "none" and kn == "clean") else ""
            o.append(f"| {new} − {base} | {k}{star} | {kn} | {mu:+.3f} | [{lo:+.3f}, {hi:+.3f}] |")
open("kadai/state/v3s2000_e2e3m_drawer_check_20261016.md", "w").write("\n".join(o) + "\n"); print("\n".join(o))
