"""F（計画 v2 §11）: 分岐点 t の状態の特徴と p（8 分岐の最終成功率）の関係。CPU のみ。判定は書かず値だけ出す。
使い方: MUJOCO_GL=egl nice -n 10 /path/to/work/.venv/bin/python kadai/scripts/analyze_state_features.py
入力: state/branches_i3_20261015.csv, state/aprime_windows_20261014.csv, state/trace_aprime/*.npz
出力: state/state_features_20261015.csv / .md, figs/state_features_20261015_<task>.png
規約: 分岐 t = sim_state 行 t を復元（branch_kadai.py）= trace の ee/quat/grip/q/tgt_dist の添字 t-1 と同じ時点。
固定具座標系 = sim の ref body の xpos/xmat（entry_geometry.py と同じ）。対象 body 位置は sim_state[t] を復元して読む（tgt_pos は使わない）。
"""
import sys, csv, json, collections
from pathlib import Path
import numpy as np
K = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(K / "scripts"))
import eval_kadai_closedloop as E
from scipy.spatial.transform import Rotation as Rot
from scipy.stats import rankdata
import torch
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

FEATS = ["x_mm", "y_mm", "z_mm", "tilt_deg", "yaw_deg", "tgtdist_mm", "dir_x", "dir_y", "dir_z", "grip_open", "fix_q", "t"]
rng = np.random.default_rng(20261015)
wins = {(w["task"], int(w["seed"]), int(w["rep"])): w for w in csv.DictReader(open(K / "state/aprime_windows_20261014.csv"))}
br = list(csv.DictReader(open(K / "state/branches_i3_20261015.csv")))
W = collections.OrderedDict()
for r in br:
    if r["kind"] not in ("t13", "t23", "succ13", "succ23"):
        continue
    d = W.setdefault(r["wid"], dict(wid=r["wid"], task=r["task"], seed=int(r["seed"]), rep=int(r["rep"]), kind=r["kind"], group=r["group"], t=int(r["t"]), succ=[]))
    assert d["t"] == int(r["t"])
    d["succ"].append(int(r["succ"]))
for d in W.values():
    d["p"] = float(np.mean(d["succ"])); d["n"] = len(d["succ"])
    d["tt"] = "13" if d["kind"].endswith("13") else "23"; d["is_succ"] = d["kind"].startswith("succ")
    if (d["task"], d["seed"], d["rep"]) in wins:   # 失敗窓（と偽陽性）は窓の開始 s がある。成功対照は A' の窓表に無いので s = t × 3 or 1.5（記録用）
        s = int(wins[(d["task"], d["seed"], d["rep"])]["win_start"])
        if not d["is_succ"]: assert abs(d["t"] - s / (3 if d["tt"] == "13" else 1.5)) <= 1, (d["wid"], d["t"], s)   # 成功対照の t は窓の s と一致しない場合がある（t 列をそのまま使う）
    else:
        assert d["is_succ"], d["wid"]; s = int(round(d["t"] * (3 if d["tt"] == "13" else 1.5)))
    d["s"] = s
print("windows", len(W), collections.Counter((d["task"], d["kind"]) for d in W.values()))

def tiltf(quat): return E.tilt_deg(quat)

def features(env_pack, d, tr, t):
    """t 時点（sim_state 行 t、trace 添字 t-1）の特徴を返す。"""
    sim, m, rid, tbid = env_pack
    i = t - 1
    sim.set_state_from_flattened(tr["sim_state"][t]); sim.forward()
    Rf = sim.data.body_xmat[rid].reshape(3, 3); pf = sim.data.body_xpos[rid].copy()
    ee = tr["ee"][i]; Re = Rot.from_quat(tr["quat"][i]).as_matrix()
    pe = Rf.T @ (ee - pf); Rrel = Rf.T @ Re
    tp = sim.data.body_xpos[tbid].copy()
    v = Rf.T @ (tp - ee); dist = float(np.linalg.norm(v))
    # 操作チェック: sim で読んだ距離は trace の tgt_dist（正しい）と一致するか
    chk = abs(dist - float(tr["tgt_dist"][i]))
    f = dict(x_mm=pe[0] * 1e3, y_mm=pe[1] * 1e3, z_mm=pe[2] * 1e3, tilt_deg=tiltf(tr["quat"][i]),
             yaw_deg=float(np.degrees(np.arctan2(Rrel[1, 0], Rrel[0, 0]))), tgtdist_mm=dist * 1e3,
             dir_x=v[0] / dist, dir_y=v[1] / dist, dir_z=v[2] / dist,
             grip_open=float(tr["grip"][i][0] - tr["grip"][i][1]), fix_q=float(tr["q"][i]), t=float(t))
    return f, chk

envs = {}; rows = []; chks = []
for d in W.values():
    task = d["task"]; spec = E.TASKS[task]
    if task not in envs:
        raw = torch.load(E.LPR / "init_files" / spec["suite"] / f"{spec['base']}.pruned_init", weights_only=False)
        envs[task] = (E.make_env(spec), raw.numpy() if hasattr(raw, "numpy") else np.asarray(raw))
    env, rowsd = envs[task]
    tr = np.load(K / f"state/trace_aprime/{task}_s{d['seed']}_r{d['rep']}.npz", allow_pickle=True)
    state = {"pos0": np.asarray(tr["ref_pos0"]), "quat0": np.asarray(tr["ref_quat0"])}
    obs, sim, m, qadr, prng = E.init_episode(env, spec, rowsd, d["seed"], str(tr["level"]), state)
    rid = m.body_name2id(spec["ref"]); tbid = int(m.jnt_bodyid[m.joint_name2id(spec["joint"])])
    f, chk = features((sim, m, rid, tbid), d, tr, d["t"]); chks.append((d["wid"], chk))
    rows.append(dict(wid=d["wid"], task=task, seed=d["seed"], rep=d["rep"], group=d["group"], kind=d["kind"], tt=d["tt"], is_succ=int(d["is_succ"]), s=d["s"], p=d["p"], n=d["n"], **f))
    print(d["wid"], round(d["p"], 3), flush=True)

# --- 座標系の検算: entry_geometry の 1 窓（t=0 → i=s, sim_state[i+1]）と同じ式を同じ窓で
cmp = None
ent = {(r["task"], int(r["seed"]), int(r["rep"])): r for r in csv.DictReader(open(K / "state/entry_geometry_20261015.csv"))}
d = next(x for x in W.values() if not x["is_succ"]); task = d["task"]; env, rowsd = envs[task]
tr = np.load(K / f"state/trace_aprime/{task}_s{d['seed']}_r{d['rep']}.npz", allow_pickle=True)
state = {"pos0": np.asarray(tr["ref_pos0"]), "quat0": np.asarray(tr["ref_quat0"])}
obs, sim, m, qadr, prng = E.init_episode(env, E.TASKS[task], rowsd, d["seed"], str(tr["level"]), state)
rid = m.body_name2id(E.TASKS[task]["ref"]); tbid = int(m.jnt_bodyid[m.joint_name2id(E.TASKS[task]["joint"])])
s = d["s"]; f, _ = features((sim, m, rid, tbid), d, tr, s + 1)   # t = s+1 → 添字 s、sim_state[s+1]
e = ent[(task, d["seed"], d["rep"])]
cmp = dict(wid=d["wid"], s=s, mine=dict(x=f["x_mm"] / 1e3, y=f["y_mm"] / 1e3, z=f["z_mm"] / 1e3, tilt=f["tilt_deg"], yaw=f["yaw_deg"]),
           entry=dict(x=float(e["x0"]), y=float(e["y0"]), z=float(e["z0"]), tilt=float(e["tilt0"]), yaw=float(e["yaw0"])))
print("検算", cmp)
print("tgt_dist 一致 最大差 (m):", max(c for _, c in chks))

# --- CSV
cols = ["wid", "task", "seed", "rep", "group", "kind", "tt", "is_succ", "s", "p", "n"] + FEATS
with open(K / "state/state_features_20261015.csv", "w", newline="") as fh:
    wr = csv.DictWriter(fh, fieldnames=cols); wr.writeheader(); wr.writerows(rows)

# --- 解析
def spearman(x, y):
    if np.ptp(x) == 0 or np.ptp(y) == 0: return np.nan
    return float(np.corrcoef(rankdata(x), rankdata(y))[0, 1])
def perm_p(x, y, n=1000):
    r0 = spearman(x, y)
    if np.isnan(r0): return np.nan
    rx = rankdata(x); cnt = 0
    for _ in range(n):
        yy = rng.permutation(y)
        if abs(np.corrcoef(rx, rankdata(yy))[0, 1]) >= abs(r0) - 1e-12: cnt += 1
    return (cnt + 1) / (n + 1)
def auroc(hi, lo):
    if len(hi) == 0 or len(lo) == 0: return np.nan
    a = np.asarray(hi)[:, None]; b = np.asarray(lo)[None, :]
    return float(((a > b).sum() + 0.5 * (a == b).sum()) / (a.size * b.size))

def sel(task, tt): return [r for r in rows if r["task"] == task and r["tt"] == tt]
md = []; res = {}
for task in ("drawer", "k6", "k4"):
    for tt in ("13", "23"):
        R = sel(task, tt); p = np.array([r["p"] for r in R]); hi = p >= 0.5; lo = p < 0.25
        md.append(f"\n### {task}  t={'s/3' if tt=='13' else '2s/3'}  n={len(R)}（失敗窓 {sum(1 for r in R if not r['is_succ'])} + 成功対照 {sum(r['is_succ'] for r in R)}）、p≥0.5: {int(hi.sum())}、p<0.25: {int(lo.sum())}、p の平均 {p.mean():.3f}")
        md.append("| 特徴 | Spearman | 順列 p | AUROC(p≥0.5 を高と見たとき) | 平均 p≥0.5 | 平均 p<0.25 |\n|---|---|---|---|---|---|")
        for ft in FEATS:
            x = np.array([r[ft] for r in R], float)
            rho = spearman(x, p); pp = perm_p(x, p) if task != "k4" else np.nan
            au = auroc(x[hi], x[lo])
            res[(task, tt, ft)] = (rho, pp, au)
            m1 = x[hi].mean() if hi.any() else np.nan; m0 = x[lo].mean() if lo.any() else np.nan
            md.append(f"| {ft} | {rho:+.3f} | {'' if np.isnan(pp) else f'{pp:.3f}'} | {au:.3f} | {m1:.3g} | {m0:.3g} |")
best = {}
for task in ("drawer", "k6"):
    for tt in ("13", "23"):
        cand = [(abs(res[(task, tt, ft)][0]), ft) for ft in FEATS if not np.isnan(res[(task, tt, ft)][0])]
        best[(task, tt)] = max(cand)[1]
md.append("\n## 最良特徴（|Spearman| 最大。t 特徴・fix_q を含む）")
for k, ft in best.items(): md.append(f"- {k[0]} t={k[1]}: {ft}  rho={res[(k[0],k[1],ft)][0]:+.3f}  順列p={res[(k[0],k[1],ft)][1]:.3f}  AUROC={res[(k[0],k[1],ft)][2]:.3f}")
# 幾何特徴のみ（t, fix_q を除く）の最良も併記（R12 は「幾何特徴」）
GEOM = [f for f in FEATS if f not in ("t", "fix_q")]
md.append("\n## 最良の幾何特徴（t・fix_q を除く）")
bestg = {}
for task in ("drawer", "k6"):
    for tt in ("13", "23"):
        cand = [(abs(res[(task, tt, ft)][0]), ft) for ft in GEOM if not np.isnan(res[(task, tt, ft)][0])]
        bestg[(task, tt)] = max(cand)[1]; ft = bestg[(task, tt)]
        md.append(f"- {task} t={tt}: {ft}  rho={res[(task,tt,ft)][0]:+.3f}  順列p={res[(task,tt,ft)][1]:.3f}  AUROC={res[(task,tt,ft)][2]:.3f}")

# R13（s/3、最良特徴＝全特徴での最良と幾何のみの最良の両方）
md.append("\n## R13: 成功未観測 seed の s/3 の最良特徴が p≥0.5 の窓の 5〜95 パーセンタイル域の外にある seed 数")
r13 = {}
for label, bd in (("全特徴の最良", best), ("幾何のみの最良", bestg)):
    tot_out = tot_n = 0
    for task in ("drawer", "k6"):
        ft = bd[(task, "13")]; R = sel(task, "13")
        ref = np.array([r[ft] for r in R if r["p"] >= 0.5])
        if len(ref) < 3:
            md.append(f"- [{label}] {task}: p≥0.5 の窓が {len(ref)} 本で域を定義できない"); continue
        lo_, hi_ = np.percentile(ref, [5, 95])
        seeds = collections.defaultdict(list)
        for r in R:
            if r["group"] == "unseen" and not r["is_succ"]: seeds[r["seed"]].append(r[ft])
        out = {sd: float(np.median(v)) for sd, v in seeds.items()}
        n_out = sum(1 for v in out.values() if v < lo_ or v > hi_)
        tot_out += n_out; tot_n += len(out)
        md.append(f"- [{label}] {task}: 特徴 {ft}、基準域 [{lo_:.4g}, {hi_:.4g}]（p≥0.5 の窓 {len(ref)} 本）、unseen seed {len(out)} 本中 域外 {n_out}（" + ", ".join(f"{sd}:{v:.4g}" for sd, v in sorted(out.items())) + "）")
        r13[(label, task)] = n_out
    md.append(f"- [{label}] 合計: {tot_out} / {tot_n}")

# 補助: 同 seed の成功対照 − 失敗窓（s/3）。seed 内平均 → 差
md.append("\n## 補助: 同じ seed の成功対照 − 失敗窓（s/3、seed 内の窓平均の差。成功対照があった seed のみ）")
for task in ("drawer", "k6", "k4"):
    R = sel(task, "13"); seeds = sorted({r["seed"] for r in R if r["is_succ"]})
    md.append(f"\n{task}: seed {len(seeds)} 本\n| seed | 群 | 失敗窓数/対照数 | " + " | ".join(FEATS) + " |\n|" + "---|" * (3 + len(FEATS)))
    dd = []
    for sd in seeds:
        F = [r for r in R if r["seed"] == sd and not r["is_succ"]]; S = [r for r in R if r["seed"] == sd and r["is_succ"]]
        if not F: continue
        diff = [np.mean([r[ft] for r in S]) - np.mean([r[ft] for r in F]) for ft in FEATS]; dd.append(diff)
        md.append(f"| {sd} | {F[0]['group']} | {len(F)}/{len(S)} | " + " | ".join(f"{v:+.3g}" for v in diff) + " |")
    if dd:
        dd = np.array(dd)
        md.append("| 平均 | | | " + " | ".join(f"{v:+.3g}" for v in dd.mean(0)) + " |")
        md.append("| 符号が正の seed 数 | | | " + " | ".join(f"{int((dd[:, j] > 0).sum())}/{len(dd)}" for j in range(len(FEATS))) + " |")

# 図
for task in ("drawer", "k6"):
    fig, axs = plt.subplots(1, 2, figsize=(11, 4.5))
    cmap = {"unseen": "tab:red", "luck": "tab:green", "mid": "tab:orange", "hold": "tab:purple"}
    for ax, tt in zip(axs, ("13", "23")):
        ft = best[(task, tt)]; R = sel(task, tt)
        for g, c in cmap.items():
            for succ, mk in ((0, "o"), (1, "s")):
                Q = [r for r in R if r["group"] == g and r["is_succ"] == succ]
                if Q: ax.scatter([r[ft] for r in Q], [r["p"] for r in Q], c=c, marker=mk, edgecolor="k" if succ else "none", alpha=.8, label=f"{g}{' 成功対照' if succ else ''}")
        rho, pp, au = res[(task, tt, ft)]
        ax.set_xlabel(ft); ax.set_ylabel("p (8 分岐の成功率)"); ax.set_title(f"{task} t={'s/3' if tt=='13' else '2s/3'}  best={ft}  rho={rho:+.2f} perm p={pp:.3f}")
    axs[0].legend(fontsize=7)
    plt.tight_layout(); plt.savefig(K / f"figs/state_features_20261015_{task}.png", dpi=110); plt.close()

hdr = ["# F: 分岐点の状態の特徴と p（2026-10-15、`scripts/analyze_state_features.py`、窓ごとの値は `state_features_20261015.csv`）",
       "p = その窓・t の 8 分岐の最終成功率。特徴 = 分岐点 t（sim_state 行 t = trace 添字 t-1）の固定具座標系の値。判定は書かない。順列 p は 1000 回・両側、補正なし（特徴 12 × t 2 × タスク 2）。AUROC は『p≥0.5 の窓の特徴が p<0.25 の窓より大きい確率』（0.5 未満 = 高 p 側で小さい）。k4 は値のみ（順列なし）。",
       f"\n## 検算\n- 座標系: 窓 {cmp['wid']}（s={cmp['s']}）を entry_geometry.py の t=0 と同じ式（i=s, sim_state[i+1]）で計算: 本スクリプト {cmp['mine']} / entry_geometry csv {cmp['entry']}",
       f"- 対象距離: sim で読んだ対象 body までの距離と trace の tgt_dist の差の最大 {max(c for _, c in chks):.2e} m（{len(chks)} 窓）"]
open(K / "state/state_features_20261015.md", "w").write("\n".join(hdr + md) + "\n")
print("done")
