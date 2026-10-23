"""進入の幾何 G1-G3（計画 v2 §9）。既存 trace のみ・CPU。
使い方: MUJOCO_GL=egl nice -n 10 /path/to/work/.venv/bin/python kadai/scripts/entry_geometry.py
出力: kadai/state/entry_geometry_20261015.csv, kadai/state/entry_geometry_stats_20261015.json（md は手書き）, kadai/figs/entry_geometry_20261015.png
固定具座標系 = 基準 body（ref）の座標系（sim から body_xpos/xmat を読む）。再現式との一致も記録。"""
import sys, csv, json
from pathlib import Path
import numpy as np
K = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(K / "scripts"))
import eval_kadai_closedloop as E
from scipy.spatial.transform import Rotation as Rot
import torch

TS = [-10, 0, 20]
wins = list(csv.DictReader(open(K / "state/aprime_windows_20261014.csv")))
envs = {}; rowsd = {}
out = []; chk = []
for w in wins:
    task, seed, rep, s = w["task"], int(w["seed"]), int(w["rep"]), int(w["win_start"])
    spec = E.TASKS[task]
    if task not in envs:
        raw = torch.load(E.LPR / "init_files" / spec["suite"] / f"{spec['base']}.pruned_init", weights_only=False)
        rowsd[task] = raw.numpy() if hasattr(raw, "numpy") else np.asarray(raw)
        envs[task] = E.make_env(spec)
    d = np.load(K / f"state/trace_aprime/{task}_s{seed}_r{rep}.npz", allow_pickle=True)
    env = envs[task]
    state = {"pos0": np.asarray(d["ref_pos0"]), "quat0": np.asarray(d["ref_quat0"])}
    level = str(d["level"]); rn, cs, cy = E.pick_level(level, seed)
    obs, sim, m, qadr, prng = E.init_episode(env, spec, rowsd[task], seed, level, state)
    rid = m.body_name2id(spec["ref"])
    # 再現式（手順 1 の独立計算）
    p0, q0 = state["pos0"], state["quat0"]; pr = np.random.default_rng((seed, 20261013))
    pos_r, quat_r, yaw_d = p0.copy(), q0.copy(), 0.0
    dxy = np.zeros(2)
    if cs or cy:
        dxy = pr.uniform(-cs, cs, 2); pos_r = p0 + np.array([dxy[0], dxy[1], 0.0])
        yaw_d = pr.uniform(-cy, cy); yw = np.radians(yaw_d)
        quat_r = (Rot.from_euler("z", yw) * Rot.from_quat(np.r_[q0[1:], q0[0]])).as_quat(); quat_r = np.r_[quat_r[3], quat_r[:3]]
    dpos = np.abs(m.body_pos[rid] - pos_r).max(); dq = min(np.abs(m.body_quat[rid] - quat_r).max(), np.abs(m.body_quat[rid] + quat_r).max())
    if len(chk) < 400: chk.append((task, seed, rep, float(dpos), float(dq)))
    row = dict(task=task, seed=seed, rep=rep, kind="fail" if w["success"] == "0" else "fp", group=w["group"], win_start=s, rn=rn, cs=cs, cy=cy,
               shift_mm=float(np.linalg.norm(dxy) * 1000), yaw_deg=float(yaw_d))
    for t in TS:
        i = min(max(s + t, 0), len(d["ee"]) - 1)
        sim.set_state_from_flattened(d["sim_state"][i + 1]); sim.forward()
        Rf = sim.data.body_xmat[rid].reshape(3, 3); pf = sim.data.body_xpos[rid].copy()
        if t == 0 and len(chk) <= 400:
            chk[-1] = chk[-1] + (float(np.abs(pf - m.body_pos[rid]).max()),)
        ee = d["ee"][i]; Re = Rot.from_quat(d["quat"][i]).as_matrix()
        pe = Rf.T @ (ee - pf); Rrel = Rf.T @ Re
        tilt = float(np.degrees(np.arccos(np.clip(-Re[2, 2], -1, 1))))
        yaw = float(np.degrees(np.arctan2(Rrel[1, 0], Rrel[0, 0])))
        tp = Rf.T @ (d["tgt_pos"][i] - pf)
        row.update({f"x{t}": pe[0], f"y{t}": pe[1], f"z{t}": pe[2], f"tilt{t}": tilt, f"yaw{t}": yaw, f"hx{t}": pe[0] - tp[0], f"hy{t}": pe[1] - tp[1], f"hz{t}": pe[2] - tp[2],
                    f"prog_q{t}": float(d["q"][i])})
        if task == "k6":
            door = m.body_name2id("microwave_1_microdoorroot")
            gb = [g for g in range(m.ngeom) if m.geom_bodyid[g] == door]
            panel = [g for g in gb if m.geom_type[g] == 6][0]; hand = [g for g in gb if m.geom_type[g] == 3]
            gp = sim.data.geom_xpos[panel].copy(); Rg = sim.data.geom_xmat[panel].reshape(3, 3); sz = m.geom_size[panel]
            corners = np.array([gp + Rg @ (np.array([a, b, c]) * sz) for a in (-1, 1) for b in (-1, 1) for c in (-1, 1)])
            hc = np.mean([sim.data.geom_xpos[g] for g in hand], axis=0)
            n = Rg[:, 1]; n = n if np.dot(n, hc - gp) > 0 else -n   # 取っ手の側 = 手前（外向き）
            row.update({f"edge_dz{t}": float(ee[2] - corners[:, 2].min()), f"door_n{t}": float(np.dot(n, ee - gp)), f"door_top_dz{t}": float(ee[2] - corners[:, 2].max())})
    out.append(row)
    print(task, seed, rep, row["kind"], flush=True)
cols = list(out[0].keys())
for r in out:
    for c in cols: r.setdefault(c, "")
all_cols = []
for r in out:
    for c in r:
        if c not in all_cols: all_cols.append(c)
with open(K / "state/entry_geometry_20261015.csv", "w", newline="") as f:
    wr = csv.DictWriter(f, fieldnames=all_cols); wr.writeheader(); wr.writerows(out)
json.dump(dict(chk=chk), open(K / "state/entry_geometry_check_20261015.json", "w"))
