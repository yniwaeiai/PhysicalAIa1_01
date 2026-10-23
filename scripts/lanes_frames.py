"""L1 補助: 窓ごとの固定具座標系 (pf, Rf) を entry_geometry.py と同じ手順で求めて保存。MUJOCO_GL=egl 必要。"""
import sys, csv
from pathlib import Path
import numpy as np, torch
K = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(K / "scripts"))
import eval_kadai_closedloop as E
rows = [r for r in csv.DictReader(open(K / "state/branches_i3_20261015.csv")) if r["kind"] == "t0" and r["k"] == "0"]
envs = {}; rowsd = {}; out = {}
for w in rows:
    task, seed, rep = w["task"], int(w["seed"]), int(w["rep"]); spec = E.TASKS[task]
    if task not in envs:
        raw = torch.load(E.LPR / "init_files" / spec["suite"] / f"{spec['base']}.pruned_init", weights_only=False)
        rowsd[task] = raw.numpy() if hasattr(raw, "numpy") else np.asarray(raw); envs[task] = E.make_env(spec)
    d = np.load(K / f"state/trace_aprime/{task}_s{seed}_r{rep}.npz", allow_pickle=True)
    state = {"pos0": np.asarray(d["ref_pos0"]), "quat0": np.asarray(d["ref_quat0"])}
    obs, sim, m, qadr, prng = E.init_episode(envs[task], spec, rowsd[task], seed, str(d["level"]), state)
    rid = m.body_name2id(spec["ref"])
    sim.set_state_from_flattened(d["sim_state"][1]); sim.forward()
    out[w["wid"]] = np.r_[sim.data.body_xpos[rid], sim.data.body_xmat[rid]]
    print(w["wid"], flush=True)
np.savez(K / "state/lanes_frames_20261016.npz", **out)
