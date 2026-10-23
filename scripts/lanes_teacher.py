"""L1d 補助: 教師 demo（kadai_demos_v1）の手先位置を固定具座標へ。配置は生成時の規則（--mix: pick_level('mix', seed)、(seed,20261013) の乱数）から再構成し、
state[0] の手先位置が再構成した sim の初期手先位置と一致するかで検証する。出力 state/lanes_teacher_20261016.npz。MUJOCO_GL=egl"""
import sys, glob
from pathlib import Path
import numpy as np, torch, h5py
K = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(K / "scripts"))
import eval_kadai_closedloop as E
FILES = {"drawer": "open_the_middle_drawer", "k4": "close_the_bottom_drawer", "k6": "close_the_microwave"}
NS = 41
out = {}
for task, key in FILES.items():
    spec = E.TASKS[task]
    f = [x for x in glob.glob(str(K.parent / "data/kadai_demos_v1/*/*.hdf5")) if key in x][0]
    raw = torch.load(E.LPR / "init_files" / spec["suite"] / f"{spec['base']}.pruned_init", weights_only=False)
    rows = raw.numpy() if hasattr(raw, "numpy") else np.asarray(raw)
    env = E.make_env(spec); state = {}
    h = h5py.File(f, "r"); P = []; err = []; seeds = []; lvl = []
    for k in h:
        seed = int(h[k].attrs["seed"]); st = h[k]["state"][:]
        obs, sim, m, qadr, prng = E.init_episode(env, spec, rows, seed, "mix", state)
        rid = m.body_name2id(spec["ref"]); pf = sim.data.body_xpos[rid].copy(); Rf = sim.data.body_xmat[rid].reshape(3, 3).copy()
        e0 = np.asarray(obs["robot0_eef_pos"]); err.append(np.linalg.norm(e0 - st[0, :3]))
        n = min(NS, len(st)); pe = (st[:n, :3] - pf) @ Rf; pe = np.vstack([pe, np.full((NS - n, 3), np.nan)])
        P.append(pe); seeds.append(seed); lvl.append(E.pick_level("mix", seed)[0])
        # y 鏡映の検査用に鏡映版の誤差も
    err = np.array(err)
    print(task, len(P), "init eef err mm: median", np.median(err) * 1000, "max", err.max() * 1000, "frac<1mm", (err < 1e-3).mean(), flush=True)
    out[task] = np.array(P); out[task + "_seed"] = np.array(seeds); out[task + "_err"] = err
np.savez(K / "state/lanes_teacher_20261016.npz", **out)
