"""閉める押し（K4 下段引き出し、K6 電子レンジ）の人間デモから、固定物の座標系でのキーポーズと接触部位を抽出する。
使い方: MUJOCO_GL=egl python extract_close_keyposes.py k4_bowl_drawer|microwave_k6
出力: kadai/state/close_keyposes_<base>.json（位置は中央値・姿勢は回転平均）と、窓の途中での手の接触部位の集計。
"""
import json, os, sys, collections
import h5py, numpy as np
from scipy.spatial.transform import Rotation as Rot
sys.path.insert(0, os.path.dirname(__file__))
from analyze_wrist_tilt import ROOT, BASES, rotvec_to_mat
from libero.libero import get_libero_path
from libero.libero.envs import OffScreenRenderEnv

SPEC = {"k4_bowl_drawer": ("white_cabinet_1_bottom_level", "white_cabinet_1_main", "white_cabinet"),
        "microwave_k6": ("microwave_1_microjoint", "microwave_1_main", "microwave")}
name = sys.argv[1]; jn, refn, tag = SPEC[name]
OFFS = [-60, -40, -25, -12, 0, 8, 16, "peak"]
f = h5py.File(os.path.join(ROOT, BASES[name]), "r")
bddl = f["data"].attrs["bddl_file_name"].split("bddl_files/")[1]
env = OffScreenRenderEnv(bddl_file_name=os.path.join(get_libero_path("bddl_files"), bddl), camera_heights=64, camera_widths=64)
env.reset(); sim = env.sim; m = sim.model; d = sim.data
qa = m.jnt_qposadr[m.joint_name2id(jn)]; rid = m.body_name2id(refn)
P = {str(o): [] for o in OFFS}; Q = {str(o): [] for o in OFFS}; contacts = collections.Counter(); wins = []
hand_part = lambda g: str(g).split("_collision")[0].replace("gripper0_", "").replace("robot0_", "")
for k in f["data"]:
    e = f["data"][k]; S = e["states"][:]; q = S[:, 1:1 + m.nq]
    sig = np.abs(q[:, qa] - q[0, qa]); ext = sig.max()
    on = int(np.argmax(sig > 0.05 * ext)); pk = int(np.argmax(sig >= 0.9 * ext)); wins.append((on, pk))
    sim.set_state_from_flattened(S[0]); sim.forward()   # 固定物は動かない
    Rf = d.body_xmat[rid].reshape(3, 3).copy(); pf = d.body_xpos[rid].copy()
    for o in OFFS:
        t = pk if o == "peak" else min(len(S) - 1, max(0, on + o))
        sim.set_state_from_flattened(S[t]); sim.forward()
        pos = d.site_xpos[m.site_name2id("gripper0_grip_site")]
        hid = m.body_name2id([m.body_id2name(i) for i in range(m.nbody) if m.body_id2name(i) and "right_hand" in m.body_id2name(i)][0])
        P[str(o)].append(Rf.T @ (pos - pf)); Q[str(o)].append(Rot.from_matrix(Rf.T @ d.body_xmat[hid].reshape(3, 3)).as_quat())
    for t in range(on, pk + 1, 3):   # 窓の中での接触部位
        sim.set_state_from_flattened(S[t]); sim.forward()
        for i in range(d.ncon):
            c = d.contact[i]; a, b = str(m.geom_id2name(c.geom1)), str(m.geom_id2name(c.geom2))
            for x, y in ((a, b), (b, a)):
                if ("gripper0" in x or "robot0" in x) and tag in y:
                    contacts[f"{hand_part(x)} -> {y}"] += 1
out = {"base": name, "joint": jn, "ref": refn, "window_onset_median": float(np.median([w[0] for w in wins])),
       "window_len_median": float(np.median([w[1] - w[0] for w in wins])), "keyposes": {}, "contacts_in_window": contacts.most_common(12)}
for o in OFFS:
    rm = Rot.from_quat(np.array(Q[str(o)])).mean()
    out["keyposes"][str(o)] = {"pos": np.median(P[str(o)], axis=0).round(4).tolist(), "pos_std": np.std(P[str(o)], axis=0).round(4).tolist(),
                               "quat_xyzw": rm.as_quat().round(5).tolist(),
                               "rot_spread_deg": float(np.degrees(np.median([(rm.inv() * Rot.from_quat(q)).magnitude() for q in Q[str(o)]])))}
    print(o, out["keyposes"][str(o)]["pos"], out["keyposes"][str(o)]["pos_std"], "rot_spread %.0f" % out["keyposes"][str(o)]["rot_spread_deg"])
print("window len median", out["window_len_median"]); print("contacts:", out["contacts_in_window"])
json.dump(out, open(os.path.join(os.path.dirname(__file__), "..", "state", f"close_keyposes_{name}.json"), "w"), indent=1)
