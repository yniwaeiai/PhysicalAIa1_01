"""中段引き出しの人間デモから、棚の座標系でのキーポーズ（手先の位置と全姿勢）を抽出する。
onset = 引き出しの変位が最大の 5% を超えた step、peak = 90% に達した step。
キーポーズは onset を基準にした step オフセットで取り、50 デモで位置は中央値、姿勢は回転の平均（scipy）。
出力: kadai/state/hook_keyposes_middle.json。CPU のみ。
"""
import json, os, sys
import h5py, numpy as np
from scipy.spatial.transform import Rotation as Rot
sys.path.insert(0, os.path.dirname(__file__))
from analyze_wrist_tilt import ROOT, BASES, rotvec_to_mat
from libero.libero import get_libero_path
from libero.libero.envs import OffScreenRenderEnv

OFFS = [-60, -40, -25, -12, -4, 0, 8, "peak"]
rel = BASES["drawer_middle"]
f = h5py.File(os.path.join(ROOT, rel), "r")
bddl = f["data"].attrs["bddl_file_name"].split("bddl_files/")[1]
env = OffScreenRenderEnv(bddl_file_name=os.path.join(get_libero_path("bddl_files"), bddl), camera_heights=64, camera_widths=64)
env.reset(); sim = env.sim; m = sim.model
qa = m.jnt_qposadr[m.joint_name2id("wooden_cabinet_1_middle_level")]
rid = m.body_name2id("wooden_cabinet_1_main")
P = {str(o): [] for o in OFFS}; Q = {str(o): [] for o in OFFS}
for k in f["data"]:
    e = f["data"][k]; S = e["states"][:]; q = S[:, 1:1 + m.nq]
    sig = np.abs(q[:, qa] - q[0, qa]); on = int(np.argmax(sig > 0.05 * sig.max())); pk = int(np.argmax(sig >= 0.9 * sig.max()))
    sim.set_state_from_flattened(S[0]); sim.forward()   # 棚は動かない（引き出しだけ動く）ので t=0 の枠で良い
    Rf = sim.data.body_xmat[rid].reshape(3, 3); pf = sim.data.body_xpos[rid].copy()
    for o in OFFS:
        t = pk if o == "peak" else max(0, on + o)
        pos = e["obs/ee_pos"][t]; R = rotvec_to_mat(e["obs/ee_ori"][t])
        P[str(o)].append(Rf.T @ (pos - pf)); Q[str(o)].append(Rot.from_matrix(Rf.T @ R).as_quat())
out = {"frame": "wooden_cabinet_1_main", "keyposes": {}}
for o in OFFS:
    pm = np.median(P[str(o)], axis=0); ps = np.std(P[str(o)], axis=0)
    rm = Rot.from_quat(np.array(Q[str(o)])).mean()
    out["keyposes"][str(o)] = {"pos": pm.round(4).tolist(), "pos_std": ps.round(4).tolist(), "quat_xyzw": rm.as_quat().round(5).tolist(),
                               "rot_spread_deg": float(np.degrees(np.median([ (rm.inv()*Rot.from_quat(q)).magnitude() for q in Q[str(o)]])))}
    print(o, out["keyposes"][str(o)])
json.dump(out, open(os.path.join(os.path.dirname(__file__), "..", "state", "hook_keyposes_middle.json"), "w"), indent=1)
