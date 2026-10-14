"""LIBERO-90 の単独タスク（K4 下段引き出しを閉める、K6 電子レンジを閉める）の人間デモ 50 本について、接触区間の手首の傾きを集計する。
窓の定義は analyze_contact_windows.py と同じ（操作対象の関節の変位が最大の 5% を超えた時点 → 90% に達した時点）。
出力: kadai/state/single_tilt_summary.json
"""
import json, os, sys
import h5py, numpy as np
sys.path.insert(0, os.path.dirname(__file__))
from analyze_wrist_tilt import rotvec_to_mat, tilt_az
from analyze_contact_windows import windows
from libero.libero import get_libero_path
from libero.libero.envs import OffScreenRenderEnv

R = "/path/to/work/data/libero_datasets/libero_90/"
T = {"k4_single_close_drawer": ("KITCHEN_SCENE4_close_the_bottom_drawer_of_the_cabinet_demo.hdf5", "white_cabinet_1_bottom_level", 0.03),
     "k6_single_close_microwave": ("KITCHEN_SCENE6_close_the_microwave_demo.hdf5", "microwave_1_microjoint", 0.2)}
out = {}
for name, (fn, jn, min_ext) in T.items():
    f = h5py.File(R + fn, "r")
    env = OffScreenRenderEnv(bddl_file_name=os.path.join(get_libero_path("bddl_files"), f["data"].attrs["bddl_file_name"].split("bddl_files/")[1]), camera_heights=64, camera_widths=64)
    env.reset(); m = env.sim.model; qa = m.jnt_qposadr[m.joint_name2id(jn)]
    med, mx, lens = [], [], []
    for k in f["data"]:
        e = f["data"][k]; S = e["states"][:]; q = S[:, 1:1 + m.nq]
        sig = np.abs(q[:, qa] - q[0, qa]); w = windows(sig, min_ext).get("out")
        if not w:
            continue
        a, b = w
        tl = np.array([tilt_az(rotvec_to_mat(o))[0] for o in e["obs/ee_ori"][a:b + 1]])
        med.append(float(np.median(tl))); mx.append(float(tl.max())); lens.append(b - a)
    out[name] = {"n": len(med), "window_len_median": float(np.median(lens)), "tilt_med_per_demo": {"p10": float(np.percentile(med, 10)), "median": float(np.median(med)), "p90": float(np.percentile(med, 90))},
                 "tilt_max_median": float(np.median(mx)), "frac_demos_median_tilt_gt_30": float(np.mean(np.array(med) > 30))}
    print(name, out[name])
json.dump(out, open(os.path.join(os.path.dirname(__file__), "..", "state", "single_tilt_summary.json"), "w"), indent=1)
