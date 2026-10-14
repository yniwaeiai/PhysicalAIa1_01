"""人間デモ（LIBERO 原本 hdf5）の手首の傾き解析。

6 基底それぞれについて、全デモで
  - ツール z 軸（指の向き）の鉛直下向きからの傾き tilt（0° = 真下）
  - 傾きの向き（水平成分の方位 az。世界 +x を 0°、+y を 90°）
  - グリッパを閉じる区間（actions[:,6] が -1→+1 に変わる）の始まりでの姿勢・位置
  - 閉じる直前 10 step の接近方向（ee の速度ベクトルの水平成分）
を出す。CPU のみ・読み取りのみ。出力: kadai/state/wrist_tilt_summary.json, kadai/state/wrist_tilt_per_demo.csv

使い方: python -I kadai/scripts/analyze_wrist_tilt.py
"""
import csv
import json
import os

import h5py
import numpy as np

ROOT = "/path/to/work/data/libero_datasets"
BASES = {
    "drawer_middle": "libero_goal/open_the_middle_drawer_of_the_cabinet_demo.hdf5",
    "drawer_top_bowl": "libero_goal/open_the_top_drawer_and_put_the_bowl_inside_demo.hdf5",
    "push_plate": "libero_goal/push_the_plate_to_the_front_of_the_stove_demo.hdf5",
    "microwave_k6": "libero_10/KITCHEN_SCENE6_put_the_yellow_and_white_mug_in_the_microwave_and_close_it_demo.hdf5",
    "k3_stove_moka": "libero_10/KITCHEN_SCENE3_turn_on_the_stove_and_put_the_moka_pot_on_it_demo.hdf5",
    "k4_bowl_drawer": "libero_10/KITCHEN_SCENE4_put_the_black_bowl_in_the_bottom_drawer_of_the_cabinet_and_close_it_demo.hdf5",
}
OUT = os.path.join(os.path.dirname(__file__), "..", "state")


def rotvec_to_mat(v):
    th = np.linalg.norm(v)
    if th < 1e-9:
        return np.eye(3)
    k = v / th
    K = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
    return np.eye(3) + np.sin(th) * K + (1 - np.cos(th)) * K @ K


def mat_to_rpy(R):
    # ZYX（yaw-pitch-roll）。degrees
    pitch = np.arcsin(-np.clip(R[2, 0], -1, 1))
    roll = np.arctan2(R[2, 1], R[2, 2])
    yaw = np.arctan2(R[1, 0], R[0, 0])
    return np.degrees([roll, pitch, yaw])


def tilt_az(R):
    z = R[:, 2]  # ツール z 軸（世界座標）
    tilt = np.degrees(np.arccos(np.clip(-z[2], -1, 1)))  # 鉛直下向きからの角
    az = np.degrees(np.arctan2(z[1], z[0])) if np.hypot(z[0], z[1]) > 1e-6 else 0.0
    return tilt, az


def close_events(grip_cmd):
    g = np.where(grip_cmd > 0, 1, 0)
    return [i for i in range(1, len(g)) if g[i] == 1 and g[i - 1] == 0]


def main():
    rows, summary = [], {}
    for name, rel in BASES.items():
        with h5py.File(os.path.join(ROOT, rel), "r") as f:
            keys = sorted(f["data"].keys(), key=lambda s: int(s.split("_")[1]))
            tilts_all, first_close_tilt, first_close_az, n_close = [], [], [], []
            appr_dir, fc_pos, lens, max_tilt = [], [], [], []
            for k in keys:
                e = f["data"][k]
                ori = e["obs/ee_ori"][:]
                pos = e["obs/ee_pos"][:]
                act = e["actions"][:]
                Rs = [rotvec_to_mat(o) for o in ori]
                ta = np.array([tilt_az(R) for R in Rs])
                tilts_all.append(ta[:, 0])
                ce = close_events(act[:, 6])
                n_close.append(len(ce))
                lens.append(len(act))
                max_tilt.append(float(ta[:, 0].max()))
                if ce:
                    i = ce[0]
                    rpy = mat_to_rpy(Rs[i])
                    j = max(0, i - 10)
                    v = pos[i] - pos[j]
                    h = np.degrees(np.arctan2(v[1], v[0])) if np.hypot(v[0], v[1]) > 1e-4 else float("nan")
                    first_close_tilt.append(ta[i, 0])
                    first_close_az.append(ta[i, 1])
                    appr_dir.append(h)
                    fc_pos.append(pos[i])
                    rows.append([name, k, len(act), len(ce), i, round(ta[i, 0], 1), round(ta[i, 1], 1),
                                 *np.round(rpy, 1), *np.round(pos[i], 3), round(h, 1), round(float(ta[:, 0].max()), 1)])
                else:
                    rows.append([name, k, len(act), 0, "", "", "", "", "", "", "", "", "", "", round(float(ta[:, 0].max()), 1)])
            allt = np.concatenate(tilts_all)
            summary[name] = {
                "n_demo": len(keys),
                "len_median": float(np.median(lens)),
                "tilt_all_deg": {"p5": float(np.percentile(allt, 5)), "median": float(np.median(allt)),
                                 "p95": float(np.percentile(allt, 95)), "max": float(allt.max())},
                "max_tilt_per_demo_median": float(np.median(max_tilt)),
                "n_close_events_median": float(np.median(n_close)),
                "first_close": None if not first_close_tilt else {
                    "tilt_deg_median": float(np.median(first_close_tilt)),
                    "tilt_deg_iqr": [float(np.percentile(first_close_tilt, 25)), float(np.percentile(first_close_tilt, 75))],
                    "az_deg_median": float(np.median(first_close_az)),
                    "pos_median": np.median(fc_pos, axis=0).round(3).tolist(),
                    "pos_std": np.std(fc_pos, axis=0).round(3).tolist(),
                },
            }
    os.makedirs(OUT, exist_ok=True)
    with open(os.path.join(OUT, "wrist_tilt_summary.json"), "w") as fp:
        json.dump(summary, fp, ensure_ascii=False, indent=1)
    with open(os.path.join(OUT, "wrist_tilt_per_demo.csv"), "w", newline="") as fp:
        w = csv.writer(fp)
        w.writerow(["base", "demo", "len", "n_close", "first_close_step", "tilt_deg", "az_deg", "roll", "pitch", "yaw",
                    "x", "y", "z", "approach_az_deg", "max_tilt_deg"])
        w.writerows(rows)
    for n, s in summary.items():
        fc = s["first_close"]
        print(f"{n:16s} n={s['n_demo']} len={s['len_median']:.0f} tilt(all) med={s['tilt_all_deg']['median']:.1f} "
              f"p95={s['tilt_all_deg']['p95']:.1f} max={s['tilt_all_deg']['max']:.1f} | "
              f"close tilt med={fc['tilt_deg_median']:.1f} iqr={fc['tilt_deg_iqr'][0]:.1f}-{fc['tilt_deg_iqr'][1]:.1f} "
              f"az={fc['az_deg_median']:.0f} pos={fc['pos_median']}" if fc else f"{n}: no close")


if __name__ == "__main__":
    main()
