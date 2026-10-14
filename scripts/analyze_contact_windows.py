"""人間デモの「接触区間」での手首姿勢の解析（analyze_wrist_tilt.py の続き）。

グリッパを閉じる瞬間では、引き出し・押し・つまみの接触を捉えられない（人間はグリッパを開いたまま引っかける）。
そこで、操作対象（可動関節または物体）の変位が出る区間を接触区間とし、そこでの姿勢を測る。

  out 窓 : 変位が onset を超えた時点 -> 変位が最大の 90% に初めて達した時点
  back 窓: 最大変位の 90% 以上に最後にいた時点 -> 終端（閉じる・戻す動き。外向きが無い基底では使わない）

手の姿勢は、操作対象を載せた固定物（棚・電子レンジ・コンロ）の座標系で表す。
  tilt  : ツール z 軸の鉛直下向きからの角（0° = 真下）
  az_f  : 傾きの水平成分の方位（固定物の座標系の x 軸を 0°、y 軸を 90°）
出力: kadai/state/contact_windows_summary.json, contact_windows_per_demo.csv
CPU のみ・読み取りのみ。使い方: MUJOCO_GL=egl python -I kadai/scripts/analyze_contact_windows.py
"""
import csv
import json
import os
import sys

import h5py
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from analyze_wrist_tilt import BASES, ROOT, rotvec_to_mat, tilt_az  # noqa: E402

from libero.libero import get_libero_path  # noqa: E402
from libero.libero.envs import OffScreenRenderEnv  # noqa: E402

OUT = os.path.join(os.path.dirname(__file__), "..", "state")

# 基底 -> (追う対象の種別, 関節名, 基準にする固定物の body 候補, 最小の変位)
SPEC = {
    "drawer_middle": ("joint", "wooden_cabinet_1_middle_level", ["wooden_cabinet_1_main"], 0.03),
    "drawer_top_bowl": ("joint", "wooden_cabinet_1_top_level", ["wooden_cabinet_1_main"], 0.03),
    "push_plate": ("free", "plate_1_joint0", ["flat_stove_1_main", "flat_stove_1_base"], 0.03),
    "microwave_k6": ("joint", "microwave_1_microjoint", ["microwave_1_main"], 0.2),
    "k3_stove_moka": ("joint", "flat_stove_1_button", ["flat_stove_1_main", "flat_stove_1_base"], 0.2),
    "k4_bowl_drawer": ("joint", "white_cabinet_1_bottom_level", ["white_cabinet_1_main"], 0.03),
}


def body_id(m, names):
    for n in names:
        try:
            i = m.body_name2id(n)
            return i, n
        except Exception:
            continue
    raise KeyError(names)


def windows(sig, min_ext):
    """sig: 変位（開始値からの差の大きさ）。returns {name: (a, b)}"""
    ext = sig.max()
    if ext < min_ext:
        return {}
    on = int(np.argmax(sig > 0.05 * ext))
    pk = int(np.argmax(sig >= 0.9 * ext))
    out = {"out": (max(0, on - 5), pk)}
    last90 = int(len(sig) - 1 - np.argmax(sig[::-1] >= 0.9 * ext))
    if last90 < len(sig) - 5 and sig[-1] < 0.5 * ext:  # 最後に戻っている = 閉じる動き
        out["back"] = (last90, len(sig) - 1)
    return out


def main():
    bd = get_libero_path("bddl_files")
    rows, summary = [], {}
    for name, rel in BASES.items():
        kind, jn, refs, min_ext = SPEC[name]
        with h5py.File(os.path.join(ROOT, rel), "r") as f:
            bddl = f["data"].attrs["bddl_file_name"].split("bddl_files/")[1]
            env = OffScreenRenderEnv(bddl_file_name=os.path.join(bd, bddl), camera_heights=64, camera_widths=64)
            env.reset()
            sim = env.sim
            m = sim.model
            j = m.joint_name2id(jn)
            qa = m.jnt_qposadr[j]
            rid, rname = body_id(m, refs)
            acc = {}
            for k in sorted(f["data"].keys(), key=lambda s: int(s.split("_")[1])):
                e = f["data"][k]
                S = e["states"][:]
                q = S[:, 1:1 + m.nq]
                if kind == "joint":
                    sig = np.abs(q[:, qa] - q[0, qa])
                else:  # 自由関節: xy 変位
                    sig = np.linalg.norm(q[:, qa:qa + 2] - q[0, qa:qa + 2], axis=1)
                ori, pos, grip = e["obs/ee_ori"][:], e["obs/ee_pos"][:], e["obs/gripper_states"][:]
                for wn, (a, b) in windows(sig, min_ext).items():
                    # 固定物の座標系での姿勢（窓の始点で再生して xmat を読む）
                    sim.set_state_from_flattened(S[a])
                    sim.forward()
                    Rf = sim.data.body_xmat[rid].reshape(3, 3)
                    pf = sim.data.body_xpos[rid]
                    idx = np.arange(a, b + 1)
                    zs, tl, azf, wid = [], [], [], []
                    for t in idx:
                        R = rotvec_to_mat(ori[t])
                        zf = Rf.T @ R[:, 2]  # ツール z 軸を固定物の座標系へ
                        tl.append(np.degrees(np.arccos(np.clip(-R[2, 2], -1, 1))))
                        azf.append(np.degrees(np.arctan2(zf[1], zf[0])))
                        wid.append(grip[t, 0] - grip[t, 1])
                    p_on = Rf.T @ (pos[a] - pf)
                    p_pk = Rf.T @ (pos[b] - pf)
                    tl, azf, wid = map(np.array, (tl, azf, wid))
                    d = acc.setdefault(wn, [])
                    d.append(dict(demo=k, a=a, b=b, n=len(idx), tilt_med=float(np.median(tl)), tilt_p90=float(np.percentile(tl, 90)),
                                  tilt_max=float(tl.max()), az_f_med=float(np.degrees(np.angle(np.mean(np.exp(1j * np.radians(azf)))))),
                                  width_med=float(np.median(wid)), p_on=p_on.round(3).tolist(), p_pk=p_pk.round(3).tolist(),
                                  ext=float(sig.max())))
            env.close()
        summary[name] = {"ref_body": rname, "joint": jn}
        for wn, lst in acc.items():
            g = lambda key: np.array([x[key] for x in lst])
            summary[name][wn] = {
                "n_demo": len(lst),
                "len_median": float(np.median(g("n"))),
                "tilt_med_deg": {"p10": float(np.percentile(g("tilt_med"), 10)), "median": float(np.median(g("tilt_med"))),
                                 "p90": float(np.percentile(g("tilt_med"), 90))},
                "tilt_max_deg_median": float(np.median(g("tilt_max"))),
                "az_f_median_deg": float(np.degrees(np.angle(np.mean(np.exp(1j * np.radians(g("az_f_med"))))))),
                "az_f_concentration": float(np.abs(np.mean(np.exp(1j * np.radians(g("az_f_med")))))),  # 1 = 全デモ同方位
                "gripper_width_median": float(np.median(g("width_med"))),
                "ee_at_start_in_ref_median": np.median([x["p_on"] for x in lst], axis=0).round(3).tolist(),
                "ee_at_end_in_ref_median": np.median([x["p_pk"] for x in lst], axis=0).round(3).tolist(),
            }
            for x in lst:
                rows.append([name, wn, x["demo"], x["a"], x["b"], x["n"], round(x["tilt_med"], 1), round(x["tilt_p90"], 1),
                             round(x["tilt_max"], 1), round(x["az_f_med"], 1), round(x["width_med"], 3), *x["p_on"], *x["p_pk"]])
    with open(os.path.join(OUT, "contact_windows_summary.json"), "w") as fp:
        json.dump(summary, fp, ensure_ascii=False, indent=1)
    with open(os.path.join(OUT, "contact_windows_per_demo.csv"), "w", newline="") as fp:
        w = csv.writer(fp)
        w.writerow(["base", "window", "demo", "a", "b", "n", "tilt_med", "tilt_p90", "tilt_max", "az_f_med", "width_med",
                    "x_on", "y_on", "z_on", "x_end", "y_end", "z_end"])
        w.writerows(rows)
    for n, s in summary.items():
        for wn in ("out", "back"):
            if wn in s:
                v = s[wn]
                print(f"{n:16s} {wn:4s} n={v['n_demo']} len={v['len_median']:.0f} tilt med={v['tilt_med_deg']['median']:.1f} "
                      f"[{v['tilt_med_deg']['p10']:.1f},{v['tilt_med_deg']['p90']:.1f}] max~{v['tilt_max_deg_median']:.0f} "
                      f"az_f={v['az_f_median_deg']:.0f} conc={v['az_f_concentration']:.2f} width={v['gripper_width_median']:.3f} "
                      f"ee_start={v['ee_at_start_in_ref_median']} end={v['ee_at_end_in_ref_median']}")


if __name__ == "__main__":
    main()
