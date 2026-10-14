"""学習用 HDF5 の書き出し（コンペ側 gen_t2_expert_demos._write_records と同じ形式: ep_N / agentview / wrist / state / actions / attrs）。
1 ステップ = (obs_t, action_t) の対。画像は env の生の観測（上下左右が逆のまま）を JPEG(q=95) で保存、state = eef 位置 3 + 軸角 3 + グリッパ 2。
読み取り側（コンペ側 train_lora_pi05.py）がこの形式を前提にしているので、キー名・順序を変えない。
"""
import io
import re
from pathlib import Path

import h5py
import numpy as np


def jpg(arr):
    from PIL import Image
    buf = io.BytesIO()
    Image.fromarray(np.ascontiguousarray(arr)).save(buf, format="JPEG", quality=95)
    return np.frombuffer(buf.getvalue(), dtype=np.uint8)


def quat2axisangle(q):
    q = np.asarray(q, dtype=np.float64).copy()
    q[3] = np.clip(q[3], -1.0, 1.0)
    den = np.sqrt(1.0 - q[3] ** 2)
    if np.isclose(den, 0.0):
        return np.zeros(3)
    return (q[:3] * 2.0 * np.arccos(q[3])) / den


class EpRecorder:
    def __init__(self):
        self.av, self.wr, self.st, self.act = [], [], [], []

    def add(self, obs, act):
        self.av.append(np.asarray(obs["agentview_image"], dtype=np.uint8).copy())
        self.wr.append(np.asarray(obs["robot0_eye_in_hand_image"], dtype=np.uint8).copy())
        self.st.append(np.concatenate([np.asarray(obs["robot0_eef_pos"], dtype=np.float64), quat2axisangle(obs["robot0_eef_quat"]),
                                       np.asarray(obs["robot0_gripper_qpos"], dtype=np.float64)]))
        self.act.append(np.asarray(act, dtype=np.float32).copy())


def language_of(bddl_path):
    m = re.search(r"\(:language\s+(.*?)\)", Path(bddl_path).read_text(), re.S)
    return " ".join(m.group(1).split()) if m else ""


def write_task_hdf5(out_dir, suite, base, bddl_path, episodes, extra_attrs=None):
    """episodes: list of dict(seed, rec=EpRecorder, T, perturb=dict)。1 タスク = 1 ファイル（<out_dir>/<suite>/<base>.hdf5）。"""
    od = Path(out_dir) / suite
    od.mkdir(parents=True, exist_ok=True)
    path = od / f"{base}.hdf5"
    tmp = Path(str(path) + ".tmp")
    vlen = h5py.vlen_dtype(np.uint8)
    with h5py.File(tmp, "w") as of:
        of.attrs.update({"suite": suite, "base_task": base, "variant": base, "instruction": language_of(bddl_path), "render_px": 128,
                         "bddl": str(bddl_path), "source": "kadai_expert", "category": "kadai_tilt_contact", **(extra_attrs or {})})
        for e in episodes:
            r = e["rec"]
            g = of.create_group(f"ep_{e['seed']}")
            g.create_dataset("agentview", data=np.array([jpg(a) for a in r.av], dtype=object), dtype=vlen)
            g.create_dataset("wrist", data=np.array([jpg(a) for a in r.wr], dtype=object), dtype=vlen)
            g.create_dataset("state", data=np.stack(r.st).astype(np.float32))
            g.create_dataset("actions", data=np.stack(r.act).astype(np.float32))
            g.attrs.update({"success": True, "length": int(e["T"]), "seed": int(e["seed"]), "source_demo": f"kadai_expert/{base}/seed{int(e['seed'])}", **{k: float(v) for k, v in e.get("perturb", {}).items()}})
    tmp.replace(path)
    return path
