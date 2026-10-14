"""中段引き出し「傾けて引っかけて引く」expert の開発用ハーネス（12 初期状態で成功 ∧ 対象外 ≤ 1 mm を測る）。

人間デモのキーポーズ（kadai/state/hook_keyposes_middle.json、棚の座標系）を、実行時の棚の位置・向きで世界座標へ直し、
手先の位置と全姿勢（世界の軸角指令）を追従する。OSC_POSE 規約: 位置デルタ [-1,1]×0.05 m、回転 [-1,1]×0.5 rad（世界座標の軸角）、
グリッパ −1 = 開（引っかけの間は開いたまま）。
使い方: LIBERO_ROOT=/path/to/work/third_party/LIBERO-plus MUJOCO_GL=egl python run_drawer_hook.py --seeds 0-11 [--trace]
"""
import argparse, json, os, sys
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation as Rot

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
sys.path.insert(0, str(ROOT))
os.environ.setdefault("LIBERO_ROOT", "/path/to/work/third_party/LIBERO-plus")
LPR = Path(os.environ["LIBERO_ROOT"]) / "libero/libero"
os.environ.setdefault("LIBERO_CONFIG_PATH", "/path/to/work/data/libero_plus_config")
from libero.libero.envs import OffScreenRenderEnv  # noqa: E402
sys.path.insert(0, str(HERE))
import kadai_record as KR  # noqa: E402

BASE, SUITE = "open_the_middle_drawer_of_the_cabinet", "libero_goal"
NUM_STEPS_WAIT = 10
ACT_ROT = 0.5


def load_rows():
    import torch
    raw = torch.load(LPR / "init_files" / SUITE / f"{BASE}.pruned_init", weights_only=False)
    return raw.numpy() if hasattr(raw, "numpy") else np.asarray(raw)


def pctl(err, v, gain=3.0):
    return float(np.clip(gain * err / 0.01 * 0.1, -v, v))


class HookRunner:
    """棚座標のキーポーズ列 -> 世界の目標。区間ごとに (pos_c, quat_c, v, tol, grip) を持つ。"""

    def __init__(self, segs, frame_fn, v_rot=0.5, gain=3.0):
        self.segs, self.frame_fn, self.v_rot, self.gain = segs, frame_fn, v_rot, gain
        self.i, self.phase_t, self.t, self.transitions = 0, 0, 0, []
        self.phase = segs[0]["name"]

    def step(self, obs):
        a = np.zeros(7)
        a[6] = -1.0
        if self.i >= len(self.segs):
            self.t += 1
            return a
        s = self.segs[self.i]
        Rf, pf = self.frame_fn()
        tgt = s["world_pos"]() if "world_pos" in s else pf + Rf @ np.asarray(s["pos"])   # world_pos: 動く物体（扉）に追従する目標
        Rt = Rot.from_matrix(Rf @ Rot.from_quat(s["quat"]).as_matrix())
        eef = np.asarray(obs["robot0_eef_pos"], dtype=np.float64)
        for k in range(3):
            a[k] = pctl(tgt[k] - eef[k], s["v"], self.gain)
        err = (Rt * Rot.from_quat(obs["robot0_eef_quat"]).inv()).as_rotvec()
        a[3:6] = np.clip(self.gain * err / ACT_ROT, -self.v_rot, self.v_rot)
        pe, re_ = float(np.linalg.norm(tgt - eef)), float(np.degrees(np.linalg.norm(err)))
        fin = pe < s["tol"] and re_ < s.get("rtol", 6.0)
        if "done" in s and s["done"]():
            fin = True
        if fin or self.phase_t >= s["max_t"]:
            self.transitions.append((s["name"], self.t, "ok" if fin else "timeout"))
            self.i += 1
            self.phase_t = 0
            self.phase = self.segs[self.i]["name"] if self.i < len(self.segs) else "HOLD"
        else:
            self.phase_t += 1
        self.t += 1
        a[0:3] = np.clip(a[0:3], -1, 1)
        a[3:6] = np.clip(a[3:6], -1, 1)
        return a


def build_segs(kp, args, qfn):
    g = lambda k: (np.array(kp[k]["pos"]), np.array(kp[k]["quat_xyzw"]))
    p_pre, q_pre = g("-40")
    p_in, q_in = g("-12")
    p_hook, q_hook = g("0")
    p_end, q_end = g("peak")
    p_pre = p_pre + np.array([0, -0.0, args.pre_dz])
    p_in = p_in + np.array([args.dx, 0, args.dz])
    p_hook = p_hook + np.array([args.dx, args.dy, args.dz])
    p_end = p_end + np.array([args.dx, 0, args.dz])
    return [
        dict(name="H_PRE", pos=p_pre, quat=q_pre, v=args.v_free, tol=0.02, rtol=10, max_t=90),
        dict(name="H_IN", pos=p_in, quat=q_in, v=args.v_contact, tol=args.in_tol, rtol=args.in_rtol, max_t=70),
        dict(name="H_HOOK", pos=p_hook, quat=q_hook, v=args.v_contact, tol=0.006, rtol=8, max_t=40),
        dict(name="H_PULL", pos=p_end, quat=q_end, v=args.v_pull, tol=0.01, rtol=15, max_t=120, done=lambda: qfn() < -0.14),
    ]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", default="0-11")
    ap.add_argument("--max-steps", type=int, default=300)
    ap.add_argument("--v-free", type=float, default=1.0)
    ap.add_argument("--v-contact", type=float, default=0.4)
    ap.add_argument("--v-pull", type=float, default=0.6)
    ap.add_argument("--pre-dz", type=float, default=0.03)
    ap.add_argument("--dx", type=float, default=0.0)
    ap.add_argument("--dy", type=float, default=0.0)
    ap.add_argument("--dz", type=float, default=-0.018)
    ap.add_argument("--robot-noise", type=float, default=0.0, help="ロボット初期関節（7 軸）への一様ノイズ幅 rad（±）")
    ap.add_argument("--cab-shift", type=float, default=0.0, help="棚の xy 位置ずれ幅 m（±、一様）")
    ap.add_argument("--cab-yaw", type=float, default=0.0, help="棚の z まわりの向きずれ deg（±、一様）")
    ap.add_argument("--in-tol", type=float, default=0.015)
    ap.add_argument("--in-rtol", type=float, default=12.0)
    ap.add_argument("--record-out", default="")
    ap.add_argument("--mix", action="store_true", help="エピソードごとに摂動水準を混合（なし 25%/中 50%/強 25%）。--robot-noise 等は無視")
    ap.add_argument("--trace", action="store_true")
    ap.add_argument("--out", default="")
    args = ap.parse_args()
    a, b = (args.seeds.split("-") + [None])[:2]
    seeds = list(range(int(a), int(b or a) + 1))
    kp = json.load(open(HERE.parent / "state/hook_keyposes_middle.json"))["keyposes"]
    rows = load_rows()
    env = OffScreenRenderEnv(bddl_file_name=str(LPR / "bddl_files" / SUITE / f"{BASE}.bddl"), camera_heights=128, camera_widths=128)
    res, episodes = [], []
    _pos0 = _quat0 = None
    LEVELS = [((0.0, 0.0, 0.0), 0.25), ((0.1, 0.03, 5.0), 0.50), ((0.2, 0.06, 10.0), 0.25)]   # --mix: 学習データの摂動の混合（なし/中/強）
    for seed in seeds:
        if args.mix:
            _r_ = np.random.default_rng((seed, 777)).random()
            _lv = LEVELS[0][0] if _r_ < 0.25 else (LEVELS[1][0] if _r_ < 0.75 else LEVELS[2][0])
            args.robot_noise, args.cab_shift, args.cab_yaw = _lv
        env.reset()
        _m0 = env.env.sim.model   # ★reset のたびに sim が作り直される。摂動は必ず reset 後の新しいモデルに適用する
        _rid0 = _m0.body_name2id("wooden_cabinet_1_main")
        if _pos0 is None:
            _pos0, _quat0 = _m0.body_pos[_rid0].copy(), _m0.body_quat[_rid0].copy()
        _m0.body_pos[_rid0], _m0.body_quat[_rid0] = _pos0, _quat0   # 基準へ戻してから摂動を絶対値で与える
        prng = np.random.default_rng((seed, 20261013))
        if args.cab_shift or args.cab_yaw:
            dxy = prng.uniform(-args.cab_shift, args.cab_shift, 2)
            _m0.body_pos[_rid0] = _pos0 + np.array([dxy[0], dxy[1], 0.0])
            yaw = np.radians(prng.uniform(-args.cab_yaw, args.cab_yaw))
            qz = np.array([np.cos(yaw / 2), 0, 0, np.sin(yaw / 2)])   # wxyz
            w0, x0, y0, z0 = _quat0
            _m0.body_quat[_rid0] = np.array([qz[0]*w0 - qz[3]*z0, qz[0]*x0 - qz[3]*y0, qz[0]*y0 + qz[3]*x0, qz[0]*z0 + qz[3]*w0])
            env.env.sim.forward()
        obs = env.set_init_state(rows[seed % len(rows)])
        if args.robot_noise:
            env.env.sim.data.qpos[0:7] += prng.uniform(-args.robot_noise, args.robot_noise, 7)
            env.env.sim.forward()
        for _ in range(NUM_STEPS_WAIT):
            obs, *_ = env.step([0.0] * 6 + [-1.0])
        sim = env.env.sim
        m = sim.model
        rid = m.body_name2id("wooden_cabinet_1_main")
        qadr = m.jnt_qposadr[m.joint_name2id("wooden_cabinet_1_middle_level")]
        qfn = lambda: float(sim.data.qpos[qadr])
        frame = lambda: (sim.data.body_xmat[rid].reshape(3, 3).copy(), sim.data.body_xpos[rid].copy())
        base = {k[:-4]: np.asarray(obs[k], float).copy() for k in obs if k.endswith("_pos") and k[:-4] != "robot0_eef" and "robot0" not in k and "gripper" not in k}
        ooi = set()  # 対象は引き出し（物体ではない）。全物体を対象外として扱う
        runner = HookRunner(build_segs(kp, args, qfn), frame)
        disp, done, T = {}, False, 0
        rec = KR.EpRecorder() if args.record_out else None
        for T in range(1, args.max_steps + 1):
            act = runner.step(obs)
            if rec is not None:
                rec.add(obs, act)
            obs, _r, done, _i = env.step(act.tolist())
            for n, p0 in base.items():
                if f"{n}_pos" in obs:
                    disp[n] = max(disp.get(n, 0.0), float(np.sum(np.abs(np.asarray(obs[f"{n}_pos"]) - p0))))
            if args.trace and T % 10 == 0:
                Rf_, pf_ = frame()
                fb = [Rf_.T @ (sim.data.body_xpos[m.body_name2id(n)] - pf_) for n in ("gripper0_leftfinger", "gripper0_rightfinger")]
                ee_c = Rf_.T @ (np.asarray(obs["robot0_eef_pos"]) - pf_)
                cn = sorted({f"{m.geom_id2name(c.geom1)}|{m.geom_id2name(c.geom2)}" for c in sim.data.contact[:sim.data.ncon]
                             if ("gripper0" in str(m.geom_id2name(c.geom1)) + str(m.geom_id2name(c.geom2))) and ("cabinet" in str(m.geom_id2name(c.geom1)) + str(m.geom_id2name(c.geom2)))})
                print(f"  T={T} {runner.phase} ee_c={np.round(ee_c, 3)} fingers_c={[np.round(x, 3).tolist() for x in fb]} q={qfn():.3f} contacts={cn}", flush=True)
            if done or runner.phase == "HOLD" and runner.t > 0 and qfn() > -0.02 and runner.i >= len(runner.segs):
                break
        dm = max(disp.values(), default=0.0)
        r = dict(seed=seed, done=bool(done), T=T, q=round(qfn(), 3), disp_mm=round(dm * 1000, 2), collided=dm > 0.001,
                 ok=bool(done) and dm <= 0.001, trans=[(n, t, w) for n, t, w in runner.transitions])
        res.append(r)
        if rec is not None and r['ok']:
            episodes.append(dict(seed=seed, rec=rec, T=T, perturb=dict(robot_noise=args.robot_noise, cab_shift=args.cab_shift, cab_yaw=args.cab_yaw)))
        print(f"seed {seed}: done={r['done']} T={T} q={r['q']} disp={r['disp_mm']}mm ok={r['ok']} {[(n, w) for n, _, w in r['trans']]}", flush=True)
    print(f"SUMMARY ok {sum(r['ok'] for r in res)}/{len(res)} done {sum(r['done'] for r in res)}/{len(res)}")
    if args.record_out and episodes:
        path = KR.write_task_hdf5(args.record_out, SUITE, BASE, LPR / 'bddl_files' / SUITE / f'{BASE}.bddl', episodes, dict(accept='clean', dz=args.dz))
        print(f'recorded {len(episodes)} episodes -> {path}')
    if args.out:
        json.dump(res, open(args.out, "w"), indent=1)


if __name__ == "__main__":
    main()
