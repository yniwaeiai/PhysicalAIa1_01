"""K4 下段引き出しを「傾けて押して閉める」expert の単独試験（ボウルは扱わない。成功 = 引き出し q > -0.004 ∧ 対象外の物体が 1 mm 以内）。
人間デモ（kadai/state/close_keyposes_k4_bowl_drawer.json）の姿勢（手首を約 30° 前傾、指は開いたまま、指先が押す向き）を、
棚座標で追従する。HookRunner は run_drawer_hook.py から再利用。
使い方: MUJOCO_GL=egl python run_close_push.py --seeds 0-11 [--robot-noise 0.1 --cab-shift 0.03 --cab-yaw 5] [--trace]
"""
import argparse, json, os, sys
from pathlib import Path
import numpy as np
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import run_drawer_hook as H   # noqa: E402
import kadai_record as KR   # noqa: E402  (LIBERO_ROOT 等の環境設定もここで行われる)
LPR = H.LPR
OOI = {"akita_black_bowl_1"}   # このタスクの対象物（衝突判定の対象外）
BOWL_C = json.load(open(HERE.parent / "state/k4_bowl_in_drawer_cabinet_frame.json"))
BASE, SUITE = "KITCHEN_SCENE4_put_the_black_bowl_in_the_bottom_drawer_of_the_cabinet_and_close_it", "libero_10"


def build_segs(kp, a, qfn):
    from scipy.spatial.transform import Rotation as Rot
    # 追加の前傾: 棚の x 軸まわりに extra_tilt deg（正 = 指先がより押す向き（棚の奥）へ倒れる）。LOW/PUSH に同じ姿勢を使う
    q_push = (Rot.from_euler("x", a.extra_tilt, degrees=True) * Rot.from_quat(kp["8"]["quat_xyzw"])).as_quat()
    return [
        dict(name="C_PRE", pos=np.array([a.x, a.y0, a.zpre]), quat=q_push, v=a.v_free, tol=0.02, rtol=12, max_t=90),
        dict(name="C_LOW", pos=np.array([a.x, a.y0, a.z]), quat=q_push, v=a.v_contact, tol=0.012, rtol=12, max_t=60),
        dict(name="C_PUSH", pos=np.array([a.x, a.y1, a.z]), quat=q_push, v=a.v_push, tol=0.01, rtol=25, max_t=120, done=lambda: qfn() > -0.004),
    ]


def main():
    global BASE, SUITE, OOI
    ap = argparse.ArgumentParser()
    ap.add_argument("--single", action="store_true", help="LIBERO-90 の単独タスク（KITCHEN_SCENE4_close_the_bottom_drawer_of_the_cabinet）を使う。対象物なし（OOI 空）、ボウル／マグは配置しない")
    ap.add_argument("--seeds", default="0-11")
    ap.add_argument("--max-steps", type=int, default=300)
    ap.add_argument("--x", type=float, default=0.0)
    ap.add_argument("--y0", type=float, default=-0.30)
    ap.add_argument("--y1", type=float, default=-0.06)
    ap.add_argument("--z", type=float, default=0.046)
    ap.add_argument("--zpre", type=float, default=0.30)
    ap.add_argument("--v-free", type=float, default=1.0)
    ap.add_argument("--v-contact", type=float, default=0.5)
    ap.add_argument("--v-push", type=float, default=0.5)
    ap.add_argument("--robot-noise", type=float, default=0.0)
    ap.add_argument("--cab-shift", type=float, default=0.0)
    ap.add_argument("--cab-yaw", type=float, default=0.0)
    ap.add_argument("--extra-tilt", type=float, default=22.0)
    ap.add_argument("--bowl-in-drawer", type=int, default=1)
    ap.add_argument("--record-out", default="", help="学習用 HDF5 の出力先（<out>/<suite>/<base>.hdf5）。受理した試行だけを書く")
    ap.add_argument("--accept", default="clean", choices=["clean", "goal"], help="clean = ゴール達成 ∧ 対象外 ≤ 1 mm、goal = ゴール達成のみ")
    ap.add_argument("--mix", action="store_true", help="エピソードごとに摂動水準を混合（なし 25%/中 50%/強 25%）。--robot-noise 等は無視")
    ap.add_argument("--trace", action="store_true")
    ap.add_argument("--out", default="")
    a = ap.parse_args()
    if a.single:
        BASE, SUITE, OOI = "KITCHEN_SCENE4_close_the_bottom_drawer_of_the_cabinet", "libero_90", set()
        a.bowl_in_drawer = 0
    s0, s1 = (a.seeds.split("-") + [None])[:2]
    seeds = list(range(int(s0), int(s1 or s0) + 1))
    kp = json.load(open(HERE.parent / "state/close_keyposes_k4_bowl_drawer.json"))["keyposes"]
    import torch
    raw = torch.load(LPR / "init_files" / SUITE / f"{BASE}.pruned_init", weights_only=False)
    rows = raw.numpy() if hasattr(raw, "numpy") else np.asarray(raw)
    env = H.OffScreenRenderEnv(bddl_file_name=str(LPR / "bddl_files" / SUITE / f"{BASE}.bddl"), camera_heights=128, camera_widths=128)
    pos0 = quat0 = None
    res, episodes = [], []
    LEVELS = [((0.0, 0.0, 0.0), 0.25), ((0.1, 0.03, 5.0), 0.50), ((0.2, 0.06, 10.0), 0.25)]   # --mix: 学習データの摂動の混合（なし/中/強）
    for seed in seeds:
        if a.mix:
            _r_ = np.random.default_rng((seed, 777)).random()
            _lv = LEVELS[0][0] if _r_ < 0.25 else (LEVELS[1][0] if _r_ < 0.75 else LEVELS[2][0])
            a.robot_noise, a.cab_shift, a.cab_yaw = _lv
        env.reset()
        sim = env.env.sim; m = sim.model   # ★reset のたびに sim が作り直される
        rid = m.body_name2id("white_cabinet_1_main"); qadr = m.jnt_qposadr[m.joint_name2id("white_cabinet_1_bottom_level")]
        if pos0 is None:
            pos0, quat0 = m.body_pos[rid].copy(), m.body_quat[rid].copy()
        m.body_pos[rid], m.body_quat[rid] = pos0, quat0
        prng = np.random.default_rng((seed, 20261013))
        if a.cab_shift or a.cab_yaw:
            d = prng.uniform(-a.cab_shift, a.cab_shift, 2); m.body_pos[rid] = pos0 + np.array([d[0], d[1], 0.0])
            yw = np.radians(prng.uniform(-a.cab_yaw, a.cab_yaw)); qz = np.array([np.cos(yw / 2), 0, 0, np.sin(yw / 2)]); w0, x0, y0, z0 = quat0
            m.body_quat[rid] = np.array([qz[0]*w0 - qz[3]*z0, qz[0]*x0 - qz[3]*y0, qz[0]*y0 + qz[3]*x0, qz[0]*z0 + qz[3]*w0]); sim.forward()
        obs = env.set_init_state(rows[seed % len(rows)])
        if a.bowl_in_drawer:   # 実タスクではこの時点でボウルは引き出しの中。人間デモ終端のボウル姿勢（棚座標）へ置く
            from scipy.spatial.transform import Rotation as Rot
            Rf0 = sim.data.body_xmat[rid].reshape(3, 3); pf0 = sim.data.body_xpos[rid]
            bj = m.jnt_qposadr[m.joint_name2id("akita_black_bowl_1_joint0")]
            bp = pf0 + Rf0 @ (np.array(BOWL_C["pos"]) + np.array([0.0, float(sim.data.qpos[qadr]), 0.0]))   # 引き出しの現在の変位ぶん手前（y_c は負が開く向き）
            bq = (Rot.from_matrix(Rf0) * Rot.from_quat(BOWL_C["quat_xyzw"])).as_quat()   # xyzw
            sim.data.qpos[bj:bj + 3] = bp; sim.data.qpos[bj + 3:bj + 7] = [bq[3], bq[0], bq[1], bq[2]]   # wxyz
            sim.data.qvel[:] = 0.0; sim.forward()
            obs = env.env._get_observations(force_update=True) if hasattr(env.env, "_get_observations") else obs
        if a.robot_noise:
            sim.data.qpos[0:7] += prng.uniform(-a.robot_noise, a.robot_noise, 7); sim.forward()
        for _ in range(H.NUM_STEPS_WAIT):
            obs, *_ = env.step([0.0] * 6 + [-1.0])
        qfn = lambda: float(sim.data.qpos[qadr])
        frame = lambda: (sim.data.body_xmat[rid].reshape(3, 3).copy(), sim.data.body_xpos[rid].copy())
        q_start = qfn()
        base = {k[:-4]: np.asarray(obs[k], float).copy() for k in obs if k.endswith("_pos") and "robot0" not in k and "gripper" not in k}
        runner = H.HookRunner(build_segs(kp, a, qfn), frame)
        disp = {}
        first_hit = None
        rec = KR.EpRecorder() if a.record_out else None
        for T in range(1, a.max_steps + 1):
            act = runner.step(obs)
            if rec is not None:
                rec.add(obs, act)
            obs, *_ = env.step(act.tolist())
            for n, p0 in base.items():
                if n in OOI:
                    continue
                if f"{n}_pos" in obs:
                    disp[n] = max(disp.get(n, 0.0), float(np.sum(np.abs(np.asarray(obs[f"{n}_pos"]) - p0))))
                    if disp[n] > 0.001 and first_hit is None:
                        d_ = sim.data; parts = sorted({(str(m.geom_id2name(c.geom1)), str(m.geom_id2name(c.geom2))) for c in d_.contact[:d_.ncon]
                                                       if n in str(m.geom_id2name(c.geom1)) + str(m.geom_id2name(c.geom2))
                                                       and ("gripper0" in str(m.geom_id2name(c.geom1)) + str(m.geom_id2name(c.geom2)) or "robot0" in str(m.geom_id2name(c.geom1)) + str(m.geom_id2name(c.geom2)))})
                        first_hit = dict(t=T, phase=runner.phase, obj=n, parts=[f"{x}|{y}" for x, y in parts][:3])
            if a.trace and T % 10 == 0:
                Rf, pf = frame(); print(f"  T={T} {runner.phase} ee_c={np.round(Rf.T @ (np.asarray(obs['robot0_eef_pos']) - pf), 3)} q={qfn():.3f}", flush=True)
            if qfn() > -0.004 or runner.i >= len(runner.segs):
                break
        dm = max(disp.values(), default=0.0); top = max(disp, key=disp.get) if disp else None
        r = dict(seed=seed, T=T, q_start=round(q_start, 3), q=round(qfn(), 3), closed=qfn() > -0.004, disp_mm=round(dm * 1000, 2), top=top,
                 ok=qfn() > -0.004 and dm <= 0.001, first_hit=first_hit, trans=[(n, t, w) for n, t, w in runner.transitions])
        res.append(r)
        if rec is not None and (r['ok'] if a.accept == 'clean' else r['closed']):
            episodes.append(dict(seed=seed, rec=rec, T=T, perturb=dict(robot_noise=a.robot_noise, cab_shift=a.cab_shift, cab_yaw=a.cab_yaw)))
        print(f"seed {seed}: closed={r['closed']} T={T} q {r['q_start']}->{r['q']} disp={r['disp_mm']}mm({top}) ok={r['ok']} hit={first_hit}", flush=True)
    print(f"SUMMARY ok {sum(r['ok'] for r in res)}/{len(res)} closed {sum(r['closed'] for r in res)}/{len(res)}")
    if a.record_out and episodes:
        path = KR.write_task_hdf5(a.record_out, SUITE, BASE, LPR / 'bddl_files' / SUITE / f'{BASE}.bddl', episodes, dict(accept=a.accept, extra_tilt=a.extra_tilt, z=a.z))
        print(f'recorded {len(episodes)} episodes -> {path}')
    if a.out:
        json.dump(res, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
