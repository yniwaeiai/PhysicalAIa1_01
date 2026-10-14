"""K6 電子レンジの扉を「傾けて押して閉める」expert の単独試験（マグは扱わない。成功 = 扉 q > -0.005（LIBERO の Close）∧ 対象外 ≤ 1 mm）。
人間デモは扉の座標系で見ると押す点が安定していた（ヒンジから約 9 cm、低い位置 z=0.054、板の −y 側から +y 方向へ押す）。
目標を扉座標の固定点にして、扉の動きに追従させる。姿勢は人間デモの窓内（onset+8）の棚（電子レンジ）座標の姿勢を固定。
使い方: MUJOCO_GL=egl python run_micro_close.py --seeds 0-11 [--robot-noise .. --cab-shift .. --cab-yaw ..] [--trace]
"""
import argparse, json, sys
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation as Rot
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import run_drawer_hook as H   # noqa: E402
import kadai_record as KR   # noqa: E402
LPR = H.LPR
BASE, SUITE = "KITCHEN_SCENE6_put_the_yellow_and_white_mug_in_the_microwave_and_close_it", "libero_10"
OOI = {"white_yellow_mug_1"}   # マグは対象物（衝突判定の対象外）


def main():
    global BASE, SUITE, OOI
    ap = argparse.ArgumentParser()
    ap.add_argument("--single", action="store_true", help="LIBERO-90 の単独タスク（KITCHEN_SCENE6_close_the_microwave）を使う。対象物なし（OOI 空）、ボウル／マグは配置しない")
    ap.add_argument("--seeds", default="0-11")
    ap.add_argument("--max-steps", type=int, default=250)
    ap.add_argument("--px", type=float, default=0.09, help="押す点の扉座標 x（ヒンジからの径方向 m）")
    ap.add_argument("--pz", type=float, default=0.054)
    ap.add_argument("--y-out", type=float, default=-0.075, help="接近時の扉座標 y（板の −y 側）")
    ap.add_argument("--depth", type=float, default=0.02, help="押し込み量: 目標 y = 板の面(−0.025) + depth")
    ap.add_argument("--pre-dz", type=float, default=0.10)
    ap.add_argument("--extra-tilt", type=float, default=0.0)
    ap.add_argument("--v-free", type=float, default=1.0)
    ap.add_argument("--v-contact", type=float, default=0.5)
    ap.add_argument("--v-push", type=float, default=0.5)
    ap.add_argument("--robot-noise", type=float, default=0.0)
    ap.add_argument("--cab-shift", type=float, default=0.0)
    ap.add_argument("--cab-yaw", type=float, default=0.0)
    ap.add_argument("--record-out", default="")
    ap.add_argument("--accept", default="clean", choices=["clean", "goal"])
    ap.add_argument("--mix", action="store_true", help="エピソードごとに摂動水準を混合（なし 25%/中 50%/強 25%）。--robot-noise 等は無視")
    ap.add_argument("--trace", action="store_true")
    ap.add_argument("--out", default="")
    a = ap.parse_args()
    if a.single:
        BASE, SUITE, OOI = "KITCHEN_SCENE6_close_the_microwave", "libero_90", {"white_yellow_mug_1"}   # 人間デモ 50 本すべてがこのマグを動かす（中央値 352 mm）ので対象物扱い
    s0, s1 = (a.seeds.split("-") + [None])[:2]
    seeds = list(range(int(s0), int(s1 or s0) + 1))
    kp = json.load(open(HERE.parent / "state/close_keyposes_microwave_k6.json"))["keyposes"]
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
        mid = m.body_name2id("microwave_1_main"); did = m.body_name2id("microwave_1_microdoorroot")
        qadr = m.jnt_qposadr[m.joint_name2id("microwave_1_microjoint")]
        if pos0 is None:
            pos0, quat0 = m.body_pos[mid].copy(), m.body_quat[mid].copy()
        m.body_pos[mid], m.body_quat[mid] = pos0, quat0
        prng = np.random.default_rng((seed, 20261013))
        if a.cab_shift or a.cab_yaw:
            d = prng.uniform(-a.cab_shift, a.cab_shift, 2); m.body_pos[mid] = pos0 + np.array([d[0], d[1], 0.0])
            yw = np.radians(prng.uniform(-a.cab_yaw, a.cab_yaw)); qz = np.array([np.cos(yw / 2), 0, 0, np.sin(yw / 2)]); w0, x0, y0, z0 = quat0
            m.body_quat[mid] = np.array([qz[0]*w0 - qz[3]*z0, qz[0]*x0 - qz[3]*y0, qz[0]*y0 + qz[3]*x0, qz[0]*z0 + qz[3]*w0]); sim.forward()
        obs = env.set_init_state(rows[seed % len(rows)])
        if a.robot_noise:
            sim.data.qpos[0:7] += prng.uniform(-a.robot_noise, a.robot_noise, 7); sim.forward()
        for _ in range(H.NUM_STEPS_WAIT):
            obs, *_ = env.step([0.0] * 6 + [-1.0])
        qfn = lambda: float(sim.data.qpos[qadr])
        frame = lambda: (sim.data.body_xmat[mid].reshape(3, 3).copy(), sim.data.body_xpos[mid].copy())
        door_pt = lambda p: sim.data.body_xpos[did] + sim.data.body_xmat[did].reshape(3, 3) @ np.asarray(p)
        Rf0, _pf0 = frame()
        q_push = (Rot.from_euler("x", a.extra_tilt, degrees=True) * Rot.from_quat(kp["8"]["quat_xyzw"])).as_quat()
        segs = [
            dict(name="M_PRE", world_pos=lambda: door_pt([a.px, a.y_out, a.pz + a.pre_dz]), pos=None, quat=q_push, v=a.v_free, tol=0.02, rtol=15, max_t=100),
            dict(name="M_LOW", world_pos=lambda: door_pt([a.px, a.y_out, a.pz]), pos=None, quat=q_push, v=a.v_contact, tol=0.012, rtol=15, max_t=60),
            dict(name="M_PUSH", world_pos=lambda: door_pt([a.px, -0.025 + a.depth, a.pz]), pos=None, quat=q_push, v=a.v_push, tol=0.004, rtol=60, max_t=150, done=lambda: qfn() > -0.005),
        ]
        base = {k[:-4]: np.asarray(obs[k], float).copy() for k in obs if k.endswith("_pos") and "robot0" not in k and "gripper" not in k}
        runner = H.HookRunner(segs, frame)
        disp, first_hit, q_start = {}, None, qfn()
        rec = KR.EpRecorder() if a.record_out else None
        for T in range(1, a.max_steps + 1):
            act = runner.step(obs)
            if rec is not None:
                rec.add(obs, act)
            obs, *_ = env.step(act.tolist())
            for n, p0 in base.items():
                if n in OOI or f"{n}_pos" not in obs:
                    continue
                disp[n] = max(disp.get(n, 0.0), float(np.sum(np.abs(np.asarray(obs[f"{n}_pos"]) - p0))))
                if disp[n] > 0.001 and first_hit is None:
                    first_hit = dict(t=T, phase=runner.phase, obj=n)
            if a.trace and T % 10 == 0:
                print(f"  T={T} {runner.phase} q={qfn():.3f} ee={np.round(obs['robot0_eef_pos'], 3)}", flush=True)
            if qfn() > -0.005 or runner.i >= len(runner.segs):
                break
        dm = max(disp.values(), default=0.0)
        r = dict(seed=seed, T=T, q_start=round(q_start, 3), q=round(qfn(), 3), closed=qfn() > -0.005, disp_mm=round(dm * 1000, 2),
                 ok=qfn() > -0.005 and dm <= 0.001, first_hit=first_hit, trans=[(n, t, w) for n, t, w in runner.transitions])
        res.append(r)
        if rec is not None and (r['ok'] if a.accept == 'clean' else r['closed']):
            episodes.append(dict(seed=seed, rec=rec, T=T, perturb=dict(robot_noise=a.robot_noise, cab_shift=a.cab_shift, cab_yaw=a.cab_yaw)))
        print(f"seed {seed}: closed={r['closed']} T={T} q {r['q_start']}->{r['q']} disp={r['disp_mm']}mm ok={r['ok']} hit={first_hit} {[(n, w) for n, _, w in r['trans']]}", flush=True)
    print(f"SUMMARY ok {sum(r['ok'] for r in res)}/{len(res)} closed {sum(r['closed'] for r in res)}/{len(res)}")
    if a.record_out and episodes:
        path = KR.write_task_hdf5(a.record_out, SUITE, BASE, LPR / 'bddl_files' / SUITE / f'{BASE}.bddl', episodes, dict(accept=a.accept))
        print(f'recorded {len(episodes)} episodes -> {path}')
    if a.out:
        json.dump(res, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
