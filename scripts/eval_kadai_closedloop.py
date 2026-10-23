"""3 つの主タスク（中段引き出しを開ける / 下段引き出しを閉める / 電子レンジを閉める）の閉ループ評価クライアント。

構成（コンペ側 scripts/eval_pi05_closedloop.py と同じ websocket 方式）:
  [openpi venv] serve_policy.py  <—— ws ——>  [LIBERO venv] このスクリプト（環境・摂動・判定・記録）

方策への入力は学習データと同一の前処理: 画像 180° 回転 → resize_with_pad(224)、状態 8 次元（eef 位置 3 + 軸角 3 + グリッパ 2）、
冒頭 10 step はダミー行動（物体の落下待ち）、方策は 10 step 分の行動を返し、先頭 replan step だけ使って再計画する。

判定: 成功 = env の done（bddl のゴール）、衝突なし = 対象外物体の最大 L1 変位 ≤ 1 mm（データ生成と同じ OOI 規則）。
摂動: データ生成と同じ水準（none / mid / strong / mix）。★評価の seed は学習データの seed（0–399）と重ならない値（既定 1000+）。
記録: エピソードごとの成否・ステップ数・手先の傾き（ツール z 軸の鉛直下向きからの角）の中央値と最大・手先の軌道長。
使い方: MUJOCO_GL=egl python eval_kadai_closedloop.py --tasks drawer,k4,k6 --episodes 20 --level mix --port 8000 --out result.json
"""
import argparse, collections, json, os, sys, time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
os.environ.setdefault("LIBERO_ROOT", "/path/to/work/third_party/LIBERO-plus")
import run_drawer_hook as H   # noqa: E402  (LIBERO の環境設定・OffScreenRenderEnv)
import kadai_record as KR      # noqa: E402
from openpi_client import image_tools, websocket_client_policy   # noqa: E402

LPR = H.LPR
LEVELS = {"none": (0.0, 0.0, 0.0), "mid": (0.1, 0.03, 5.0), "strong": (0.2, 0.06, 10.0)}
TASKS = {
    "drawer": dict(suite="libero_goal", base="open_the_middle_drawer_of_the_cabinet", ref="wooden_cabinet_1_main", joint="wooden_cabinet_1_middle_level", ooi=set()),
    "k4": dict(suite="libero_90", base="KITCHEN_SCENE4_close_the_bottom_drawer_of_the_cabinet", ref="white_cabinet_1_main", joint="white_cabinet_1_bottom_level", ooi=set()),
    "k6": dict(suite="libero_90", base="KITCHEN_SCENE6_close_the_microwave", ref="microwave_1_main", joint="microwave_1_microjoint", ooi={"white_yellow_mug_1"}),
}


def pick_level(level, seed):
    if level != "mix":
        return LEVELS[level]
    r = np.random.default_rng((seed, 777)).random()
    return LEVELS["none"] if r < 0.25 else (LEVELS["mid"] if r < 0.75 else LEVELS["strong"])


def tilt_deg(quat_xyzw):
    from scipy.spatial.transform import Rotation as Rot
    z = Rot.from_quat(quat_xyzw).as_matrix()[:, 2]
    return float(np.degrees(np.arccos(np.clip(-z[2], -1, 1))))


def make_env(spec):
    bddl = LPR / "bddl_files" / spec["suite"] / f"{spec['base']}.bddl"
    return H.OffScreenRenderEnv(bddl_file_name=str(bddl), camera_heights=128, camera_widths=128)


def init_episode(env, spec, rows, seed, level, state):
    """エピソードの初期化（reset → 摂動 → 初期状態 → 待機 10 step）。run_episode と branch_kadai.py が共用する。
    返り値: obs, sim, m, qadr, prng。★順序（pick_level → reset → prng → 摂動 → set_init_state → 関節ノイズ → 待機）は動かさない。"""
    rn, cs, cy = pick_level(level, seed)
    env.reset()
    sim = env.env.sim; m = sim.model   # ★reset のたびに sim が作り直される
    rid = m.body_name2id(spec["ref"]); qadr = m.jnt_qposadr[m.joint_name2id(spec["joint"])]
    if state.get("pos0") is None:
        state["pos0"], state["quat0"] = m.body_pos[rid].copy(), m.body_quat[rid].copy()
    pos0, quat0 = state["pos0"], state["quat0"]
    m.body_pos[rid], m.body_quat[rid] = pos0, quat0
    prng = np.random.default_rng((seed, 20261013))
    if cs or cy:
        d = prng.uniform(-cs, cs, 2); m.body_pos[rid] = pos0 + np.array([d[0], d[1], 0.0])
        yw = np.radians(prng.uniform(-cy, cy)); qz = np.array([np.cos(yw / 2), 0, 0, np.sin(yw / 2)]); w0, x0, y0, z0 = quat0
        m.body_quat[rid] = np.array([qz[0]*w0 - qz[3]*z0, qz[0]*x0 - qz[3]*y0, qz[0]*y0 + qz[3]*x0, qz[0]*z0 + qz[3]*w0]); sim.forward()
    obs = env.set_init_state(rows[seed % len(rows)])
    if rn:
        sim.data.qpos[0:7] += prng.uniform(-rn, rn, 7); sim.forward()
    for _ in range(H.NUM_STEPS_WAIT):
        obs, *_ = env.step([0.0] * 6 + [-1.0])
    return obs, sim, m, qadr, prng


def make_element(obs, prompt):
    """方策への入力（学習データと同一の前処理）。run_episode と branch_kadai.py が共用する。"""
    img = np.ascontiguousarray(obs["agentview_image"][::-1, ::-1]); wr = np.ascontiguousarray(obs["robot0_eye_in_hand_image"][::-1, ::-1])
    return {"observation/image": image_tools.convert_to_uint8(image_tools.resize_with_pad(img, 224, 224)),
            "observation/wrist_image": image_tools.convert_to_uint8(image_tools.resize_with_pad(wr, 224, 224)),
            "observation/state": np.concatenate((obs["robot0_eef_pos"], KR.quat2axisangle(obs["robot0_eef_quat"]), obs["robot0_gripper_qpos"])),
            "prompt": prompt}


def _is_robot_body(name):
    return (name or "").startswith(("robot0_", "gripper0_"))


def contact_detail(sim, m):
    """ロボットと非ロボットの接触を geom 単位で: [[ロボット側 geom, 相手 geom, 相手 body, 法線力 N], ...] の JSON 文字列。"""
    import mujoco
    out = []; f6 = np.zeros(6)
    for i in range(sim.data.ncon):
        c = sim.data.contact[i]
        g1, g2 = int(c.geom1), int(c.geom2)
        b1, b2 = (m.body_id2name(m.geom_bodyid[g1]) or ""), (m.body_id2name(m.geom_bodyid[g2]) or "")
        r1, r2 = _is_robot_body(b1), _is_robot_body(b2)
        if r1 == r2:
            continue
        mujoco.mj_contactForce(m._model, sim.data._data, i, f6)
        gr, go, bo = (g1, g2, b2) if r1 else (g2, g1, b1)
        nm = lambda g: m.geom_id2name(g) or f"{m.body_id2name(m.geom_bodyid[g])}#g{g}"
        out.append([nm(gr), nm(go), bo, round(float(f6[0]), 4)])
    return json.dumps(out)


def target_info(sim, m, bid, cid, ee):
    """対象（タスク関節の body）の位置・手先からの距離、手首カメラへの射影（保存した img_wrist と同じ向きの画素座標 u 右・v 下、128 px）と画角内か。"""
    p = np.array(sim.data.body_xpos[bid], dtype=float)   # ★copy（view のままだと全 step が終端の値になる。10/15 発見）
    cp, R = np.asarray(sim.data.cam_xpos[cid], float), np.asarray(sim.data.cam_xmat[cid], float).reshape(3, 3)
    pc = R.T @ (p - cp)   # カメラ座標（-z が視線方向）
    depth = -pc[2]; W = Hh = 128; f = (Hh / 2) / np.tan(np.radians(m.cam_fovy[cid]) / 2)
    if depth > 1e-6:
        u, v = W / 2 - f * pc[0] / depth, Hh / 2 - f * pc[1] / depth   # ★u は左右反転: 保存画像（obs を 180° 回転した方策の入力向き）の画素座標に一致する（物体投影で確認済み）
    else:
        u = v = float("nan")
    inv = bool(depth > 1e-6 and 0 <= u < W and 0 <= v < Hh)
    return p, float(np.linalg.norm(p - ee)), np.array([u, v, depth]), inv


def parse_retry_n(spec_str, name):
    """--retry-n: 整数（全タスク共通）または 'drawer=54,k4=13,k6=57'。"""
    if "=" not in spec_str:
        return int(spec_str)
    return int(dict(kv.split("=") for kv in spec_str.split(","))[name])


def ref_state(args, name):
    """--ref-pose の json から、そのタスクの基準 body の素の位置姿勢を init_episode の state に渡す形で返す。指定なしは空（従来どおり）。"""
    if not args.ref_pose:
        return {}
    r = json.load(open(args.ref_pose))[name]
    return {"pos0": np.asarray(r["pos0"], float), "quat0": np.asarray(r["quat0"], float)}


def run_episode(env, name, spec, args, client, rows, prompt, seed, rep, state):
    obs, sim, m, qadr, prng = init_episode(env, spec, rows, seed, args.level, state)
    rn, cs, cy = pick_level(args.level, seed)
    home_pos, home_quat = np.asarray(obs["robot0_eef_pos"], float).copy(), np.asarray(obs["robot0_eef_quat"], float).copy()   # 再試行ラッパの退避先（待機後・最初の行動の直前の手先）
    base_pos = {kk[:-4]: np.asarray(obs[kk], float).copy() for kk in obs if kk.endswith("_pos") and "robot0" not in kk and "gripper" not in kk}
    disp, tilts, path, prev = {}, [], 0.0, np.asarray(obs["robot0_eef_pos"], float)
    plan = collections.deque(); done = False; n_infer = 0; t0 = time.time()
    q_start = float(sim.data.qpos[qadr])
    fr = dict(step=[], agent=[], wrist=[], q=[], tilt=[], ee=[]) if args.save_frames else None

    def grab(T_, o_):
        # 表示用に 180° 回転して上下左右を直す（学習・推論の前処理と同じ向き）
        fr["step"].append(T_); fr["agent"].append(np.ascontiguousarray(o_["agentview_image"][::-1, ::-1])); fr["wrist"].append(np.ascontiguousarray(o_["robot0_eye_in_hand_image"][::-1, ::-1]))
        fr["q"].append(float(sim.data.qpos[qadr])); fr["tilt"].append(tilt_deg(o_["robot0_eef_quat"])); fr["ee"].append(np.asarray(o_["robot0_eef_pos"], float))
    if fr is not None:
        grab(0, obs)
    tr = None
    if args.trace:
        jn = [f"robot0_joint{i}" for i in range(1, 8)]
        jadr = [m.jnt_qposadr[m.joint_name2id(j)] for j in jn]; jrng = np.array([m.jnt_range[m.joint_name2id(j)] for j in jn])
        dadr = [m.jnt_dofadr[m.joint_name2id(j)] for j in jn]
        tr = dict(act_raw=[], act_exec=[], ee=[], quat=[], grip=[], jq=[], jv=[], jmargin=[], q=[], contact=[], chunk_at=[], chunks=[], contact_geom=[], tgt_pos=[], tgt_dist=[], tgt_uv=[], in_view=[], img_agent=[], img_wrist=[])
        tbid = int(m.jnt_bodyid[m.joint_name2id(spec["joint"])]); cam_id = m.camera_name2id("robot0_eye_in_hand")
        tr["sim_state"] = [sim.get_state().flatten().copy()]   # 行 k = k step 実行後の完全状態（行 0 = 待機後・最初の行動の直前）
        def rob_contacts():
            # ロボット（robot0_/gripper0_）と、ロボット以外の物体との接触先の body 名（重複なし）
            names = set()
            for i in range(sim.data.ncon):
                c = sim.data.contact[i]
                b1, b2 = (m.body_id2name(m.geom_bodyid[c.geom1]) or ""), (m.body_id2name(m.geom_bodyid[c.geom2]) or "")
                r1, r2 = b1.startswith(("robot0_", "gripper0_")), b2.startswith(("robot0_", "gripper0_"))
                if r1 != r2:
                    names.add(b2 if r1 else b1)
            return ";".join(sorted(names))
    # ---- 推論時の摂動（停留対策の試験）: 既定は無効 ----
    nrng = np.random.default_rng((seed, 4242, rep))
    ee_hist = [np.asarray(obs["robot0_eef_pos"], float)]
    kick_left, kick_vec, last_kick, n_kicks, dev_sum = 0, np.zeros(3), -10**9, 0, 0.0
    resample_left, last_resample, n_resample, tau_used, q_hist = 0, -10**9, 0, [], [float(sim.data.qpos[qadr])]
    def infer_chunk():
        # 方策に問い合わせて行動チャンクを得る。初期ノイズのスケール τ（常時 / 停留からの再サンプリング）があれば要求に付ける
        nonlocal n_infer, resample_left
        el = make_element(obs, prompt)
        tau = None
        if resample_left > 0:
            tau = args.qstall_resample; resample_left -= 1   # 停留からの脱出: 数チャンクだけ初期ノイズを大きくして引き直す
        elif args.flow_noise > 0:
            tau = args.flow_noise
        if tau is not None:
            el["_noise_scale"] = float(tau); el["_noise_seed"] = int(nrng.integers(0, 2**31 - 1)); tau_used.append(float(tau))
        n_infer += 1
        ch = np.asarray(client.infer(el)["actions"])
        if tr is not None:
            tr["chunk_at"].append(len(tr["act_raw"])); tr["chunks"].append(ch.copy())   # chunk_at = この推論の直後に実行される step の添字
            tr["img_agent"].append(np.ascontiguousarray(obs["agentview_image"][::-1, ::-1])); tr["img_wrist"].append(np.ascontiguousarray(obs["robot0_eye_in_hand_image"][::-1, ::-1]))   # 方策に渡した向き（180° 回転）の 128×128
        plan.extend(ch[:args.replan])

    # ---- 停留検出 D と再試行ラッパ（計画 v2 §11 W）: 既定は無効。特権情報は使わない（手先位置と方策の指令のみ）----
    det_n = parse_retry_n(args.retry_n, name) if (args.retry or args.shadow_detect) else 0
    cmd_hist, fire_step, wr = [], None, None   # wr: 退避の記録

    def after_step(act0, act, obs_):
        # 1 step 実行後の共通の記録（元のループ本体と同じ内容）
        nonlocal obs, prev, path
        obs = obs_
        ee = np.asarray(obs["robot0_eef_pos"], float); path += float(np.linalg.norm(ee - prev)); prev = ee; ee_hist.append(ee); q_hist.append(float(sim.data.qpos[qadr]))
        tilts.append(tilt_deg(obs["robot0_eef_quat"]))
        if tr is not None:
            jq = sim.data.qpos[jadr].copy(); tr["act_raw"].append(act0); tr["act_exec"].append(act.copy()); tr["ee"].append(ee); tr["quat"].append(np.asarray(obs["robot0_eef_quat"], float))
            tr["grip"].append(np.asarray(obs["robot0_gripper_qpos"], float)); tr["jq"].append(jq); tr["jv"].append(sim.data.qvel[dadr].copy())
            tr["jmargin"].append(np.minimum(jq - jrng[:, 0], jrng[:, 1] - jq)); tr["q"].append(float(sim.data.qpos[qadr])); tr["contact"].append(rob_contacts())
            tr["contact_geom"].append(contact_detail(sim, m)); tp, td, tuv, tin = target_info(sim, m, tbid, cam_id, ee)
            tr["tgt_pos"].append(tp); tr["tgt_dist"].append(td); tr["tgt_uv"].append(tuv); tr["in_view"].append(tin); tr["sim_state"].append(sim.get_state().flatten().copy())
        if fr is not None and (T % args.frame_every == 0 or done or T == args.max_steps):
            grab(T, obs)
        for n, p0 in base_pos.items():
            if n in spec["ooi"] or f"{n}_pos" not in obs:
                continue
            disp[n] = max(disp.get(n, 0.0), float(np.sum(np.abs(np.asarray(obs[f"{n}_pos"]) - p0))))

    def stall_fires():
        # D: 直近 N step の手先速度の平均 < 1.5 mm/step かつ 位置指令（方策の生の出力）のノルムの中央値 >= 0.3。T > N を要求（stall_detector_offline.py と同じ）
        if T <= det_n or len(cmd_hist) < det_n:
            return False
        w = np.asarray(ee_hist[-(det_n + 1):])
        return bool(np.linalg.norm(np.diff(w, axis=0), axis=1).mean() * 1000 < 1.5 and np.median(cmd_hist[-det_n:]) >= 0.3)

    def retreat():
        # 退避 R: (i) +z 指令 1.0 を 6 step、(ii) 初期姿勢の位置・姿勢へ OSC 指令で（最大 60 step、位置 1.5 cm・回転 0.05 rad で終了）。グリッパ指令は直前の値を維持
        nonlocal T, done
        from scipy.spatial.transform import Rotation as Rot
        grip = float(last_grip); ee0 = np.asarray(obs["robot0_eef_pos"], float).copy(); n_up = n_home = 0
        def step_(a):
            nonlocal T, done
            if T >= args.max_steps or done:
                return False
            T += 1; a = np.asarray(a, float); o, _r, d, _i = env.step(a.tolist()); done = bool(d)
            cmd_hist.append(float(np.linalg.norm(a[:3]))); after_step(a, a, o)
            return not done
        for _ in range(6):
            if not step_([0, 0, 1.0, 0, 0, 0, grip]):
                break
            n_up += 1
        def errs():
            ep = home_pos - np.asarray(obs["robot0_eef_pos"], float)
            rv = (Rot.from_quat(home_quat) * Rot.from_quat(obs["robot0_eef_quat"]).inv()).as_rotvec()
            return ep, rv
        for _ in range(60):
            ep, rv = errs()
            if done or T >= args.max_steps or (np.linalg.norm(ep) < 0.015 and np.linalg.norm(rv) < 0.05):
                break
            step_(np.concatenate([np.clip(ep / 0.05, -1, 1), np.clip(rv / 0.5, -1, 1), [grip]])); n_home += 1
        ep, rv = errs()
        return dict(up_steps=n_up, home_steps=n_home, err_pos_mm=round(float(np.linalg.norm(ep) * 1000), 2), err_rot_rad=round(float(np.linalg.norm(rv)), 4),
                    reached=bool(np.linalg.norm(ep) < 0.015 and np.linalg.norm(rv) < 0.05), moved_mm=round(float(np.linalg.norm(np.asarray(obs["robot0_eef_pos"], float) - ee0) * 1000), 1), resume_T=T)

    T = 0; last_grip = -1.0; n_infer_at_resume = None
    while T < args.max_steps:
        T += 1
        if args.qstall_resample > 0 and resample_left == 0 and T >= args.qstall_min_t and T - last_resample >= args.qstall_cooldown and len(q_hist) > args.qstall_window:
            if abs(q_hist[-1] - q_hist[-1 - args.qstall_window]) < args.qstall_dq:   # 窓内のタスクの関節がほとんど動かない = 進捗の停留
                resample_left, last_resample = args.resample_chunks, T; n_resample += 1; plan.clear()   # 計画を捨てて引き直す
        if not plan:
            infer_chunk()
        act = np.array(plan.popleft(), dtype=np.float64)   # ★コピーを取る（サーバから届く配列は読み取り専用で、摂動の代入が ValueError になる）
        act0 = act.copy()
        if args.stall_kick and kick_left == 0 and T >= args.stall_min_t and T - last_kick >= args.kick_cooldown and len(ee_hist) > args.stall_window:
            w = np.asarray(ee_hist[-(args.stall_window + 1):])
            if float(np.linalg.norm(np.diff(w, axis=0), axis=1).sum()) < args.stall_dist:   # 窓内の手先の移動量が小さい = 停留
                kick_left, last_kick = args.kick_steps, T
                v = nrng.normal(size=3); kick_vec = v / np.linalg.norm(v) * args.kick_mag; n_kicks += 1
                plan.clear()   # 停留から抜けたら再推論させる
        if kick_left > 0:
            act[:3] += kick_vec; act[3:6] += nrng.normal(0, args.kick_mag * 0.5, 3); kick_left -= 1
        if args.act_noise > 0:
            act[:6] += nrng.normal(0, args.act_noise, 6)
        act[:6] = np.clip(act[:6], -1.0, 1.0)
        dev_sum += float(np.abs(act[:6] - act0[:6]).mean())   # ★操作チェック: 実際に加わった摂動量
        obs_, _r, done, _i = env.step(act.tolist())
        cmd_hist.append(float(np.linalg.norm(act0[:3]))); last_grip = float(act[6])
        after_step(act0, act, obs_)
        if done:
            break
        if det_n and fire_step is None and stall_fires():
            fire_step = T
            if args.retry:
                wr = retreat(); plan.clear(); n_infer_at_resume = n_infer   # 行動チャンクは破棄して新規に推論させる
                if done:
                    break
    dm = max(disp.values(), default=0.0)
    if tr is not None:
        os.makedirs(args.trace, exist_ok=True)
        np.savez_compressed(os.path.join(args.trace, f"{name}_s{seed}_r{rep}.npz"), act_raw=np.array(tr["act_raw"]), act_exec=np.array(tr["act_exec"]), ee=np.array(tr["ee"]), quat=np.array(tr["quat"]),
                            grip=np.array(tr["grip"]), jq=np.array(tr["jq"]), jv=np.array(tr["jv"]), jmargin=np.array(tr["jmargin"]), q=np.array(tr["q"]), contact=np.array(tr["contact"]),
                            chunk_at=np.array(tr["chunk_at"]), chunks=np.array(tr["chunks"]), success=bool(done), seed=seed, rep=rep,
                            contact_geom=np.array(tr["contact_geom"]), tgt_pos=np.array(tr["tgt_pos"]), tgt_dist=np.array(tr["tgt_dist"]), tgt_uv=np.array(tr["tgt_uv"]), in_view=np.array(tr["in_view"]),
                            sim_state=np.array(tr["sim_state"]), img_agent=np.array(tr["img_agent"]), img_wrist=np.array(tr["img_wrist"]),
                            task=name, level=args.level, replan=args.replan, trace_version=2,
                            ref_pos0=np.asarray(state["pos0"]), ref_quat0=np.asarray(state["quat0"]))   # ★基準 body の素の位置姿勢（reset ごとに乱数で変わるのでプロセスの最初の reset の値を記録。再生側が同じ値で固定する）
    if fr is not None:
        os.makedirs(args.save_frames, exist_ok=True)
        np.savez_compressed(os.path.join(args.save_frames, f"{name}_s{seed}.npz"), step=np.array(fr["step"]), agent=np.stack(fr["agent"]), wrist=np.stack(fr["wrist"]),
                            q=np.array(fr["q"]), tilt=np.array(fr["tilt"]), ee=np.stack(fr["ee"]))
    return dict(task=name, seed=seed, level=[rn, cs, cy], success=bool(done), clean=bool(done) and dm <= 0.001, steps=T, disp_mm=round(dm * 1000, 2),
                top=(max(disp, key=disp.get) if disp else None), tilt_med=round(float(np.median(tilts)), 1), tilt_max=round(float(np.max(tilts)), 1),
                path_m=round(path, 3), n_infer=n_infer, n_kicks=n_kicks, act_dev=round(dev_sum / max(T, 1), 4), n_resample=n_resample, tau_mean=(round(float(np.mean(tau_used)), 3) if tau_used else None), rep=rep, q_start=round(q_start, 3), q_end=round(float(sim.data.qpos[qadr]), 3), sec=round(time.time() - t0, 1),
                **(dict(det_n=det_n, retry=bool(args.retry), fire_step=fire_step, retreat=wr, n_infer_after_resume=(None if n_infer_at_resume is None else n_infer - n_infer_at_resume),
                        success250=bool(done) and T <= 250, clean250=bool(done) and T <= 250 and dm <= 0.001, success_end=bool(done), done_step=(T if done else None), max_steps=args.max_steps) if det_n else {}))


def run_task(name, spec, args, client):
    import torch
    suite, base = spec["suite"], spec["base"]
    raw = torch.load(LPR / "init_files" / suite / f"{base}.pruned_init", weights_only=False)
    rows = raw.numpy() if hasattr(raw, "numpy") else np.asarray(raw)
    prompt = KR.language_of(LPR / "bddl_files" / suite / f"{base}.bddl")
    env = make_env(spec); state = ref_state(args, name)
    out = []
    if args.seed_list:
        plan_eps = [(int(x), r) for x in args.seed_list.split(",") for r in range(args.repeat)]
        plan_eps = plan_eps[args.shard_start:args.shard_start + args.episodes] if args.shard_start >= 0 else plan_eps
    else:
        plan_eps = [(args.seed_offset + k, 0) for k in range(args.episodes)]
    for k, (seed_k, rep_k) in enumerate(plan_eps):
        for attempt in range(3):   # ネットワークFS上の一時的な読み込み失敗（メッシュ等）に備えて環境を作り直して再試行
            try:
                r = run_episode(env, name, spec, args, client, rows, prompt, seed_k, rep_k, state); break
            except (ValueError, OSError, RuntimeError) as e:
                print(f"{name} ep {k} attempt {attempt}: {type(e).__name__}: {str(e)[:120]} -> env 再生成", flush=True)
                try:
                    env.close()
                except Exception:
                    pass
                env = make_env(spec); state = ref_state(args, name)
        else:
            r = dict(task=name, seed=seed_k, rep=rep_k, error=True, success=False, clean=False)
        out.append(r)
        if args.out:
            with open(args.out + ".jsonl", "a") as fp:
                fp.write(json.dumps(r) + "\n")
        print(f"{name} seed {r['seed']}: success={r.get('success')} clean={r.get('clean')} T={r.get('steps')} disp={r.get('disp_mm')}mm tilt_med={r.get('tilt_med')} max={r.get('tilt_max')} q {r.get('q_start')}->{r.get('q_end')} {r.get('sec')}s", flush=True)
    try:
        env.close()
    except Exception:
        pass
    return out


def build_parser():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tasks", default="drawer,k4,k6")
    ap.add_argument("--episodes", type=int, default=20)
    ap.add_argument("--seed-offset", type=int, default=1000)
    ap.add_argument("--level", default="mix", choices=["none", "mid", "strong", "mix"])
    ap.add_argument("--replan", type=int, default=5)
    ap.add_argument("--max-steps", type=int, default=250)
    ap.add_argument("--host", default="localhost")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--out", default="")
    ap.add_argument("--save-frames", default="", help="コマ送り用にエピソードごとの画像（agentview・wrist）と q・傾き・手先位置を npz で保存する先のディレクトリ")
    ap.add_argument("--frame-every", type=int, default=6)
    ap.add_argument("--trace", default="", help="原因追究用: エピソードごとに毎 step の行動（生・実行）・手先・関節・関節限界までの余裕・ロボットの接触先・各推論の行動チャンク全体を npz で保存する先のディレクトリ")
    ap.add_argument("--act-noise", type=float, default=0.0, help="常時の行動ノイズ: 実行する行動の位置・回転 6 成分に N(0, σ) を足す（正規化行動の単位）")
    ap.add_argument("--stall-kick", action="store_true", help="停留（手先がほとんど動かない）を検出したら、短い乱れ（kick）を入れて再推論させる")
    ap.add_argument("--stall-window", type=int, default=20)
    ap.add_argument("--stall-dist", type=float, default=0.02, help="窓内の手先の移動量（m）がこれ未満なら停留")
    ap.add_argument("--stall-min-t", type=int, default=30)
    ap.add_argument("--kick-mag", type=float, default=0.2, help="kick の位置成分の大きさ（正規化行動。0.2 ≈ 1 cm/step）")
    ap.add_argument("--kick-steps", type=int, default=8)
    ap.add_argument("--kick-cooldown", type=int, default=40)
    ap.add_argument("--flow-noise", type=float, default=0.0, help="常時: フローマッチングの初期ノイズを τ 倍にする（サーバが serve_policy_noise.py のとき）。0 = 無効")
    ap.add_argument("--qstall-resample", type=float, default=0.0, help="停留（タスク関節 q が窓内でほぼ動かない）を検出したら、次の数チャンクの初期ノイズを τ 倍にして引き直す。0 = 無効")
    ap.add_argument("--qstall-window", type=int, default=30)
    ap.add_argument("--qstall-dq", type=float, default=0.004)
    ap.add_argument("--qstall-min-t", type=int, default=60)
    ap.add_argument("--qstall-cooldown", type=int, default=30)
    ap.add_argument("--resample-chunks", type=int, default=3)
    ap.add_argument("--ref-pose", default="", help="基準 body の素の位置姿勢の json（タスク名 -> {pos0, quat0}）。指定すると init_episode の state に渡して固定する（kadai/state/ref_pose_20261015.json）")
    ap.add_argument("--retry", action="store_true", help="再試行ラッパ（計画 v2 §11 W）: 停留検出 D が発火したら +z 6 step → 初期姿勢へ → 方策再開。1 ep 1 回。--retry-n が必要")
    ap.add_argument("--shadow-detect", action="store_true", help="D を計算して発火 step だけ記録する（行動は変えない。基準側の対照用）")
    ap.add_argument("--retry-n", default="drawer=54,k4=13,k6=57", help="D の窓長 N（整数、または 'drawer=54,k4=13,k6=57'）")
    ap.add_argument("--seed-list", default="", help="カンマ区切りの seed を明示（失敗 seed の反復実行用）。--repeat 回ずつ、方策のノイズだけ変えて走らせる")
    ap.add_argument("--repeat", type=int, default=1)
    ap.add_argument("--shard-start", type=int, default=-1, help="seed-list × repeat の列のうち、この位置から --episodes 本を担当（シャード用）")
    return ap


def main():
    args = build_parser().parse_args()
    # ★初回推論は torch.compile で数分かかり、websockets 既定の keepalive（ping 20 秒）が先に切れる。keepalive を無効にする
    import functools, websockets.sync.client as _wsc
    _wsc.connect = functools.partial(_wsc.connect, ping_interval=None)
    client = websocket_client_policy.WebsocketClientPolicy(args.host, args.port)
    allr = []
    for name in args.tasks.split(","):
        allr += run_task(name, TASKS[name], args, client)
    for name in args.tasks.split(","):
        rs = [r for r in allr if r["task"] == name]
        print(f"SUMMARY {name}: success {sum(r['success'] for r in rs)}/{len(rs)}  clean {sum(r['clean'] for r in rs)}/{len(rs)}  "
              f"steps_med {np.median([r['steps'] for r in rs]):.0f}  tilt_med {np.median([r['tilt_med'] for r in rs]):.1f}")
    if args.out:
        json.dump(dict(args=vars(args), episodes=allr), open(args.out, "w"), indent=1)


if __name__ == "__main__":
    main()
