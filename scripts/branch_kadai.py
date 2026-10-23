"""trace v2 の npz から、同じ seed・摂動で環境を用意し、再生 / 復元 / 継続 / 再サンプルを行う分岐ハーネス（計画 v2 §2・§4）。

規約:
  - sim_state の行 k = k step 実行後の完全状態（行 0 = 待機後）。「step t で復元」= 行 t を戻す → 次に実行するのは act_exec[t]。
  - replay(t) = 初期化 → act_exec[0:t] を順に与える（復元と同じ地点に着く）。
  - 前置き: toward = 対象方向へ 8 step 並進（単位ベクトル × 0.3、行動の単位）／retreat = +z 3 cm（手先 z が +0.03 になるまで 最大 20 step、続けて初期姿勢へ回転 P 制御 最大 15 step）。
  - 位置指令 1.0 = 0.05 m（osc_pose.json の output_max。実際の手先変位は制御の追従分だけ小さい）。回転指令 1.0 = 0.5 rad。
使い方:
  python branch_kadai.py --trace-file X.npz --mode selfcheck                        # C2・C3・C4
  python branch_kadai.py --trace-file X.npz --mode continue --via restore --t 80 --H 60 --replan 5 --prefix toward --port 18040 --out o.npz
  python branch_kadai.py --trace-file X.npz --mode resample --via restore --t 80 --n 16 --port 18040 --out o.npz
"""
import argparse, collections, functools, json, os, sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import eval_kadai_closedloop as E   # noqa: E402
import kadai_record as KR            # noqa: E402

H = E.H


class Ctx:
    """1 エピソード分の環境。open_ctx → (replay | restore) → step / continue_policy / resample。"""
    def __init__(self, task, seed, level, env=None, pos0=None, quat0=None):
        import torch
        self.name, self.seed, self.level = task, int(seed), level
        self.spec = E.TASKS[task]; spec = self.spec
        raw = torch.load(E.LPR / "init_files" / spec["suite"] / f"{spec['base']}.pruned_init", weights_only=False)
        self.rows = raw.numpy() if hasattr(raw, "numpy") else np.asarray(raw)
        self.prompt = KR.language_of(E.LPR / "bddl_files" / spec["suite"] / f"{spec['base']}.bddl")
        self.env = env or E.make_env(spec)
        # ★基準 body の素の位置姿勢は run_episode ではプロセス最初の reset の乱数サンプルなので、記録した値で固定する（接触局面の再現に効く）
        self.state = {} if pos0 is None else {"pos0": np.asarray(pos0).copy(), "quat0": np.asarray(quat0).copy()}
        self.fresh()

    def fresh(self):
        """reset → 摂動 → 初期状態 → 待機（run_episode と同じ共用関数）。"""
        self.obs, self.sim, self.m, self.qadr, self.prng = E.init_episode(self.env, self.spec, self.rows, self.seed, self.level, self.state)
        self.tbid = int(self.m.jnt_bodyid[self.m.joint_name2id(self.spec["joint"])])
        self.log = dict(ee=[], quat=[], q=[], jq=[], act=[], phase=[], chunks=[], chunk_at=[])
        self.n = 0; self.done = False
        self._rec()
        return self.obs

    def _rec(self, act=None, phase=""):
        o = self.obs
        self.log["ee"].append(np.asarray(o["robot0_eef_pos"], float)); self.log["quat"].append(np.asarray(o["robot0_eef_quat"], float))
        self.log["q"].append(float(self.sim.data.qpos[self.qadr])); self.log["jq"].append(np.asarray(o["robot0_joint_pos"], float))
        self.log["act"].append(np.full(7, np.nan) if act is None else np.asarray(act, float)); self.log["phase"].append(phase)

    def step(self, act, phase="replay"):
        act = np.asarray(act, float).copy()
        self.obs, _r, d, _i = self.env.step(act.tolist()); self.done = bool(d or self.done); self.n += 1
        self._rec(act, phase)
        return self.obs

    def replay(self, acts, t):
        for k in range(t):
            self.step(acts[k], "replay")
        return self.obs

    def restore(self, sim_state_row):
        """完全状態を戻す（物理のみ。OSC 制御器の内部状態は戻らない）。事前に fresh() で摂動つきのモデルにしておくこと。"""
        self.sim.set_state_from_flattened(np.asarray(sim_state_row, float)); self.sim.forward()
        self.env.env._update_observables(force=True); self.obs = self.env.env._get_observations()
        self._rec(None, "restore")
        return self.obs

    def robot_joint_idx(self):
        """ロボットの関節（robot0_joint1〜7 + 指 2 関節）の (qpos index, qvel index) を flattened 状態 [time, qpos, qvel, ...] の添字で返す。"""
        if not hasattr(self, "_rji"):
            m = self.sim.model; nq = m.nq; names = [n for n in m.joint_names if n.startswith("robot0_joint") or "finger_joint" in n]
            assert len(names) == 9, names
            qi = []; vi = []
            for n in names:
                qa = m.get_joint_qpos_addr(n); va = m.get_joint_qvel_addr(n)
                qi += [int(qa)] if isinstance(qa, (int, np.integer)) else list(range(*qa))
                vi += [int(va)] if isinstance(va, (int, np.integer)) else list(range(*va))
            self._rji = (names, np.array(qi) + 1, np.array(vi) + 1 + nq)
        return self._rji

    def restore_swap_robot(self, row_base, row_robot):
        """I4: base の完全状態を戻し、ロボットの関節 qpos/qvel（7 関節 + 指 2）だけ row_robot の値にする → forward → 観測更新。"""
        _n, qi, vi = self.robot_joint_idx()
        row = np.asarray(row_base, float).copy(); rr = np.asarray(row_robot, float)
        row[qi] = rr[qi]; row[vi] = rr[vi]
        return self.restore(row)

    # ---- I5: 腕 7 関節の IK と手先姿勢（計画 v2 §16） ----
    def arm_joint_idx(self):
        """腕 7 関節（robot0_joint1〜7、指を除く）の (names, flattened qpos 添字, flattened qvel 添字, qpos addr, dof 添字(jacobian 列), jnt_range)。"""
        if not hasattr(self, "_aji"):
            m = self.sim.model; nq = m.nq
            names = [f"robot0_joint{i}" for i in range(1, 8)]
            qa = np.array([int(m.get_joint_qpos_addr(n)) for n in names]); va = np.array([int(m.get_joint_qvel_addr(n)) for n in names])
            rng = np.array([m.jnt_range[m.joint_name2id(n)] for n in names], float)
            self._aji = (names, qa + 1, va + 1 + nq, qa, va, rng)
        return self._aji

    def _eef_site(self):
        if not hasattr(self, "_esite"):
            m = self.sim.model; sid = int(m.site_name2id("gripper0_grip_site"))
            r0 = self.env.env.robots[0]
            esid = getattr(r0, "eef_site_id", None)
            if isinstance(esid, dict):
                esid = esid.get("right", list(esid.values())[0])
            if esid is not None:
                assert int(esid) == sid, (esid, sid)
            self._esite = sid
        return self._esite

    def eef_pose(self):
        """eef site（robot0_eef_pos の元）の位置 (3,) と回転行列 (3,3)。forward 済みの sim から読む。"""
        sid = self._eef_site(); d = self.sim.data
        return np.array(d.site_xpos[sid], float).copy(), np.array(d.site_xmat[sid], float).reshape(3, 3).copy()

    def _jac_site(self):
        sid = self._eef_site()
        try:
            import mujoco
            jp = np.zeros((3, self.sim.model.nv)); jr = np.zeros((3, self.sim.model.nv))
            mujoco.mj_jacSite(self.sim.model._model, self.sim.data._data, jp, jr, sid)
        except Exception:
            nv = self.sim.model.nv
            jp = np.array(self.sim.data.get_site_jacp(self.sim.model.site_id2name(sid))).reshape(3, nv)
            jr = np.array(self.sim.data.get_site_jacr(self.sim.model.site_id2name(sid))).reshape(3, nv)
        return jp, jr

    @staticmethod
    def rot_err_vec(R, Rt):
        """世界座標の回転誤差ベクトル（R を Rt へ回す小回転）。0.5 Σ_i R[:,i] × Rt[:,i]。"""
        return 0.5 * sum(np.cross(R[:, i], Rt[:, i]) for i in range(3))

    @staticmethod
    def rot_angle_deg(R, Rt):
        c = (np.trace(Rt @ R.T) - 1) / 2
        return float(np.degrees(np.arccos(np.clip(c, -1, 1))))

    def ik_arm(self, target_pos, target_rot, q_init, mask_pos=True, mask_rot=True, lam=1e-3, iters=200, tol_pos=5e-4, tol_rot_deg=0.2):
        """減衰最小二乗 IK（腕 7 関節。他の qpos・qvel は触らない）。呼び出し前後で sim の状態は同じ（復元は呼び出し側）。
        返り値: dict(q(7,), conv, iters, err_pos_mm（拘束した成分のみでなく常に位置 3 の誤差）, err_rot_deg（常に回転の誤差）, err_pos_hist, err_rot_hist)。"""
        _n, _qi, _vi, qa, va, rng = self.arm_joint_idx()
        saved = self.sim.get_state().flatten().copy()
        q = np.clip(np.asarray(q_init, float).copy(), rng[:, 0], rng[:, 1])
        hp = []; hr = []; conv = False; it = 0
        try:
            for it in range(iters + 1):
                self.sim.data.qpos[qa] = q; self.sim.forward()
                p, R = self.eef_pose(); ep = np.asarray(target_pos, float) - p; er = self.rot_err_vec(R, np.asarray(target_rot, float))
                hp.append(float(np.linalg.norm(ep))); hr.append(self.rot_angle_deg(R, np.asarray(target_rot, float)))
                ok_p = (not mask_pos) or hp[-1] < tol_pos; ok_r = (not mask_rot) or hr[-1] < tol_rot_deg
                if ok_p and ok_r:
                    conv = True; break
                if it == iters:
                    break
                jp, jr = self._jac_site(); rows = []; e = []
                if mask_pos:
                    rows.append(jp[:, va]); e.append(ep)
                if mask_rot:
                    rows.append(jr[:, va]); e.append(er)
                J = np.vstack(rows); e = np.concatenate(e)
                dq = J.T @ np.linalg.solve(J @ J.T + lam * np.eye(len(e)), e)
                q = np.clip(q + dq, rng[:, 0], rng[:, 1])
        finally:
            self.sim.set_state_from_flattened(saved); self.sim.forward()
        return dict(q=q, conv=bool(conv), iters=int(it), err_pos_mm=hp[-1] * 1000, err_rot_deg=hr[-1], err_pos_hist=np.array(hp) * 1000, err_rot_hist=np.array(hr))

    def restore_set_arm(self, row_base, q_arm7):
        """I5: base の完全状態を戻し、腕 7 関節の qpos だけ q_arm7 にする（qvel・指・固定具は base のまま）→ forward → 観測更新。"""
        _n, qi, _vi, _qa, _va, _r = self.arm_joint_idx()
        row = np.asarray(row_base, float).copy(); row[qi] = np.asarray(q_arm7, float)
        return self.restore(row)

    def contact_fmax(self):
        """現在の接触（forward 直後）のロボット対非ロボットの法線力の最大 [N]（相手は物体・机などすべて）。"""
        fs = [c[3] for c in json.loads(E.contact_detail(self.sim, self.m))]
        return float(max(fs)) if fs else 0.0

    def ee(self):
        return np.asarray(self.obs["robot0_eef_pos"], float)

    def infer(self, client):
        ch = np.asarray(client.infer(E.make_element(self.obs, self.prompt))["actions"])
        return ch

    def prefix_toward(self, grip, n=8, mag=0.3):
        d = np.asarray(self.sim.data.body_xpos[self.tbid], float) - self.ee(); u = d / (np.linalg.norm(d) + 1e-9)
        for _ in range(n):
            self.step(np.concatenate([u * mag, np.zeros(3), [grip]]), "prefix_toward")

    def prefix_retreat(self, grip, quat_ref, dz=0.03, max_up=20, max_rot=15):
        from scipy.spatial.transform import Rotation as Rot
        z0 = self.ee()[2]
        for _ in range(max_up):
            if self.ee()[2] >= z0 + dz:
                break
            self.step(np.array([0, 0, 1.0, 0, 0, 0, grip]), "prefix_retreat_up")
        for _ in range(max_rot):
            r = Rot.from_quat(quat_ref) * Rot.from_quat(self.obs["robot0_eef_quat"]).inv()   # 世界座標の回転誤差
            rv = r.as_rotvec()
            if np.linalg.norm(rv) < 0.02:
                break
            self.step(np.concatenate([[0, 0, 0], np.clip(rv / 0.5 * 1.0, -1, 1), [grip]]), "prefix_retreat_rot")

    def prefix_home(self, grip, pos_ref, quat_ref, max_steps=40, tol_pos=0.01, tol_rot=0.05):
        """I3 (f): 手先を基準の位置・姿勢へ OSC 指令で戻す（位置誤差/0.05 と回転誤差/0.5 を clip して同時に出す）。位置誤差 < tol_pos かつ回転誤差 < tol_rot で終了。"""
        from scipy.spatial.transform import Rotation as Rot
        pos_ref = np.asarray(pos_ref, float); n0 = self.n
        def errs():
            ep = pos_ref - self.ee()
            rv = (Rot.from_quat(quat_ref) * Rot.from_quat(self.obs["robot0_eef_quat"]).inv()).as_rotvec()
            return ep, rv
        for _ in range(max_steps):
            ep, rv = errs()
            if np.linalg.norm(ep) < tol_pos and np.linalg.norm(rv) < tol_rot:
                break
            self.step(np.concatenate([np.clip(ep / 0.05, -1, 1), np.clip(rv / 0.5, -1, 1), [grip]]), "prefix_home")
        ep, rv = errs()
        self.home = dict(err_pos_mm=float(np.linalg.norm(ep) * 1000), err_rot_rad=float(np.linalg.norm(rv)), steps=self.n - n0)
        return self.home

    def continue_policy(self, client, H_, replan):
        plan = collections.deque(); steps = 0
        while steps < H_ and not self.done:
            if not plan:
                ch = self.infer(client); self.log["chunks"].append(ch.copy()); self.log["chunk_at"].append(self.n); plan.extend(ch[:replan])
            act = np.array(plan.popleft(), float); act[:6] = np.clip(act[:6], -1, 1)
            self.step(act, "policy"); steps += 1

    def resample(self, client, n):
        """同じ観測で n 回問い合わせる。物理は進めない。"""
        return np.stack([self.infer(client) for _ in range(n)])

    def result(self):
        L = {k: (np.array(v) if k != "phase" else np.array(v)) for k, v in self.log.items() if v}
        return L


def load_trace(path):
    z = np.load(path, allow_pickle=False)
    return {k: z[k] for k in z.files}


def open_from_trace(tr, env=None):
    return Ctx(str(tr["task"]), int(tr["seed"]), str(tr["level"]), env=env, pos0=tr.get("ref_pos0"), quat0=tr.get("ref_quat0"))


def goto(ctx, tr, via, t):
    """step t の地点へ。via = restore（sim_state）／replay（act_exec）。"""
    if via == "restore":
        return ctx.restore(tr["sim_state"][t])
    return ctx.replay(tr["act_exec"], t)


def dev_mm(a, b):
    a, b = np.asarray(a), np.asarray(b); n = min(len(a), len(b))
    d = np.linalg.norm(a[:n] - b[:n], axis=1)
    return float(d.max() * 1000), d


class StubClient:
    """記録したチャンクを順に返す（C4: trace あり/なしの比較用）。"""
    def __init__(self, chunks):
        self.chunks = list(chunks); self.i = 0
    def infer(self, el):
        ch = self.chunks[self.i]; self.i += 1
        return {"actions": ch}


def selfcheck(tr, out_json=None):
    res = {}
    T = len(tr["act_exec"]); ee_tr = tr["ee"]
    ctx = open_from_trace(tr)
    # C2: 再生の忠実度
    ctx.replay(tr["act_exec"], T)
    ee_re = np.array(ctx.log["ee"][1:])
    m2, d2 = dev_mm(ee_re, ee_tr)
    res["C2_replay_max_dev_mm"] = m2; res["C2_q_max_dev"] = float(np.abs(np.array(ctx.log["q"][1:]) - tr["q"]).max())
    res["C2_first_step_over_1mm"] = int(np.argmax(d2 > 1e-3)) if (d2 > 1e-3).any() else -1
    # C3: 復元の忠実度（新品の環境へ復元 / 再生し終えた後の同じ環境へ復元＝制御器の内部状態が古いまま）
    for variant in ("fresh", "stale"):
        for t in (40, 80):
            if t + 10 > T:
                continue
            if variant == "fresh":
                ctx.fresh()
            else:
                ctx.fresh(); ctx.replay(tr["act_exec"], min(T, t + 37))   # 別の地点まで進めた後で戻す
                ctx.log = dict(ee=[], quat=[], q=[], jq=[], act=[], phase=[], chunks=[], chunk_at=[])
            ctx.restore(tr["sim_state"][t])
            d_pre = float(np.linalg.norm(ctx.ee() - ee_tr[t - 1]) * 1000)
            for k in range(t, t + 10):
                ctx.step(tr["act_exec"][k], "c3")
            ee_c3 = np.array(ctx.log["ee"][-10:])   # 復元後に実行した 10 step
            m3, _ = dev_mm(ee_c3, ee_tr[t:t + 10])
            res[f"C3_{variant}_t{t}_max_dev_mm"] = m3; res[f"C3_{variant}_t{t}_restore_ee_err_mm"] = d_pre
    # C4: trace の無害性（記録チャンクを返す stub で run_episode を trace あり/なしで実行）
    import tempfile
    ap = E.build_parser(); base = ap.parse_args(["--tasks", str(tr["task"]), "--level", str(tr["level"]), "--replan", str(int(tr["replan"])), "--max-steps", str(T)])
    spec = E.TASKS[str(tr["task"])]
    outs = {}
    for flag in (False, True):
        a = ap.parse_args(["--tasks", str(tr["task"]), "--level", str(tr["level"]), "--replan", str(int(tr["replan"])), "--max-steps", str(T)] + (["--trace", tempfile.mkdtemp()] if flag else []))
        r = E.run_episode(ctx.env, str(tr["task"]), spec, a, StubClient(tr["chunks"]), ctx.rows, ctx.prompt, int(tr["seed"]), int(tr["rep"]), {"pos0": np.asarray(tr["ref_pos0"]).copy(), "quat0": np.asarray(tr["ref_quat0"]).copy()})
        outs[flag] = r
        if flag:
            z = np.load(os.path.join(a.trace, f"{tr['task']}_s{int(tr['seed'])}_r{int(tr['rep'])}.npz"))
            res["C4_trace_on_ee_max_dev_vs_original_mm"], _ = dev_mm(z["ee"], ee_tr)
            res["C4_trace_on_sim_state_max_abs_diff"] = float(np.abs(z["sim_state"] - tr["sim_state"]).max()) if z["sim_state"].shape == tr["sim_state"].shape else "shape_mismatch"
    keys = ("steps", "path_m", "q_end", "success", "disp_mm")
    res["C4_off_vs_on"] = {k: [outs[False][k], outs[True][k]] for k in keys}
    res["C4_off_equals_on"] = all(outs[False][k] == outs[True][k] for k in keys)
    res["C4_off_path_m_vs_original"] = float(outs[False]["path_m"])
    if out_json:
        json.dump(res, open(out_json, "w"), indent=1)
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--trace-file", required=True)
    ap.add_argument("--mode", choices=["selfcheck", "continue", "resample", "none"], default="none")
    ap.add_argument("--via", choices=["restore", "replay"], default="restore")
    ap.add_argument("--t", type=int, default=0)
    ap.add_argument("--H", type=int, default=60, help="continue: 方策で続ける step 数")
    ap.add_argument("--replan", type=int, default=5)
    ap.add_argument("--prefix", choices=["none", "toward", "retreat"], default="none")
    ap.add_argument("--n", type=int, default=16, help="resample: 問い合わせ回数")
    ap.add_argument("--host", default="localhost"); ap.add_argument("--port", type=int, default=18040)
    ap.add_argument("--out", default="")
    a = ap.parse_args()
    tr = load_trace(a.trace_file)
    if a.mode == "selfcheck":
        print(json.dumps(selfcheck(tr, a.out or None), indent=1, ensure_ascii=False)); return
    ctx = open_from_trace(tr)
    goto(ctx, tr, a.via, a.t)
    client = None
    if a.mode in ("continue", "resample"):
        import websockets.sync.client as _wsc
        _wsc.connect = functools.partial(_wsc.connect, ping_interval=None)
        from openpi_client import websocket_client_policy
        client = websocket_client_policy.WebsocketClientPolicy(a.host, a.port)
    out = {}
    if a.mode == "continue":
        grip = float(tr["act_exec"][max(a.t - 1, 0)][6])
        if a.prefix == "toward":
            ctx.prefix_toward(grip)
        elif a.prefix == "retreat":
            ctx.prefix_retreat(grip, tr["quat"][0])
        ctx.continue_policy(client, a.H, a.replan)
        out = ctx.result()
    elif a.mode == "resample":
        out = dict(chunks=ctx.resample(client, a.n), ee=ctx.ee(), q=float(ctx.sim.data.qpos[ctx.qadr]))
    out.update(success=ctx.done, task=str(tr["task"]), seed=int(tr["seed"]), t=a.t, via=a.via, prefix=a.prefix, replan=a.replan, H=a.H)
    if a.out:
        np.savez_compressed(a.out, **out)
    print({k: (v.shape if hasattr(v, "shape") else v) for k, v in out.items()})


if __name__ == "__main__":
    main()
