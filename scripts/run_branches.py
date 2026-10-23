"""I1（固定観測の再サンプル）と I2（同一状態からの短い分岐）を回す（計画 v2 §8）。Pod で実行。
使い方（Pod）:
  python kadai/scripts/run_branches.py --shard 0 --shards 4 [--port 18040] [--out data/branches]
  python kadai/scripts/run_branches.py --check-window drawer_s1014_r0_rec --k 1 --out data/branches_check   # 操作チェック（1 窓・各条件 k 本）
分岐点: rec = win_start+20（回復）、entry = max(win_start-20, 10)（進入。(a) のみ）、fp = 成功 ep の偽陽性窓の win_start+20（(a) のみ）。
各分岐はエピソード絶対 step 250 まで（または done まで）。出力: <out>/b_<id>.npz（軌跡）, <out>/summary_s<shard>.jsonl（1 分岐 1 行。再開時はこれを読んで済みをスキップ）,
  <out>/i1_<window>.npz（16 チャンク）。完了時にログへ BRANCHES_DONE。
prog の定義は analyze_aprime.py と同じ（q0 = 元 trace の q[0]、td0 = 元 trace の tgt_dist[0]）。分岐中は q と tgt_dist を保存し解析側で計算する。
保存軌跡の添字 0 = 復元直後（分岐点 t）、添字 k = 分岐後 k step 目（前置きを含む。絶対 step = t + k）。
"""
import argparse, collections, csv, functools, json, os, random, sys, time
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import branch_kadai as B   # noqa: E402
import eval_kadai_closedloop as E   # noqa: E402

MAXSTEP = 250
CONDS = {"a": 8, "b": 4, "d": 8, "e": 8}


def load_windows(csv_path, trace_dir):
    rows = list(csv.DictReader(open(csv_path)))
    wins = []
    for r in rows:
        base = dict(task=r["task"], seed=int(r["seed"]), rep=int(r["rep"]), group=r["group"], ws=int(r["win_start"]), succ=int(r["success"]))
        base["file"] = os.path.join(trace_dir, f"{r['task']}_s{r['seed']}_r{r['rep']}.npz")
        if base["succ"] == 0:
            wins.append(dict(base, kind="rec", t=base["ws"] + 20, conds=dict(CONDS)))
            wins.append(dict(base, kind="entry", t=max(base["ws"] - 20, 10), conds={"a": 8}))
        else:
            wins.append(dict(base, kind="fp", t=base["ws"] + 20, conds={"a": 8}))
    for w in wins:
        w["wid"] = f"{w['task']}_s{w['seed']}_r{w['rep']}_{w['kind']}"
        w["cost"] = sum(w["conds"].values()) + 1.0
    return wins


def load_windows_i3(csv_path, trace_dir):
    """I3（計画 v2 §10）: home / t0 / t13 / t23（失敗窓 56 本）と succ13 / succ23（運・中間・保留 seed の成功 ep）。"""
    rows = list(csv.DictReader(open(csv_path)))
    grp = {(r["task"], int(r["seed"])): r["group"] for r in rows}
    wins = []; fails = collections.defaultdict(list)
    for r in rows:
        if int(r["success"]) == 0:
            fails[(r["task"], int(r["seed"]))].append(int(r["win_start"]))
            base = dict(task=r["task"], seed=int(r["seed"]), rep=int(r["rep"]), group=r["group"], ws=int(r["win_start"]), succ=0,
                        file=os.path.join(trace_dir, f"{r['task']}_s{r['seed']}_r{r['rep']}.npz"))
            ws = base["ws"]
            wins.append(dict(base, kind="home", t=ws + 20, conds={"f": 8}))
            wins.append(dict(base, kind="t0", t=0, conds={"a": 8}))
            wins.append(dict(base, kind="t13", t=int(round(ws / 3)), conds={"a": 8}))
            wins.append(dict(base, kind="t23", t=int(round(2 * ws / 3)), conds={"a": 8}))
    skipped = []
    for (task, seed), g in sorted(grp.items()):
        if g not in ("luck", "mid", "hold"):
            continue
        sreps = []
        for f_ in sorted(Path(trace_dir).glob(f"{task}_s{seed}_r*.npz")):
            z = np.load(f_, allow_pickle=False)
            if bool(z["success"]):
                sreps.append(int(z["rep"]))
        if not sreps:
            continue
        if not fails[(task, seed)]:
            skipped.append((task, seed, "no failure window")); continue
        sbar = float(np.median(fails[(task, seed)]))
        for rep in sorted(sreps)[:2]:
            for kind, fr in (("succ13", 1 / 3), ("succ23", 2 / 3)):
                wins.append(dict(task=task, seed=seed, rep=rep, group=g, ws=int(round(sbar)), succ=1, kind=kind, t=int(round(fr * sbar)), conds={"a": 8},
                                 file=os.path.join(trace_dir, f"{task}_s{seed}_r{rep}.npz")))
    for w in wins:
        w["wid"] = f"{w['task']}_s{w['seed']}_r{w['rep']}_{w['kind']}"
        w["cost"] = sum(w["conds"].values()) + 1.0
    return wins, skipped


def load_windows_i4(csv_path, trace_dir):
    """I4（計画 v2 §14）: I3 の成功対照がある seed の失敗窓 × 成功 ep（I3 と同じ最大 2 本）× t ∈ {t13, t23}（t は失敗窓のもの）。条件 s / r 各 8 本。"""
    w3, skipped = load_windows_i3(csv_path, trace_dir)
    succ = collections.defaultdict(set)
    for w in w3:
        if w["kind"] in ("succ13", "succ23"):
            succ[(w["task"], w["seed"])].add(w["rep"])
    wins = []
    for w in w3:
        if w["kind"] not in ("t13", "t23") or (w["task"], w["seed"]) not in succ:
            continue
        for srep in sorted(succ[(w["task"], w["seed"])])[:2]:
            wid = f"{w['task']}_s{w['seed']}_r{w['rep']}_x_r{srep}_{w['kind']}"
            wins.append(dict(task=w["task"], seed=w["seed"], rep=w["rep"], srep=srep, group=w["group"], ws=w["ws"], succ=0, kind=w["kind"], t=w["t"],
                             fwid=w["wid"], file=w["file"], file2=os.path.join(trace_dir, f"{w['task']}_s{w['seed']}_r{srep}.npz"),
                             conds={"s": 8, "r": 8}, wid=wid, cost=17.0))
    return wins, skipped


I5_CONDS = ["q", "p", "pp", "po", "n"]


def load_windows_i5(csv_path, trace_dir):
    """I5（計画 v2 §16）: I4 と同じ 28 組、t = t13 のみ、条件 q/p/pp/po/n 各 8 本。"""
    w4, skipped = load_windows_i4(csv_path, trace_dir)
    wins = []
    for w in w4:
        if w["kind"] != "t13":
            continue
        wins.append(dict(w, conds={c: 8 for c in I5_CONDS}, cost=41.0))
    return wins, skipped


def shard_windows(wins, shards, seed=20261015):
    order = list(wins); random.Random(seed).shuffle(order); order.sort(key=lambda w: -w["cost"])
    loads = [0.0] * shards; buckets = [[] for _ in range(shards)]
    for w in order:
        i = int(np.argmin(loads)); buckets[i].append(w); loads[i] += w["cost"]
    return buckets


class Runner:
    def __init__(self, port, host="localhost"):
        import websockets.sync.client as _wsc
        _wsc.connect = functools.partial(_wsc.connect, ping_interval=None)
        from openpi_client import websocket_client_policy
        self.client = websocket_client_policy.WebsocketClientPolicy(host, port)
        self.ctx = None

    def open(self, tr):
        task = str(tr["task"])
        if self.ctx is None or self.ctx.name != task:
            self.ctx = B.Ctx(task, int(tr["seed"]), str(tr["level"]), pos0=tr["ref_pos0"], quat0=tr["ref_quat0"])
        c = self.ctx
        c.seed = int(tr["seed"]); c.level = str(tr["level"])
        c.state = {"pos0": np.asarray(tr["ref_pos0"]).copy(), "quat0": np.asarray(tr["ref_quat0"]).copy()}
        return c

    def branch(self, tr, t, cond):
        """cond ∈ a / b / d / e / f（f = 初期姿勢へ退避、I3）。"""
        c = self.open(tr); spec = c.spec
        c.fresh()
        base_pos = {kk[:-4]: np.asarray(c.obs[kk], float).copy() for kk in c.obs if kk.endswith("_pos") and "robot0" not in kk and "gripper" not in kk}
        ee_fresh = c.ee().copy()
        c.restore(tr["sim_state"][t])
        # 行 t = t step 実行後 = trace の添字 t-1。t=0 は trace に待機後の手先が無いので fresh() 直後の手先と比べる
        restore_err_mm = float(np.linalg.norm(c.ee() - (tr["ee"][t - 1] if t >= 1 else ee_fresh)) * 1000)
        disp = {}
        def upd():
            for n, p0 in base_pos.items():
                if n in spec["ooi"] or f"{n}_pos" not in c.obs:
                    continue
                disp[n] = max(disp.get(n, 0.0), float(np.sum(np.abs(np.asarray(c.obs[f"{n}_pos"]) - p0))))
        upd(); disp_start = max(disp.values(), default=0.0)
        tbid = c.tbid
        tdist = lambda: float(np.linalg.norm(np.asarray(c.sim.data.body_xpos[tbid], float) - c.ee()))
        td = [tdist()]; qs = [float(c.sim.data.qpos[c.qadr])]
        orig = c.step
        def step(act, phase="policy"):
            o = orig(act, phase); upd(); td.append(tdist()); qs.append(float(c.sim.data.qpos[c.qadr]))
            return o
        c.step = step
        home = None
        try:
            grip = float(tr["act_exec"][max(t - 1, 0)][6])
            ee0 = c.ee().copy()
            if cond == "d":
                c.prefix_toward(grip)
            elif cond == "e":
                c.prefix_retreat(grip, tr["quat"][0])
            elif cond == "f":
                c.prefix_home(grip, tr["ee"][0], tr["quat"][0])
                home = dict(c.home, q_after=float(c.sim.data.qpos[c.qadr]), disp_after_mm=max(disp.values(), default=0.0) * 1000)
            n_pref = c.n
            pref_ee = np.array(c.log["ee"][2:2 + n_pref]) if n_pref else np.zeros((0, 3))   # log["ee"] = [fresh, restore, step1, ...]
            pref_move_mm = float(np.linalg.norm(pref_ee - ee0, axis=1).max() * 1000) if n_pref else 0.0
            pref_path_mm = float(np.linalg.norm(np.diff(np.vstack([ee0, pref_ee]), axis=0), axis=1).sum() * 1000) if n_pref else 0.0
            replan = 1 if cond == "b" else int(tr["replan"])
            plan = collections.deque(); n_infer = 0
            while t + c.n < MAXSTEP and not c.done:
                if not plan:
                    ch = c.infer(self.client); n_infer += 1; plan.extend(ch[:replan])
                act = np.array(plan.popleft(), float); act[:6] = np.clip(act[:6], -1, 1)
                c.step(act, "policy")
        finally:
            c.step = orig
        L = c.result()
        hx = {} if home is None else dict(home_err_pos_mm=home["err_pos_mm"], home_err_rot_rad=home["err_rot_rad"], home_steps=home["steps"],
                                          q_after_home=home["q_after"], disp_after_home_mm=home["disp_after_mm"])
        return dict(**hx, ee=L["ee"][1:], q=np.array(qs), td=np.array(td), act=L["act"][1:], policy=(L["phase"][1:] == "policy").astype(np.int8),
                    n_pref=n_pref, n_infer=n_infer, done=bool(c.done), disp_mm=max(disp.values(), default=0.0) * 1000, disp_start_mm=disp_start * 1000,
                    top=(max(disp, key=disp.get) if disp else ""), restore_err_mm=restore_err_mm, pref_move_mm=pref_move_mm, pref_path_mm=pref_path_mm,
                    t=t, n_steps=c.n, cond=cond, replan=replan, q0ref=float(tr["q"][0]), td0ref=float(tr["tgt_dist"][0]))

    def branch_swap(self, trb, trr, t, cond):
        """I4: base（trb）の完全状態に、ロボットの関節だけ trr の同 t の値を入れる → 方策で終端（250）まで。cond は s / r（記録上の区別のみ）。"""
        c = self.open(trb); spec = c.spec
        c.fresh()
        base_pos = {kk[:-4]: np.asarray(c.obs[kk], float).copy() for kk in c.obs if kk.endswith("_pos") and "robot0" not in kk and "gripper" not in kk}
        c.restore_swap_robot(trb["sim_state"][t], trr["sim_state"][t])
        names, qi, vi = c.robot_joint_idx()
        # 操作チェック: 入れ替え後の flattened 状態の並びを検算（ロボット関節 = robot 側、固定具 = base 側）
        flat = np.asarray(c.sim.get_state().flatten(), float)
        jerr_q = float(np.abs(flat[qi] - np.asarray(trr["sim_state"][t], float)[qi]).max())
        jerr_v = float(np.abs(flat[vi] - np.asarray(trr["sim_state"][t], float)[vi]).max())
        oth = np.setdiff1d(np.arange(1, 1 + c.m.nq), qi)
        oerr_q = float(np.abs(flat[oth] - np.asarray(trb["sim_state"][t], float)[oth]).max())   # ロボット以外の qpos は base のまま
        ee_robot_err = float(np.linalg.norm(c.ee() - trr["ee"][t - 1]) * 1000); ee_base_gap = float(np.linalg.norm(c.ee() - trb["ee"][t - 1]) * 1000)
        q_base_err = float(c.sim.data.qpos[c.qadr] - trb["q"][t - 1])
        jq_robot_err = float(np.abs(np.asarray(c.obs["robot0_joint_pos"], float) - trr["jq"][t - 1]).max()) if "jq" in trr else float("nan")
        f0 = c.contact_fmax()
        disp = {}
        def upd():
            for n, p0 in base_pos.items():
                if n in spec["ooi"] or f"{n}_pos" not in c.obs:
                    continue
                disp[n] = max(disp.get(n, 0.0), float(np.sum(np.abs(np.asarray(c.obs[f"{n}_pos"]) - p0))))
        upd(); disp_start = max(disp.values(), default=0.0)
        tbid = c.tbid
        tdist = lambda: float(np.linalg.norm(np.asarray(c.sim.data.body_xpos[tbid], float) - c.ee()))
        td = [tdist()]; qs = [float(c.sim.data.qpos[c.qadr])]; fl = [f0]
        orig = c.step
        def step(act, phase="policy"):
            o = orig(act, phase); upd(); td.append(tdist()); qs.append(float(c.sim.data.qpos[c.qadr]))
            if len(fl) < 2:
                fl.append(c.contact_fmax())
            return o
        c.step = step
        try:
            replan = int(trb["replan"]); plan = collections.deque(); n_infer = 0
            while t + c.n < MAXSTEP and not c.done:
                if not plan:
                    ch = c.infer(self.client); n_infer += 1; plan.extend(ch[:replan])
                act = np.array(plan.popleft(), float); act[:6] = np.clip(act[:6], -1, 1)
                c.step(act, "policy")
        finally:
            c.step = orig
        L = c.result()
        return dict(ee=L["ee"][1:], q=np.array(qs), td=np.array(td), act=L["act"][1:], f=np.array(fl),
                    n_pref=0, n_infer=n_infer, done=bool(c.done), disp_mm=max(disp.values(), default=0.0) * 1000, disp_start_mm=disp_start * 1000,
                    top=(max(disp, key=disp.get) if disp else ""), t=t, n_steps=c.n, cond=cond, replan=replan,
                    ee_robot_err_mm=ee_robot_err, ee_base_gap_mm=ee_base_gap, q_base_err=q_base_err, jq_robot_err=jq_robot_err,
                    swap_jerr_q=jerr_q, swap_jerr_v=jerr_v, swap_oerr_q=oerr_q, f0=f0, f1=(fl[1] if len(fl) > 1 else float("nan")),
                    q0ref=float(trb["q"][0]), td0ref=float(trb["tgt_dist"][0]))

    def branch_ik(self, trb, trs, t, cond):
        """I5: base = 失敗 ep（trb）の完全状態、腕 7 関節 qpos だけ cond に応じて作る（trs = 成功 ep）→ 方策で終端まで。
        q: 成功の q ／ p: IK(成功の手先 位置+回転, 初期 q_f) ／ pp: 位置のみ ／ po: 回転のみ ／ n: IK(失敗の手先, 初期 q_s)。"""
        c = self.open(trb); spec = c.spec
        c.fresh()
        base_pos = {kk[:-4]: np.asarray(c.obs[kk], float).copy() for kk in c.obs if kk.endswith("_pos") and "robot0" not in kk and "gripper" not in kk}
        names, qi, vi, qa, va, rng = c.arm_joint_idx()
        row_f = np.asarray(trb["sim_state"][t], float); row_s = np.asarray(trs["sim_state"][t], float)
        q_f = row_f[qi].copy(); q_s = row_s[qi].copy()
        c.restore(row_s); ps, Rs = c.eef_pose()
        c.restore(row_f); pf, Rf = c.eef_pose()
        ik = None
        if cond == "q":
            q_c = q_s.copy()
        elif cond == "n":
            ik = c.ik_arm(pf, Rf, q_s, True, True)
        else:
            ik = c.ik_arm(ps, Rs, q_f, mask_pos=(cond != "po"), mask_rot=(cond != "pp"))
        if ik is not None:
            q_c = ik["q"]
        meta = dict(ik_conv=(True if ik is None else ik["conv"]), ik_iters=(0 if ik is None else ik["iters"]),
                    ik_err_pos_mm=(0.0 if ik is None else ik["err_pos_mm"]), ik_err_rot_deg=(0.0 if ik is None else ik["err_rot_deg"]),
                    d_qc_qf=float(np.linalg.norm(q_c - q_f)), d_qc_qs=float(np.linalg.norm(q_c - q_s)), d_qs_qf=float(np.linalg.norm(q_s - q_f)),
                    t=t, cond=cond, replan=int(trb["replan"]), q0ref=float(trb["q"][0]), td0ref=float(trb["tgt_dist"][0]))
        if ik is not None and not ik["conv"]:
            return dict(skipped=True, **meta)
        c.restore_set_arm(row_f, q_c)
        pc, Rc = c.eef_pose()
        meta.update(
            err_vs_s_pos_mm=float(np.linalg.norm(pc - ps) * 1000), err_vs_s_rot_deg=c.rot_angle_deg(Rc, Rs),
            err_vs_f_pos_mm=float(np.linalg.norm(pc - pf) * 1000), err_vs_f_rot_deg=c.rot_angle_deg(Rc, Rf),
            ee_obs_vs_site_mm=float(np.linalg.norm(c.ee() - pc) * 1000),
            fixture_q_err=float(c.sim.data.qpos[c.qadr] - trb["q"][t - 1]),
            finger_qv_ok=float(np.abs(np.asarray(c.sim.get_state().flatten(), float)[vi] - row_f[vi]).max()))
        # 目標誤差（条件ごとの目標: q/p/pp/po = 成功の手先、n = 失敗の手先）
        tg = "f" if cond == "n" else "s"
        meta["tgt_err_pos_mm"] = meta[f"err_vs_{tg}_pos_mm"]; meta["tgt_err_rot_deg"] = meta[f"err_vs_{tg}_rot_deg"]
        flat = np.asarray(c.sim.get_state().flatten(), float)
        oth = np.setdiff1d(np.arange(1, 1 + c.m.nq), qi)
        meta["oerr_q"] = float(np.abs(flat[oth] - row_f[oth]).max())            # 腕以外の qpos は失敗側のまま
        meta["arm_q_err"] = float(np.abs(flat[qi] - q_c).max())
        meta["arm_v_err"] = float(np.abs(flat[vi] - row_f[vi]).max())
        f0 = c.contact_fmax()
        disp = {}
        def upd():
            for n, p0 in base_pos.items():
                if n in spec["ooi"] or f"{n}_pos" not in c.obs:
                    continue
                disp[n] = max(disp.get(n, 0.0), float(np.sum(np.abs(np.asarray(c.obs[f"{n}_pos"]) - p0))))
        upd(); disp_start = max(disp.values(), default=0.0)
        tbid = c.tbid
        tdist = lambda: float(np.linalg.norm(np.asarray(c.sim.data.body_xpos[tbid], float) - c.ee()))
        td = [tdist()]; qs = [float(c.sim.data.qpos[c.qadr])]; fl = [f0]
        orig = c.step
        def step(act, phase="policy"):
            o = orig(act, phase); upd(); td.append(tdist()); qs.append(float(c.sim.data.qpos[c.qadr]))
            if len(fl) < 2:
                fl.append(c.contact_fmax())
            return o
        c.step = step
        try:
            replan = int(trb["replan"]); plan = collections.deque(); n_infer = 0
            while t + c.n < MAXSTEP and not c.done:
                if not plan:
                    ch = c.infer(self.client); n_infer += 1; plan.extend(ch[:replan])
                act = np.array(plan.popleft(), float); act[:6] = np.clip(act[:6], -1, 1)
                c.step(act, "policy")
        finally:
            c.step = orig
        L = c.result()
        return dict(ee=L["ee"][1:], q=np.array(qs), td=np.array(td), act=L["act"][1:], f=np.array(fl), skipped=False,
                    n_pref=0, n_infer=n_infer, done=bool(c.done), disp_mm=max(disp.values(), default=0.0) * 1000, disp_start_mm=disp_start * 1000,
                    top=(max(disp, key=disp.get) if disp else ""), n_steps=c.n, f0=f0, f1=(fl[1] if len(fl) > 1 else float("nan")), **meta)

    def resample(self, tr, t, n=16):
        c = self.open(tr); c.fresh(); c.restore(tr["sim_state"][t])
        return c.resample(self.client, n), c.ee().copy()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shard", type=int, default=0); ap.add_argument("--shards", type=int, default=4)
    ap.add_argument("--port", type=int, default=18040)
    ap.add_argument("--csv", default="kadai/state/aprime_windows_20261014.csv")
    ap.add_argument("--traces", default="data/trace_aprime")
    ap.add_argument("--out", default="data/branches")
    ap.add_argument("--check-window", default="", help="<task>_s<seed>_r<rep>_<kind> の 1 窓だけ、各条件 --k 本")
    ap.add_argument("--k", type=int, default=0)
    ap.add_argument("--i3", action="store_true", help="I3（計画 v2 §10）の窓を使う")
    ap.add_argument("--i5", action="store_true", help="I5（計画 v2 §16、腕の状態の分解）の窓を使う")
    ap.add_argument("--i4", action="store_true", help="I4（計画 v2 §14、腕の状態の入れ替え）の窓を使う")
    a = ap.parse_args()
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    if a.i5:
        wins, skipped = load_windows_i5(a.csv, a.traces); print("I5 skipped:", skipped, "windows:", len(wins), flush=True)
    elif a.i4:
        wins, skipped = load_windows_i4(a.csv, a.traces); print("I4 skipped:", skipped, "windows:", len(wins), flush=True)
    elif a.i3:
        wins, skipped = load_windows_i3(a.csv, a.traces); print("I3 skipped succ seeds:", skipped, flush=True)
    else:
        wins = load_windows(a.csv, a.traces)
    if a.check_window:
        mine = [w for w in wins if w["wid"] == a.check_window]; assert mine, "窓が見つからない"; tag = "check"
    else:
        mine = shard_windows(wins, a.shards)[a.shard]; tag = f"s{a.shard}"
    summ = out / f"summary_{tag}.jsonl"
    done_ids = {json.loads(l)["id"] for f_ in out.glob("summary_s*.jsonl") for l in open(f_) if l.strip()} if tag != "check" else (  # 再開は全シャードの記録を見る（シャード数を変えても重複しない）
        {json.loads(l)["id"] for l in open(summ) if l.strip()} if summ.exists() else set())
    R = Runner(a.port)
    rng = random.Random(1000 + a.shard); rng.shuffle(mine)
    total = sum(sum((a.k or n) for n in w["conds"].values()) for w in mine)
    print(f"[shard {tag}] windows={len(mine)} branches={total} done_before={len(done_ids)}", flush=True)
    t_start = time.time(); nb = 0
    for w in mine:
        tr = B.load_trace(w["file"]); t = w["t"]; tr2 = B.load_trace(w["file2"]) if (a.i4 or a.i5) else None
        if not (a.i3 or a.i4 or a.i5) and not (out / f"i1_{w['wid']}.npz").exists():
            chs, ee = R.resample(tr, t)
            np.savez_compressed(out / f"i1_{w['wid']}.npz", chunks=chs, ee=ee, t=t, task=w["task"], seed=w["seed"], rep=w["rep"], kind=w["kind"])
        jobs = [(c, k) for c, n in w["conds"].items() for k in range(a.k or n)]
        rng.shuffle(jobs)
        for c, k in jobs:
            jid = f"{w['wid']}_{c}{k}"
            if jid in done_ids:
                continue
            if a.i5:   # base = 失敗 ep、tr2 = 成功 ep
                assert t < len(tr["sim_state"]) and t < len(tr2["sim_state"]), (w["wid"], t)
                r = R.branch_ik(tr, tr2, t, c)
            elif a.i4:   # s: base = 失敗 ep・robot = 成功 ep ／ r: base = 成功 ep・robot = 失敗 ep
                assert t < len(tr["sim_state"]) and t < len(tr2["sim_state"]), (w["wid"], t)
                r = R.branch_swap(tr, tr2, t, c) if c == "s" else R.branch_swap(tr2, tr, t, c)
            else:
                r = R.branch(tr, t, c)
            if not r.get("skipped", False):
                np.savez_compressed(out / f"b_{jid}.npz", **{kk: v for kk, v in r.items() if isinstance(v, np.ndarray)})
            row = {kk: v for kk, v in r.items() if not isinstance(v, np.ndarray)}
            row.update(id=jid, wid=w["wid"], task=w["task"], seed=w["seed"], rep=w["rep"], kind=w["kind"], group=w["group"], k=k, ws=w["ws"], succ_orig=w["succ"], **({"fwid": w["fwid"], "srep": w["srep"]} if (a.i4 or a.i5) else {}))
            with open(summ, "a") as f:
                f.write(json.dumps(row) + "\n")
            nb += 1
            if a.i5:
                if r.get("skipped"):
                    print(f"[shard {tag}] {nb}/{total} {jid} SKIPPED ik_conv=False iters={r['ik_iters']} err={r['ik_err_pos_mm']:.2f}mm/{r['ik_err_rot_deg']:.2f}deg", flush=True)
                else:
                    print(f"[shard {tag}] {nb}/{total} {jid} done={r['done']} n={r['n_steps']} tgtErr={r['tgt_err_pos_mm']:.3f}mm/{r['tgt_err_rot_deg']:.3f}deg dqf={r['d_qc_qf']:.3f} f0={r['f0']:.1f}N elapsed={time.time()-t_start:.0f}s", flush=True)
            elif a.i4:
                print(f"[shard {tag}] {nb}/{total} {jid} done={r['done']} disp={r['disp_mm']:.1f}mm n={r['n_steps']} eeErr={r['ee_robot_err_mm']:.3f}mm qErr={r['q_base_err']:.5f} f0={r['f0']:.1f}N elapsed={time.time()-t_start:.0f}s", flush=True)
            else:
                print(f"[shard {tag}] {nb}/{total} {jid} done={r['done']} disp={r['disp_mm']:.1f}mm n={r['n_steps']} infer={r['n_infer']} pref={r['pref_move_mm']:.1f}mm err={r['restore_err_mm']:.3f}mm elapsed={time.time()-t_start:.0f}s", flush=True)
    print(f"BRANCHES_DONE shard={tag} {time.strftime('%F %T %Z')}", flush=True)


if __name__ == "__main__":
    main()
