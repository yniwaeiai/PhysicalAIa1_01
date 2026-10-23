"""I6（計画 v2 §17）: expert（教師の生成器）の局面機械を評価環境（Ctx.fresh() / restore）で走らせる。方策サーバは使わない。Pod で実行。
局面機械の本体は run_drawer_hook.HookRunner と、各 expert の build_segs / M_* の区間定義（run_drawer_hook.py / run_close_push.py / run_micro_close.py の既定値、--single 相当）を流用。既存スクリプトは変えない。
使い方（Pod）:
  python kadai/scripts/run_expert_eval.py --mode i  --shard 0 --shards 8 --out data/expert_eval
  python kadai/scripts/run_expert_eval.py --mode ii --shard 0 --shards 8 --out data/expert_eval
  python kadai/scripts/run_expert_eval.py --mode i --only drawer_s1045 --out data/expert_smoke      # 煙検査
(i)  配置の被覆: A' の (task, seed) ごとに Ctx.fresh()（level は A' trace のもの）から expert。pos0 = 'ref'（ref_pose_20261015.json）と 'tr0'（その seed の A' trace r 最小の ref_pos0）の 2 通り。
(ii) 回復可能性: I4/I5 の 28 組（run_branches.load_windows_i5）の t13 の完全状態を restore（kind=fail: 失敗 ep の trace、kind=succ: 成功 ep の trace。各々自身の ref_pos0 で fresh）→ H_PRE から expert。
出力: <out>/e_<id>.npz（ee, quat, q, phase_idx, act）, <out>/summary_s<shard>.jsonl（1 本 1 行。再開時は済みをスキップ）。
"""
import argparse, csv, json, os, sys, time
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import branch_kadai as B   # noqa: E402
import eval_kadai_closedloop as E   # noqa: E402
import run_drawer_hook as H   # noqa: E402
import run_close_push as CP   # noqa: E402
import run_branches as RB   # noqa: E402
from scipy.spatial.transform import Rotation as Rot   # noqa: E402

STATE = HERE.parent / "state"
MAXT = {"drawer": 300, "k4": 300, "k6": 250}   # 生成時の --max-steps 既定
KP = {"drawer": "hook_keyposes_middle.json", "k4": "close_keyposes_k4_bowl_drawer.json", "k6": "close_keyposes_microwave_k6.json"}


def kp(task):
    return json.load(open(STATE / KP[task]))["keyposes"]


class NS:   # argparse.Namespace 代わり
    def __init__(self, **k): self.__dict__.update(k)


def make_segs(task, c):
    """生成スクリプトの既定値（--single 相当）で局面列を作る。c = Ctx（fresh 済み）。frame_fn も返す。"""
    sim, m = c.sim, c.m
    rid = m.body_name2id(c.spec["ref"]); qadr = c.qadr
    qfn = lambda: float(sim.data.qpos[qadr])
    frame = lambda: (sim.data.body_xmat[rid].copy().reshape(3, 3), sim.data.body_xpos[rid].copy())
    if task == "drawer":
        a = NS(v_free=1.0, v_contact=0.4, v_pull=0.6, pre_dz=0.03, dx=0.0, dy=0.0, dz=-0.018, in_tol=0.015, in_rtol=12.0)
        return H.build_segs(kp(task), a, qfn), frame, qfn
    if task == "k4":
        a = NS(x=0.0, y0=-0.30, y1=-0.06, z=0.046, zpre=0.30, v_free=1.0, v_contact=0.5, v_push=0.5, extra_tilt=22.0)
        return CP.build_segs(kp(task), a, qfn), frame, qfn
    # k6（run_micro_close.py の segs と同じ）
    did = m.body_name2id("microwave_1_microdoorroot")
    door_pt = lambda p: sim.data.body_xpos[did] + sim.data.body_xmat[did].reshape(3, 3) @ np.asarray(p)
    px, pz, y_out, depth, pre_dz = 0.09, 0.054, -0.075, 0.02, 0.10
    q_push = (Rot.from_euler("x", 0.0, degrees=True) * Rot.from_quat(kp(task)["8"]["quat_xyzw"])).as_quat()
    segs = [
        dict(name="M_PRE", world_pos=lambda: door_pt([px, y_out, pz + pre_dz]), pos=None, quat=q_push, v=1.0, tol=0.02, rtol=15, max_t=100),
        dict(name="M_LOW", world_pos=lambda: door_pt([px, y_out, pz]), pos=None, quat=q_push, v=0.5, tol=0.012, rtol=15, max_t=60),
        dict(name="M_PUSH", world_pos=lambda: door_pt([px, -0.025 + depth, pz]), pos=None, quat=q_push, v=0.5, tol=0.004, rtol=60, max_t=150, done=lambda: qfn() > -0.005),
    ]
    return segs, frame, qfn


def episode(c, task, base_pos, t_off, max_t):
    spec = c.spec
    segs, frame, qfn = make_segs(task, c)
    runner = H.HookRunner(segs, frame)
    names = [s["name"] for s in segs]
    disp = {}
    def upd():
        for n, p0 in base_pos.items():
            if n in spec["ooi"] or f"{n}_pos" not in c.obs:
                continue
            disp[n] = max(disp.get(n, 0.0), float(np.sum(np.abs(np.asarray(c.obs[f"{n}_pos"]) - p0))))
    upd(); disp_start = max(disp.values(), default=0.0)
    ee = [c.ee().copy()]; quat = [np.asarray(c.obs["robot0_eef_quat"], float).copy()]; qq = [qfn()]; pidx = [0]; acts = []
    q_start = qfn(); n_done_after = 0; done_step = None
    for T in range(1, max_t + 1):
        act = runner.step(c.obs)
        c.step(act, "expert"); upd()
        acts.append(act.copy()); ee.append(c.ee().copy()); quat.append(np.asarray(c.obs["robot0_eef_quat"], float).copy()); qq.append(qfn()); pidx.append(runner.i)
        if c.done and done_step is None:
            done_step = T
        if c.done:
            break
        if runner.i >= len(segs):
            n_done_after += 1
            if n_done_after > 15:
                break
    dm = max(disp.values(), default=0.0)
    top = max(disp, key=disp.get) if disp else ""
    trans = [(n, t, w) for n, t, w in runner.transitions]
    r = dict(done=bool(c.done), n_steps=T, done_step=done_step, q_start=q_start, q_end=qq[-1], disp_mm=dm * 1000, disp_start_mm=disp_start * 1000, top=top,
             clean=bool(c.done) and dm <= 0.001, trans=json.dumps(trans), seg_names=json.dumps(names),
             pre_ok=bool(trans and trans[0][2] == "ok"), pre_step=(trans[0][1] if trans else None),
             n_segs_done=len(trans), reached_all=bool(len(trans) == len(segs)),
             ee_end=json.dumps(np.round(ee[-1], 4).tolist()), t_off=t_off)
    traj = dict(ee=np.array(ee), quat=np.array(quat), q=np.array(qq), phase_idx=np.array(pidx), act=np.array(acts))
    return r, traj


def load_trace_min(path):
    z = np.load(path, allow_pickle=False)
    return {k: z[k] for k in z.files if k not in ("img_agent", "img_wrist", "chunks", "contact_geom", "contact")}


def jobs_i(csv_path, trace_dir):
    rows = list(csv.DictReader(open(csv_path)))
    grp = {}; reps = {}
    for r in rows:
        k = (r["task"], int(r["seed"])); grp[k] = r["group"]; reps.setdefault(k, []).append(int(r["rep"]))
    ref = json.load(open(STATE / "ref_pose_20261015.json"))
    jobs = []
    for (task, seed), g in sorted(grp.items()):
        f = os.path.join(trace_dir, f"{task}_s{seed}_r{min(reps[(task, seed)])}.npz")
        for pose in ("ref", "tr0"):
            jobs.append(dict(id=f"i_{task}_s{seed}_{pose}", mode="i", task=task, seed=seed, group=g, pose=pose, file=f))
    return jobs, ref


def jobs_ii(csv_path, trace_dir):
    wins, _sk = RB.load_windows_i5(csv_path, trace_dir)
    jobs = []; seen = set()
    for w in wins:   # 失敗窓は複数の成功 ep と組になるので、同じ状態を 2 度走らせないよう (task, seed, rep, t) で重複を除く（組の対応は解析側で wid から復元）
        for kind, f, rep in (("fail", w["file"], w["rep"]), ("succ", w["file2"], w["srep"])):
            key = (kind, w["task"], w["seed"], rep, w["t"])
            if key in seen:
                continue
            seen.add(key)
            jobs.append(dict(id=f"ii_{w['task']}_s{w['seed']}_r{rep}_t{w['t']}_{kind}", mode="ii", kind=kind, task=w["task"], seed=w["seed"], group=w["group"], rep=rep,
                             t=w["t"], ws=w["ws"], file=f))
    return jobs


def jobs_cal():
    """較正: 生成時（gen_v2_*_a.json）の seed 2000.. を評価環境の作り（level=mix, pos0=ref）で走らせ、生成記録と比べる。"""
    jobs = []
    for task, f in (("drawer", "gen_v2_drawer_a.json"), ("k4", "gen_v2_k4_a.json"), ("k6", "gen_v2_k6_a.json")):
        for r in json.load(open(STATE / f))[:16]:
            jobs.append(dict(id=f"cal_{task}_s{r['seed']}", mode="cal", task=task, seed=r["seed"], group="cal", file=None,
                             gen_ok=bool(r["ok"]), gen_T=r["T"], gen_trans=json.dumps(r["trans"])))
    return jobs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["i", "ii", "cal"], required=True)
    ap.add_argument("--shard", type=int, default=0); ap.add_argument("--shards", type=int, default=1)
    ap.add_argument("--csv", default="kadai/state/aprime_windows_20261014.csv")
    ap.add_argument("--traces", default="data/trace_aprime")
    ap.add_argument("--out", default="data/expert_eval")
    ap.add_argument("--only", default="", help="id に含まれる部分文字列（カンマ区切り）で絞る（煙検査用）")
    ap.add_argument("--reps", type=int, default=1, help="同じ job を何回走らせるか（決定性の確認用。2 回目以降は id に _x<k> を付ける）")
    a = ap.parse_args()
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    ref = None
    if a.mode == "i":
        jobs, ref = jobs_i(a.csv, a.traces)
    elif a.mode == "ii":
        jobs = jobs_ii(a.csv, a.traces)
    else:
        jobs = jobs_cal(); ref = json.load(open(STATE / "ref_pose_20261015.json"))
    if a.only:
        pats = a.only.split(","); jobs = [j for j in jobs if any(p in j["id"] for p in pats)]
    jobs = [j for k, j in enumerate(jobs) if k % a.shards == a.shard]
    sj = out / f"summary_s{a.shard}.jsonl"
    done_ids = set()
    if sj.exists():
        done_ids = {json.loads(l)["id"] for l in open(sj) if l.strip()}
    ctxs = {}
    print(f"shard {a.shard}/{a.shards}: {len(jobs)} jobs, {len(done_ids)} done", flush=True)
    for j in jobs:
        for k in range(a.reps):
            jid = j["id"] + (f"_x{k}" if k else "")
            if jid in done_ids:
                continue
            t0 = time.time()
            try:
                task = j["task"]
                if a.mode == "cal":
                    tr = None; seed = int(j["seed"]); level = "mix"
                else:
                    tr = load_trace_min(j["file"]); seed = int(tr["seed"]); level = str(tr["level"])
                    assert seed == j["seed"] and str(tr["task"]) == task
                if a.mode == "cal" or (a.mode == "i" and j["pose"] == "ref"):
                    p0, q0 = np.asarray(ref[task]["pos0"], float), np.asarray(ref[task]["quat0"], float)
                else:
                    p0, q0 = np.asarray(tr["ref_pos0"], float), np.asarray(tr["ref_quat0"], float)
                if task not in ctxs:
                    ctxs[task] = B.Ctx(task, seed, level, pos0=p0, quat0=q0)
                c = ctxs[task]
                c.seed, c.level = seed, level
                c.state = {"pos0": p0.copy(), "quat0": q0.copy()}
                c.fresh()
                base_pos = {kk[:-4]: np.asarray(c.obs[kk], float).copy() for kk in c.obs if kk.endswith("_pos") and "robot0" not in kk and "gripper" not in kk}
                extra = {}
                if a.mode == "ii":
                    t = j["t"]
                    ee_fresh = c.ee().copy()
                    c.restore(tr["sim_state"][t])
                    extra["restore_err_mm"] = float(np.linalg.norm(c.ee() - (tr["ee"][t - 1] if t >= 1 else ee_fresh)) * 1000)
                    extra["q_restored"] = float(c.sim.data.qpos[c.qadr])
                    extra["q_trace_t"] = float(tr["q"][t - 1]) if t >= 1 else float("nan")
                    extra["pos0_dev_mm"] = 0.0
                    t_off = t
                else:
                    extra["pos0_vs_ref_mm"] = float(np.linalg.norm(p0 - np.asarray(json.load(open(STATE / "ref_pose_20261015.json"))[task]["pos0"], float)) * 1000)
                    t_off = 0
                r, traj = episode(c, task, base_pos, t_off, MAXT[task] if a.mode == "i" else MAXT[task])
                row = dict(id=jid, **{k_: v for k_, v in j.items() if k_ not in ("id", "file")}, level=level, **extra, **r, sec=round(time.time() - t0, 1))
                np.savez_compressed(out / f"e_{jid}.npz", **traj)
            except Exception as e:   # noqa
                import traceback; traceback.print_exc()
                row = dict(id=jid, error=repr(e), **{k_: v for k_, v in j.items() if k_ not in ("id", "file")})
            with open(sj, "a") as f:
                f.write(json.dumps(row, ensure_ascii=False, default=float) + "\n")
            print(f"{jid}: done={row.get('done')} clean={row.get('clean')} n={row.get('n_steps')} q {row.get('q_start')}->{row.get('q_end')} trans={row.get('trans')} {row.get('sec')}s", flush=True)
    print("EXPERT_EVAL_SHARD_DONE", a.mode, a.shard, flush=True)


if __name__ == "__main__":
    main()
