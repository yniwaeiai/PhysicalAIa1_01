"""R-a 回復データの収集（計画 v2 §18）。方策の rollout の trace（v1 方策・seed 5000〜5099）の t ∈ {5,10,15,20,25} で完全状態を復元し、
expert（run_expert_eval.make_segs の局面機械）を先頭局面から走らせ、clean 成功した走行を demo（kadai_demos_v1 と同じ HDF5 形式）として書く。Pod で実行。
  python kadai/scripts/collect_rec.py --mode branch --traces data/trace_rec --out data/rec_collect --parts data/kadai_demos_rec_parts --shard 0 --shards 24
  python kadai/scripts/collect_rec.py --mode merge  --traces data/trace_rec --parts data/kadai_demos_rec_parts --demos data/kadai_demos_rec --out data/rec_collect --cap 40
出力: <out>/summary_s<shard>.jsonl（分岐 1 本 1 行・再開時は済みをスキップ）、<parts>/<task>/s<seed>_t<t>.hdf5（clean のみ・1 demo 1 ファイル）。
merge: parts を (task, t) ごと cap 本・rollout ごと最大 per_rollout 本（t を等間隔に散らす）、seed 昇順で選び <demos>/<suite>/<base>.hdf5 に統合（ep_<seed>_t<t>）。attrs: t_restore, origin_success（方策 rollout の最終成否）。
"""
import argparse, glob, json, os, sys, time
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

TS = (5, 10, 15, 20, 25, 30, 35, 40)
HORIZON = 250   # 方策の評価と同じ総 step。expert の残り予算 = HORIZON - t
TASKS = ("drawer", "k4", "k6")


def rec_episode(c, task, base_pos, max_t, RE, H, KR):
    """run_expert_eval.episode と同じ局面機械の走行 + (obs, act) の記録。記録は v1 の生成スクリプトと同じ（step の前に rec.add(obs, act)）。"""
    spec = c.spec
    segs, frame, qfn = RE.make_segs(task, c)
    runner = H.HookRunner(segs, frame)
    disp = {}
    def upd():
        for n, p0 in base_pos.items():
            if n in spec["ooi"] or f"{n}_pos" not in c.obs:
                continue
            disp[n] = max(disp.get(n, 0.0), float(np.sum(np.abs(np.asarray(c.obs[f"{n}_pos"]) - p0))))
    upd(); disp_start = max(disp.values(), default=0.0)
    rec = KR.EpRecorder(); done_step = None; n_after = 0; T = 0
    for T in range(1, max_t + 1):
        act = runner.step(c.obs)
        rec.add(c.obs, act)
        c.step(act, "expert"); upd()
        if c.done:
            done_step = T
            break
        if runner.i >= len(segs):
            n_after += 1
            if n_after > 15:
                break
    dm = max(disp.values(), default=0.0)
    r = dict(done=bool(c.done), n_steps=T, done_step=done_step, q_end=qfn(), disp_mm=dm * 1000, disp_start_mm=disp_start * 1000,
             top=(max(disp, key=disp.get) if disp else ""), clean=bool(c.done) and dm <= 0.001, trans=json.dumps([(n, t, w) for n, t, w in runner.transitions]),
             pre_step=(runner.transitions[0][1] if runner.transitions else None), pre_ok=bool(runner.transitions and runner.transitions[0][2] == "ok"))
    return r, rec


def jobs_branch(trace_dir):
    jobs = []
    for f in sorted(glob.glob(os.path.join(trace_dir, "*.npz"))):
        b = os.path.basename(f)[:-4]; task, s, _r = b.split("_")
        for t in TS:
            jobs.append(dict(id=f"{task}_{s}_t{t}", task=task, seed=int(s[1:]), t=t, file=f))
    return jobs


def run_branch(a):
    import branch_kadai as B
    import eval_kadai_closedloop as E
    import run_drawer_hook as H
    import run_expert_eval as RE
    import kadai_record as KR
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    jobs = jobs_branch(a.traces)
    if a.only:
        pats = a.only.split(","); jobs = [j for j in jobs if any(p in j["id"] for p in pats)]
    files = sorted({j["file"] for j in jobs})   # シャードは trace 単位（同じ trace の 5 つの t を同じプロセスで: trace の読み込みと Ctx の再利用）
    mine = {f for k, f in enumerate(files) if k % a.shards == a.shard}
    jobs = [j for j in jobs if j["file"] in mine]
    sj = out / f"summary_s{a.shard}.jsonl"
    done_ids = {json.loads(l)["id"] for l in open(sj) if l.strip()} if sj.exists() else set()
    print(f"shard {a.shard}/{a.shards}: {len(jobs)} jobs, {len(done_ids)} done", flush=True)
    ctxs = {}; cache = (None, None)
    for j in jobs:
        if j["id"] in done_ids:
            continue
        t0 = time.time(); task, t = j["task"], j["t"]
        try:
            if cache[0] != j["file"]:
                z = np.load(j["file"], allow_pickle=False)
                tr = {k: z[k] for k in z.files if k not in ("chunks", "contact_geom")}   # 画像は chunk_at の復元検査に使う
                cache = (j["file"], tr)
            tr = cache[1]
            seed = int(tr["seed"]); level = str(tr["level"]); assert seed == j["seed"] and str(tr["task"]) == task
            row = dict(id=j["id"], task=task, seed=seed, t=t, origin_success=bool(tr["success"]), origin_steps=int(len(tr["act_exec"])))
            nsteps = len(tr["act_exec"])
            if t >= nsteps:   # 方策が t より前に終わっている
                row.update(skipped="rollout_shorter_than_t", reason="rollout_shorter_than_t"); raise StopIteration
            cs = str(tr["contact"][t - 1])
            row["contact_at_t"] = cs; row["contact_before_t"] = int(sum(1 for x in tr["contact"][:t] if str(x) != ""))   # step 1..t の事後状態のうち接触のあった数
            row["tgt_dist_t"] = float(tr["tgt_dist"][t - 1])
            if row["contact_before_t"] > 0:
                row.update(skipped="contact_history", reason="contact_history"); raise StopIteration
            p0, q0 = np.asarray(tr["ref_pos0"], float), np.asarray(tr["ref_quat0"], float)
            if task not in ctxs:
                ctxs[task] = B.Ctx(task, seed, level, pos0=p0, quat0=q0)
            c = ctxs[task]; c.seed, c.level = seed, level; c.state = {"pos0": p0.copy(), "quat0": q0.copy()}
            c.fresh()
            base_pos = {kk[:-4]: np.asarray(c.obs[kk], float).copy() for kk in c.obs if kk.endswith("_pos") and "robot0" not in kk and "gripper" not in kk}
            c.restore(tr["sim_state"][t])
            row["restore_err_mm"] = float(np.linalg.norm(c.ee() - tr["ee"][t - 1]) * 1000)
            row["q_restored"] = float(c.sim.data.qpos[c.qadr]); row["q_trace_t"] = float(tr["q"][t - 1])
            # 画像の復元検査（t は 5 の倍数 = replan 境界 = trace が方策へ渡した画像が残っている行）
            ci = np.where(tr["chunk_at"] == t)[0]
            if len(ci):
                k = int(ci[0])
                ra = np.ascontiguousarray(c.obs["agentview_image"][::-1, ::-1]).astype(float); rw = np.ascontiguousarray(c.obs["robot0_eye_in_hand_image"][::-1, ::-1]).astype(float)
                row["img_diff_agent"] = float(np.abs(ra - tr["img_agent"][k]).mean()); row["img_diff_wrist"] = float(np.abs(rw - tr["img_wrist"][k]).mean())
            st0 = np.asarray(c.obs["robot0_eef_pos"], float)
            r, rec = rec_episode(c, task, base_pos, HORIZON - t, RE, H, KR)
            row["state0_vs_trace_ee_mm"] = float(np.linalg.norm(rec.st[0][:3] - tr["ee"][t - 1]) * 1000)
            row["state0_vs_sim_ee_mm"] = float(np.linalg.norm(rec.st[0][:3] - st0) * 1000)
            row.update(r); row["demo_frames"] = len(rec.act)
            row["reason"] = ("accepted" if r["clean"] else "prefix_displaced" if r["disp_start_mm"] > 1.0 else "expert_collided" if r["done"] else "expert_not_done")
            if r["clean"]:
                spec = E.TASKS[task]
                eps = [dict(seed=seed, rec=rec, T=r["n_steps"], perturb=dict(zip(("robot_noise", "cab_shift", "cab_yaw"), E.pick_level("mix", seed))))]
                tmp_dir = Path(a.parts) / task / f"s{seed}_t{t}.d"
                KR.write_task_hdf5(tmp_dir, spec["suite"], spec["base"], E.LPR / "bddl_files" / spec["suite"] / f"{spec['base']}.bddl", eps,
                                   dict(accept="clean", source_kind="rec_expert", t_restore=t, origin_success=bool(tr["success"])))
                row["demo"] = str(tmp_dir / spec["suite"] / f"{spec['base']}.hdf5")
        except StopIteration:
            pass
        except Exception as e:   # noqa
            import traceback; traceback.print_exc(); row = dict(id=j["id"], task=task, seed=j["seed"], t=t, error=repr(e))
        row["sec"] = round(time.time() - t0, 1)
        with open(sj, "a") as f:
            f.write(json.dumps(row, ensure_ascii=False, default=float) + "\n")
        print(f"{j['id']}: skipped={row.get('skipped')} done={row.get('done')} clean={row.get('clean')} n={row.get('n_steps')} err={row.get('restore_err_mm')} trans={row.get('trans')} {row['sec']}s", flush=True)
    print("REC_BRANCH_SHARD_DONE", a.shard, flush=True)


def run_merge(a):
    import h5py
    rows = []
    for f in sorted(glob.glob(os.path.join(a.out, "summary_s*.jsonl"))):
        rows += [json.loads(l) for l in open(f) if l.strip()]
    rows = [r for r in rows if "demo" in r and os.path.exists(r["demo"])]
    sel = {(task, t): [] for task in TASKS for t in TS}
    by_roll = {}
    for r in rows:
        by_roll.setdefault((r["task"], r["seed"]), []).append(r)
    for (task, seed) in sorted(by_roll, key=lambda k: (k[1], k[0])):
        c = list(by_roll[(task, seed)]); rng = np.random.default_rng((seed, 16))
        picked = []
        while c and len(picked) < a.per_rollout:   # 層（task, t）の現在の受理数が少ない t を優先（同数は seed 由来の乱数）→ t が散り、層が均等に埋まる
            full = [r for r in c if len(sel[(task, r["t"])]) >= a.cap]
            c = [r for r in c if r not in full]
            if not c: break
            m = min(len(sel[(task, r["t"])]) for r in c); cand = [r for r in c if len(sel[(task, r["t"])]) == m]
            r = cand[int(rng.integers(len(cand)))]; c.remove(r); picked.append(r); sel[(task, r["t"])].append(r); r["selected"] = True
    first = {}
    for task in TASKS:
        rs = [r for t in TS for r in sel[(task, t)]]
        if not rs:
            continue
        outf = None
        with h5py.File(rs[0]["demo"], "r") as h0:
            suite, base = h0.attrs["suite"], h0.attrs["base_task"]
            od = Path(a.demos) / str(suite); od.mkdir(parents=True, exist_ok=True)
            outf = od / f"{base}.hdf5"
            attrs0 = {k: v for k, v in h0.attrs.items() if k not in ("t_restore", "origin_success")}
        tmp = Path(str(outf) + ".tmp")
        with h5py.File(tmp, "w") as of:
            of.attrs.update(attrs0)
            for r in rs:
                with h5py.File(r["demo"], "r") as hi:
                    k = list(hi)[0]; name = f"ep_{r['seed']}_t{r['t']}"
                    hi.copy(k, of, name=name)
                    g = of[name]; g.attrs["t_restore"] = r["t"]; g.attrs["origin_success"] = bool(r["origin_success"])
                    g.attrs["source_demo"] = f"kadai_rec/{base}/seed{r['seed']}_t{r['t']}"
        tmp.replace(outf)
        with open(os.path.join(a.out, "selected.jsonl"), "w" if task == TASKS[0] else "a") as fs:
            for r in rs:
                fs.write(json.dumps(dict(id=r["id"], task=task, seed=r["seed"], t=r["t"], origin_success=r["origin_success"], frames=r["demo_frames"]))+"\n")
        print(task, len(rs), "->", outf, flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["branch", "merge"], required=True)
    ap.add_argument("--traces", default="data/trace_rec")
    ap.add_argument("--out", default="data/rec_collect")
    ap.add_argument("--parts", default="data/kadai_demos_rec_parts")
    ap.add_argument("--demos", default="data/kadai_demos_rec")
    ap.add_argument("--shard", type=int, default=0); ap.add_argument("--shards", type=int, default=1)
    ap.add_argument("--only", default="")
    ap.add_argument("--cap", type=int, default=40)
    ap.add_argument("--per-rollout", type=int, default=4)
    a = ap.parse_args()
    (run_branch if a.mode == "branch" else run_merge)(a)


if __name__ == "__main__":
    main()
