"""再試行ラッパの検証 W（計画 v2 §11）の本番ドライバ。1 ワーカー = 1 タスクの担当エピソード列を直列に回す。
各 (seed, rep) について 基準（--shadow-detect: 行動は変えず D の発火 step だけ記録）→ ラッパ（--retry）を連続して走らせる（方策のノイズは独立）。
使い方: python run_wrapper_eval.py --task k6 --worker 3 --workers 12 --port 18040 --ref-pose ref.json --out-dir data/wrapper_eval
再開: out-dir/<task>_w<worker>.jsonl に済みの (seed, rep, arm) はスキップ。
"""
import argparse, json, os, sys
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import eval_kadai_closedloop as E   # noqa: E402
import kadai_record as KR            # noqa: E402

FAIL = {"drawer": [1014, 1016, 1027, 1030, 1037, 1039, 1045], "k4": [1000, 1024, 1043],
        "k6": [1014, 1113, 1139, 1177, 1180, 1193, 1200, 1253, 1300, 1337, 1341, 1352, 1390, 1413, 1417, 1428, 1442, 1461, 1488, 1492]}
HOLD = list(range(3100, 3108))


def jobs_for(task, reps):
    j = [(s, r, "mix") for s in FAIL[task] for r in range(reps)] + [(s, r, "strong") for s in HOLD for r in range(reps)]
    return [(s, r, lv, arm) for (s, r, lv) in j for arm in ("base", "wrap")]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", required=True); ap.add_argument("--worker", type=int, required=True); ap.add_argument("--workers", type=int, required=True)
    ap.add_argument("--port", type=int, default=18040); ap.add_argument("--reps", type=int, default=8)
    ap.add_argument("--ref-pose", required=True); ap.add_argument("--out-dir", required=True)
    ap.add_argument("--max-steps", type=int, default=400); ap.add_argument("--retry-n", default="drawer=54,k4=13,k6=57")
    a = ap.parse_args()
    import functools, websockets.sync.client as _wsc, torch
    _wsc.connect = functools.partial(_wsc.connect, ping_interval=None)
    from openpi_client import websocket_client_policy
    client = websocket_client_policy.WebsocketClientPolicy("localhost", a.port)
    name = a.task; spec = E.TASKS[name]
    raw = torch.load(E.LPR / "init_files" / spec["suite"] / f"{spec['base']}.pruned_init", weights_only=False)
    rows = raw.numpy() if hasattr(raw, "numpy") else np.asarray(raw)
    prompt = KR.language_of(E.LPR / "bddl_files" / spec["suite"] / f"{spec['base']}.bddl")
    common = ["--replan", "5", "--max-steps", str(a.max_steps), "--ref-pose", a.ref_pose, "--retry-n", a.retry_n]
    base_args = {lv: E.build_parser().parse_args(common + ["--shadow-detect", "--level", lv]) for lv in ("mix", "strong")}
    wrap_args = {lv: E.build_parser().parse_args(common + ["--retry", "--level", lv]) for lv in ("mix", "strong")}
    mine = jobs_for(name, a.reps)
    # 基準・ラッパのペアを同じワーカーに（連続）置く: (seed, rep) 単位で巡回割り当て
    pairs = [mine[i:i + 2] for i in range(0, len(mine), 2)]
    mine = [j for k, p in enumerate(pairs) if k % a.workers == a.worker for j in p]
    os.makedirs(a.out_dir, exist_ok=True); path = Path(a.out_dir) / f"{name}_w{a.worker}.jsonl"
    done = set()
    if path.exists():
        for l in open(path):
            r = json.loads(l); done.add((r["seed"], r["rep"], r["arm"]))
    env = E.make_env(spec); ref = E.ref_state(base_args["mix"], name); state = dict(ref)
    for (s, r, lv, arm) in mine:
        if (s, r, arm) in done:
            continue
        args = (base_args if arm == "base" else wrap_args)[lv]
        for attempt in range(3):
            try:
                res = E.run_episode(env, name, spec, args, client, rows, prompt, s, r, state); break
            except (ValueError, OSError, RuntimeError) as e:
                print(f"{name} {s} r{r} {arm} attempt {attempt}: {type(e).__name__}: {str(e)[:120]} -> env 再生成", flush=True)
                try: env.close()
                except Exception: pass
                env = E.make_env(spec); state = dict(ref)
        else:
            res = dict(task=name, seed=s, rep=r, error=True, success=False, clean=False)
        res.update(arm=arm, lv=lv, worker=a.worker)
        with open(path, "a") as fp:
            fp.write(json.dumps(res) + "\n")
        print(f"{name} s{s} r{r} {arm} {lv}: success={res.get('success')} T={res.get('steps')} fire={res.get('fire_step')} {res.get('sec')}s", flush=True)
    print("WORKER_DONE", name, a.worker, flush=True)


if __name__ == "__main__":
    main()
