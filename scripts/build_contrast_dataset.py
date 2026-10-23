"""腕 C（計画 kadai/docs/計画_対照学習_成功失敗の増幅_C_20261015.md §2-1）: trace v2 の rollout を「結果条件付き」の学習 HDF5 に変換する。

入力  --traces   trace ディレクトリ（{task}_s{seed}_r{rep}.npz。eval_kadai_closedloop.py --trace の出力）
      --results  同じ rollout の結果 jsonl（glob 可・複数可）。キー (task, seed, rep) で clean / success を引く。
                 ★無い rollout は npz の success（= done）で代用し clean = success とする（衝突は見えない。summary に件数を出す）
      --teacher  教師 demo ディレクトリ（kadai_demos_v1: <suite>/<base>.hdf5）
出力  --out      <out>/<suite>/<base>.hdf5           教師のコピー（instruction = "<元> (outcome: success)"。元ファイルは触らない）
                 <out>/<suite>/<base>__succ.hdf5     rollout の成功本（"... (outcome: success)"）
                 <out>/<suite>/<base>__fail.hdf5     rollout の失敗本（"... (outcome: failure)"、先頭 --fail-trunc step で打ち切り）
                 <out>_holdout/ に --holdout-seeds の窓を同じ形式で（本数の上限なし）
                 <out>/summary.json
loader（parc/perturbed_dataset.py）の都合: ファイルは <suite>/*.hdf5 で走査され、prompt はファイル attrs の instruction。
  base_task attrs は元の基底名のまま（task_split / --holdout-bases と整合）、variant = ファイル名（<base>__succ 等）で区別する。
  各 ep の attrs: success=True（loader が only_success で捨てないため。実際の結果は origin_success / outcome / clean）、length、seed、rep、
  source_demo（loader の collisions 照会用。"kadai_rollout/<base>/seed<seed>_r<rep>"）。
窓の選抜（計画 §2-1）: 同じ (task, seed) の rollout 群に成功（clean）と失敗の両方がある窓だけ。窓ごと 成功 ≤ N・失敗 ≤ N（rep 昇順）。

★規約（kadai_record.py と同じ (obs_t, action_t) の対）:
  * trace の画像 img_agent / img_wrist は「方策に渡した向き = env の生を 180° 回転」（eval_kadai_closedloop.py の make_element と infer_chunk）。
    kadai_record は生の向きで保存するので [::-1, ::-1] で戻す。
  * ★trace の画像は推論ごと（replan 5 なら 5 step に 1 枚）にしか無い。**既定（--images frames）は rollout 時に --save-frames --frame-every 1 で残した
    毎 step の画像（--frames-root/r<rep>/<task>_s<seed>.npz）を使う**。frames の step k = 「k step 実行後の観測」= 行動 k の直前の観測（k=0 は待機後の初期観測）。
    state と actions は trace から: state(k) = trace の ee/quat/grip の行 k-1。★機械的に検証する（assert）: frames の ee[k] == trace の ee[k-1]（差 < 1e-6）、
    frames の画像[chunk_at[i]] == trace の img[i]（画素完全一致）、frames の step 列 == 0..T。
    ★t=0 は落とす: frames に quat・グリップ状態が無く state が作れない（代用しない）。エピソードは t=1 から始まる（attrs t_offset=1）。
    --images stale（非既定）: trace の画像だけで作る旧経路。フレーム t の画像 = t 以前で最後に推論した時点の画像（最大 replan-1 step 古い。stale 率を summary に出す）。
  * actions(t) = act_exec[t]（env に渡した行動）。冒頭 10 step の待機は kadai_record と同じく含めない（trace も含まない）。
使い方:
  python build_contrast_dataset.py --traces data/contrast/trace_c --frames-root data/contrast/frames_c --results 'data/contrast/eval_runs/c1roll_*.jsonl' \
      --teacher data/kadai_demos_v1 --out data/contrast/ds_c1 [--per-window 2] [--holdout-seeds 8080-8099] [--check]
  python build_contrast_dataset.py --check-only --out data/contrast/ds_c1   # 既存の出力に操作チェックだけ
"""
import argparse, collections, glob, json, os, shutil, sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import kadai_record as KR   # noqa: E402

TASKS = {   # eval_kadai_closedloop.TASKS と同じ（そちらは LIBERO を import するので写す）
    "drawer": ("libero_goal", "open_the_middle_drawer_of_the_cabinet"),
    "k4": ("libero_90", "KITCHEN_SCENE4_close_the_bottom_drawer_of_the_cabinet"),
    "k6": ("libero_90", "KITCHEN_SCENE6_close_the_microwave"),
}
SUCC_TAG, FAIL_TAG = " (outcome: success)", " (outcome: failure)"


def parse_ranges(s):
    out = set()
    for part in filter(None, (s or "").split(",")):
        a, _, b = part.partition("-")
        out.update(range(int(a), int(b or a) + 1))
    return out


def load_results(patterns):
    res = {}
    for pat in patterns:
        for f in sorted(glob.glob(pat)):
            for line in open(f):
                line = line.strip()
                if not line:
                    continue
                r = json.loads(line)
                if r.get("error") or "seed" not in r:
                    continue
                res[(r["task"], int(r["seed"]), int(r.get("rep", 0)))] = r   # 再実行があれば後勝ち
    return res


def read_trace(path):
    d = np.load(path, allow_pickle=True)
    need = ("act_exec", "ee", "quat", "grip", "chunk_at", "img_agent", "img_wrist", "success")
    miss = [k for k in need if k not in d.files]
    assert not miss, f"{path}: trace v2 のキーが無い {miss}"
    return d


def episode_arrays(d, n_steps, frames=None):
    """trace → (agentview_raw[M,...], wrist_raw[M,...], state[n,8], actions[n,7], stale[n], idx[n], t_offset)。
    frames あり: フレーム j ↔ t = j+1（t=0 を落とす）、画像は毎 step。idx = 画像配列の添字。
    frames なし(stale): フレーム j ↔ t = j、画像は推論ごと、idx = 最後の推論の添字。n = フレーム数。"""
    T = len(d["act_exec"])
    ca = np.asarray(d["chunk_at"], int)
    assert len(ca) == len(d["img_agent"]) == len(d["img_wrist"]) and ca[0] == 0 and np.all(np.diff(ca) > 0), "chunk_at と画像の対応が崩れている"
    quat_aa = np.stack([KR.quat2axisangle(q) for q in d["quat"][:T]])
    st_all = np.concatenate([np.asarray(d["ee"][:T], float), quat_aa, np.asarray(d["grip"][:T], float)], axis=1)   # 行 k = k+1 step 実行後
    nn = min(T, n_steps)
    if frames is not None:
        fs = np.asarray(frames["step"], int)
        assert np.array_equal(fs, np.arange(len(fs))) and len(fs) >= T + 1, f"frames の step が 0..T で連続でない（frame_every 1 で保存したか）: len {len(fs)} T {T}"
        fe = np.asarray(frames["ee"], float)
        err = float(np.abs(fe[1:T + 1] - np.asarray(d["ee"][:T], float)).max())
        assert err < 1e-6, f"★frames の ee と trace の ee がずれている（フレーム k ↔ trace 行 k-1 の対応が違う）: {err:.2e}"
        for i, c in enumerate(ca):
            assert np.array_equal(frames["agent"][c], d["img_agent"][i]) and np.array_equal(frames["wrist"][c], d["img_wrist"][i]), f"frames と trace の推論時画像が不一致 chunk {i} step {c}"
        t = np.arange(1, nn)                  # t = 1..nn-1（行動 t の直前の観測 = frames[t]）
        state = st_all[t - 1].astype(np.float32)
        actions = np.asarray(d["act_exec"][t], np.float32)
        av = frames["agent"][:, ::-1, ::-1]; wr = frames["wrist"][:, ::-1, ::-1]
        return av, wr, state, actions, np.zeros(len(t), bool), t.copy(), 1
    idx = np.searchsorted(ca, np.arange(nn), side="right") - 1   # フレーム t の画像 = ca[i] <= t の最後の i
    stale = (np.arange(nn) != ca[idx])
    prev = np.maximum(np.arange(nn) - 1, 0)
    state = st_all[prev].astype(np.float32)
    actions = np.asarray(d["act_exec"][:nn], np.float32)
    return d["img_agent"][:, ::-1, ::-1], d["img_wrist"][:, ::-1, ::-1], state, actions, stale, idx, 0


def write_group(f, name, av, wr, state, actions, idx, attrs):
    import h5py
    vlen = h5py.vlen_dtype(np.uint8)
    n = len(actions)
    ja = {int(i): KR.jpg(av[int(i)]) for i in np.unique(idx)}   # 同じ推論の画像は JPEG を 1 回だけ作って使い回す
    jw = {int(i): KR.jpg(wr[int(i)]) for i in np.unique(idx)}
    g = f.create_group(name)
    g.create_dataset("agentview", data=np.array([ja[int(idx[t])] for t in range(n)], dtype=object), dtype=vlen)
    g.create_dataset("wrist", data=np.array([jw[int(idx[t])] for t in range(n)], dtype=object), dtype=vlen)
    g.create_dataset("state", data=state)
    g.create_dataset("actions", data=actions)
    g.attrs.update(attrs)


def select_windows(by_window, per_window):
    """by_window[(task,seed)] = [(rep, clean, info)] → 両ラベルのある窓だけ。窓ごと 成功/失敗 各 ≤ per_window（rep 昇順）。0 = 上限なし。"""
    sel = {}
    for key, rs in by_window.items():
        rs = sorted(rs, key=lambda x: x[0])
        succ = [r for r in rs if r[1]]
        fail = [r for r in rs if not r[1]]
        if not succ or not fail:
            continue
        if per_window:
            succ, fail = succ[:per_window], fail[:per_window]
        sel[key] = (succ, fail)
    return sel


def check_dataset(out_dir, ckpt):
    """操作チェック（計画 §3-1）: loader（学習と同じ経路・同じ transform）で読み、成功/失敗ファイルの tokenized_prompt が異なることを assert。
    openpi の venv（third_party/openpi/.venv）で実行する。tokenizer は CPU。"""
    repo = Path(os.environ.get("KADAI_REPO", HERE.parent.parent))
    cands = [repo, repo / "third_party/openpi/src", Path("/path/to/work"), Path("/path/to/work/third_party/openpi/src"), repo / "scripts"]
    for p in cands:
        if p.exists() and str(p) not in sys.path:
            sys.path.insert(0, str(p))
    from parc.perturbed_dataset import PerturbedDemoDataset
    from openpi.training import config as _config
    import train_lora_kadai as TL
    tc = _config.get_config("pi05_libero")
    tf, _ = TL.build_transform(tc, Path(ckpt))
    ds = PerturbedDemoDataset(out_dir, action_horizon=tc.model.action_horizon, tasks=None, transform=tf)
    by_file = collections.defaultdict(list)
    for i, s in enumerate(ds.index):
        by_file[s.file].append(i)
    toks, rows = {}, []
    for f, ii in sorted(by_file.items()):
        it = ds[ii[0]]
        tk = np.asarray(it["tokenized_prompt"]); m = np.asarray(it["tokenized_prompt_mask"]).astype(bool)
        toks[f] = tuple(tk[m].tolist())
        rows.append((Path(f).name, ds.meta[f]["instruction"], int(m.sum()), len(ii)))
        assert it["state"].shape[-1] >= 8 and it["actions"].shape[0] == 10, "変換後の形"
    for name, ins, nt, n in rows:
        print(f"  {name:72s} frames={n:6d} ntok={nt:3d} | {ins}")
    n_pairs = 0
    for f in toks:
        if f.endswith("__succ.hdf5"):
            g = f.replace("__succ.hdf5", "__fail.hdf5")
            if g in toks:
                assert toks[f] != toks[g], f"★操作チェック失敗: 成功/失敗の tokenized_prompt が同一 {f}"
                n_pairs += 1
            b = f.replace("__succ.hdf5", ".hdf5")
            if b in toks:
                assert toks[f] == toks[b], "教師コピーと成功ファイルの prompt が違う"
    assert n_pairs > 0, "成功/失敗の対になるファイルが無い"
    print(f"[check] OK: 成功/失敗ファイルの対 {n_pairs} で tokenized_prompt が異なる。サンプル総数 {len(ds)}")
    return dict(pairs=n_pairs, samples=len(ds))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--traces")
    ap.add_argument("--results", nargs="*", default=[], help="結果 jsonl（glob 可）")
    ap.add_argument("--teacher")
    ap.add_argument("--out", required=True)
    ap.add_argument("--per-window", type=int, default=2, help="窓ごとの成功/失敗の上限（計画 2。フレーム比が教師を超えるなら 1）")
    ap.add_argument("--holdout-seeds", default="8080-8099", help="前提試験用に <out>_holdout/ へ回す seed（例 8080-8099）。窓の選抜は同じ・本数の上限なし")
    ap.add_argument("--images", choices=["frames", "stale"], default="frames", help="frames（既定）: 毎 step の画像（--frames-root）。stale: trace の推論時画像を次の推論まで使い回す旧経路")
    ap.add_argument("--frames-root", default="", help="rollout の --save-frames の親ディレクトリ（r<rep>/<task>_s<seed>.npz）。--images frames で必須")
    ap.add_argument("--fail-trunc", type=int, default=100, help="失敗本を先頭 N step で打ち切る")
    ap.add_argument("--check", action="store_true", help="作成後に loader で操作チェック")
    ap.add_argument("--check-only", action="store_true")
    ap.add_argument("--ckpt", default=os.environ.get("KADAI_BASE_CKPT", "/path/to/work/data/openpi/pi05_libero_pytorch"), help="assets（norm_stats）の置き場")
    ap.add_argument("--no-teacher", action="store_true", help="教師コピーを作らない（手元の小さな検査用）")
    a = ap.parse_args()
    out = Path(a.out)
    if a.check_only:
        check_dataset(out, a.ckpt); return
    assert a.traces and (a.teacher or a.no_teacher), "--traces と --teacher（または --no-teacher）が要る"
    assert a.images == "stale" or a.frames_root, "--images frames には --frames-root が要る"
    hold = parse_ranges(a.holdout_seeds)
    res = load_results(a.results)
    import h5py

    files = sorted(glob.glob(os.path.join(a.traces, "*.npz")))
    assert files, f"trace が無い: {a.traces}"
    by_split = {"train": collections.defaultdict(list), "holdout": collections.defaultdict(list)}
    n_no_jsonl = 0
    act_hash = collections.defaultdict(list)   # (task, seed) -> [act_exec のハッシュ]。同じ seed の rep が別標本か（方策の標本化が効いているか）の操作チェック
    for p in files:
        b = os.path.basename(p)[:-4]; task, s, r = b.split("_")
        seed, rep = int(s[1:]), int(r[1:])
        d = np.load(p, allow_pickle=True)
        done = bool(d["success"]); rr = res.get((task, seed, rep))
        if rr is not None:
            assert bool(rr["success"]) == done, f"jsonl と trace の success が不一致 {b}"
        else:
            n_no_jsonl += 1
        clean = bool(rr["clean"]) if rr is not None else done
        act_hash[(task, seed)].append(hash(np.asarray(d["act_exec"]).tobytes()))
        by_split["holdout" if seed in hold else "train"][(task, seed)].append((rep, clean, dict(path=p, done=done)))

    multi = {k: v for k, v in act_hash.items() if len(v) >= 2}
    n_same = sum(1 for v in multi.values() if len(set(v)) == 1)
    n_dup_pairs = sum(len(v) - len(set(v)) for v in multi.values())
    print(f"[操作チェック] 同じ seed の rep 間で act_exec が全て同一の窓 {n_same}/{len(multi)}、同一の rep 対 {n_dup_pairs}（0 に近いはず。方策の標本化が rep ごとに別になっていること）")
    assert len(multi) == 0 or n_same <= 0.1 * len(multi), "★rep が別標本になっていない（方策の乱数が固定されている？）"
    summary = dict(args=vars(a), n_traces=len(files), n_traces_without_jsonl=n_no_jsonl, windows_with_identical_reps=n_same, windows_multi_rep=len(multi), identical_rep_pairs=n_dup_pairs, splits={})
    teacher_frames = {}
    instr = {}
    for task, (suite, base) in TASKS.items():
        if a.no_teacher:
            instr[task] = f"(no teacher) {task}"; continue
        src = Path(a.teacher) / suite / f"{base}.hdf5"
        dst = out / suite / f"{base}.hdf5"
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(src, dst)
        with h5py.File(dst, "r+") as f:
            instr[task] = str(f.attrs["instruction"])
            f.attrs["instruction"] = instr[task] + SUCC_TAG
            teacher_frames[task] = int(sum(int(f[k].attrs["length"]) for k in f))
    if teacher_frames:
        summary["teacher_frames"] = teacher_frames

    for split, by_window in by_split.items():
        root = Path(a.out + "_holdout") if split == "holdout" else out
        sel = select_windows(by_window, 0 if split == "holdout" else a.per_window)
        tab = collections.defaultdict(lambda: collections.Counter())
        open_f = {}
        for (task, seed), (succ, fail) in sorted(sel.items()):
            suite, base = TASKS[task]
            t = tab[task]; t["windows"] += 1
            for label, group in (("succ", succ), ("fail", fail)):
                path = root / suite / f"{base}__{label}.hdf5"
                if path not in open_f:
                    path.parent.mkdir(parents=True, exist_ok=True)
                    fh = h5py.File(path, "w")
                    fh.attrs.update({"suite": suite, "base_task": base, "variant": f"{base}__{label}",
                                     "instruction": instr[task] + (SUCC_TAG if label == "succ" else FAIL_TAG), "render_px": 128,
                                     "source": "kadai_rollout_v1", "category": "kadai_tilt_contact",
                                     "outcome": "success" if label == "succ" else "failure"})
                    open_f[path] = fh
                fh = open_f[path]
                for rep, clean, info in group:
                    d = read_trace(info["path"])
                    T = len(d["act_exec"])
                    trunc = (label == "fail" and T > a.fail_trunc)
                    fz = None
                    if a.images == "frames":
                        fz = np.load(os.path.join(a.frames_root, f"r{rep}", f"{task}_s{seed}.npz"))
                        fz = {k: fz[k] for k in ("step", "agent", "wrist", "ee")}
                    av, wr, state, actions, stale, idx, toff = episode_arrays(d, a.fail_trunc if label == "fail" else T, fz)
                    n = len(actions)
                    if n < 1:
                        continue
                    write_group(fh, f"ep_{seed}_r{rep}", av, wr, state, actions, idx,
                                dict(success=True, length=n, seed=seed, rep=rep, origin_success=bool(info["done"]), clean=bool(clean), full_length=T,
                                     truncated=bool(trunc), outcome=label, t_offset=toff, window=f"{task}_s{seed}",
                                     source_demo=f"kadai_rollout/{base}/seed{seed}_r{rep}"))
                    t[f"{label}_ep"] += 1; t[f"{label}_frames"] += n; t["frames"] += n; t["stale_frames"] += int(stale.sum())
                    if label == "fail":
                        t["fail_truncated"] += int(trunc)
                        t["fail_done_collided" if info["done"] else "fail_not_done"] += 1
        for fh in open_f.values():
            fh.close()
        n_win_all = collections.Counter(k[0] for k in by_window)
        n_mixed = collections.Counter(k[0] for k, v in by_window.items() if any(r[1] for r in v) and any(not r[1] for r in v))
        rows = {}
        for task in TASKS:
            t = dict(tab[task]); t["windows_mixed"] = n_mixed.get(task, 0); t["windows_all"] = n_win_all.get(task, 0)
            tf_ = teacher_frames.get(task)
            t["rollout_to_teacher_frame_ratio"] = (round(t.get("frames", 0) / tf_, 3) if tf_ else None)
            rows[task] = t
        tot = sum(r.get("frames", 0) for r in rows.values()); tt = sum(teacher_frames.values())
        summary["splits"][split] = dict(root=str(root), per_task=rows, rollout_frames=tot, teacher_frames=tt or None, ratio=(round(tot / tt, 3) if tt else None))
        print(f"\n[{split}] -> {root}  per-window={'無制限' if split == 'holdout' else a.per_window}")
        print(f"{'task':7s} {'win(選/混/全)':>13s} {'succ ep':>8s} {'fail ep':>8s} {'succ fr':>9s} {'fail fr':>9s} {'fail未done':>10s} {'done&衝突':>9s} {'打切':>5s} {'stale%':>7s} {'/教師':>7s}")
        for task, t in rows.items():
            print(f"{task:7s} {t.get('windows',0):>4d}/{t['windows_mixed']:>3d}/{t['windows_all']:<4d} {t.get('succ_ep',0):>8d} {t.get('fail_ep',0):>8d} {t.get('succ_frames',0):>9d} {t.get('fail_frames',0):>9d} "
                  f"{t.get('fail_not_done',0):>10d} {t.get('fail_done_collided',0):>9d} {t.get('fail_truncated',0):>5d} "
                  f"{100*t.get('stale_frames',0)/max(t.get('frames',0),1):>6.1f}% {str(t['rollout_to_teacher_frame_ratio']):>7s}")
        msg = f"  rollout フレーム合計 {tot}"
        if tt:
            msg += f" / 教師 {tt} = {tot/tt:.2f}" + ("  ★教師を超える: --per-window 1 を検討" if split == "train" and tot > tt else "")
        print(msg)
    if n_no_jsonl:
        print(f"\n[注意] jsonl の無い trace {n_no_jsonl}/{len(files)} 本: clean = done で代用（done かつ衝突を成功に数える）")
    out.mkdir(parents=True, exist_ok=True)
    json.dump(summary, open(out / "summary.json", "w"), indent=1, ensure_ascii=False, default=str)
    if a.check:
        check_dataset(out, a.ckpt)


if __name__ == "__main__":
    main()
