"""実験 A: --trace の npz から、失敗エピソードの終端の停留を分類する（定義は kadai/docs/計画_確率的失敗の原因追究_20261014.md §1.3）。
使い方: python analyze_trace.py <trace ディレクトリ> [--p0-json 各 seed の成功確率を含む jsonl のディレクトリ接頭辞]
出力: 分類表（タスク × 分類）、seed の群（p=0 / 運）別の分類、P1〜P3 の判定に要る数値。
"""
import argparse, collections, glob, json, os
import numpy as np

WIN = 40
STALL_DIST = 0.02
JM_LIMIT = 0.05


def load(path):
    z = np.load(path, allow_pickle=False)
    d = {k: z[k] for k in z.files}
    d["success"] = bool(d["success"]); d["seed"] = int(d["seed"]); d["rep"] = int(d["rep"])
    d["task"] = os.path.basename(path).split("_s")[0]
    return d


def tilt_deg(quat_xyzw):
    # eval_kadai_closedloop.tilt_deg と同じ定義
    from scipy.spatial.transform import Rotation as Rot
    z = Rot.from_quat(quat_xyzw).as_matrix()[:, 2]
    return float(np.degrees(np.arccos(np.clip(-z[2], -1, 1))))


def moving_ref(eps):
    """成功エピソードの、動いている step（手先速度が上位半分）での位置行動のノルム中央値 = a_ref。"""
    vals = []
    for e in eps:
        if not e["success"] or len(e["ee"]) < 10:
            continue
        v = np.linalg.norm(np.diff(e["ee"], axis=0), axis=1)
        a = np.linalg.norm(e["act_raw"][1:, :3], axis=1)
        vals.append(a[v >= np.median(v)])
    return float(np.median(np.concatenate(vals))) if vals else float("nan")


def classify(e, a_ref):
    n = len(e["ee"])
    if e["success"] or n < WIN + 1:
        return None
    w = slice(n - WIN, n)
    dist = float(np.linalg.norm(np.diff(e["ee"][w], axis=0), axis=1).sum())
    if dist >= STALL_DIST:
        return dict(cls="停留でない失敗", dist=dist)
    a = np.linalg.norm(e["act_raw"][w, :3], axis=1)
    a_pos = float(np.median(a))
    ar = e["act_raw"][w, :3]
    cs = [float(np.dot(ar[i], ar[i + 1]) / (np.linalg.norm(ar[i]) * np.linalg.norm(ar[i + 1]) + 1e-9)) for i in range(WIN - 1)]
    dither = float(np.mean(np.array(cs) < 0))
    jm = float(e["jmargin"][w].min())
    cnt = collections.Counter(c for cc in e["contact"][w] for c in (cc.split(";") if cc else []))
    top = cnt.most_common(1)[0][0] if cnt else ""
    if jm < JM_LIMIT:
        cls = "関節限界"
    elif a_pos >= 0.5 * a_ref:
        cls = "詰まり"
    elif a_pos < 0.25 * a_ref:
        cls = "方策の静止"
    elif dither > 0.3:
        cls = "迷い"
    else:
        cls = "未分類"
    return dict(cls=cls, dist=dist, a_pos=a_pos, a_ratio=a_pos / a_ref, dither=dither, jm=jm, contact=top,
                tilt=tilt_deg(e["quat"][n - 1]), q_end=float(e["q"][-1]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dir")
    ap.add_argument("--groups", default="", help="seed の群を与える json: {\"drawer\": {\"1045\": \"p0\", ...}}（p0 = 決定的、luck = 運）")
    args = ap.parse_args()
    eps = [load(p) for p in sorted(glob.glob(os.path.join(args.dir, "*.npz")))]
    groups = json.load(open(args.groups)) if args.groups else {}
    by_task = collections.defaultdict(list)
    for e in eps:
        by_task[e["task"]].append(e)
    rows = []
    for t, es in by_task.items():
        a_ref = moving_ref(es)
        print(f"== {t}: {len(es)} 本、成功 {sum(e['success'] for e in es)}、a_ref={a_ref:.3f}")
        for e in es:
            r = classify(e, a_ref)
            if r:
                r.update(task=t, seed=e["seed"], rep=e["rep"], grp=groups.get(t, {}).get(str(e["seed"]), "?"))
                rows.append(r)
    tab = collections.Counter((r["task"], r["grp"], r["cls"]) for r in rows)
    print("\n失敗の分類（task, 群, 分類）: 本数")
    for k, v in sorted(tab.items()):
        print(" ", k, v)
    allc = collections.Counter(r["cls"] for r in rows)
    print("\n全体:", dict(allc))
    json.dump(rows, open(os.path.join(args.dir, "classification.json"), "w"), ensure_ascii=False, indent=1)


if __name__ == "__main__":
    main()
