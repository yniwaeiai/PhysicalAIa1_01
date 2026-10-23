"""腕 C の E1 集計（標準ライブラリのみ。Pod でも手元でも動く）。
入力: <runs-dir>/c1_<cond>_E1_<task>_s<k>.json.jsonl（cond = v1 / w0 / w15 / w3）。出力: markdown（stdout、--out で保存）。
表: 条件 × タスクの clean 本数 / 総数、done・衝突(done かつ not clean)・打ち切り(not done)の内訳、成功 ep(clean) の step 中央値、
    seed 対応差（w0−v1、w15−w0 など。clean の 0/1 の seed ごとの差の合計と、勝ち/負け/同点の seed 数）、v1 の E1 が過去値（drawer 27 / k4 42 / k6 42）±4 以内か。
使い方: python summarize_contrast_e1.py --runs-dir /workspace/Kadai/data/contrast/eval_runs --conds v1,w0,w15 --out summary.md
"""
import argparse, collections, glob, json, statistics

PAST = {"drawer": 27, "k4": 42, "k6": 42}
TASKS = ["drawer", "k4", "k6"]


def load(d, cond):
    rows = {}
    for f in sorted(glob.glob(f"{d}/c1_{cond}_E1_*_s*.json.jsonl")):
        for line in open(f):
            r = json.loads(line)
            if r.get("error"):
                rows.setdefault(("ERR", r["task"]), []).append(r); continue
            rows[(r["task"], r["seed"])] = r
    return rows


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--runs-dir", required=True); ap.add_argument("--conds", default="v1,w0,w15"); ap.add_argument("--out", default="")
    a = ap.parse_args(); conds = a.conds.split(",")
    D = {c: load(a.runs_dir, c) for c in conds}
    L = ["# 腕 C: E1 集計（strong 3000〜3047、各タスク 48、250 step、replan 5）", "", "clean = done かつ対象外 ≤ 1 mm。衝突 = done かつ not clean。打ち切り = not done（250 step で未完了）。", "",
         "| 条件 | タスク | n | clean | done | 衝突 | 打ち切り | error | 成功(clean) step 中央値 |", "|---|---|---|---|---|---|---|---|---|"]
    clean = {}
    for c in conds:
        for t in TASKS:
            rs = [r for (tk, s), r in D[c].items() if tk == t and tk != "ERR"]
            n = len(rs); cl = sum(r["clean"] for r in rs); dn = sum(r["success"] for r in rs); col = sum(r["success"] and not r["clean"] for r in rs); to = sum(not r["success"] for r in rs)
            st = [r["steps"] for r in rs if r["clean"]]
            clean[(c, t)] = {r["seed"]: int(r["clean"]) for r in rs}
            L.append(f"| {c} | {t} | {n} | {cl} | {dn} | {col} | {to} | {len(D[c].get(('ERR', t), []))} | {statistics.median(st) if st else '-'} |")
    L += ["", "## seed 対応差（clean の 0/1。同じ seed どうし）", "", "| 比較 | タスク | 共通 seed | 差の合計 | 勝(+) | 負(−) | 同点 |", "|---|---|---|---|---|---|---|"]
    pairs = [(conds[i + 1], conds[i]) for i in range(len(conds) - 1)] if len(conds) > 1 else []
    if "v1" in conds and "w0" in conds and ("w0", "v1") not in pairs: pairs.append(("w0", "v1"))
    for x, y in pairs:
        tot = [0, 0, 0, 0]
        for t in TASKS + ["drawer+k6"]:
            ts = ["drawer", "k6"] if t == "drawer+k6" else [t]
            diffs = [clean[(x, tt)][s] - clean[(y, tt)][s] for tt in ts for s in clean[(x, tt)] if s in clean[(y, tt)]]
            L.append(f"| {x} − {y} | {t} | {len(diffs)} | {sum(diffs):+d} | {sum(d > 0 for d in diffs)} | {sum(d < 0 for d in diffs)} | {sum(d == 0 for d in diffs)} |")
    if "v1" in conds:
        L += ["", "## v1 の E1 と過去値（物差しの連続性、±4）", ""]
        for t in TASKS:
            v = sum(clean[("v1", t)].values()); L.append(f"- {t}: v1 clean {v}/{len(clean[('v1', t)])}、過去値 {PAST[t]}、差 {v - PAST[t]:+d} → {'±4 以内' if abs(v - PAST[t]) <= 4 else '★±4 の外'}")
    out = "\n".join(L) + "\n"
    print(out)
    if a.out: open(a.out, "w").write(out)


if __name__ == "__main__":
    main()
