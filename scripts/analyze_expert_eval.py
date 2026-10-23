"""I6（計画 v2 §17）の集計。入力 kadai/state/expert_eval_20261016/{i,ii,cal}/summary_s*.jsonl、出力 expert_eval_summary_20261016.md / .csv。判定は書かない（数値のみ）。"""
import collections, csv, json, sys
from pathlib import Path
import numpy as np

ST = Path(__file__).resolve().parent.parent / "state"
IN = ST / "expert_eval_20261016"
GN = {"unseen": "成功未観測", "luck": "運", "mid": "中間", "hold": "保留"}


def load(mode):
    out = []
    for f in sorted((IN / mode).glob("summary_s*.jsonl")):
        out += [json.loads(l) for l in open(f) if l.strip()]
    return out


def frac(xs):
    xs = list(xs)
    return (sum(xs) / len(xs), len(xs)) if xs else (float("nan"), 0)


def fm(p):
    return f"{p[0]:.3f} ({int(round(p[0] * p[1]))}/{p[1]})" if p[1] else "-"


def main():
    I, II, CAL = load("i"), load("ii"), load("cal")
    L = ["# I6 expert を評価環境で走らせた集計（2026-10-16、`scripts/analyze_expert_eval.py`。判定は書かない）", ""]
    err = [r for r in I + II + CAL if "error" in r]
    L.append(f"件数: (i) {len(I)}、(ii) {len(II)}、較正 {len(CAL)}、エラー {len(err)}"); L.append("")
    for e in err[:5]:
        L.append(f"- error {e['id']}: {e['error'][:200]}")
    rows_csv = []
    # ---- 較正（生成記録との照合） ----
    L += ["## 操作チェック 1: 生成時の seed（gen_v2_*_a の先頭 16 本）を評価環境の作りで走らせた較正", "",
          "| task | n | 生成記録の ok 率 | 評価環境 clean 率 | done 率 | 局面転換 step の中央値（評価環境）| 同（生成記録）| 局面 step の一致（±2 step 以内）|", "|---|---|---|---|---|---|---|---|"]
    for task in ("drawer", "k4", "k6"):
        c = [r for r in CAL if r.get("task") == task and "error" not in r]
        if not c:
            continue
        gt = [json.loads(r["gen_trans"]) for r in c]; et = [json.loads(r["trans"]) for r in c]
        def med(tt):
            d = collections.defaultdict(list)
            for t in tt:
                for n, s, _w in t:
                    d[n].append(s)
            return {n: float(np.median(v)) for n, v in d.items()}
        match = np.mean([len(g) == len(e) and all(abs(a[1] - b[1]) <= 2 and a[2] == b[2] for a, b in zip(g, e)) for g, e in zip(gt, et)])
        L.append(f"| {task} | {len(c)} | {fm(frac(r['gen_ok'] for r in c))} | {fm(frac(r['clean'] for r in c))} | {fm(frac(r['done'] for r in c))} | {med(et)} | {med(gt)} | {match:.2f} |")
        for r in c:
            rows_csv.append(dict(mode="cal", id=r["id"], task=task, seed=r["seed"], group="cal", done=r["done"], clean=r["clean"], gen_ok=r["gen_ok"]))
    L.append("")
    # ---- (i) ----
    I = [r for r in I if "error" not in r]
    p_reset = {}
    try:
        for r in csv.DictReader(open(ST / "branches_i3_20261015.csv")):
            if r["kind"] == "t0" and r["cond"] == "a":
                p_reset.setdefault((r["task"], int(r["seed"])), []).append(int(r["succ"]))
    except Exception as e:   # noqa
        L.append(f"(p_reset 読み込み失敗 {e})")
    orig = collections.defaultdict(list)
    for r in csv.DictReader(open(ST / "aprime_windows_20261014.csv")):
        orig[(r["task"], int(r["seed"]))].append(int(r["success"]))
    L += ["## (i) 配置の被覆: A' の seed を評価環境の初期状態から expert（pos0 = ref の列を主、tr0 = A' trace の pos0 の列を副）", "",
          "| 群 | pose | seed 数 | done 率 | clean 率（done ∧ 対象外 ≤ 1 mm）| 局面が最後まで進んだ率 |", "|---|---|---|---|---|---|"]
    for g in ("unseen", "luck", "mid", "hold"):
        for pose in ("ref", "tr0"):
            c = [r for r in I if r["group"] == g and r["pose"] == pose]
            if c:
                L.append(f"| {GN[g]} | {pose} | {len(c)} | {fm(frac(r['done'] for r in c))} | {fm(frac(r['clean'] for r in c))} | {fm(frac(r['reached_all'] for r in c))} |")
    L.append("")
    L += ["### タスク別（pose = ref）", "", "| task | 群 | seed 数 | done | clean |", "|---|---|---|---|---|"]
    for task in ("drawer", "k4", "k6"):
        for g in ("unseen", "luck", "mid", "hold"):
            c = [r for r in I if r["task"] == task and r["group"] == g and r["pose"] == "ref"]
            if c:
                L.append(f"| {task} | {GN[g]} | {len(c)} | {fm(frac(r['done'] for r in c))} | {fm(frac(r['clean'] for r in c))} |")
    L.append("")
    L += ["### seed 別（pose = ref / tr0。方策の p_reset = I3 の t0 分岐の成功率、A' 元 ep の成功率と並べる）", "",
          "| task | seed | 群 | p_reset（I3 t0）| A' 元成功率 | expert ref: done / clean / disp mm / q_end / 局面 | expert tr0: done / clean / disp mm |", "|---|---|---|---|---|---|---|"]
    for r in sorted([r for r in I if r["pose"] == "ref"], key=lambda r: (["unseen", "luck", "mid", "hold"].index(r["group"]), r["task"], r["seed"])):
        t = next((x for x in I if x["task"] == r["task"] and x["seed"] == r["seed"] and x["pose"] == "tr0"), None)
        pr = p_reset.get((r["task"], r["seed"])); po = orig.get((r["task"], r["seed"]))
        trs = "->".join(f"{n}{'' if w == 'ok' else '!'}@{s}" for n, s, w in json.loads(r["trans"]))
        L.append(f"| {r['task']} | {r['seed']} | {GN[r['group']]} | {('%.2f (%d)' % (np.mean(pr), len(pr))) if pr else '-'} | {('%.2f' % np.mean(po)) if po else '-'} | "
                 f"{int(r['done'])} / {int(r['clean'])} / {r['disp_mm']:.1f} / {r['q_end']:.3f} / {trs} | "
                 f"{('%d / %d / %.1f' % (t['done'], t['clean'], t['disp_mm'])) if t else '-'} |")
        rows_csv.append(dict(mode="i", id=r["id"], task=r["task"], seed=r["seed"], group=r["group"], done=r["done"], clean=r["clean"], disp_mm=r["disp_mm"], q_end=r["q_end"],
                             p_reset=(np.mean(pr) if pr else ""), tr0_done=(t["done"] if t else ""), tr0_clean=(t["clean"] if t else "")))
    L.append("")
    # 失敗の様式（unseen のみ）
    L += ["### 失敗した expert の様式（pose = ref、done = False のもの）", "", "| task | seed | 群 | 終了 step | timeout した局面 | 最終 q | disp mm | 接触先 |", "|---|---|---|---|---|---|---|---|"]
    for r in sorted([r for r in I if r["pose"] == "ref" and not r["done"]], key=lambda r: (r["group"], r["task"], r["seed"])):
        to = [n for n, s, w in json.loads(r["trans"]) if w == "timeout"]
        L.append(f"| {r['task']} | {r['seed']} | {GN[r['group']]} | {r['n_steps']} | {to} | {r['q_end']:.3f} | {r['disp_mm']:.1f} | {r['top']} |")
    L.append("")
    # ---- (ii) ----
    II = [r for r in II if "error" not in r]
    key = lambda r: (r["task"], r["seed"], r["rep"], r["t"], r["kind"])
    D = {key(r): r for r in II}
    pairs = []
    # 組は run_branches.load_windows_i5 から復元（Pod と同じ csv/trace が無いのでローカルの branches_i5 csv の fwid/srep/t を使う）
    seen = set()
    for r in csv.DictReader(open(ST / "branches_i5_20261015.csv")):
        k = (r["wid"],)
        if k in seen:
            continue
        seen.add(k)
        f = D.get((r["task"], int(r["seed"]), int(r["rep"]), int(r["t"]), "fail")); s = D.get((r["task"], int(r["seed"]), int(r["srep"]), int(r["t"]), "succ"))
        pairs.append((r["wid"], r["task"], int(r["seed"]), r["group"], f, s))
    L += ["## (ii) 回復可能性: t13 の完全状態を復元して expert を H_PRE から", "",
          f"組数 {len(pairs)}（欠損 {sum(1 for p in pairs if p[4] is None or p[5] is None)}）。実走行数 失敗 ep 側 {sum(r['kind']=='fail' for r in II)}・成功 ep 側 {sum(r['kind']=='succ' for r in II)}（同じ状態は 1 度だけ走らせ、組へは複製して数える）", ""]
    ok = [p for p in pairs if p[4] is not None and p[5] is not None]
    for lab, fn in (("done", lambda r: r["done"]), ("clean（done ∧ 対象外 ≤ 1 mm）", lambda r: r["clean"])):
        fa = np.array([fn(p[4]) for p in ok], float); su = np.array([fn(p[5]) for p in ok], float)
        L.append(f"- {lab}: 失敗 ep t13 から {fa.mean():.3f}、成功 ep t13 から {su.mean():.3f}、組対応の差（失敗 − 成功）の組平均 {np.mean(fa - su):+.3f}")
    # seed 平均
    L.append("")
    L += ["### seed 平均の差（失敗側 − 成功側、clean。seed 内で組を平均してから seed 平均）", ""]
    bys = collections.defaultdict(list)
    for p in ok:
        bys[(p[1], p[2])].append(float(p[4]["clean"]) - float(p[5]["clean"]))
    sd = [np.mean(v) for v in bys.values()]
    L.append(f"seed {len(sd)} 本、平均 {np.mean(sd):+.3f}、最小 {np.min(sd):+.3f}、最大 {np.max(sd):+.3f}")
    L += ["", "### タスク別", "", "| task | 組数 | 失敗側 done | 失敗側 clean | 成功側 done | 成功側 clean | 差(clean, 組平均) |", "|---|---|---|---|---|---|---|"]
    for task in ("drawer", "k4", "k6"):
        c = [p for p in ok if p[1] == task]
        if c:
            L.append(f"| {task} | {len(c)} | {np.mean([p[4]['done'] for p in c]):.3f} | {np.mean([p[4]['clean'] for p in c]):.3f} | {np.mean([p[5]['done'] for p in c]):.3f} | {np.mean([p[5]['clean'] for p in c]):.3f} | {np.mean([float(p[4]['clean']) - float(p[5]['clean']) for p in c]):+.3f} |")
    L += ["", "### 群別（clean）", "", "| 群 | 組数 | 失敗側 | 成功側 |", "|---|---|---|---|"]
    for g in ("unseen", "luck", "mid", "hold"):
        c = [p for p in ok if p[3] == g]
        if c:
            L.append(f"| {GN[g]} | {len(c)} | {np.mean([p[4]['clean'] for p in c]):.3f} | {np.mean([p[5]['clean'] for p in c]):.3f} |")
    # 操作チェック
    L += ["", "## 操作チェック 2: 復元と局面の到達（実走行ぶん）", ""]
    re_ = np.array([r["restore_err_mm"] for r in II])
    L.append(f"- 復元誤差（restore 直後の手先 vs trace の記録、mm）: 中央値 {np.median(re_):.3f}、最大 {re_.max():.3f}")
    L.append(f"- 復元した固定具 q vs trace の q: 最大 |差| {max(abs(r['q_restored'] - r['q_trace_t']) for r in II):.2e}")
    L.append(f"- 復元時点で対象外物体が既に動いている（disp_start > 1 mm）: {sum(r['disp_start_mm'] > 1 for r in II)}/{len(II)}")
    L.append("- 復元時点で動いていた実走行: " + ", ".join(f"{r['id']}（{r['top']} が fresh 比 {r['disp_start_mm']:.0f} mm。この 2 本の clean は意味を持たないので done を見る）" for r in II if r["disp_start_mm"] > 1))
    for kind in ("fail", "succ"):
        c = [r for r in II if r["kind"] == kind]
        L.append(f"- {kind} 側: 先頭局面（H_PRE / C_PRE / M_PRE）を ok で通過 {fm(frac(r['pre_ok'] for r in c))}、先頭局面の通過 step 中央値 {np.median([r['pre_step'] for r in c if r['pre_step'] is not None]):.0f}、全局面を通過 {fm(frac(r['reached_all'] for r in c))}")
    L.append("")
    L += ["### 組別", "", "| wid | 群 | 失敗側 done/clean/disp mm/局面 | 成功側 done/clean/disp mm/局面 | restore誤差 mm (f/s) |", "|---|---|---|---|---|"]
    tr_ = lambda r: "->".join(f"{n}{'' if w == 'ok' else '!'}@{s}" for n, s, w in json.loads(r["trans"]))
    for wid, task, seed, g, f, s in pairs:
        if f is None or s is None:
            L.append(f"| {wid} | {GN[g]} | 欠損 | 欠損 | |"); continue
        L.append(f"| {wid} | {GN[g]} | {int(f['done'])}/{int(f['clean'])}/{f['disp_mm']:.1f}/{tr_(f)} | {int(s['done'])}/{int(s['clean'])}/{s['disp_mm']:.1f}/{tr_(s)} | {f['restore_err_mm']:.2f}/{s['restore_err_mm']:.2f} |")
        rows_csv.append(dict(mode="ii", id=wid, task=task, seed=seed, group=g, fail_done=f["done"], fail_clean=f["clean"], succ_done=s["done"], succ_clean=s["clean"],
                             fail_disp_mm=f["disp_mm"], succ_disp_mm=s["disp_mm"], fail_restore_err_mm=f["restore_err_mm"], succ_restore_err_mm=s["restore_err_mm"]))
    (ST / "expert_eval_summary_20261016.md").write_text("\n".join(L) + "\n")
    keys = []
    for r in rows_csv:
        for k in r:
            if k not in keys:
                keys.append(k)
    with open(ST / "expert_eval_summary_20261016.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys); w.writeheader(); w.writerows(rows_csv)
    print("\n".join(L))


if __name__ == "__main__":
    main()
