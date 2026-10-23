"""I5（計画 v2 §16、腕の状態の分解）の分岐を集計する。CPU のみ。判定は書かず値だけ出す。
使い方: python kadai/scripts/analyze_branches_i5.py
入力: kadai/state/branches_i5_20261015/summary_s*.jsonl + b_*.npz、kadai/state/branches_i4_20261015/summary_s*.jsonl（(s)）、kadai/state/branches_i3_20261015.csv（(a)）
出力: kadai/state/branches_i5_summary_20261015.md, kadai/state/branches_i5_20261015.csv
窓 = (失敗窓 f, 成功 ep srep)、t = s/3。窓の値 = 分岐 8 本の平均 → seed の値 = 窓の平均 → 群/タスクの値 = seed の平均。括弧は (seed 数/窓数/分岐数)。差は窓対応の seed 平均。
(a) = その失敗窓の I3 t13 の最終成功、(s) = I4 の (s) t13。最終成功 = done。衝突なし成功 = done かつ 対象外物体の最大変位 <= 1 mm。貫入 = 復元直後の接触法線力 f0 > 20 N。
"""
import csv, json, glob, collections
from pathlib import Path
import numpy as np

K = Path(__file__).resolve().parents[1]
BR = K / "state/branches_i5_20261015"; B4 = K / "state/branches_i4_20261015"
GN = {"luck": "運", "mid": "中間", "hold": "保留"}
CONDS = ["q", "p", "pp", "po", "n"]
CN = {"q": "(q) 関節位置", "p": "(p) 手先 位置+回転", "pp": "(pp) 手先位置", "po": "(po) 手先回転", "n": "(n) 冗長自由度"}
FTH = 20.0


def mean(x):
    x = [v for v in x if v is not None and not (isinstance(v, float) and np.isnan(v))]
    return float(np.mean(x)) if x else float("nan")


def readjl(d):
    rows = []
    for f in sorted(glob.glob(str(d / "summary_s*.jsonl"))):
        rows += [json.loads(l) for l in open(f) if l.strip()]
    return rows


def load():
    rows = readjl(BR); last = {r["id"]: r for r in rows}; n_dup = len(rows) - len(last)
    run = []; skip = []
    for r in last.values():
        if r.get("skipped"):
            skip.append(r); continue
        if not (BR / f"b_{r['id']}.npz").exists():
            continue
        r["succ"] = int(r["done"]); r["clean"] = int(r["done"] and r["disp_mm"] <= 1.0); r["pen"] = int(r["f0"] > FTH)
        run.append(r)
    return run, skip, n_dup, len(last) - len(run) - len(skip)


def load_ref():
    rows = list(csv.DictReader(open(K / "state/branches_i3_20261015.csv")))
    a = collections.defaultdict(list)
    for r in rows:
        if r["cond"] == "a" and r["kind"] == "t13":
            a[(r["wid"], "succ")].append(int(float(r["succ"]))); a[(r["wid"], "clean")].append(int(float(r["clean"])))
    s = collections.defaultdict(list)
    for r in readjl(B4):
        if r["cond"] == "s" and r["kind"] == "t13" and (B4 / f"b_{r['id']}.npz").exists():
            f0 = r["f0"]
            s[(r["wid"], "succ")].append((int(r["done"]), f0)); s[(r["wid"], "clean")].append((int(r["done"] and r["disp_mm"] <= 1.0), f0))
    return {k: mean(v) for k, v in a.items()}, s


def windows(rows, key, excl):
    A, S4 = REF
    by = collections.defaultdict(lambda: {c: [] for c in CONDS}); meta = {}
    for r in rows:
        if excl and r["pen"]:
            continue
        by[r["wid"]][r["cond"]].append(r[key]); meta[r["wid"]] = r
    out = {}
    for w, d in by.items():
        m = meta[w]
        sv = [v for v, f in S4.get((w, key), []) if not (excl and f > FTH)]
        out[w] = dict(task=m["task"], seed=m["seed"], group=m["group"], a=A.get((m["fwid"], key)), s=mean(sv), **{c: mean(d[c]) for c in CONDS}, **{"n_" + c: len(d[c]) for c in CONDS})
    return out


def agg(W, f, task=None, group=None):
    bys = collections.defaultdict(list); nb = 0; nw = 0
    for w, v in W.items():
        if (task and v["task"] != task) or (group and v["group"] != group):
            continue
        x = v[f]
        if x is None or np.isnan(x):
            continue
        bys[(v["task"], v["seed"])].append(x); nw += 1; nb += v.get("n_" + f, 8)
    sv = {s: mean(x) for s, x in bys.items()}
    return mean(list(sv.values())), len(sv), nw, nb


def diff(W, f1, f2, task=None, group=None):
    bys = collections.defaultdict(list)
    for w, v in W.items():
        if (task and v["task"] != task) or (group and v["group"] != group):
            continue
        if v[f1] is None or v[f2] is None or np.isnan(v[f1]) or np.isnan(v[f2]):
            continue
        bys[(v["task"], v["seed"])].append(v[f1] - v[f2])
    sv = [mean(x) for x in bys.values()]
    return mean(sv), len(sv)


def fmt(a):
    v, ns, nw, nb = a
    return "—" if nw == 0 else f"{v:.3f} ({ns}/{nw}/{nb})"


def q3(x):
    x = np.asarray(x, float)
    return f"{np.median(x):.4g} [{np.percentile(x,25):.4g}, {np.percentile(x,75):.4g}]" if len(x) else "—"


def main():
    global REF
    rows, skip, n_dup, n_miss = load(); REF = load_ref()
    cols = ["id", "wid", "fwid", "task", "seed", "rep", "srep", "group", "cond", "k", "t", "ws", "done", "succ", "clean", "disp_mm", "n_steps", "n_infer", "ik_conv", "ik_iters", "ik_err_pos_mm", "ik_err_rot_deg",
            "tgt_err_pos_mm", "tgt_err_rot_deg", "err_vs_s_pos_mm", "err_vs_s_rot_deg", "err_vs_f_pos_mm", "err_vs_f_rot_deg", "d_qc_qf", "d_qc_qs", "d_qs_qf", "f0", "f1", "pen",
            "ee_obs_vs_site_mm", "fixture_q_err", "oerr_q", "arm_q_err", "arm_v_err", "top"]
    with open(K / "state/branches_i5_20261015.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore"); w.writeheader(); [w.writerow(r) for r in rows + skip]
    L = []; P = L.append
    P("# I5 の集計（2026-10-15、`scripts/analyze_branches_i5.py`、分岐ごとの値 `state/branches_i5_20261015.csv`）\n")
    P("定義（s/3 のみ。base = 失敗 ep の完全状態、qvel・指は失敗側、腕 7 関節 qpos だけ作る）: (q) 成功 ep の q ／ (p) IK(成功の手先 位置+回転、初期 q_f) ／ (pp) IK(位置 3 のみ) ／ (po) IK(回転 3 のみ) ／ (n) IK(失敗の手先、初期 q_s)。"
      "(a) = I3 t13、(s) = I4 の (s) t13。窓の値 = 分岐 8 本の平均 → seed → タスク/群（seed 平均）。括弧は (seed 数/窓数/分岐数)。差は窓対応の seed 平均。判定は書かない。\n")
    kc = collections.Counter(r["cond"] for r in rows); ks = collections.Counter(r["cond"] for r in skip)
    P(f"I5 分岐（走行）{len(rows)} 本（{dict(kc)}）、IK 不収束で除外 {len(skip)} 本（{dict(ks)}）。重複行 {n_dup}、npz 欠損 {n_miss}。窓 {len({r['wid'] for r in rows + skip})}。\n")
    allr = rows + skip

    P("## 操作チェックの実測")
    P("### IK の収束（分岐単位。8 本は同一の IK なので窓単位も併記）")
    P("| 条件 | 分岐数 | 収束 | 収束率 | 反復数 中央値 [四分位] / 最大 | IK 最終誤差 位置 mm 中央値/最大 | IK 最終誤差 回転 ° 中央値/最大 |"); P("|---|---|---|---|---|---|---|")
    for c in CONDS:
        rc = [r for r in allr if r["cond"] == c]
        if not rc:
            continue
        cv = sum(1 for r in rc if r["ik_conv"]); it = [r["ik_iters"] for r in rc]; ep = [r["ik_err_pos_mm"] for r in rc]; er = [r["ik_err_rot_deg"] for r in rc]
        wk = {r["wid"] for r in rc}; wnc = {r["wid"] for r in rc if not r["ik_conv"]}
        P(f"| {c} | {len(rc)} | {cv} | {cv/len(rc):.3f}（窓 {len(wk)-len(wnc)}/{len(wk)}） | {q3(it)} / {max(it)} | {np.median(ep):.4f} / {max(ep):.4f}（拘束外成分を含む） | {np.median(er):.4f} / {max(er):.4f}（拘束外成分を含む） |")
    P("")
    P("### 復元後に実測した手先の目標誤差（走行した分岐。目標 = q/p/pp/po は成功の手先、n は失敗の手先。拘束した成分のみ）")
    P("| 条件 | 成分 | 本数 | 中央値 | 最大 | > 閾値の本数（位置 > 1 mm / 回転 > 0.5°） |"); P("|---|---|---|---|---|---|")
    for c in CONDS:
        rc = [r for r in rows if r["cond"] == c]
        if not rc:
            continue
        comps = [("位置 mm", "tgt_err_pos_mm", 1.0)] * (c != "po") + [("回転 °", "tgt_err_rot_deg", 0.5)] * (c != "pp")
        for nm, k_, th in comps:
            x = np.array([r[k_] for r in rc], float)
            P(f"| {c} | {nm} | {len(x)} | {np.median(x):.4f} | {x.max():.4f} | {int((x>th).sum())} |")
    P("")
    P("参考（拘束していない成分、または対象外の手先との差）:")
    for c in CONDS:
        rc = [r for r in rows if r["cond"] == c]
        if rc:
            P(f"- ({c}) 成功の手先との差: 位置 {q3([r['err_vs_s_pos_mm'] for r in rc])} mm、回転 {q3([r['err_vs_s_rot_deg'] for r in rc])} °；失敗の手先との差: 位置 {q3([r['err_vs_f_pos_mm'] for r in rc])} mm、回転 {q3([r['err_vs_f_rot_deg'] for r in rc])} °（中央値 [四分位]）")
    P("")
    P("### 関節空間距離（7 関節 L2、rad。窓単位 = 28 組のうち当該条件の IK が収束した窓）")
    P("| 量 | 窓数 | 中央値 [四分位] | 最小 / 最大 |"); P("|---|---|---|---|")
    wsel = {}
    for r in rows:
        wsel[(r["wid"], r["cond"])] = r
    for nm, k_, c in (("d(q_s, q_f)", "d_qs_qf", "q"), ("d(q_q, q_f)", "d_qc_qf", "q"), ("d(q_p, q_f)", "d_qc_qf", "p"), ("d(q_p, q_s)", "d_qc_qs", "p"), ("d(q_pp, q_f)", "d_qc_qf", "pp"), ("d(q_pp, q_s)", "d_qc_qs", "pp"),
                      ("d(q_po, q_f)", "d_qc_qf", "po"), ("d(q_po, q_s)", "d_qc_qs", "po"), ("**d(q_n, q_f)**", "d_qc_qf", "n"), ("d(q_n, q_s)", "d_qc_qs", "n")):
        x = [r[k_] for (w, cc), r in wsel.items() if cc == c]
        if x:
            P(f"| {nm} | {len(x)} | {q3(x)} | {min(x):.4f} / {max(x):.4f} |")
    P("")
    P("### 状態の整合（全走行分岐）")
    P(f"- 腕以外の qpos が失敗側と一致（oerr_q）最大 {max(r['oerr_q'] for r in rows):.2e}；固定具 q − 失敗 trace の q[t−1] 最大 |差| {max(abs(r['fixture_q_err']) for r in rows):.2e}；腕 qpos が作った q と一致（arm_q_err）最大 {max(r['arm_q_err'] for r in rows):.2e}；腕 qvel・指が失敗側（arm_v_err）最大 {max(r['arm_v_err'] for r in rows):.2e}；観測の手先と site の差 最大 {max(r['ee_obs_vs_site_mm'] for r in rows):.2e} mm")
    f0 = np.array([r["f0"] for r in rows]); f1 = np.array([r["f1"] for r in rows], float)
    P(f"- 復元直後の接触法線力 f0: 0 N が {int((f0==0).sum())}/{len(f0)} 本、中央値 {np.median(f0):.2f} / 90% {np.percentile(f0,90):.1f} / 最大 {f0.max():.1f} N、> 20 N {int((f0>FTH).sum())} 本（" + "、".join(f"{c} {sum(1 for r in rows if r['cond']==c and r['pen'])}" for c in CONDS) + f"）；f1 最大 {np.nanmax(f1):.1f} N、> 20 N {int((f1>FTH).sum())} 本")
    P(f"- 走行: done {sum(r['done'] for r in rows)}/{len(rows)}、n_steps 中央値 {np.median([r['n_steps'] for r in rows]):.0f}\n")

    # 本表
    for key, ttl in (("succ", "最終成功率"), ("clean", "衝突なし成功率")):
        for excl in (False, True):
            W = windows(rows, key, excl); sfx = "（貫入除外 f0 > 20 N）" if excl else ""
            P(f"## {ttl}{sfx}: タスク × 条件（s/3）")
            P("| タスク | (a) | (s) I4 | " + " | ".join(f"{CN[c]}" for c in CONDS) + " |"); P("|---|---|---|" + "---|" * len(CONDS))
            for tk in (None, "k6", "drawer", "k4"):
                P(f"| {tk or '全タスク'} | {fmt(agg(W,'a',tk))} | {fmt(agg(W,'s',tk))} | " + " | ".join(fmt(agg(W, c, tk)) for c in CONDS) + " |")
            P("")
            P(f"差 −(a)（窓対応・seed 平均、括弧は seed 数）{sfx}")
            P("| タスク | (s)−(a) | " + " | ".join(f"{c}−(a)" for c in CONDS) + " |"); P("|---|---|" + "---|" * len(CONDS))
            for tk in (None, "k6", "drawer", "k4"):
                P(f"| {tk or '全タスク'} | " + " | ".join(f"{diff(W,f,'a',tk)[0]:+.3f} ({diff(W,f,'a',tk)[1]})" for f in ["s"] + CONDS) + " |")
            P("")
            if key == "succ":
                P(f"### 群別{sfx}")
                P("| 群 | (a) | (s) | " + " | ".join(CN[c] for c in CONDS) + " |"); P("|---|---|---|" + "---|" * len(CONDS))
                for g in ("luck", "mid", "hold"):
                    P(f"| {GN[g]} | {fmt(agg(W,'a',None,g))} | {fmt(agg(W,'s',None,g))} | " + " | ".join(fmt(agg(W, c, None, g)) for c in CONDS) + " |")
                P("")
                P(f"群別の差 −(a){sfx}")
                P("| 群 | (s)−(a) | " + " | ".join(f"{c}−(a)" for c in CONDS) + " |"); P("|---|---|" + "---|" * len(CONDS))
                for g in ("luck", "mid", "hold"):
                    P(f"| {GN[g]} | " + " | ".join(f"{diff(W,f,'a',None,g)[0]:+.3f}" for f in ["s"] + CONDS) + " |")
                P("")
    # seed 別
    W = windows(rows, "succ", False)
    P("## seed 別（最終成功率、除外なし。窓平均）")
    P("| タスク | seed | 群 | 窓 | (a) | (s) | " + " | ".join(CONDS) + " |"); P("|---|---|---|---|---|---|" + "---|" * len(CONDS))
    for tk, sd, g in sorted({(v["task"], v["seed"], v["group"]) for v in W.values()}, key=lambda x: (x[0], x[1])):
        ws = [v for v in W.values() if v["task"] == tk and v["seed"] == sd]
        P(f"| {tk} | {sd} | {GN[g]} | {len(ws)} | {mean([v['a'] for v in ws]):.3f} | {mean([v['s'] for v in ws]):.3f} | " + " | ".join(f"{mean([v[c] for v in ws]):.3f}" for c in CONDS) + " |")
    P("")
    # R17-R20
    P("## R17〜R20 の値（線は書かない。計画 §16 の表。s/3、最終成功率、貫入除外なし、seed 平均）")
    dn = [r["d_qc_qf"] for (w, cc), r in wsel.items() if cc == "n"]
    for tk, nm in ((None, "全タスク"), ("k6", "k6"), ("drawer", "drawer"), ("k4", "k4")):
        parts = []
        for tag, f in (("R17 (q)−(a)", "q"), ("R18 (p)−(a)", "p"), ("R19 (n)−(a)", "n"), ("R20 (pp)−(a)", "pp"), ("R20 (po)−(a)", "po"), ("参考 (s)−(a)", "s")):
            d, n = diff(W, f, "a", tk); parts.append(f"{tag} = **{d:+.3f}** (seed {n})")
        P(f"- {nm}: " + "、".join(parts))
    P(f"- R19 付記: d(q_n, q_f) の中央値（窓 {len(dn)}） = **{np.median(dn):.4f} rad**（[四分位 {np.percentile(dn,25):.4f}, {np.percentile(dn,75):.4f}]、最大 {max(dn):.4f}）。d(q_s, q_f) 中央値 {np.median([r['d_qs_qf'] for (w, cc), r in wsel.items() if cc == 'q']):.4f} rad")
    (K / "state/branches_i5_summary_20261015.md").write_text("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
