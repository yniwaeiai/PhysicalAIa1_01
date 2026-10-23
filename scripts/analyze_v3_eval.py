#!/usr/bin/env python3
"""v3 評価の集計（計画 v2 §18 + GPT-6 第 2 回 採用 (6)〜(8)）。判定は書かない（数値と線を並べるだけ）。

入力: kadai/state/v3_eval_20261016/ に回収した jsonl（Pod の data/eval_runs/ の <名前>_<条件>_<task>_s<k>.json.jsonl）。
      名前 = v1r / v3、条件 = E1 / E2 / E3s / E3m / T（mix の標的 20 seed）/ Ts（strong の保留 14 seed）。T と Ts は合わせて「T」(34 seed) として扱う。
出力: kadai/state/v3_eval_summary_20261016.md と .csv
使い方: /path/to/work/.venv/bin/python kadai/scripts/analyze_v3_eval.py [--dir DIR] [--out-prefix kadai/state/v3_eval_summary_20261016]
        [--base v1r --new v3]（差 = new − base）

単位と区間: 差は seed 単位（E 系は 1 seed = 1 本、T は 1 seed = その seed の全反復の平均）。
  タスクを跨ぐプールは (タスク, seed) を 1 単位とする。区間は単位の対応差の平均に対する seed リサンプリングの percentile bootstrap（B=10000, 乱数固定）。反復・窓を独立に数えない。
群ラベルは過去のものを固定: kadai/state/expert_eval_summary_20261016.csv の mode=i（id が _ref）の group 列
  （unseen=成功未観測 7 / luck=運 8 / mid=中間 5 / hold=保留 14。出典は W/A' の群分け = wrapper_eval_summary_20261015.md・aprime_summary_20261014.md を I6 で 34 seed に絞ったもの）。v3 の結果で再分類しない。
expert 可否 = 同 csv の mode=i（_ref）の clean 列（expert が ref の配置で clean 成功したか）。
"""
import argparse, collections, csv, glob, json, os, re, sys
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
STATE = os.path.join(os.path.dirname(HERE), "state")
NAME_RE = re.compile(r"^(?P<model>[^_]+)_(?P<cond>E1|E2|E3s|E3m|Ts|T)_(?P<task>drawer|k4|k6)_s(?P<shard>\d+)\.json(\.jsonl)?$")
LEVELS = {"none": (0.0, 0.0, 0.0), "mid": (0.1, 0.03, 5.0), "strong": (0.2, 0.06, 10.0)}
TASKS = ["drawer", "k4", "k6"]
GN = {"unseen": "成功未観測", "luck": "運", "mid": "中間", "hold": "保留"}
E_CONDS = ["E1", "E2", "E3s", "E3m"]
# 条件ごとの期待する seed 範囲と水準（操作チェック: 設計どおりの seed・水準で走ったか）
E_SPEC = {"E1": (3000, "strong"), "E2": (1000, "mix"), "E3s": (7000, "strong"), "E3m": (6000, "mix")}
B = 10000


def pick_level_label(level, seed):
    """eval_kadai_closedloop.pick_level と同じ式（mix の水準を seed から決める）。ラベル none/mid/strong を返す。"""
    if level != "mix":
        return level
    r = np.random.default_rng((seed, 777)).random()
    return "none" if r < 0.25 else ("mid" if r < 0.75 else "strong")


def level_of_row(r):
    for k, v in LEVELS.items():
        if tuple(r["level"]) == v:
            return k
    return "?"


def load(d):
    """{(model, cond, task, seed, rep): row}。重複は後勝ちで数を返す。"""
    rows, dup, errs, skipped = {}, 0, collections.Counter(), []
    for p in sorted(glob.glob(os.path.join(d, "*.jsonl"))):
        m = NAME_RE.match(os.path.basename(p))
        if not m:
            skipped.append(os.path.basename(p)); continue
        for l in open(p):
            if not l.strip():
                continue
            r = json.loads(l)
            if r.get("error"):
                errs[(m["model"], m["cond"], m["task"])] += 1; continue
            cond = "T" if m["cond"] in ("T", "Ts") else m["cond"]
            k = (m["model"], cond, m["task"], int(r["seed"]), int(r.get("rep", 0)))
            if k in rows:
                dup += 1
            r["_src"] = m["cond"]
            rows[k] = r
    return rows, dup, errs, skipped


def labels():
    g, ex = {}, {}
    for r in csv.DictReader(open(os.path.join(STATE, "expert_eval_summary_20261016.csv"), encoding="utf-8")):
        if r["mode"] == "i" and r["id"].endswith("_ref"):
            g[(r["task"], int(r["seed"]))] = r["group"]
            ex[(r["task"], int(r["seed"]))] = r["clean"] in ("True", "1", "true")
    return g, ex


def per_seed(rows, model, cond, metric):
    """{(task, seed): 反復平均の指標} （E 系は 1 本）。"""
    acc = collections.defaultdict(list)
    for (m, c, t, s, rep), r in rows.items():
        if m == model and c == cond:
            acc[(t, s)].append(float(r[metric]))
    return {k: float(np.mean(v)) for k, v in acc.items()}


def boot_diff(base, new, units, rng):
    """units: [(task, seed)]。base/new は per_seed の辞書。両方にある単位だけ。返り値 (n, 平均差, lo, hi, base 平均, new 平均)。"""
    u = [k for k in units if k in base and k in new]
    if not u:
        return (0, np.nan, np.nan, np.nan, np.nan, np.nan)
    d = np.array([new[k] - base[k] for k in u])
    idx = rng.integers(0, len(d), size=(B, len(d)))
    bm = d[idx].mean(axis=1)
    return (len(u), d.mean(), np.percentile(bm, 2.5), np.percentile(bm, 97.5),
            float(np.mean([base[k] for k in u])), float(np.mean([new[k] for k in u])))


def fmt(x, p=3):
    return "-" if x is None or (isinstance(x, float) and np.isnan(x)) else f"{x:+.{p}f}"


def fm(x):
    return "-" if x is None or (isinstance(x, float) and np.isnan(x)) else f"{x:.3f}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default=os.path.join(STATE, "v3_eval_20261016"))
    ap.add_argument("--out-prefix", default=os.path.join(STATE, "v3_eval_summary_20261016"))
    ap.add_argument("--base", default="v1r"); ap.add_argument("--new", default="v3")
    ap.add_argument("--title-note", default="")
    a = ap.parse_args()
    rows, dup, errs, skipped = load(a.dir)
    grp, exp = labels()
    rng = np.random.default_rng(20261016)
    BM, NM = a.base, a.new
    L, C = [], [["table", "cond", "task", "group", "metric", "n_units", "base", "new", "diff", "lo", "hi"]]
    L += [f"# v3 評価の集計（{NM} − {BM}、seed 対応）{a.title_note}", "",
          f"入力: `{os.path.relpath(a.dir)}`、行数 {len(rows)}、重複（後勝ち）{dup}、環境エラーで除外 {sum(errs.values())}、名前が規約外でスキップ {len(skipped)}"
          + (f"（{', '.join(skipped[:5])}…）" if skipped else ""), ""]
    if errs:
        L += ["環境エラー（除外した件数）: " + ", ".join(f"{'/'.join(k)}={v}" for k, v in sorted(errs.items())), ""]

    # ---- 操作チェック ----
    L += ["## 0. 操作チェック（結果の前に）", "",
          "| 条件 | モデル | タスク | 本数 | seed 範囲 | 期待 | 水準の内訳（行の level） | 期待どおり |", "|---|---|---|---|---|---|---|---|"]
    allok = True
    for c in E_CONDS:
        off, lv = E_SPEC[c]
        for m in (BM, NM):
            for t in TASKS:
                rs = [(k, r) for k, r in rows.items() if k[0] == m and k[1] == c and k[2] == t]
                if not rs:
                    L.append(f"| {c} | {m} | {t} | 0 | - | {off}〜{off+47} | - | なし |"); continue
                seeds = sorted(k[3] for k, _ in rs)
                lc = collections.Counter(level_of_row(r) for _, r in rs)
                exp_lv = collections.Counter(pick_level_label(lv, s) for s in seeds)
                ok = (seeds == list(range(off, off + 48)) and lc == exp_lv)
                allok &= ok
                L.append(f"| {c} | {m} | {t} | {len(rs)} | {seeds[0]}〜{seeds[-1]} | {off}〜{off+47} ×48 | {dict(lc)} | {'OK' if ok else '不一致/欠け'} |")
    tseeds = collections.defaultdict(set)
    for (m, c, t, s, rep) in rows:
        if c == "T":
            tseeds[(m, t)].add((s, rep))
    L += ["", "T（標的）: 期待 = 34 seed × 8 反復 = 272（ラベルのある seed 34）", "",
          "| モデル | 本数 | seed 数 | ラベルに無い seed | ラベル seed のうち欠け | 反復数が 8 でない seed 数 |", "|---|---|---|---|---|---|"]
    for m in (BM, NM):
        ss = collections.defaultdict(set)
        for (mm, c, t, s, rep) in rows:
            if mm == m and c == "T":
                ss[(t, s)].add(rep)
        unl = [k for k in ss if k not in grp]
        miss = [k for k in grp if k not in ss]
        bad = [k for k, v in ss.items() if len(v) != 8]
        L.append(f"| {m} | {sum(len(v) for v in ss.values())} | {len(ss)} | {len(unl)} | {len(miss)} | {len(bad)} |")
    L += ["", f"E 系の seed・水準の一致: {'全て一致' if allok else '**不一致あり（上表）**'}", ""]

    # ---- 1. 条件 × タスク × モデル ----
    L += ["## 1. 条件 × タスク × モデル: done 率・clean 率・成功分の step 中央値", "",
          "| 条件 | タスク | モデル | n | done | clean | step 中央値（成功分） |", "|---|---|---|---|---|---|---|"]
    conds_all = E_CONDS + ["T"]
    for c in conds_all:
        for t in TASKS + ["all"]:
            for m in (BM, NM):
                rs = [r for k, r in rows.items() if k[0] == m and k[1] == c and (t == "all" or k[2] == t)]
                if not rs:
                    continue
                d = np.mean([r["success"] for r in rs]); cl = np.mean([r["clean"] for r in rs])
                st = [r["steps"] for r in rs if r["success"]]
                med = f"{np.median(st):.0f}" if st else "-"
                L.append(f"| {c} | {t} | {m} | {len(rs)} | {d:.3f} | {cl:.3f} | {med} |")
                C.append(["t1", c, t, "", "done/clean/steps_med", len(rs), "", "", "", "", f"{d:.4f}|{cl:.4f}|{med}"])

    # ---- 2. seed 対応差 ----
    L += ["", f"## 2. seed 対応差（{NM} − {BM}）と seed 単位 bootstrap 95% 区間", "",
          "E 系は 1 seed = 1 本、T は seed ごとの反復平均。「all」は (タスク, seed) をプールした単位。", "",
          "| 条件 | タスク | 指標 | 単位数 | " + BM + " | " + NM + " | 差 | 95% 区間 |", "|---|---|---|---|---|---|---|---|"]
    ps = {}
    for c in conds_all:
        for met in ("success", "clean"):
            for m in (BM, NM):
                ps[(m, c, met)] = per_seed(rows, m, c, met)
    for c in conds_all:
        for t in TASKS + ["all"]:
            for met in ("success", "clean"):
                b, n_ = ps[(BM, c, met)], ps[(NM, c, met)]
                units = [k for k in b if (t == "all" or k[0] == t) and (c != "T" or k in grp)]   # T はラベルのある 34 seed のみ
                n, d, lo, hi, bm_, nm_ = boot_diff(b, n_, units, rng)
                if n == 0:
                    continue
                L.append(f"| {c} | {t} | {'done' if met == 'success' else 'clean'} | {n} | {fm(bm_)} | {fm(nm_)} | {fmt(d)} | [{fmt(lo)}, {fmt(hi)}] |")
                C.append(["t2", c, t, "", met, n, f"{bm_:.4f}", f"{nm_:.4f}", f"{d:.4f}", f"{lo:.4f}", f"{hi:.4f}"])

    # ---- 3. 線のための数値 ----
    def row_line(name, cond, units, met, note):
        b, n_ = ps[(BM, cond, met)], ps[(NM, cond, met)]
        n, d, lo, hi, bm_, nm_ = boot_diff(b, n_, units, rng)
        C.append(["lines", cond, name, "", met, n, f"{bm_:.4f}", f"{nm_:.4f}", f"{d:.4f}", f"{lo:.4f}", f"{hi:.4f}"])
        return n, d, lo, hi, bm_, nm_

    L += ["", "## 3. 線のための数値（判定はしない。計画 v2 §18 の線と並べるだけ。線は変えない）", ""]
    # V1
    L += ["### V1 主: drawer + k6 の clean（プール）。線: 差 ≥ +0.10 支持 / ≤ +0.03 反証 / その間は保留。**主は E3s**（E1 は A' 標的の選抜 seed を含むので副）", "",
          "| 集合 | 単位数 | " + BM + " | " + NM + " | 差 | 95% 区間 | 点推定の位置（機械的）|", "|---|---|---|---|---|---|---|"]
    for c in ("E3s", "E1"):
        units = [k for k in ps[(BM, c, "clean")] if k[0] in ("drawer", "k6")]
        n, d, lo, hi, bm_, nm_ = row_line("V1_drawer+k6", c, units, "clean", "")
        pos = "-" if n == 0 else ("≥ +0.10" if d >= 0.10 else ("≤ +0.03" if d <= 0.03 else "+0.03〜+0.10"))
        L.append(f"| {c}（{'主' if c == 'E3s' else '副'}）| {n} | {fm(bm_)} | {fm(nm_)} | {fmt(d)} | [{fmt(lo)}, {fmt(hi)}] | {pos} |")
    for c in ("E3s", "E1"):  # タスク別の内訳（参考）
        for t in ("drawer", "k6", "k4"):
            units = [k for k in ps[(BM, c, "clean")] if k[0] == t]
            n, d, lo, hi, bm_, nm_ = row_line(f"V1_{t}", c, units, "clean", "")
            if n:
                L.append(f"| （内訳）{c} {t} | {n} | {fm(bm_)} | {fm(nm_)} | {fmt(d)} | [{fmt(lo)}, {fmt(hi)}] | |")
    # V2 / V3
    def tgroup(g):
        return [k for k, v in grp.items() if v == g]
    L += ["", "### V2 機構: T の運群 8 seed の seed 対応差（線: ≥ +0.15 支持 / ≤ +0.05 反証）。V3 対照: 成功未観測 7 seed（線: |Δ| ≤ 0.05 が予測どおり、≥ +0.15 なら機構以外を疑う）", "",
          "指標は計画に明記が無いので done と clean の両方を出す（p_reset 系の先行結果は clean を主に読んでいる。どちらを主とするかは判定担当が決める）。", "",
          "| 線 | 群 | 指標 | 単位数 | " + BM + " | " + NM + " | 差 | 95% 区間 | 点推定の位置（機械的）|", "|---|---|---|---|---|---|---|---|---|"]
    for line, g in (("V2", "luck"), ("V3", "unseen")):
        for met in ("clean", "success"):
            n, d, lo, hi, bm_, nm_ = row_line(f"{line}_{g}", "T", tgroup(g), met, "")
            if line == "V2":
                pos = "-" if n == 0 else ("≥ +0.15" if d >= 0.15 else ("≤ +0.05" if d <= 0.05 else "+0.05〜+0.15"))
            else:
                pos = "-" if n == 0 else ("|Δ| ≤ 0.05" if abs(d) <= 0.05 else ("≥ +0.15" if d >= 0.15 else "その他"))
            L.append(f"| {line} | {GN[g]} | {'clean' if met == 'clean' else 'done'} | {n} | {fm(bm_)} | {fm(nm_)} | {fmt(d)} | [{fmt(lo)}, {fmt(hi)}] | {pos} |")
    # V4
    L += ["", "### V4 退行なし: E2 と E3m の各タスクの clean 差（線: 全タスクで区間下限 > −0.06 を非劣性とする。点推定も併記）", "",
          "| 条件 | タスク | 単位数 | " + BM + " | " + NM + " | 差 | 95% 区間 | 区間下限 > −0.06 | 点推定 > −0.06 |", "|---|---|---|---|---|---|---|---|---|"]
    for c in ("E2", "E3m"):
        for t in TASKS:
            units = [k for k in ps[(BM, c, "clean")] if k[0] == t]
            n, d, lo, hi, bm_, nm_ = row_line(f"V4_{t}", c, units, "clean", "")
            if n:
                L.append(f"| {c} | {t} | {n} | {fm(bm_)} | {fm(nm_)} | {fmt(d)} | [{fmt(lo)}, {fmt(hi)}] | {'はい' if lo > -0.06 else 'いいえ'} | {'はい' if d > -0.06 else 'いいえ'} |")

    # ---- 4. 補助 ----
    L += ["", "## 4. 補助", "", "### 4.1 T の全群別（群ラベルは固定）", "",
          "| 群 | 指標 | 単位数 | " + BM + " | " + NM + " | 差 | 95% 区間 |", "|---|---|---|---|---|---|---|"]
    for g in ("luck", "mid", "unseen", "hold"):
        for met in ("clean", "success"):
            n, d, lo, hi, bm_, nm_ = row_line(f"aux_group_{g}", "T", tgroup(g), met, "")
            if n:
                L.append(f"| {GN[g]} | {'clean' if met == 'clean' else 'done'} | {n} | {fm(bm_)} | {fm(nm_)} | {fmt(d)} | [{fmt(lo)}, {fmt(hi)}] |")
    L += ["", "### 4.2 T の expert 可否別（I6 の (i) ref の clean。ラベルは固定）", "",
          "| expert | 指標 | 単位数 | " + BM + " | " + NM + " | 差 | 95% 区間 |", "|---|---|---|---|---|---|---|"]
    for e, name in ((True, "expert clean 成功"), (False, "expert 失敗")):
        units = [k for k, v in exp.items() if v == e]
        for met in ("clean", "success"):
            n, d, lo, hi, bm_, nm_ = row_line(f"aux_expert_{e}", "T", units, met, "")
            if n:
                L.append(f"| {name} | {'clean' if met == 'clean' else 'done'} | {n} | {fm(bm_)} | {fm(nm_)} | {fmt(d)} | [{fmt(lo)}, {fmt(hi)}] |")
    L += ["", "### 4.3 E2・E3m の摂動水準別（水準は行の level。seed から決まる pick_level('mix', seed) と一致を 0 章で確認）", "",
          "| 条件 | 水準 | 指標 | 単位数 | " + BM + " | " + NM + " | 差 | 95% 区間 |", "|---|---|---|---|---|---|---|---|"]
    for c in ("E2", "E3m"):
        for lv in ("none", "mid", "strong"):
            for met in ("clean", "success"):
                b, n_ = ps[(BM, c, met)], ps[(NM, c, met)]
                units = [k for k in b if pick_level_label("mix", k[1]) == lv]
                n, d, lo, hi, bm_, nm_ = boot_diff(b, n_, units, rng)
                if n:
                    L.append(f"| {c} | {lv} | {'clean' if met == 'clean' else 'done'} | {n} | {fm(bm_)} | {fm(nm_)} | {fmt(d)} | [{fmt(lo)}, {fmt(hi)}] |")
                    C.append(["aux_level", c, "all", lv, met, n, f"{bm_:.4f}", f"{nm_:.4f}", f"{d:.4f}", f"{lo:.4f}", f"{hi:.4f}"])
    L += ["", "注: 単位数が小さい行（T の群別、水準別の none など）は区間が広い。区間は seed リサンプリングで、同一 seed の反復・同一 ep の窓を独立とは数えていない。", ""]
    open(a.out_prefix + ".md", "w", encoding="utf-8").write("\n".join(L))
    with open(a.out_prefix + ".csv", "w", encoding="utf-8", newline="") as f:
        csv.writer(f).writerows(C)
    print("\n".join(L))


if __name__ == "__main__":
    main()
