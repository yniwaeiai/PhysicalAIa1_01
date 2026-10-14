"""レポート用の図を作る（静的 PNG）。データは kadai/state/ の記録から。v2 のデータがあれば自動で加える。
配色は dataviz スキルの検証済み既定パレット（カテゴリ順固定）: 素の π0.5 = 青、LoRA v1 = オレンジ、LoRA v2 = 緑青。エンティティに色を固定する。
使い方: python make_figures.py   （出力: kadai/report/figs/*.png）
"""
import glob, json, math, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager

HERE = os.path.dirname(os.path.abspath(__file__))
ST = os.path.join(HERE, "..", "state")
EV = os.path.join(ST, "eval_runs_20261013")
OUT = os.path.join(HERE, "figs")
os.makedirs(OUT, exist_ok=True)
for fam in ("Noto Sans CJK JP", "Noto Sans CJK HK", "Droid Sans Fallback"):
    if any(f.name == fam for f in font_manager.fontManager.ttflist):
        plt.rcParams["font.family"] = fam
        break
plt.rcParams.update({"axes.spines.top": False, "axes.spines.right": False, "axes.edgecolor": "#8a8984", "axes.labelcolor": "#52514e",
                     "xtick.color": "#52514e", "ytick.color": "#52514e", "text.color": "#0b0b0b", "figure.facecolor": "#fcfcfb", "axes.facecolor": "#fcfcfb",
                     "axes.grid": True, "grid.color": "#e6e5e1", "grid.linewidth": 0.8, "axes.axisbelow": True, "savefig.dpi": 200})
BLUE, ORANGE, AQUA, INK, MUTED = "#2a78d6", "#eb6834", "#1baf7a", "#0b0b0b", "#8a8984"
TASKS = [("drawer", "中段引き出しを開ける"), ("k4", "下段引き出しを閉める（K4）"), ("k6", "電子レンジを閉める（K6）")]


def load(pattern):
    return [json.loads(l) for f in sorted(glob.glob(os.path.join(EV, pattern))) for l in open(f) if l.strip()]


def wilson(k, n, z=1.96):
    p = k / n; d = 1 + z * z / n; c = p + z * z / (2 * n); h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return (c - h) / d, (c + h) / d


# ---------------- 図 1: 人間デモの接触区間での手首の傾き ----------------
def fig_tilt():  # noqa
    cw = json.load(open(os.path.join(ST, "contact_windows_summary.json")))
    sg = json.load(open(os.path.join(ST, "single_tilt_summary.json")))
    rows = [("中段引き出しを開ける（本課題）", cw["drawer_middle"]["out"]["tilt_med_deg"], True),
            ("下段引き出しを閉める（K4、単独・本課題）", sg["k4_single_close_drawer"]["tilt_med_per_demo"], True),
            ("電子レンジを閉める（K6、単独・本課題）", sg["k6_single_close_microwave"]["tilt_med_per_demo"], True),
            ("K4 複合（ボウルを入れて閉める）", cw["k4_bowl_drawer"]["out"]["tilt_med_deg"], False),
            ("K6 複合（マグを入れて閉める）", cw["microwave_k6"]["out"]["tilt_med_deg"], False),
            ("上段引き出しを開ける", cw["drawer_top_bowl"]["out"]["tilt_med_deg"], False),
            ("皿を押す", cw["push_plate"]["out"]["tilt_med_deg"], False),
            ("コンロのつまみを回す", cw["k3_stove_moka"]["out"]["tilt_med_deg"], False)]
    fig, ax = plt.subplots(figsize=(8.2, 4.0))
    for i, (name, d, main) in enumerate(rows[::-1]):
        c = BLUE if main else MUTED
        ax.plot([d["p10"], d["p90"]], [i, i], color=c, lw=2.2, solid_capstyle="round")
        ax.plot([d["median"]], [i], "o", color=c, ms=8, mec="#fcfcfb", mew=1.5)
        ax.text(d["p90"] + 2, i, f"{d['median']:.0f}°", va="center", fontsize=9, color=INK)
    ax.set_yticks(range(len(rows))); ax.set_yticklabels([r[0] for r in rows[::-1]], fontsize=9)
    ax.set_xlim(0, 90); ax.set_xlabel("接触区間の手首の傾き（度）"); ax.grid(axis="y", visible=False)
    fig.text(0.01, 0.005, "傾き = ツール軸の鉛直下向きからの角（0° = 真下）。点 = 中央値、線 = 10〜90 パーセンタイル（各タスクの人間デモ 50 本）。", fontsize=7.5, color="#52514e")
    fig.tight_layout(rect=(0, 0.04, 1, 1)); fig.savefig(os.path.join(OUT, "fig1_human_tilt.png")); plt.close(fig)


# ---------------- 図 2: 主結果（成功率・衝突なし率・ステップ数） ----------------
def fig_main():
    runs = [("素の π0.5-LIBERO", BLUE, load("base_f1_*.jsonl")), ("LoRA v1（2400 step）", ORANGE, load("kadai_v1_final_*.jsonl"))]
    if load("v2_mix_*.jsonl"):
        runs.append(("LoRA v2（強い摂動を追加）", AQUA, load("v2_mix_*.jsonl")))
    fig, axes = plt.subplots(1, 3, figsize=(10.5, 3.9))
    w = 0.8 / len(runs)
    for ax, (key, title, kind) in zip(axes, [("success", "成功率（%）", "rate"), ("clean", "成功かつ衝突なし（%）", "rate"), ("steps", "成功時のステップ数（中央値）", "steps")]):
        for ri, (lab, col, rs) in enumerate(runs):
            for ti, (t, _) in enumerate(TASKS):
                xs = [r for r in rs if r["task"] == t and not r.get("error")]
                x = ti + (ri - (len(runs) - 1) / 2) * w
                if kind == "rate":
                    k = sum(r[key] for r in xs); n = len(xs); lo, hi = wilson(k, n)
                    ax.bar(x, 100 * k / n, width=w * 0.92, color=col, label=lab if ti == 0 else None)
                    ax.plot([x, x], [100 * lo, 100 * hi], color=INK, lw=1)
                    ax.text(x, 100 * hi + 2, f"{100*k/n:.0f}", ha="center", fontsize=8, color=INK)
                else:
                    v = np.median([r["steps"] for r in xs if r["success"]]) if any(r["success"] for r in xs) else 0
                    ax.bar(x, v, width=w * 0.92, color=col, label=lab if ti == 0 else None)
                    ax.text(x, v + 4, f"{v:.0f}", ha="center", fontsize=8, color=INK)
        ax.set_xticks(range(3)); ax.set_xticklabels(["中段\n引き出し", "K4\n下段引き出し", "K6\n電子レンジ"], fontsize=9)
        ax.set_title(title, fontsize=10.5, loc="left"); ax.grid(axis="x", visible=False)
        if kind == "rate":
            ax.set_ylim(0, 112)
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, frameon=False, fontsize=9.5, loc="upper center", ncol=len(runs), bbox_to_anchor=(0.5, 1.0))   # 凡例は棒に重ならないよう図の上に置く
    fig.text(0.01, 0.005, "評価: 混合摂動（なし/中/強 = 25/50/25%）、seed 1000–1023（素）/ 1000–1047（LoRA）、各タスク 24〜48 本、OSMesa 描画。縦線は Wilson 95% 区間。", fontsize=7.5, color="#52514e")
    fig.tight_layout(rect=(0, 0.03, 1, 0.93)); fig.savefig(os.path.join(OUT, "fig2_main_results.png")); plt.close(fig)


# ---------------- 図 3: 失敗の構成（標的 seed の反復成功確率） ----------------
def fig_taxonomy():
    rs = load("t2_T0_*.jsonl")
    fig, axes = plt.subplots(1, 3, figsize=(10.5, 3.5), sharey=True)
    for ax, (t, name) in zip(axes, TASKS):
        g = {}
        for r in rs:
            if r["task"] == t and not r.get("error"):
                g.setdefault(r["seed"], []).append(r["success"])
        ps = sorted(sum(v) / len(v) for v in g.values())
        n0 = sum(p == 0 for p in ps); n1 = sum(p == 1 for p in ps)
        ax.scatter(range(len(ps)), ps, s=46, color=BLUE, edgecolor="#fcfcfb", linewidth=1.2, zorder=3)
        ax.axhline(0, color=MUTED, lw=1); ax.axhline(1, color=MUTED, lw=1)
        ax.set_title(f"{name}（失敗 seed {len(ps)} 本）", fontsize=9.5, loc="left")
        ax.text(0.03, 0.55, f"毎回失敗 p=0: {n0} 本\n毎回成功 p=1: {n1} 本", transform=ax.transAxes, fontsize=9, va="top", color=INK)
        ax.set_xticks([]); ax.set_xlabel("seed（成功確率の昇順）", fontsize=8.5); ax.grid(axis="x", visible=False)
        ax.set_ylim(-0.08, 1.08)
    axes[0].set_ylabel("同じ初期条件を 8 回再実行したときの成功確率 p", fontsize=8.5)
    fig.text(0.01, 0.005, "最終評価で失敗した seed を、同じ初期条件・摂動で方策のノイズだけ変えて 8 回ずつ再実行。p=1 は運の失敗、p=0 は決定的な失敗（全て強い摂動の seed）。", fontsize=7.5, color="#52514e")
    fig.tight_layout(rect=(0, 0.04, 1, 1)); fig.savefig(os.path.join(OUT, "fig3_failure_taxonomy.png")); plt.close(fig)


# ---------------- 図 4: 摂動（停留対策）の効果: N0 との同一 seed の差 ----------------
def fig_perturb():
    n0 = {(r["task"], r["seed"]): r for r in load("kadai_v1_final_*.jsonl") if not r.get("error")}
    conds = [("N0b", "再実行（ノイズ床）", "v1f_N0b_*.jsonl"), ("A05", "行動ノイズ σ=0.05", "v1f_A05_*.jsonl"), ("A15", "行動ノイズ σ=0.15", "v1f_A15_*.jsonl"),
             ("K", "停留キック（手先ベース）", "v1f_K_*.jsonl"), ("F10", "初期ノイズ τ=1.0（対照）", "v2_F10_*.jsonl"), ("F15", "初期ノイズ τ=1.5", "v2_F15_*.jsonl"),
             ("S25", "q 停留 → τ=2.5 で引き直し", "v2_S25_*.jsonl")]
    rng = np.random.default_rng(0)
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.8), sharey=True)
    for ax, (t, name) in zip(axes, TASKS):
        for i, (k, lab, pat) in enumerate(conds[::-1]):
            rs = {(r["task"], r["seed"]): r for r in load(pat) if not r.get("error")}
            seeds = [s for (tt, s) in n0 if tt == t and (t, s) in rs]
            d = np.array([float(rs[(t, s)]["success"]) - float(n0[(t, s)]["success"]) for s in seeds]) * 100
            boot = [rng.choice(d, len(d)).mean() for _ in range(2000)]
            lo, hi = np.percentile(boot, [2.5, 97.5])
            col = MUTED if k == "N0b" else BLUE
            ax.plot([lo, hi], [i, i], color=col, lw=2.2, solid_capstyle="round"); ax.plot([d.mean()], [i], "o", color=col, ms=8, mec="#fcfcfb", mew=1.5)
        ax.axvline(0, color=INK, lw=1)
        ax.set_title(name, fontsize=10, loc="left"); ax.set_xlabel("成功率の差（pt）vs 無摂動 N0、95% ブートストラップ区間", fontsize=8)
        ax.grid(axis="y", visible=False); ax.set_xlim(-65, 25)
    axes[0].set_yticks(range(len(conds))); axes[0].set_yticklabels([c[1] for c in conds[::-1]], fontsize=8.5)
    fig.text(0.01, 0.005, "同じ seed（1000–1047）での対応つき差。灰 = 同じ設定の再実行（ノイズ床）。行動ノイズ σ=0.15 は中段引き出しで −48 pt と大きく悪化。他は差を検出できない。", fontsize=7.5, color="#52514e")
    fig.tight_layout(rect=(0, 0.04, 1, 1)); fig.savefig(os.path.join(OUT, "fig4_perturbation.png")); plt.close(fig)


# ---------------- 図 5: 学習曲線 ----------------
def fig_train():
    fig, ax = plt.subplots(figsize=(6.2, 3.4))
    h1 = json.load(open(os.path.join(ST, "train_history_kadai_v1.json")))
    ax.axhline(h1["baseline_test_loss"], color=MUTED, lw=1, ls=(0, (4, 3))); ax.text(2400, h1["baseline_test_loss"] + 0.006, "未学習 0.320", ha="right", fontsize=8.5, color="#52514e")
    ax.plot(h1["step"], h1["test_loss"], "-o", color=ORANGE, lw=2, ms=6, mec="#fcfcfb", mew=1.2, label="LoRA v1")
    p2 = os.path.join(ST, "train_history_kadai_v2.json")
    if os.path.exists(p2):
        h2 = json.load(open(p2)); ax.plot(h2["step"], h2["test_loss"], "-o", color=AQUA, lw=2, ms=6, mec="#fcfcfb", mew=1.2, label="LoRA v2")
    ax.set_yscale("log"); ax.set_xlabel("学習 step"); ax.set_ylabel("損失（学習データ上、対数軸）"); ax.legend(frameon=False, fontsize=9)
    ax.set_title("フローマッチングの損失（学習データ上の値。汎化の指標ではない）", fontsize=9.5, loc="left")
    fig.tight_layout(); fig.savefig(os.path.join(OUT, "fig5_training.png")); plt.close(fig)


if __name__ == "__main__":
    fig_tilt(); fig_main(); fig_taxonomy(); fig_perturb(); fig_train()
    print(sorted(os.listdir(OUT)))
