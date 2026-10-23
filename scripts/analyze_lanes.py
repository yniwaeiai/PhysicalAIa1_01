"""L1（計画 v2 §17）: 同一初期状態からの t0 分岐 8 本 × 56 窓の手先位置（固定具座標）の揺らぎと成否。CPU のみ。判定は書かず値だけ。
前提: scripts/lanes_frames.py が state/lanes_frames_20261016.npz（窓ごとの固定具の pf, Rf）を作ってある。
出力: state/lanes_20261016.md / .csv、figs/lanes_20261016_{drawer,k4,k6}.png。
"""
import csv, collections, json
from pathlib import Path
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

K = Path(__file__).resolve().parents[1]
BR = K / "state/branches_i3_20261015"
F = np.load(K / "state/lanes_frames_20261016.npz")
STEPS = [8, 13, 18, 23, 30]
TASKS = ["drawer", "k4", "k6"]
rng = np.random.default_rng(0)
plt.rcParams["font.family"] = ["Noto Sans CJK JP", "IPAexGothic", "DejaVu Sans"]


def tofix(wid, x):
    f = F[wid]; return (np.asarray(x) - f[:3]) @ f[3:].reshape(3, 3)   # = Rf^T (x - pf)

# ---- 読み込み
rows = [r for r in csv.DictReader(open(K / "state/branches_i3_20261015.csv")) if r["kind"] == "t0"]
W = collections.OrderedDict()
for r in rows:
    w = W.setdefault(r["wid"], dict(task=r["task"], seed=int(r["seed"]), rep=int(r["rep"]), group=r["group"], ws=int(r["ws"]), P=[], succ=[], clean=[]))
    ee = np.load(BR / f"b_{r['id']}.npz")["ee"]
    w["P"].append(tofix(r["wid"], ee[STEPS])); w["succ"].append(int(r["succ"])); w["clean"].append(int(r["clean"]))
for w in W.values():
    w["P"] = np.array(w["P"]); w["succ"] = np.array(w["succ"]); w["clean"] = np.array(w["clean"])
assert all(len(w["succ"]) == 8 for w in W.values())
L = []
def P(s=""): L.append(s)
csvrows = []

def pct(x): return f"{np.median(x):.2f} [{np.percentile(x,25):.2f}, {np.percentile(x,75):.2f}]"

P("# L1 経路の揺らぎと成否（2026-10-16、`kadai/scripts/analyze_lanes.py`。判定なし、数値のみ）")
P("固定具座標 = 基準 body 座標（entry_geometry.py と同じ手順。窓 trace の sim_state[1] から pf, Rf。窓内の ee 位置を Rf^T(x−pf)）。位置は mm。")
P(f"窓 {len(W)}（{', '.join(f'{t} {sum(1 for w in W.values() if w['task']==t)}' for t in TASKS)}）× 8 本。step = 分岐 ee の添字（t0 = エピソード先頭からの分岐）。成功 = done。")
P()
# ---- 成功本数
P("## 窓の成功本数（8 本中）")
for t in TASKS:
    c = collections.Counter(int(w["succ"].sum()) for w in W.values() if w["task"] == t)
    mixed = sum(v for k, v in c.items() if 0 < k < 8)
    P(f"- {t}: " + ", ".join(f"{k}本:{c[k]}窓" for k in sorted(c)) + f"　（混在 = L1b 対象 {mixed} 窓）")
P()

# ---- L1a
P("## L1a 窓内 SD（mm、8 本、ddof=1。値 = タスク別の窓の 中央値 [Q1, Q3]）")
P("| task | step | SD x | SD y | SD z | 合成 RMS（√ΣVar） | 合成 ≥8mm の窓 | 合成 <3mm の窓 |")
P("|---|---|---|---|---|---|---|---|")
for t in TASKS:
    ws = [w for w in W.values() if w["task"] == t]
    for si, s in enumerate(STEPS):
        sd = np.array([w["P"][:, si].std(0, ddof=1) * 1000 for w in ws])
        rm = np.sqrt((sd ** 2).sum(1))
        P(f"| {t} | {s} | {pct(sd[:,0])} | {pct(sd[:,1])} | {pct(sd[:,2])} | {pct(rm)} | {(rm>=8).sum()}/{len(ws)} | {(rm<3).sum()}/{len(ws)} |")
        for w, r_, v in zip(ws, rm, sd):
            pass
P()
for wid, w in W.items():
    for si, s in enumerate(STEPS):
        sd = w["P"][:, si].std(0, ddof=1) * 1000
        csvrows.append(dict(wid=wid, task=w["task"], group=w["group"], step=s, n_succ=int(w["succ"].sum()), sd_x=sd[0], sd_y=sd[1], sd_z=sd[2], sd_rms=float(np.sqrt((sd**2).sum()))))

# ---- L1b / L1c
def auroc(score, y):
    p = score[y == 1]; n = score[y == 0]
    if len(p) == 0 or len(n) == 0: return float("nan")
    d = p[:, None] - n[None, :]
    return float(((d > 0).sum() + 0.5 * (d == 0).sum()) / d.size)

def direction(ws, si, lab):
    C = np.concatenate([w["P"][:, si] - w["P"][:, si].mean(0) for w in ws]); y = np.concatenate([w[lab] for w in ws])
    u = C[y == 1].mean(0) - C[y == 0].mean(0); return u / np.linalg.norm(u)

def lowo(ws, si, lab, perm=False):
    sc, ys = [], []
    for i, h in enumerate(ws):
        tr = [w for j, w in enumerate(ws) if j != i]
        if perm:
            tr = [dict(P=w["P"], **{lab: rng.permutation(w[lab])}) for w in tr]
        u = direction(tr, si, lab)
        sc.append((h["P"][:, si] - h["P"][:, si].mean(0)) @ u); ys.append(h[lab])
    sc = np.concatenate(sc); ys = np.concatenate(ys)
    return auroc(sc, ys), sc, ys

P("## L1b 経路→成否（窓平均で中心化、leave-one-window-out の u、成功も失敗もある窓のみでプール）")
P("AUROC（成功=正）。perm = 学習窓内でラベルを置換した LOWO AUROC の帰無（100 回、平均±SD）。u = 全混在窓で作った単位ベクトル（固定具座標の x,y,z 成分）。")
P("| task | 混在窓 | 本数 | step | LOWO AUROC | perm 帰無 | AUROC(clean=正) | u（全窓, x/y/z） | 軸別の窓内 AUROC 平均（x/y/z） |")
P("|---|---|---|---|---|---|---|---|---|")
U = {}; PROJ = {}
for t in TASKS:
    ws = [w for w in W.values() if w["task"] == t and 0 < w["succ"].sum() < 8]
    wc = [w for w in W.values() if w["task"] == t and 0 < w["clean"].sum() < 8]
    nb = 8 * len(ws)
    for si, s in enumerate(STEPS):
        a, sc, ys = lowo(ws, si, "succ")
        ac, _, _ = lowo(wc, si, "clean") if len(wc) >= 3 else (float("nan"),) * 3
        nulls = [lowo(ws, si, "succ", perm=True)[0] for _ in range(100)] if s == 13 else []
        u = direction(ws, si, "succ"); U[(t, s)] = u
        if s == 13: PROJ[t] = (ws, u)
        ax = []
        for c in range(3):
            ax.append(np.nanmean([auroc(w["P"][:, si, c], w["succ"]) for w in ws]))
        nl = f"{np.mean(nulls):.3f}±{np.std(nulls):.3f}" if nulls else "—"
        P(f"| {t} | {len(ws)} | {nb} | {s} | {a:.3f} | {nl} | {ac:.3f} | ({u[0]:+.2f}, {u[1]:+.2f}, {u[2]:+.2f}) | {ax[0]:.2f}/{ax[1]:.2f}/{ax[2]:.2f} |")
        csvrows.append(dict(task=t, step=s, wid="LOWO", n_succ=len(ws), sd_x=a, sd_y=ac, sd_z=u[0], sd_rms=float("nan")))
P()

# 窓ごとの AUROC（参考: 符号の一貫性）
P("## 参考: step 13 の窓内 AUROC（u_全窓への射影、窓ごと、in-sample）の分布")
for t in TASKS:
    ws, u = PROJ[t]; si = STEPS.index(13)
    a = np.array([auroc((w["P"][:, si] - w["P"][:, si].mean(0)) @ u, w["succ"]) for w in ws])
    P(f"- {t}: 窓 {len(ws)}、窓内 AUROC 中央値 {np.median(a):.2f}、>0.5 の窓 {(a>0.5).sum()}、<0.5 の窓 {(a<0.5).sum()}、=0.5 {(a==0.5).sum()}")
P()

P("## L1c 片側性（step 13。窓平均から見た u への射影 proj。正 = 成功側）")
P("in = 全混在窓で作った u への射影（in-sample。失敗の中央値は定義上負になりやすい）、lowo = 残した窓の射影（その窓を除いた u）。")
P("| task | 射影 | 失敗 n | 失敗 proj<0 の割合 | 失敗 proj 中央値 (mm) | 成功 n | 成功 proj<0 の割合 | 成功 proj 中央値 (mm) | |proj| 失敗 中央値 [Q1,Q3] | |proj| 成功 中央値 [Q1,Q3] | |proj|成功/失敗 比 | 窓ごと: 失敗が全て proj<0 の窓 | 窓ごと: 成功が全て proj>0 の窓 |")
P("|---|---|---|---|---|---|---|---|---|---|---|---|---|")
si = STEPS.index(13)
for t in TASKS:
    ws, u = PROJ[t]
    for kind in ("in", "lowo"):
        pr, ys, pw = [], [], []
        for i, w in enumerate(ws):
            uu = u if kind == "in" else direction([x for j, x in enumerate(ws) if j != i], si, "succ")
            p = (w["P"][:, si] - w["P"][:, si].mean(0)) @ uu * 1000
            pr.append(p); ys.append(w["succ"]); pw.append((w["succ"], p))
        pr = np.concatenate(pr); ys = np.concatenate(ys)
        pf_, ps_ = pr[ys == 0], pr[ys == 1]
        allf = sum(1 for sc, p in pw if (p[sc == 0] < 0).all()); alls = sum(1 for sc, p in pw if (p[sc == 1] > 0).all())
        P(f"| {t} | {kind} | {len(pf_)} | {(pf_<0).mean():.2f} | {np.median(pf_):+.2f} | {len(ps_)} | {(ps_<0).mean():.2f} | {np.median(ps_):+.2f} | {pct(np.abs(pf_))} | {pct(np.abs(ps_))} | {np.median(np.abs(ps_))/np.median(np.abs(pf_)):.2f} | {allf}/{len(ws)} | {alls}/{len(ws)} |")
P()

# ---- 補助: I4/I5 の組
P("## 補助: I4/I5 の組（成功 ep − 失敗 ep、step 13 の手先位置、各 ep 自身の固定具座標）と u（step 13、全窓）の内積")
i4 = list(csv.DictReader(open(K / "state/branches_i4_20261015.csv")))
pairs = {}
for r in i4:
    fe = r["fwid"].rsplit("_t", 1)[0]; pairs[(fe, int(r["srep"]))] = r["task"]
import sys
sys.path.insert(0, str(K / "scripts"))
res = collections.defaultdict(list); detail = []
def ep_ee13(task, seed, rep):
    d = np.load(K / f"state/trace_aprime/{task}_s{seed}_r{rep}.npz", allow_pickle=True)
    f = np.r_[np.asarray(d["ref_pos0"])]   # 固定具の位置は窓 frame から（t0 窓があれば）
    return d
FR = {}
def frame_of(task, seed, rep):
    wid = f"{task}_s{seed}_r{rep}_t0"
    return F[wid] if wid in F.files else None
nmiss = 0
for (fe, srep), task in sorted(pairs.items()):
    t_, s_, r_ = fe.split("_"); seed = int(s_[1:]); frep = int(r_[1:])
    d_f = np.load(K / f"state/trace_aprime/{task}_s{seed}_r{frep}.npz", allow_pickle=True)
    d_s = np.load(K / f"state/trace_aprime/{task}_s{seed}_r{srep}.npz", allow_pickle=True)
    ff, fs = frame_of(task, seed, frep), frame_of(task, seed, srep)
    if ff is None or fs is None:
        # t0 窓が無い ep は fixture frame を同 seed の別 ep で代用（ref_pos0 が同じ場合のみ）
        cand = [x for x in F.files if x.startswith(f"{task}_s{seed}_")]
        fx = ff if ff is not None else (fs if fs is not None else (F[cand[0]] if cand else None))
        if fx is None: nmiss += 1; continue
        ff = ff if ff is not None else fx; fs = fs if fs is not None else fx
        same = bool(np.allclose(ff, fx, atol=1e-6))
    pe_f = (d_f["ee"][13] - ff[:3]) @ ff[3:].reshape(3, 3); pe_s = (d_s["ee"][13] - fs[:3]) @ fs[3:].reshape(3, 3)
    dv = (pe_s - pe_f)
    u = U[(task, 13)]
    res[task].append(float(dv @ u)); detail.append((task, fe, srep, dv * 1000, float(dv @ u) * 1000, float(np.linalg.norm(ff - fs))))
P(f"(fwid から窓の接尾辞を除いた失敗 ep × 成功 ep の一意な組 {len(pairs)}、欠損 {nmiss}。I4 csv は fwid×srep で {len({(x['fwid'],x['srep']) for x in i4})} 行（t13/t23 窓が同一 ep を共有）。)")
P("| task | 組数 | 内積>0（位置差が u と同向 = 成功側）の割合 | 内積 中央値 (mm) | 位置差ノルム 中央値 (mm) | 失敗/成功 ep の固定具フレーム差 最大 |")
P("|---|---|---|---|---|---|")
for t in TASKS:
    v = np.array(res[t]) * 1000
    if len(v) == 0: continue
    nz = np.median([np.linalg.norm(x[3]) for x in detail if x[0] == t]); fd = max(x[5] for x in detail if x[0] == t)
    P(f"| {t} | {len(v)} | {(v>0).mean():.2f} | {np.median(v):+.2f} | {nz:.1f} | {fd:.4g} |")
allv = np.concatenate([np.array(res[t]) for t in TASKS if res[t]]) * 1000
P(f"| 全体 | {len(allv)} | {(allv>0).mean():.2f} | {np.median(allv):+.2f} | | |")
P()
for t, fe, sr, dv, dp, _ in detail[:0]: pass


# ---- L1d 教師 demo との照合（state/lanes_teacher_20261016.npz = scripts/lanes_teacher.py）
T = np.load(K / "state/lanes_teacher_20261016.npz")
P("## L1d 教師 demo（kadai_demos_v1）との照合（固定具座標、Mahalanobis 距離²、教師の経験分布内パーセンタイル）")
P("配置は --mix の規則から再構成（state[0] の手先と再構成 sim の初期手先が 1 mm 以内に一致した demo のみ使用。drawer 349/349、k4 172/172、k6 383/389 → 除外 6）。")
P("対応: 分岐 ee[i] = i+1 回行動後の位置 ⇒ 教師 state[i+1] と対応づける（主 = 分岐 step 13 ↔ 教師 state[14]。参考 = state[13]）。教師の共分散は demo プール（摂動 none/mid/strong 混合）。")
P("percentile = その分岐の D² が教師 demo 自身の D² のうち何割より大きいか（0.5 = 教師の中央値と同程度、1 = 教師より外れる）。")
P("| task | 教師対応 | 教師 n | 教師平均 (x,y,z mm) | 教師 SD (x,y,z mm) | 分岐 平均−教師平均 (x,y,z mm) | 成功分岐 n | 成功 percentile 中央値 | 成功 >0.95 | 失敗分岐 n | 失敗 percentile 中央値 | 失敗 >0.95 | AUROC(失敗の D² > 成功の D²) |")
P("|---|---|---|---|---|---|---|---|---|---|---|---|---|")
si = STEPS.index(13)
for t in TASKS:
    for off in (14, 13):
        X = T[t][:, off] * 1000; ok = T[t + "_err"] < 1e-3; X = X[ok]; X = X[~np.isnan(X).any(1)]
        mu = X.mean(0); C = np.cov(X.T); Ci = np.linalg.inv(C)
        d2t = np.einsum("ij,jk,ik->i", X - mu, Ci, X - mu)
        B = np.concatenate([w["P"][:, si] for w in W.values() if w["task"] == t]) * 1000
        yb = np.concatenate([w["succ"] for w in W.values() if w["task"] == t])
        d2 = np.einsum("ij,jk,ik->i", B - mu, Ci, B - mu)
        pc = np.array([(d2t < v).mean() for v in d2])
        a = auroc(d2, 1 - yb)
        P(f"| {t} | state[{off}]{' (主)' if off==14 else ''} | {len(X)} | ({mu[0]:.1f}, {mu[1]:.1f}, {mu[2]:.1f}) | ({X[:,0].std():.1f}, {X[:,1].std():.1f}, {X[:,2].std():.1f}) | ({(B-mu).mean(0)[0]:+.1f}, {(B-mu).mean(0)[1]:+.1f}, {(B-mu).mean(0)[2]:+.1f}) | {(yb==1).sum()} | {np.median(pc[yb==1]) if (yb==1).any() else float('nan'):.2f} | {(pc[yb==1]>.95).mean() if (yb==1).any() else float('nan'):.2f} | {(yb==0).sum()} | {np.median(pc[yb==0]):.2f} | {(pc[yb==0]>.95).mean():.2f} | {a:.3f} |")
P("注: 教師 (scripted expert) の step 14 の位置は z 方向の SD が 0.6〜0.8 mm（drawer/k4）と極端に小さく、共分散はほぼ特異。分岐の SD は 10 mm 級なので、percentile は飽和（≈1）しやすい。")
P()
# ---- 図
for t in TASKS:
    ws, u = PROJ[t]; si = STEPS.index(13)
    fig, ax = plt.subplots(1, 3, figsize=(15, 4.6))
    C = np.concatenate([w["P"][:, si] - w["P"][:, si].mean(0) for w in ws]) * 1000; y = np.concatenate([w["succ"] for w in ws])
    for a_, (i, j, nm) in zip(ax[:2], [(0, 1, "x–y"), (0, 2, "x–z")]):
        a_.scatter(C[y == 0, i], C[y == 0, j], s=10, c="#c0392b", label="失敗", alpha=.7)
        a_.scatter(C[y == 1, i], C[y == 1, j], s=10, c="#2471a3", label="成功", alpha=.7)
        a_.arrow(0, 0, u[i] * 10, u[j] * 10, color="k", width=.15, head_width=.8)
        a_.set_xlabel(f"{'xyz'[i]} (mm)"); a_.set_ylabel(f"{'xyz'[j]} (mm)"); a_.set_title(f"{t} step13 窓平均から（{nm}、黒 = u×10mm）"); a_.set_aspect("equal"); a_.legend()
    pr = C @ u
    ax[2].hist([pr[y == 0], pr[y == 1]], bins=25, color=["#c0392b", "#2471a3"], label=["失敗", "成功"]); ax[2].axvline(0, c="k"); ax[2].set_xlabel("u への射影 (mm)"); ax[2].legend()
    ax[2].set_title(f"{t} 混在窓 {len(ws)} の射影")
    fig.tight_layout(); fig.savefig(K / f"figs/lanes_20261016_{t}.png", dpi=110); plt.close(fig)

(K / "state/lanes_20261016.md").write_text("\n".join(L) + "\n")
keys = ["wid", "task", "group", "step", "n_succ", "sd_x", "sd_y", "sd_z", "sd_rms"]
with open(K / "state/lanes_20261016.csv", "w", newline="") as f:
    wr = csv.DictWriter(f, fieldnames=keys); wr.writeheader(); [wr.writerow({k: r.get(k, "") for k in keys}) for r in csvrows]
print("\n".join(L))
