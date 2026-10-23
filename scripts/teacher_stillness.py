"""教師デモ（kadai_demos_v1/v2）の静止区間の測定。読み取り専用。出力: kadai/state/teacher_stillness_20261014.json"""
import glob, json, sys
from collections import defaultdict
import h5py, numpy as np

SEGS = {"d": ["H_PRE", "H_IN", "H_HOOK", "H_PULL"], "k4": ["C_PRE", "C_LOW", "C_PUSH"], "k6": ["M_PRE", "M_LOW", "M_PUSH"]}
CONTACT = {"d": {"H_HOOK", "H_PULL"}, "k4": {"C_PUSH"}, "k6": {"M_PUSH"}}
NAME = {"d": "中段引き出しを開ける(libero_goal)", "k4": "下段引き出しを閉める(K4)", "k6": "電子レンジを閉める(K6)"}
TH = (0.05, 0.1)

J = defaultdict(dict)
for f in glob.glob("kadai/state/gen_v*_*.json"):
    v = f.split("gen_")[1][:2]
    t = "d" if ("drawer" in f or "hook" in f) else "k4" if ("k4" in f or "push" in f) else "k6"
    for r in json.load(open(f)):
        J[(v, t)][r["seed"]] = r

def task_of(n):
    return "k4" if "bottom" in n else "k6" if "microwave" in n else "d"

def phases(T, trans, t):
    segs = SEGS[t]; lab = np.empty(T, dtype=object); prev = 0
    for i, (n, te, w) in enumerate(trans):
        lab[prev:te + 1] = n; prev = te + 1
    nxt = segs[len(trans)] if len(trans) < len(segs) else "HOLD"
    lab[prev:] = nxt
    return lab

def frac(m): return float(np.mean(m)) if len(m) else float("nan")
def q(x, p): return float(np.percentile(x, p)) if len(x) else float("nan")

out = {}
for grp in ("v1", "v2_strong", "v2_base"):
    pass
files = sorted(glob.glob("data/kadai_demos_v*/*/*.hdf5"))
agg = {}
for f in files:
    ver = "v1" if "demos_v1" in f else "v2"
    n = f.split("/")[-1]; t = task_of(n)
    strong = "_strong_" in n
    grp = "v1" if ver == "v1" else ("v2_strong" if strong else "v2_base_copy_of_v1")
    if grp == "v2_base_copy_of_v1":
        continue  # 後で v1 と同一であることを検査
    key = (grp, t); d = agg.setdefault(key, dict(L=[], A=[], lab=[], eef=[], trail=[], trail_strict=[], trans=[], ep_first5=[], seeds=0))
    h = h5py.File(f, "r")
    for k in h:
        s = int(k[3:]); a = h[k]["actions"][:].astype(np.float64); st = h[k]["state"][:]
        r = J[("v1" if ver == "v1" else "v2", t)][s] if (("v1" if ver == "v1" else "v2"), t) in J and s in J[("v1" if ver == "v1" else "v2", t)] else J[("v1", t)][s]
        assert r["T"] == a.shape[0]
        lab = phases(len(a), r["trans"], t)
        d["L"].append(len(a)); d["A"].append(a); d["lab"].append(lab); d["eef"].append(st[:, :3]); d["trans"].append(r["trans"])
        still = (np.linalg.norm(a[:, :3], axis=1) < TH[1]) & (np.linalg.norm(a[:, 3:6], axis=1) < TH[1])
        stricts = (np.linalg.norm(a[:, :3], axis=1) < TH[0]) & (np.linalg.norm(a[:, 3:6], axis=1) < TH[0])
        for m, lst in ((still, d["trail"]), (stricts, d["trail_strict"])):
            c = 0
            for x in m[::-1]:
                if x: c += 1
                else: break
            lst.append(c)
# v2 base が v1 と同一かの検査
same = {}
for t, fn in (("d", "libero_goal/open_the_middle_drawer_of_the_cabinet"), ("k4", "libero_90/KITCHEN_SCENE4_close_the_bottom_drawer_of_the_cabinet"), ("k6", "libero_90/KITCHEN_SCENE6_close_the_microwave")):
    a = h5py.File(f"data/kadai_demos_v1/{fn}.hdf5", "r"); b = h5py.File(f"data/kadai_demos_v2/{fn}.hdf5", "r")
    same[t] = bool(set(a) == set(b) and all(np.array_equal(a[k]["actions"][:], b[k]["actions"][:]) for k in a))

def dist(x): return dict(n=len(x), min=q(x, 0), p25=q(x, 25), med=q(x, 50), p75=q(x, 75), max=q(x, 100), mean=float(np.mean(x)) if len(x) else float("nan"))

res = dict(v2_base_identical_to_v1=same, thresholds=TH, tasks={})
for (grp, t), d in sorted(agg.items()):
    A = np.concatenate(d["A"]); lab = np.concatenate(d["lab"]); pn = np.linalg.norm(A[:, :3], axis=1); rn = np.linalg.norm(A[:, 3:6], axis=1)
    pos_in = np.concatenate([np.arange(len(a)) / len(a) for a in d["A"]]); idx = np.concatenate([np.arange(len(a)) for a in d["A"]])
    e = dict(n_ep=len(d["L"]), n_frames=int(len(A)), length=dist(d["L"]), pos_norm_median=q(pn, 50), rot_norm_median=q(rn, 50),
             frac_pos_lt=dict(zip(map(str, TH), [frac(pn < x) for x in TH])), frac_rot_lt=dict(zip(map(str, TH), [frac(rn < x) for x in TH])),
             frac_both_lt=dict(zip(map(str, TH), [frac((pn < x) & (rn < x)) for x in TH])))
    still = (pn < TH[1]) & (rn < TH[1]); still5 = (pn < TH[0]) & (rn < TH[0])
    ph = {}
    for p in SEGS[t] + ["HOLD"]:
        m = lab == p
        if m.sum() == 0: continue
        ph[p] = dict(frames=int(m.sum()), share_of_frames=float(m.mean()), pos_med=q(pn[m], 50), rot_med=q(rn[m], 50), still01=frac(still[m]), still005=frac(still5[m]),
                     n_ep_with=int(sum((l == p).any() for l in d["lab"])), len_dist=dist([int((l == p).sum()) for l in d["lab"] if (l == p).any()]),
                     n_timeout=int(sum(1 for tr in d["trans"] for (n_, _, w) in tr if n_ == p and w == "timeout")))
    e["by_phase"] = ph
    e["still_share_by_loc_01"] = dict(first5=float(still[idx < 5].sum() / max(still.sum(), 1)), contact=float(still[np.isin(lab, list(CONTACT[t]))].sum() / max(still.sum(), 1)),
                                     precontact=float(still[~np.isin(lab, list(CONTACT[t]) + ["HOLD"]) & (idx >= 5)].sum() / max(still.sum(), 1)), hold=float(still[lab == "HOLD"].sum() / max(still.sum(), 1)))
    e["first5_still_frac_01"] = frac(still[idx < 5]); e["first5_pos_med"] = q(pn[idx < 5], 50)
    e["trailing_still_run_01"] = dist(d["trail"]); e["trailing_still_run_005"] = dist(d["trail_strict"])
    e["frac_ep_trailing_ge1_01"] = float(np.mean(np.array(d["trail"]) >= 1)); e["frac_ep_trailing_ge5_01"] = float(np.mean(np.array(d["trail"]) >= 5))
    e["last_step_pos_norm_med"] = q([np.linalg.norm(a[-1, :3]) for a in d["A"]], 50)
    e["last5_pos_norm_med"] = q(np.concatenate([np.linalg.norm(a[-5:, :3], axis=1) for a in d["A"]]), 50)
    # item 6 代理: 接触局面の中の進み具合
    cm = np.isin(lab, list(CONTACT[t]))
    if t == "d":
        # H_PULL 開始からの手先の累積移動量（cm）を -q の代理にする（q は記録に無い）
        disp = np.concatenate([np.where(l == "H_PULL", np.linalg.norm(ee - ee[np.argmax(l == "H_PULL")], axis=1) if (l == "H_PULL").any() else 0, np.nan) for l, ee in zip(d["lab"], d["eef"])])
        bands = [(0, .02), (.02, .04), (.04, .08), (.08, .13), (.13, 1)]
        e["item6_proxy_disp_from_pull_start_m"] = {f"{lo}-{hi}": dict(n=int(((disp >= lo) & (disp < hi)).sum()), pos_med=q(pn[(disp >= lo) & (disp < hi)], 50), still01=frac(still[(disp >= lo) & (disp < hi)])) for lo, hi in bands}
    prog = np.concatenate([np.where(np.isin(l, list(CONTACT[t])), np.cumsum(np.isin(l, list(CONTACT[t]))) / max(np.isin(l, list(CONTACT[t])).sum(), 1), np.nan) for l in d["lab"]])
    e["item6_proxy_contact_progress_quintiles"] = {f"{i*20}-{i*20+20}%": dict(n=int(((prog > i/5) & (prog <= (i+1)/5)).sum()), pos_med=q(pn[(prog > i/5) & (prog <= (i+1)/5)], 50), still01=frac(still[(prog > i/5) & (prog <= (i+1)/5)])) for i in range(5)}
    res["tasks"][f"{grp}|{t}"] = dict(name=NAME[t], **e)
json.dump(res, open("kadai/state/teacher_stillness_20261014.json", "w"), indent=1, ensure_ascii=False)
print(json.dumps(res, indent=1, ensure_ascii=False))

# --- 追加: 手先の実速度（状態の差分）で見た静止。行動が非ゼロでも接触でブロックされて動かない step を拾う
vel = {}
for (grp, t), d in sorted(agg.items()):
    sp = {p: [] for p in SEGS[t] + ["HOLD"]}; tail = []
    for lab_, ee in zip(d["lab"], d["eef"]):
        v = np.r_[np.linalg.norm(np.diff(ee, axis=0), axis=1), np.nan]  # step i の行動後の移動（最終 step は nan）
        for p in sp: sp[p].append(v[lab_ == p])
        c = 0
        for x in v[:-1][::-1]:
            if x < 0.0005: c += 1
            else: break
        tail.append(c)
    vel[f"{grp}|{t}"] = dict(by_phase={p: dict(speed_med_mm=float(np.nanmedian(np.concatenate(x)) * 1000), frac_lt_0p5mm=float(np.nanmean(np.concatenate(x) < 0.0005)), frac_lt_1mm=float(np.nanmean(np.concatenate(x) < 0.001)), n=int(sum(len(y) for y in x))) for p, x in sp.items() if sum(len(y) for y in x)},
                             trailing_run_speed_lt_0p5mm=dist(tail))
res["eef_speed"] = vel
json.dump(res, open("kadai/state/teacher_stillness_20261014.json", "w"), indent=1, ensure_ascii=False)
for k, v in vel.items():
    print(k, {p: (round(x["speed_med_mm"], 2), round(x["frac_lt_0p5mm"], 3), round(x["frac_lt_1mm"], 3)) for p, x in v["by_phase"].items()}, "trail", v["trailing_run_speed_lt_0p5mm"]["mean"], v["trailing_run_speed_lt_0p5mm"]["max"])
