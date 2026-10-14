"""評価の失敗例からコマ送り画像（コンタクトシート）を作る。

入力: 評価結果の jsonl（<prefix>*.jsonl）と、--save-frames で保存したエピソード別 npz（<task>_s<seed>.npz）。
各タスクについて失敗を型に分け、型ごとに代表（その型の傾き中央値に最も近いエピソード）を選んで最大 3 件を出す。
  型 A  timeout_no_contact : 成功せず、対象外物体は 1 mm 以内（動けない・迷う）
  型 B  timeout_collision  : 成功せず、対象外物体を 1 mm 超動かした
  型 C  success_collision  : ゴールは達成したが、対象外物体を 1 mm 超動かした
型が 3 に満たなければ、最も多い型から傾きの離れたエピソードで補う。
1 件 = 1 枚の PNG: 上にヘッダ（タスク・seed・結果・ステップ・傾き・q・対象外物体の移動）、下に 12 コマ（6 列 × 2 行）。
各コマは上段 agentview・下段 wrist（どちらも上下左右を直した向き）、左上に step・q・傾きを重ねる。
使い方: python make_failure_sheets.py <評価jsonlの接頭辞> <フレームのディレクトリ> <出力ディレクトリ> [件数=3]
"""
import glob, json, os, sys
import numpy as np
from PIL import Image, ImageDraw, ImageFont

CELL = 192
NCOL, NROW = 6, 2
TASK_TITLE = {"drawer": "Open the middle drawer (libero_goal)", "k4": "Close the bottom drawer (libero_90 K4)", "k6": "Close the microwave (libero_90 K6)"}


def font(size):
    for p in ("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", "/usr/share/fonts/dejavu/DejaVuSans.ttf"):
        if os.path.exists(p):
            return ImageFont.truetype(p, size)
    return ImageFont.load_default()


def ftype(r):
    if not r["success"]:
        return "A_timeout_no_contact" if r["disp_mm"] <= 1.0 else "B_timeout_collision"
    return "C_success_collision" if r["disp_mm"] > 1.0 else None


def pick(rs, k):
    fails = [r for r in rs if ftype(r)]
    by = {}
    for r in fails:
        by.setdefault(ftype(r), []).append(r)
    chosen = []
    for t in sorted(by, key=lambda t: -len(by[t])):
        if len(chosen) >= k:
            break
        med = np.median([r["tilt_med"] for r in by[t]])
        chosen.append(min(by[t], key=lambda r: abs(r["tilt_med"] - med)))
    pool = [r for t in sorted(by, key=lambda t: -len(by[t])) for r in by[t] if r not in chosen]
    while len(chosen) < k and pool:
        # 既に選んだものと傾きが最も離れたものを足す
        r = max(pool, key=lambda r: min(abs(r["tilt_med"] - c["tilt_med"]) for c in chosen) if chosen else 0)
        chosen.append(r); pool.remove(r)
    return chosen, len(fails), {t: len(v) for t, v in by.items()}


def sheet(r, npz, path):
    d = np.load(npz)
    n = len(d["step"])
    idx = np.unique(np.linspace(0, n - 1, NCOL * NROW).round().astype(int))
    fs, fb = font(13), font(11)
    W = CELL * NCOL
    H0 = 78
    img = Image.new("RGB", (W, H0 + CELL * 2 * NROW), (20, 20, 24))
    dr = ImageDraw.Draw(img)
    outcome = {"A_timeout_no_contact": "FAIL: not completed in %d steps (no contact with other objects)",
               "B_timeout_collision": "FAIL: not completed in %d steps, and moved another object",
               "C_success_collision": "Goal reached in %d steps, but moved another object (collision)"}[ftype(r)] % r["steps"]
    dr.text((8, 6), f"{TASK_TITLE[r['task']]}   |   seed {r['seed']}   |   perturbation (robot noise rad, cabinet shift m, yaw deg) = {tuple(r['level'])}", fill=(255, 255, 255), font=fs)
    dr.text((8, 26), outcome, fill=(255, 120, 100), font=fs)
    obj = f", moved '{r['top']}' by {r['disp_mm']:.0f} mm (L1)" if r["disp_mm"] > 1.0 else ""
    dr.text((8, 46), f"steps {r['steps']}   wrist tilt median {r['tilt_med']:.0f} deg / max {r['tilt_max']:.0f} deg   joint q {r['q_start']:.3f} -> {r['q_end']:.3f}{obj}", fill=(200, 220, 255), font=fs)
    dr.text((8, 62), "each cell: top = agentview, bottom = wrist camera (both upright).  t = control step, q = target joint, tilt = angle of the tool axis from straight down",
            fill=(150, 150, 160), font=fb)
    for k, i in enumerate(idx):
        col, row = k % NCOL, k // NCOL
        x, y = col * CELL, H0 + row * CELL * 2
        for j, cam in enumerate(("agent", "wrist")):
            im = Image.fromarray(d[cam][i]).resize((CELL, CELL), Image.BICUBIC)
            img.paste(im, (x, y + j * CELL))
        t = ImageDraw.Draw(img)
        t.rectangle([x, y, x + CELL, y + 16], fill=(0, 0, 0))
        t.text((x + 3, y + 1), f"t={int(d['step'][i])}  q={d['q'][i]:.3f}  tilt={d['tilt'][i]:.0f}", fill=(255, 255, 0), font=fb)
    img.save(path)


def main():
    prefix, fdir, odir = sys.argv[1], sys.argv[2], sys.argv[3]
    k = int(sys.argv[4]) if len(sys.argv) > 4 else 3
    rows = []
    for p in sorted(glob.glob(prefix + "*.jsonl")):
        rows += [json.loads(l) for l in open(p) if l.strip()]
    os.makedirs(odir, exist_ok=True)
    lines = ["| task | case | seed | type | steps | tilt med/max | q | other-object move | file |", "|---|---|---|---|---|---|---|---|---|"]
    for task in ("drawer", "k4", "k6"):
        rs = [r for r in rows if r["task"] == task and not r.get("error") and os.path.exists(os.path.join(fdir, f"{task}_s{r['seed']}.npz"))]
        ch, nfail, bytype = pick(rs, k)
        print(f"{task}: episodes {len(rs)}, failures {nfail} {bytype}, chosen seeds {[c['seed'] for c in ch]}")
        for c, r in enumerate(ch, 1):
            fn = f"{task}_case{c}_seed{r['seed']}_{ftype(r)}.png"
            sheet(r, os.path.join(fdir, f"{task}_s{r['seed']}.npz"), os.path.join(odir, fn))
            lines.append(f"| {task} | {c} | {r['seed']} | {ftype(r)} | {r['steps']} | {r['tilt_med']:.0f}/{r['tilt_max']:.0f} | {r['q_start']:.3f}→{r['q_end']:.3f} | {r['top']} {r['disp_mm']:.0f} mm | {fn} |")
    open(os.path.join(odir, "INDEX.md"), "w").write("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
