"""提出動画（3 分以内、2 倍速まで可）を組み立てる。静止カード（タイトル・図・結果）と、素 / LoRA の並置クリップをつなぎ、H.264 の MP4 にする。
入力: eval_kadai_closedloop.py --save-frames で保存したエピソード別 npz（agent・wrist・step・q・tilt）。--frame-every 1 だと実時間（20 Hz）で滑らか。
クリップの定義は JSON（--spec）: {"fps": 20, "speed": 2, "segments": [{"type": "card", "title": "...", "lines": ["..."], "sec": 4}, {"type": "image", "path": "...", "caption": "...", "sec": 8},
  {"type": "pair", "title": "...", "left": {"label": "...", "npz": "..."}, "right": {...}, "note": "..."}]}
使い方: python make_video.py --spec video_spec.json --out kadai_video.mp4
個人情報（氏名・所属・メール）をタイトル・文字に入れないこと。
"""
import argparse, json, os
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import imageio

W, H = 1280, 720
BG, INK, SUB, ACC, OK, NG = (252, 252, 251), (11, 11, 11), (82, 81, 78), (42, 120, 214), (27, 130, 90), (200, 60, 50)
FONT = "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"


def font(sz, bold=False):
    return ImageFont.truetype(FONT, sz, index=0)


def card(title, lines, sec, fps):
    im = Image.new("RGB", (W, H), BG); d = ImageDraw.Draw(im)
    d.text((80, 200), title, font=font(44), fill=INK)
    for i, l in enumerate(lines):
        d.text((80, 300 + 46 * i), l, font=font(28), fill=SUB)
    return [np.asarray(im)] * int(sec * fps)


def image_seg(path, caption, sec, fps):
    im = Image.new("RGB", (W, H), BG); d = ImageDraw.Draw(im)
    src = Image.open(path).convert("RGB"); src.thumbnail((W - 80, H - 130), Image.LANCZOS)
    im.paste(src, ((W - src.width) // 2, 20)); d.text((60, H - 80), caption, font=font(24), fill=SUB)
    return [np.asarray(im)] * int(sec * fps)


def panel(npz_frame_agent, npz_frame_wrist, label, info, outcome):
    """左右 1 つ分のパネル（agentview 大 + wrist 小）。"""
    pw = 600
    im = Image.new("RGB", (pw, 540), BG); d = ImageDraw.Draw(im)
    a = Image.fromarray(npz_frame_agent).resize((420, 420), Image.BICUBIC); im.paste(a, (0, 70))
    w = Image.fromarray(npz_frame_wrist).resize((170, 170), Image.BICUBIC); im.paste(w, (430, 70))
    d.text((0, 8), label, font=font(26), fill=INK)
    d.text((430, 250), "手首カメラ", font=font(16), fill=SUB)
    d.text((430, 275), info, font=font(15), fill=SUB)
    if outcome:
        col = OK if outcome[0] else NG
        d.rectangle([430, 330, 598, 372], outline=col, width=3); d.text((438, 340), outcome[1], font=font(19), fill=col)   # 文字が箱からはみ出さない大きさ
    return im


def pair_seg(seg, fps, speed):
    L, R = np.load(seg["left"]["npz"]), np.load(seg["right"]["npz"])
    nl, nr = len(L["step"]), len(R["step"])
    n = max(nl, nr)
    # npz のコマ間隔（step）から実時間を作る。step の間隔は frame-every。fps と speed で動画の長さを決める。
    steps_per_frame = max(1, int(round(speed)))   # 2 倍速 = 2 step ごとに 1 コマ（20 Hz の動画で 40 step/秒）
    frames = []
    ls, rs = L["step"], R["step"]
    tmax = max(ls[-1], rs[-1])
    for t in range(0, int(tmax) + steps_per_frame, steps_per_frame):
        il = int(np.searchsorted(ls, min(t, ls[-1]), side="right") - 1); ir = int(np.searchsorted(rs, min(t, rs[-1]), side="right") - 1)
        fl_done, fr_done = t >= ls[-1], t >= rs[-1]
        left = panel(L["agent"][il], L["wrist"][il], seg["left"]["label"], f"step {int(ls[il])}  q={L['q'][il]:.2f}\n傾き {L['tilt'][il]:.0f}°",
                     (seg["left"]["success"], seg["left"]["result"]) if fl_done else None)
        right = panel(R["agent"][ir], R["wrist"][ir], seg["right"]["label"], f"step {int(rs[ir])}  q={R['q'][ir]:.2f}\n傾き {R['tilt'][ir]:.0f}°",
                      (seg["right"]["success"], seg["right"]["result"]) if fr_done else None)
        im = Image.new("RGB", (W, H), BG); d = ImageDraw.Draw(im)
        d.text((40, 12), seg["title"], font=font(30), fill=INK)
        d.text((W - 260, 18), f"{int(speed)}倍速" if speed > 1 else "実時間", font=font(22), fill=ACC)
        im.paste(left, (40, 90)); im.paste(right, (660, 90))
        d.text((40, H - 70), seg.get("note", ""), font=font(22), fill=SUB)
        frames.append(np.asarray(im))
    frames += [frames[-1]] * int(0.8 * fps)   # 結果を少し見せる
    return frames


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--spec", required=True); ap.add_argument("--out", required=True)
    a = ap.parse_args()
    spec = json.load(open(a.spec)); fps, speed = spec.get("fps", 20), spec.get("speed", 2)
    frames = []
    for seg in spec["segments"]:
        if seg["type"] == "card":
            frames += card(seg["title"], seg.get("lines", []), seg["sec"], fps)
        elif seg["type"] == "image":
            frames += image_seg(seg["path"], seg.get("caption", ""), seg["sec"], fps)
        elif seg["type"] == "pair":
            frames += pair_seg(seg, fps, seg.get("speed", speed))
    dur = len(frames) / fps
    print(f"frames {len(frames)}  duration {dur:.1f} s")
    assert dur <= 180, f"動画が 3 分を超える: {dur:.1f} s"
    w = imageio.get_writer(a.out, fps=fps, codec="libx264", quality=8, macro_block_size=16, pixelformat="yuv420p")
    for f in frames:
        w.append_data(f)
    w.close()
    print("wrote", a.out, round(os.path.getsize(a.out) / 1e6, 1), "MB")


if __name__ == "__main__":
    main()
