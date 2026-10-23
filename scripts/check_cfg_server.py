"""誘導サーバ（serve_policy_cfg.py, 既定 port 18042）の操作チェック（計画 §3-4）。★GPU のある Pod で使う（手元では --help まで）。
  (a) 同じ obs・同じ _noise_seed を、通常サーバと誘導サーバ(w=0) に送って |Δ| が線内（正規化行動空間で median<1e-3 かつ max<1e-2。手動ループと compile 済みの bf16 差を許す）。
      ★比較対象の通常サーバは c1 の merge 済み重みを serve_policy_noise.py で別ポート（例 18043）に立てたもの。v1 (18040) とは重みが違うので比較にならない。
      通常サーバを立てられない場合は --plain-port を省略: (b) だけ行う。
  (b) w = 0 / 1.5 / 3 の行動の大きさ（|Δ位置| = 行動チャンク先頭 5 step の位置成分ノルムの中央値）と、‖v_s−v_f‖/‖v_s‖ の step 別（応答の cfg_ratio）を数 obs で。w=3 が w=0 の 2 倍を超えるか。
obs は合成ではなく **実際の trace の画像・状態**（--trace-file の img_agent[0] / img_wrist[0] = 方策向き 128px）から作る（学習・評価と同じ resize_with_pad(224)）。
使い方: python check_cfg_server.py --trace-file data/contrast/trace_c/drawer_s8000_r0.npz --task drawer --cfg-port 18042 [--plain-port 18043]
"""
import argparse, sys
import numpy as np

INSTR = {"drawer": "Open the middle layer of the drawer", "k4": "close the bottom drawer of the cabinet", "k6": "close the microwave"}


def make_obs(d, i, task):
    from openpi_client import image_tools
    from kadai_record import quat2axisangle
    ca = int(d["chunk_at"][i]); prev = max(ca - 1, 0)
    st = np.concatenate([d["ee"][prev], quat2axisangle(d["quat"][prev]), d["grip"][prev]])
    f = lambda im: image_tools.convert_to_uint8(image_tools.resize_with_pad(np.ascontiguousarray(im), 224, 224))
    return {"observation/image": f(d["img_agent"][i]), "observation/wrist_image": f(d["img_wrist"][i]), "observation/state": st.astype(np.float64), "prompt": INSTR[task]}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--trace-file", required=True, nargs="+")
    ap.add_argument("--task", default="drawer", choices=list(INSTR))
    ap.add_argument("--host", default="localhost")
    ap.add_argument("--cfg-port", type=int, default=18042)
    ap.add_argument("--plain-port", type=int, default=0, help="同じ重みの別の通常サーバ（serve_policy_noise.py）。0 = 誘導サーバ自身の通常経路（_cfg_* を付けない要求は Policy.infer に落ちる）と比べる")
    ap.add_argument("--n-obs", type=int, default=3, help="trace あたりの obs 数（チャンク先頭を等間隔に）")
    a = ap.parse_args()
    import functools, websockets.sync.client as _wsc
    _wsc.connect = functools.partial(_wsc.connect, ping_interval=None)
    from openpi_client import websocket_client_policy as W
    sys.path.insert(0, __import__("os").path.dirname(__import__("os").path.abspath(__file__)))
    cfg = W.WebsocketClientPolicy(a.host, a.cfg_port)
    plain = W.WebsocketClientPolicy(a.host, a.plain_port) if a.plain_port else cfg   # 既定: 同じサーバの通常経路（_cfg_* 無し = Policy.infer）と比べる
    SUF_S, SUF_F = " (outcome: success)", " (outcome: failure)"
    diffs = []; bad = 0; mags = {0.0: [], 1.5: [], 3.0: []}; ratios = []
    k = 0
    for tf in a.trace_file:
        d = np.load(tf, allow_pickle=True)
        for i in np.linspace(0, len(d["chunk_at"]) - 1, a.n_obs).astype(int):
            base = make_obs(d, int(i), a.task); seed = 1000 + k; k += 1
            for w in (0.0, 1.5, 3.0):
                o = dict(base); o["prompt"] = base["prompt"] + SUF_S
                o.update(_cfg_w=w, _cfg_neg_prompt=base["prompt"] + SUF_F, _noise_seed=seed)
                r = cfg.infer(o); act = np.asarray(r["actions"])
                mags[w].append(float(np.median(np.linalg.norm(act[:5, :3], axis=1))))
                if w == 0.0:
                    ratios.append(np.asarray(r["cfg_ratio"]))
                    if True:
                        p = dict(base); p["prompt"] = base["prompt"] + SUF_S; p["_noise_seed"] = seed
                        ap_ = np.asarray(plain.infer(p)["actions"]); dd = np.abs(ap_ - act)
                        diffs.append((k - 1, int(i), float(np.median(dd)), float(dd.max())))
    print("\n(b) 位置行動ノルム（先頭 5 step の中央値）の obs 平均:", {w: round(float(np.mean(v)), 4) for w, v in mags.items()})
    r0 = np.mean(mags[0.0]); r3 = np.mean(mags[3.0])
    print(f"    w=3 / w=0 = {r3 / max(r0, 1e-9):.2f}（> 2 なら格子を {{0, 0.75, 1.5}} に下げる）")
    R = np.stack([x for x in ratios if x.size]) if any(x.size for x in ratios) else None
    if R is not None:
        print("    ‖v_s−v_f‖/‖v_s‖ step別中央値:", np.round(np.median(R, axis=0), 3).tolist())
    # (a) 手動ループ（denoise_step 直呼び）と compile 済み sample_actions は bf16 の数値で一致しない。線: 全 obs 平均でなく各 obs で 中央値 < 1e-3 かつ 最大 < 1e-2
    #     （返る行動は出力変換後 = env に渡す行動空間。正規化前後の差は quantile 幅で数倍まで）
    print("\n(a) w=0（誘導経路）vs 通常経路（_cfg_* 無し、同じ _noise_seed）:  obs | trace idx | median|Δ| | max|Δ|")
    for o, i, md, mx in diffs:
        print(f"    {o:3d} | {i:3d} | {md:.2e} | {mx:.2e}")
    bad = sum(1 for _o, _i, md, mx in diffs if not (md < 1e-3 and mx < 1e-2))
    print("(a) 合格（各 obs で median<1e-3 かつ max<1e-2）" if bad == 0 else f"(a) ★線を超えた obs {bad}/{len(diffs)} → 数値を報告して止める（直すのは親の判断）")
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
