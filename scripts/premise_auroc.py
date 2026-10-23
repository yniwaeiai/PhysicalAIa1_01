"""腕 C の前提試験（計画 §3-3）: held-out 窓の各チャンクで、成功 prompt と失敗 prompt の流れ損失を「同じ (t, ε)」で測り、
score = L_fail − L_succ（成功ラベルの本ほど正になるはず）と真のラベル（ファイルの succ / fail）との AUROC を、step の bin × タスク別に出す。
帰無 = 窓（同じ task, seed）内でエピソードのラベルを入れ替える順列（既定 200 回）。★Pod の GPU で動かす前提（手元では --help と構文まで）。

データ: build_contrast_dataset.py の <out>_holdout/（各 ep の attrs: outcome, window, seed。ファイル attrs: instruction = "<元> (outcome: ...)"）。
標本: 各エピソードの t % --stride == 0（既定 10 = チャンク単位）。学習と同じ前処理（train_lora_kadai.build_transform = repack → data → Normalize → tokenize）。
同じ (t, ε) と同じ画像拡張: 2 つの prompt の呼び出しの直前に同じ torch seed を置く（PI0Pytorch.forward は train=True 固定で画像拡張の乱数を引くため）。
  --n-draws 回の (t, ε) の平均を 1 標本の損失にする。損失 = モデルの流れ損失（MSE）の先頭 --dims 次元の平均（既定 7 = 実際の行動次元。32 だと学習損失と同じ）。
重み: --ckpt に merge 済みのディレクトリ（pod_run_contrast_train.sh が作る data/contrast/policy/c1_s2400 等）を渡すのが基本。
      未 merge の LoRA（lora_s2400.pt）を使うなら --ckpt を素のベース、--lora を .pt、--rank/--alpha/--targets を学習と同じにする（train_lora_kadai.py の注入と同じ）。
      ★どちらでも同じ結果になるはず（merge は数値的にほぼ恒等）。報告では merge 済みを使う。
出力: --out <md>（表）と <md の拡張子を .csv にした>（サンプル別 score）。
使い方: python premise_auroc.py --data data/contrast/ds_c1_holdout --ckpt data/contrast/policy/c1_s2400 --out data/contrast/premise_c1_s2400.md
"""
import argparse, collections, dataclasses, json, os, sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
REPO = Path(os.environ.get("KADAI_REPO", HERE.parent.parent))
SUCC_TAG, FAIL_TAG = " (outcome: success)", " (outcome: failure)"
BINS = [("<=15", 0, 15), ("16-40", 16, 40), (">40", 41, 10**9)]


def auroc(score, label):
    """label 1 = 成功。成功の score が失敗より大きい確率（同点は 0.5）。"""
    score = np.asarray(score, float); label = np.asarray(label, int)
    pos, neg = score[label == 1], score[label == 0]
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    allv = np.concatenate([pos, neg]); order = allv.argsort(kind="mergesort")
    ranks = np.empty(len(allv)); i = 0
    sv = allv[order]
    while i < len(sv):
        j = i
        while j + 1 < len(sv) and sv[j + 1] == sv[i]:
            j += 1
        ranks[order[i:j + 1]] = (i + j) / 2 + 1; i = j + 1
    return float((ranks[:len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))


def summarize(rows, n_perm, seed):
    """rows: dict(task, window, ep, t, label, score)。bin × タスク（+ 全タスク）で AUROC と窓内順列の帰無（ラベル入れ替えはエピソード単位）。"""
    rng = np.random.default_rng(seed)
    out = []
    eps = collections.defaultdict(list)   # (window) -> [ep ids]
    ep_label = {}
    for r in rows:
        ep_label[(r["window"], r["ep"])] = r["label"]
    for (w, e) in ep_label:
        eps[w].append(e)
    perms = []   # n_perm 個の {(window, ep): label}
    for _ in range(n_perm):
        m = {}
        for w, es in eps.items():
            labs = [ep_label[(w, e)] for e in es]; rng.shuffle(labs)
            m.update({(w, e): l for e, l in zip(es, labs)})
        perms.append(m)
    for task in ["ALL"] + sorted({r["task"] for r in rows}):
        for bname, lo, hi in BINS:
            sel = [r for r in rows if (task == "ALL" or r["task"] == task) and lo <= r["t"] <= hi]
            if not sel:
                continue
            sc = np.array([r["score"] for r in sel]); lab = np.array([r["label"] for r in sel])
            a = auroc(sc, lab)
            null = np.array([auroc(sc, [p[(r["window"], r["ep"])] for r in sel]) for p in perms])
            null = null[~np.isnan(null)]
            out.append(dict(task=task, bin=bname, n=len(sel), n_pos=int(lab.sum()), n_eps=len({(r["window"], r["ep"]) for r in sel}), auroc=a,
                            null_mean=float(null.mean()) if len(null) else float("nan"), null_sd=float(null.std()) if len(null) else float("nan"),
                            p=float((1 + (null >= a).sum()) / (1 + len(null))) if len(null) else float("nan")))
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", type=Path, required=True)
    ap.add_argument("--ckpt", type=Path, required=True)
    ap.add_argument("--lora", type=Path, default=None)
    ap.add_argument("--rank", type=int, default=16)
    ap.add_argument("--alpha", type=float, default=32.0)
    ap.add_argument("--targets", default="attn-mlp", choices=["attn", "attn-mlp"])
    ap.add_argument("--assets-from", type=Path, default=None, help="norm_stats の置き場（既定 --ckpt/assets。merge 済みならそのまま）")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--stride", type=int, default=10)
    ap.add_argument("--n-draws", type=int, default=4)
    ap.add_argument("--dims", type=int, default=7)
    ap.add_argument("--perm", type=int, default=200)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max-samples", type=int, default=0, help="煙検査用: 先頭 N 標本だけ")
    a = ap.parse_args()

    import torch, h5py
    for p in (REPO, REPO / "third_party/openpi/src", REPO / "scripts"):
        sys.path.insert(0, str(p))
    from openpi.training import config as _config
    import openpi.models_pytorch.pi0_pytorch as pi0_pytorch
    import safetensors.torch
    from parc import lora as parc_lora
    from parc.perturbed_dataset import PerturbedDemoDataset
    import train_lora_kadai as TL

    device = torch.device("cuda")
    tc = _config.get_config("pi05_libero")
    mcfg = dataclasses.replace(tc.model, pytorch_compile_mode=None, dtype="bfloat16")
    tf, _ = TL.build_transform(tc, a.assets_from or a.ckpt)
    model = pi0_pytorch.PI0Pytorch(mcfg).to(device)
    safetensors.torch.load_model(model, str(a.ckpt / "model.safetensors"), strict=False)
    if a.lora is not None:
        targets = parc_lora.ATTN_MLP if a.targets == "attn-mlp" else parc_lora.ATTN_PROJ
        parc_lora.inject(model, rank=a.rank, alpha=a.alpha, targets=targets)
        sd = torch.load(a.lora, map_location="cpu")
        assert set(sd) == set(parc_lora.lora_state_dict(model)), "LoRA のキー集合が違う（rank/targets の取り違え）"
        parc_lora.load_lora_state_dict(model, sd)
        print(f"[lora] {a.lora} を読み込み")
    model.eval()

    ds = PerturbedDemoDataset(a.data, action_horizon=mcfg.action_horizon, transform=None)   # transform=None: 生の dict（prompt 付き）が返る
    idx = [i for i, s in enumerate(ds.index) if s.t % a.stride == 0]
    if a.max_samples:
        idx = idx[:a.max_samples]
    print(f"[data] {a.data}: 全 {len(ds)} フレーム中 {len(idx)} 標本（stride {a.stride}）")
    task_of = {"open_the_middle_drawer_of_the_cabinet": "drawer", "KITCHEN_SCENE4_close_the_bottom_drawer_of_the_cabinet": "k4", "KITCHEN_SCENE6_close_the_microwave": "k6"}
    hf = {}
    def ep_attrs(s):
        if s.file not in hf:
            hf[s.file] = h5py.File(s.file, "r")
        g = hf[s.file][s.ep]
        return dict(window=str(g.attrs["window"]), outcome=str(g.attrs["outcome"]), seed=int(g.attrs["seed"]))

    rows, B = [], a.batch
    H, D = mcfg.action_horizon, mcfg.action_dim
    gen = torch.Generator(device="cpu").manual_seed(a.seed)
    for b0 in range(0, len(idx), B):
        ii = idx[b0:b0 + B]
        items = [ds[i] for i in ii]
        base_instr = [it["prompt"].replace(SUCC_TAG, "").replace(FAIL_TAG, "") for it in items]
        def batch_for(tag):
            its = []
            for it, bi in zip(items, base_instr):
                x = dict(it); x["prompt"] = bi + tag; its.append(tf(x))
            return TL.to_observation(TL.collate(its), device)
        (obs_s, act), (obs_f, act_f) = batch_for(SUCC_TAG), batch_for(FAIL_TAG)
        assert torch.equal(act, act_f)
        tok_s, tok_f = obs_s.tokenized_prompt, obs_f.tokenized_prompt
        assert not torch.equal(tok_s, tok_f), "成功/失敗の tokenized_prompt が同一"
        ls, lf = 0.0, 0.0
        with torch.no_grad():
            for k in range(a.n_draws):
                noise = torch.randn(act.shape, generator=gen).to(device)
                time_ = (torch.distributions.Beta(1.5, 1.0).sample((act.shape[0],)) * 0.999 + 0.001).to(device)   # openpi の sample_time と同じ分布
                sd_ = int(torch.randint(0, 2**31 - 1, (1,), generator=gen))
                torch.manual_seed(sd_); l_s = model(obs_s, act, noise=noise, time=time_)[..., :a.dims].float().mean(dim=(1, 2))
                torch.manual_seed(sd_); l_f = model(obs_f, act, noise=noise, time=time_)[..., :a.dims].float().mean(dim=(1, 2))
                ls = ls + l_s; lf = lf + l_f
        ls, lf = (ls / a.n_draws).cpu().numpy(), (lf / a.n_draws).cpu().numpy()
        for j, i in enumerate(ii):
            s = ds.index[i]; at = ep_attrs(s)
            rows.append(dict(task=task_of[ds.meta[s.file]["base_task"]], window=at["window"], ep=s.ep, t=s.t, seed=at["seed"],
                             label=int(at["outcome"] in ("success", "succ")), l_succ=float(ls[j]), l_fail=float(lf[j]), score=float(lf[j] - ls[j])))
        if (b0 // B) % 20 == 0:
            print(f"  {b0 + len(ii)}/{len(idx)}", flush=True)
    res = summarize(rows, a.perm, a.seed)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    import csv
    with open(a.out.with_suffix(".csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    L = [f"# 前提試験 AUROC（score = L_fail − L_succ、成功ラベルが正）", "",
         f"- 重み: ckpt={a.ckpt} lora={a.lora}、データ {a.data}、stride {a.stride}、n_draws {a.n_draws}、dims {a.dims}、perm {a.perm}（窓内でエピソードのラベルを入れ替え）", "",
         "| タスク | step bin | 標本 | 成功標本 | ep | AUROC | 帰無平均 | 帰無 SD | p(片側) |", "|---|---|---|---|---|---|---|---|---|"]
    for r in res:
        L.append(f"| {r['task']} | {r['bin']} | {r['n']} | {r['n_pos']} | {r['n_eps']} | {r['auroc']:.3f} | {r['null_mean']:.3f} | {r['null_sd']:.3f} | {r['p']:.3f} |")
    allr = {(r["task"], r["bin"]): r["auroc"] for r in res}
    b1, b2 = allr.get(("ALL", "<=15"), float("nan")), allr.get(("ALL", "16-40"), float("nan"))
    verdict = ("通過" if (b1 >= 0.60 and b2 >= 0.70) else "不通過" if (b1 <= 0.55 and b2 <= 0.60) else "中間（線の間）")
    L += ["", f"線（計画 §3-3、全タスク）: bin<=15 AUROC {b1:.3f}（通過 ≥ 0.60 / 不通過 ≤ 0.55）、16-40 {b2:.3f}（通過 ≥ 0.70 / 不通過 ≤ 0.60）→ **{verdict}**"]
    a.out.write_text("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
