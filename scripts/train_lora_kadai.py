#!/usr/bin/env python3
"""[Kadai 派生: scripts/train_lora_pi05.py の複製。差分は ds_te の代用のみ] π0.5 を摂動下データで LoRA 追加学習する（PARC2026 予選の必須要件）。

## なぜ自前の学習ループか

openpi の `scripts/train_pytorch.py` は **LoRA に対応していない**
（`models_pytorch/` に LoRA 実装が無く、`AdamW(model.parameters())` で 3.617B 全部を
学習するため 5090 32GB に載らない）。LoRA 部分は `parc/lora.py` を使う。

## 規約上の位置づけ

運営は独自学習として「LoRA，adapter等の学習済みパラメータ」を認め、
「公開重みを実質的に変更せず，そのまま推論する構成」を**認めていない**
（`docs/予選ルール_確定事実.md` §1）。これは提出要件を満たすための学習である。

## ★前処理は openpi の変換スタックをそのまま使う

    repack → LiberoInputs → Normalize(norm_stats) → model_transforms(tokenize)

`data_loader.transform_dataset` と同じ順序・同じ実体を通す。ここを自作すると
学習と推論がずれ、「学習したのに悪化した」という最も debug しにくい失敗になる。

## 使い方

    third_party/openpi/.venv/bin/python scripts/train_lora_pi05.py \\
        --data data/perturbed_demos/libero_spatial \\
        --steps 3000 --batch 16 --lr 1e-4 --exp-name lora_track1_v1

    # 学習後、ベース重みへ畳み込んで提出用チェックポイントを作る
    ... --merge-out data/openpi/pi05_libero_lora_merged
"""

from __future__ import annotations

import argparse
import random
import dataclasses
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "third_party" / "openpi" / "src"))


def build_transform(train_config, ckpt_dir: Path):
    """`data_loader.transform_dataset` と**同じ順序**の変換を組む。"""
    import openpi.transforms as _transforms

    data_config = train_config.data.create(ckpt_dir / "assets", train_config.model)
    if data_config.norm_stats is None:
        raise RuntimeError(
            f"norm_stats が見つからない: {ckpt_dir/'assets'}。"
            " 重み変換時に assets がコピーされているか確認すること"
            "（scripts/convert_pi05_libero.sh の既知バグ）"
        )
    return _transforms.compose([
        *data_config.repack_transforms.inputs,
        *data_config.data_transforms.inputs,
        _transforms.Normalize(data_config.norm_stats, use_quantiles=data_config.use_quantile_norm),
        *data_config.model_transforms.inputs,
    ]), data_config


def collate(items: list[dict]) -> dict:
    """変換後の dict をバッチにまとめる。

    ★`image` と `image_mask` は**入れ子の dict**（カメラ名 → 配列）なので再帰する。
    文字列（prompt 等）はモデルに渡さないので落とす。
    """
    def stack(key_path: list[str], vals: list):
        if isinstance(vals[0], dict):
            return {k: stack(key_path + [k], [v[k] for v in vals]) for k in vals[0]}
        return torch.from_numpy(np.stack([np.asarray(v) for v in vals]))

    return {
        k: stack([k], [it[k] for it in items])
        for k in items[0]
        if not isinstance(items[0][k], str)
    }


def to_observation(batch: dict, device):
    """openpi の Observation に詰める。dtype は float32 のまま（本家 train_pytorch と同じ）。"""
    from openpi.models import model as _model

    def dev(v):
        return v.to(device) if torch.is_tensor(v) else v

    # ★画像は uint8 [b,h,w,c] で来る。openpi の Observation.from_dict と同じく
    #   [-1,1] の float32・[b,c,h,w] に直す。ここを外すと入力分布が丸ごとずれる。
    images = {
        k: (dev(v).to(torch.float32).permute(0, 3, 1, 2) / 255.0 * 2.0 - 1.0)
        for k, v in batch["image"].items()
    }
    masks = {k: dev(v) for k, v in batch["image_mask"].items()}
    obs = _model.Observation(
        images=images,
        image_masks=masks,
        state=dev(batch["state"]).to(torch.float32),
        tokenized_prompt=dev(batch.get("tokenized_prompt")),
        tokenized_prompt_mask=dev(batch.get("tokenized_prompt_mask")),
    )
    return obs, dev(batch["actions"]).to(torch.float32)


class ContrastBatchSampler(torch.utils.data.Sampler):
    """★2026-09-13（ユーザ提案、DC1 事後対策）: 対照グループ内の異なる base_task から
    毎バッチ最低 n_pair 件ずつを強制的に混在させ、残りは全体からランダム抽出するバッチサンプラー。

    背景: `state/predictions_t2_drawer_20260913.md` Stage 2（DC1）。drawer 専用の運動パターンを
    大量投入すると、LoRA が instruction でなく **物体の視覚特徴**にその運動を汎化し、同じ物体
    （akita_black_bowl）を使う別タスク（cookie box→plate）まで巻き込んで崩壊した。
    同じバッチ・同じ勾配ステップで両タスクの損失を同時に見せれば、物体だけに依存する
    ショートカットでは両方の損失を下げられないため、instruction 条件付けを使わざるを得なくなる、
    という仮説（[[parc2026-dc1-object-contamination]] の対策候補）。
    """

    def __init__(self, dataset, batch_size: int, groups: list[list[str]], n_pair: int = 4,
                 seed: int = 0, drop_last: bool = True):
        self.batch_size = batch_size
        self.n = len(dataset)
        self.drop_last = drop_last
        self.rng = random.Random(seed)

        base_of: list[str] = [dataset.meta[s.file]["base_task"] for s in dataset.index]
        self.group_idx: list[list[list[int]]] = []   # [group][base_task 内] -> インデックス列
        covered: set[int] = set()
        for group in groups:
            per_base = []
            for bt in group:
                idx = [i for i, b in enumerate(base_of) if b == bt]
                if not idx:
                    print(f"[contrast] ★警告: base_task '{bt}' がデータセットに無い（0件）", flush=True)
                per_base.append(idx)
                covered.update(idx)
            self.group_idx.append(per_base)
        self.rest = [i for i in range(self.n) if i not in covered]
        n_forced = sum(n_pair * len(g) for g in self.group_idx)
        print(f"[contrast] グループ数={len(groups)} 強制混在={n_forced}/batch（batch_size={batch_size}）"
              f" 対象サンプル={len(covered)} 残り={len(self.rest)}", flush=True)
        assert n_forced < batch_size, f"★n_pair×グループ内タスク数がbatch_size以上: {n_forced} >= {batch_size}"
        for gi, per_base in enumerate(self.group_idx):
            for bt, idx in zip(groups[gi], per_base):
                assert len(idx) >= n_pair, f"★base_task '{bt}' のサンプル数 {len(idx)} < n_pair {n_pair}"
        self.n_pair = n_pair

    def __iter__(self):
        n_batches = len(self)
        for _ in range(n_batches):
            batch = []
            for per_base in self.group_idx:
                for idx in per_base:
                    batch.extend(self.rng.sample(idx, self.n_pair))
            need = self.batch_size - len(batch)
            batch.extend(self.rng.sample(self.rest, min(need, len(self.rest))))
            while len(batch) < self.batch_size:   # rest が尽きたら全体から補充（稀）
                batch.append(self.rng.randrange(self.n))
            self.rng.shuffle(batch)
            yield batch

    def __len__(self):
        return self.n // self.batch_size if self.drop_last else (self.n + self.batch_size - 1) // self.batch_size


def cat_batches(a: dict, b: dict) -> dict:
    """collate 済みの 2 バッチを先頭の次元で連結する（入れ子の dict は再帰）。"""
    return {k: (cat_batches(a[k], b[k]) if isinstance(a[k], dict) else torch.cat([a[k], b[k]], dim=0)) for k in a}


class DPOPairDataset(torch.utils.data.Dataset):
    """★2026-10-12 DPO v1: `scripts/build_t2_dpo_pairs.py` の pairs.json から (勝ち, 負け, 重み) を返す。

    読み出しと前処理は `PerturbedDemoDataset.__getitem__` をそのまま使う（画像の回転・resize・state・openpi 変換）。
    対の中の時刻は勝ちと負けで**独立**に引く（軌跡単位の DPO = GRAPE の TPO と同じく軌跡を集合として扱う。
    同一時刻で揃えない）。乱数は torch から取る（DataLoader の worker ごとに seed が変わる）。
    ★負けの時刻の窓（§15 (i)、分岐点以降）: 衝突負けは最初の接触（hdf5 attrs first_contact_step = 対象外物体の
      変位が初めて > 1 mm になった行動の step）の lose_pre 前から末尾、timeout 負けは末尾 lose_tail step。
      どちらも負の値なら全区間（96a54b2 の煙検査と同じ）。勝ちは常に全区間。
    ★重み（§15 (ii)）: 1/（その条件の対の数）を、対の上での平均が 1 になるよう正規化（= N_pairs / (N_cond · n_c)）。
      対を一様に引くので、期待値で各条件が等しく寄与する。
    fixed: [(対の index, u_w, u_l)] を渡すと時刻を u ∈ [0,1) で決め打ちし、重みは 1（held-out の評価用、決定的）。
    """

    def __init__(self, pairs_json: Path, action_horizon: int, transform, pairs: list[dict],
                 lose_pre: int = -1, lose_tail: int = -1, fixed: list | None = None):
        import collections as _c

        import h5py
        from parc.perturbed_dataset import PerturbedDemoDataset
        root = Path(pairs_json).parent
        self.pairs = pairs
        self.fixed = fixed
        self.sides = {}
        self.span: dict[tuple[str, str, str], tuple[int, int]] = {}
        for side in ("win", "lose"):
            ds = PerturbedDemoDataset(root / side, action_horizon=action_horizon, only_success=False, transform=transform)
            self.sides[side] = ds
            for i, smp in enumerate(ds.index):
                k = (side, smp.file, smp.ep)
                if k not in self.span:
                    self.span[k] = (i, 0)
                st, n = self.span[k]
                assert i == st + n and smp.t == n, (k, i, smp.t)   # 1 軌跡の時刻は index 上で連続
                self.span[k] = (st, n + 1)
        # 時刻の窓 [lo, hi)（index 上の絶対位置）
        self.win_range: dict[tuple, tuple[int, int]] = {}
        self.n_win_frames = {"win": 0, "collided": 0, "timeout": 0}
        for p in pairs:
            for side, f, ep in (("win", p["win"], p["win_ep"]), ("lose", p["lose"], p["lose_ep"])):
                k = (side, str(root / f), ep)
                assert k in self.span, k
                if k in self.win_range:
                    continue
                st, n = self.span[k]
                lo = 0
                if side == "lose":
                    with h5py.File(k[1], "r") as hf:
                        at = hf[ep].attrs
                        outcome, fc, L = str(at["outcome"]), int(at["first_contact_step"]), int(at["length"])
                    assert L == n and outcome == p["type"], (k, L, n, outcome, p["type"])
                    if outcome == "collided" and lose_pre >= 0:
                        assert 0 <= fc < n, (k, fc, n)
                        lo = max(0, fc - lose_pre)
                    elif outcome == "timeout" and lose_tail >= 0:
                        lo = max(0, n - lose_tail)
                self.win_range[k] = (st + lo, st + n)
                self.n_win_frames["win" if side == "win" else outcome] += n - lo
        self.keys = [(("win", str(root / p["win"]), p["win_ep"]), ("lose", str(root / p["lose"]), p["lose_ep"]))
                     for p in pairs]
        cond = [(p["task"], p["seed"], p["episode_id"]) for p in pairs]
        nc = _c.Counter(cond)
        self.n_cond = len(nc)
        self.weight = [len(pairs) / (len(nc) * nc[c]) for c in cond]

    def __len__(self) -> int:
        return len(self.fixed) if self.fixed is not None else len(self.keys)

    def __getitem__(self, i: int):
        if self.fixed is not None:
            j, uw, ul = self.fixed[i]
            us, w = (uw, ul), 1.0
        else:
            j, us, w = i, None, self.weight[i]
        out = []
        for s_i, (side, k) in enumerate(zip(("win", "lose"), self.keys[j])):
            lo, hi = self.win_range[k]
            off = int(us[s_i] * (hi - lo)) if us is not None else int(torch.randint(hi - lo, ()))
            out.append(self.sides[side][lo + off])
        return out[0], out[1], w


def collate_pairs(items: list) -> tuple[dict, dict, torch.Tensor]:
    return (collate([x[0] for x in items]), collate([x[1] for x in items]),
            torch.tensor([x[2] for x in items], dtype=torch.float32))


def dpo_loss_fn(model, parc_lora, bw: dict, bl: dict, device, beta: float, n_dims: int, weight=None):
    """勝ち w・負け l の chunk について、同じ (ε, t)・同じ画像拡張で方策 θ と参照（LoRA を切った同じモデル）の
    flow の MSE を測り、Diffusion-DPO / CrossVLA 型の損失を返す。

        L(x) = mean_{h, d<n_dims} (v(x_t, t, obs) − (ε − a))²      （openpi の forward の要素ごとの MSE）
        margin = (L_ref(w) − L_θ(w)) − (L_ref(l) − L_θ(l))           （= 報酬差 log p̃θ/p̃ref の勝ち − 負け）
        loss  = mean_i weight_i · (−log σ(β · margin_i))             （weight は条件の重み、無ければ 1）

    戻り値 (loss, margin[N] detach, 参照の L_w−L_l 平均)。"""
    obs, act = to_observation(cat_batches(bw, bl), device)
    n = act.shape[0] // 2
    noise = model.sample_noise(act.shape, device)
    time_ = model.sample_time(act.shape[0], device)
    # ★画像拡張（preprocess_observation_pytorch の train=True）は CUDA の大域乱数から引く。
    #   参照と方策で同じ拡張になるよう、乱数の状態を保存して参照の後に戻す。
    rng_cpu, rng_cuda = torch.get_rng_state(), torch.cuda.get_rng_state(device)
    with torch.no_grad(), parc_lora.disabled(model):
        l_ref = model(obs, act, noise=noise, time=time_)[..., :n_dims].mean(dim=(1, 2))
    torch.set_rng_state(rng_cpu)
    torch.cuda.set_rng_state(rng_cuda, device)
    l_pol = model(obs, act, noise=noise, time=time_)[..., :n_dims].mean(dim=(1, 2))
    margin = (l_ref[:n] - l_pol[:n]) - (l_ref[n:] - l_pol[n:])
    per = -torch.nn.functional.logsigmoid(beta * margin)
    loss = (per * weight.to(device)).mean() if weight is not None else per.mean()
    return loss, margin.detach(), float((l_ref[:n] - l_ref[n:]).mean())


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, required=True)
    ap.add_argument("--ckpt", type=Path, default=REPO_ROOT / "data/openpi/pi05_libero_pytorch")
    ap.add_argument("--exp-name", default="lora_track1")
    ap.add_argument("--steps", type=int, default=3000)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--warmup", type=int, default=200)
    ap.add_argument("--rank", type=int, default=16)
    ap.add_argument("--alpha", type=float, default=32.0)
    ap.add_argument("--targets", default="attn", choices=["attn", "attn-mlp", "attn-vision"],
                    help="attn-vision は既存の attn(全層) に加え、画像エンコーダ(SigLIP/"
                         "vision_tower)の MLP(fc1/fc2) にも LoRA を付ける。★注意: 既存の "
                         "ATTN_PROJ (q/k/v/o/out_proj) は include/exclude 無指定だと "
                         "vision_tower の注意射影にも既に一致するため（実測: 252層中108層 / "
                         "10.17M中3.98M が vision_tower 配下）、'attn' 単体でも視覚側の注意層は"
                         "既に学習対象。attn-vision が追加するのは vision の MLP 容量")
    ap.add_argument("--ema-decay", type=float, default=None,
                    help="LoRA パラメータの指数移動平均 (decay)。既定 None = 従来どおり EMA 無し"
                         "（このときのコードパスは EMA 導入前と同一）。有効にすると学習中 "
                         "shadow = decay*shadow + (1-decay)*param を保持し、--save-every の "
                         "chekpoint と --merge-out の出力に EMA 側の重みを使う")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--n-test-tasks", type=int, default=2)
    ap.add_argument("--wrist-mask-closed-p", type=float, default=0.0,
                    help="AC-VLA 型: グリッパ閉指令の時刻のサンプルで手首カメラを確率 p でマスク（学習側のみ、既定 0 = 無効）")
    ap.add_argument("--collisions", type=Path, default=None,
                    help="check_demo_collisions.py の出力。渡すと運営の衝突ルールに"
                         "触れるデモを学習から外す（★成功のゲートなので効きが確実）")
    ap.add_argument("--contrast-groups", type=Path, default=None,
                    help="★2026-09-13 対照バッチ学習（ユーザ提案、DC1 の物体単位汚染への対策）。"
                         "JSON: [[\"base_task_A\", \"base_task_B\", ...], ...]（同一物体を共有する base_task 名の集合を列挙）。"
                         "各バッチに、各グループの複数 base_task から --contrast-n-pair 件ずつを強制的に混在させ、"
                         "同じ勾配ステップで『同じ物体でも instruction が違えば行動が違う』ことを学習させる。")
    ap.add_argument("--contrast-n-pair", type=int, default=2,
                    help="対照グループの各 base_task からバッチに強制的に入れる件数（既定 4）")
    ap.add_argument("--eval-every", type=int, default=250)
    ap.add_argument("--save-every", type=int, default=500)
    ap.add_argument("--snapshot-every", type=int, default=0,
                    help="0 以外なら、この step ごとに lora_s<step>.pt を別名で残す（lora.pt は従来どおり上書き）")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--split-seed", type=int, default=None,
                    help="task_split（train/test タスク分割）専用の seed。既定 None は --seed を"
                         "そのまま使う（従来動作、後方互換）。model soup の seed replica など、"
                         "学習の stochasticity（LoRA 初期化・バッチ順）だけを変えて train/test の"
                         "タスク分割は固定したいときに使う（★test は選別しない、B と同じ test で"
                         "測る、という既存方針の延長）")
    ap.add_argument("--holdout-bases", default=None,
                    help="held-out（test）にする基底名のカンマ区切り。指定時は task_split の乱択の代わりにこれを使う"
                         "（データの基底集合が変わっても既存の分割に固定するため。2026-10-12 si1）。"
                         "既定 None は従来どおり task_split")
    ap.add_argument("--out-dir", type=Path, default=REPO_ROOT / "data/lora_runs")
    ap.add_argument("--merge-out", type=Path, default=None,
                    help="学習後、ベース重みへ畳み込んだ提出用チェックポイントを書く先")
    # ---- ★resume（既定は無効。無効時は以降の分岐が一切実行されず、従来のコードパスと同一）----
    ap.add_argument("--resume-lora", type=Path, default=None,
                    help="途中の LoRA（lora_s<N>.pt）から続ける。--start-step と併用（提供 GPU の 12 時間枠をまたぐため）")
    ap.add_argument("--start-step", type=int, default=0,
                    help="--resume-lora の重みが何 step 目のものか。LR の位置をそこへ合わせ、step N+1 から --steps まで続ける")
    ap.add_argument("--resume-warmup", type=int, default=100,
                    help="resume 直後に LR を線形に立ち上げる step 数（AdamW のモーメントが 0 から始まるため）。"
                         "--resume-optim を渡したときは使わない")
    ap.add_argument("--resume-optim", type=Path, default=None,
                    help="--save-optim で保存した optim.pt。渡すと AdamW の状態も復元する（立ち上げ無し）")
    ap.add_argument("--save-optim", action="store_true",
                    help="保存点で AdamW の状態も optim.pt に書く（次回の厳密な resume 用。既定は書かない）")
    ap.add_argument("--stop-after", type=int, default=0,
                    help="この起動で N step 進めたら保存して止める（resume の動作確認や時間枠の区切り用。既定 0 = 無効）")
    ap.add_argument("--log-every", type=int, default=50, help="学習 loss の表示間隔（既定 50 = 従来どおり）")
    # ---- ★DPO（2026-10-12、既定は無効。無効時は以降の分岐が一切実行されず、従来のコードパスと同一）----
    ap.add_argument("--dpo-pairs", type=Path, default=None,
                    help="scripts/build_t2_dpo_pairs.py の pairs.json。渡すと各 step で BC バッチ（--data）1 つ + "
                         "対バッチ（--dpo-batch 組）1 つを処理し、loss = bc_weight·BC + DPO")
    ap.add_argument("--dpo-beta", type=float, default=0.1)
    ap.add_argument("--dpo-bc-weight", type=float, default=1.0)
    ap.add_argument("--dpo-ref", choices=["lora-off"], default="lora-off",
                    help="参照モデル。lora-off = 同じモデルで LoRA を切った前向き（= --ckpt のベース、重みのコピー無し）")
    ap.add_argument("--dpo-batch", type=int, default=4, help="1 step あたりの対の数（勝ち・負けで計 2 倍の chunk）")
    ap.add_argument("--dpo-max-pairs", type=int, default=0, help="0 以外なら --seed で並べ替えた先頭 N 対だけ使う（煙検査用）")
    ap.add_argument("--dpo-action-dims", type=int, default=7,
                    help="DPO の MSE を平均する行動の次元（LIBERO は 7。残りは openpi が 0 で詰めた次元）")
    ap.add_argument("--dpo-workers", type=int, default=4)
    ap.add_argument("--dpo-lose-pre-contact", type=int, default=-1,
                    help="衝突負けの chunk を最初の接触の N step 前から末尾に絞る（§15 (i) は 20）。負なら全区間")
    ap.add_argument("--dpo-lose-timeout-tail", type=int, default=-1,
                    help="timeout 負けの chunk を末尾 N step に絞る（§15 (i) は 100）。負なら全区間")
    ap.add_argument("--dpo-cond-weight", action="store_true",
                    help="DPO 項に条件の重み 1/（その条件の対の数）を掛ける（平均 1 に正規化、§15 (ii)）")
    ap.add_argument("--dpo-heldout-conds", type=int, default=0,
                    help="この数の条件（seed --dpo-heldout-seed で選ぶ）の対を学習から外し、評価で margin を出す（§15 (iii)）")
    ap.add_argument("--dpo-heldout-seed", type=int, default=0)
    ap.add_argument("--dpo-eval-n", type=int, default=128,
                    help="held-out（と学習対の同じ作りの比較用）の margin を測る試料数（条件一様に決め打ち）")
    args = ap.parse_args()
    resume = args.resume_lora is not None
    if resume:
        assert 0 < args.start_step < args.steps, f"--start-step {args.start_step} は 1〜{args.steps - 1} の範囲で指定する"
    else:
        assert args.start_step == 0 and args.resume_optim is None, "--start-step / --resume-optim は --resume-lora と併用する"

    torch.manual_seed(args.seed + (args.start_step if resume else 0))
    np.random.seed(args.seed + (args.start_step if resume else 0))
    run_dir = args.out_dir / args.exp_name
    run_dir.mkdir(parents=True, exist_ok=True)

    from openpi.training import config as _config
    import openpi.models_pytorch.pi0_pytorch as pi0_pytorch
    import safetensors.torch
    from torch.utils.data import DataLoader

    from parc import lora as parc_lora
    from parc.perturbed_dataset import PerturbedDemoDataset, task_split, summarize

    device = torch.device("cuda")
    train_config = _config.get_config("pi05_libero")
    model_cfg = dataclasses.replace(train_config.model, pytorch_compile_mode=None,
                                    dtype="bfloat16")

    # ---- データ -----------------------------------------------------------
    print(f"[data] {json.dumps(summarize(args.data), ensure_ascii=False)}")
    tf, data_config = build_transform(train_config, args.ckpt)
    split_seed = args.split_seed if args.split_seed is not None else args.seed
    if args.holdout_bases:
        te_tasks = {x.strip() for x in args.holdout_bases.split(",") if x.strip()}
        all_bases, _ = task_split(args.data, n_test=0, seed=split_seed)   # n_test=0 → 全基底（test は空）
        missing = te_tasks - all_bases
        assert not missing, f"--holdout-bases の基底がデータに無い: {sorted(missing)}"
        assert len(te_tasks) == args.n_test_tasks, f"--holdout-bases {len(te_tasks)} 件 != --n-test-tasks {args.n_test_tasks}"
        tr_tasks = all_bases - te_tasks
        print(f"[data] held-out 基底を固定（--holdout-bases）: {sorted(te_tasks)}")
    else:
        tr_tasks, te_tasks = task_split(args.data, n_test=args.n_test_tasks, seed=split_seed)
    print(f"[data] ★タスク単位で分割: train {len(tr_tasks)} / test {len(te_tasks)}")
    print(f"[data] test: {sorted(te_tasks)}")

    ds_tr = PerturbedDemoDataset(args.data, action_horizon=model_cfg.action_horizon,
                                 tasks=tr_tasks, transform=tf, collisions=args.collisions,
                                 wrist_mask_closed_p=args.wrist_mask_closed_p)
    # ★test は選別しない。「衝突しないデモだけで測る」と自分に都合の良い
    #   test set になり、比較が成立しなくなる。B と同じ test で測る。
    # [kadai 派生] 検証用タスクが空（3 タスクすべてを学習に使う）のときは学習セットで代用する。
    #   この検証損失は汎化の指標ではない（参考値）。汎化は未使用 seed・摂動での閉ループ評価で測る
    ds_te = (PerturbedDemoDataset(args.data, action_horizon=model_cfg.action_horizon, tasks=te_tasks, transform=tf)
             if te_tasks else ds_tr)
    print(f"[data] train {len(ds_tr)} / test {len(ds_te)} サンプル")

    if args.contrast_groups is not None:
        groups = json.load(open(args.contrast_groups, encoding="utf-8"))
        sampler = ContrastBatchSampler(ds_tr, args.batch, groups, n_pair=args.contrast_n_pair, seed=args.seed)
        dl_tr = DataLoader(ds_tr, batch_sampler=sampler, num_workers=args.workers,
                           collate_fn=collate, persistent_workers=args.workers > 0, pin_memory=True)
    else:
        dl_tr = DataLoader(ds_tr, batch_size=args.batch, shuffle=True, num_workers=args.workers,
                           collate_fn=collate, drop_last=True, persistent_workers=args.workers > 0,
                           pin_memory=True)
    dl_te = DataLoader(ds_te, batch_size=args.batch, shuffle=False, num_workers=2,
                       collate_fn=collate, drop_last=True)

    dl_dpo = None
    dpo_eval_sets = {}
    if args.dpo_pairs is not None:
        import collections as _c
        blob = json.load(open(args.dpo_pairs, encoding="utf-8"))
        pairs = blob["pairs"]
        n_all_pairs = len(pairs)
        cond_of = lambda p: (p["task"], p["seed"], p["episode_id"])  # noqa: E731
        # ★held-out の条件（§15 (iii)）。BC の --holdout-bases とは独立（DPO の対は基底では外さない）
        ho_conds = set()
        if args.dpo_heldout_conds:
            all_conds = sorted({cond_of(p) for p in pairs})
            ho_conds = set(random.Random(args.dpo_heldout_seed).sample(all_conds, args.dpo_heldout_conds))
            print(f"[dpo] held-out 条件 {len(ho_conds)} / {len(all_conds)}（seed {args.dpo_heldout_seed}）: "
                  + "; ".join(f"{t}#{e}" for t, _s, e in sorted(ho_conds)), flush=True)
        pairs_ho = [p for p in pairs if cond_of(p) in ho_conds]
        pairs = [p for p in pairs if cond_of(p) not in ho_conds]
        if args.dpo_max_pairs:
            rng_p = random.Random(args.seed)
            pairs = rng_p.sample(pairs, min(args.dpo_max_pairs, len(pairs)))
        print(f"[dpo] 学習の対 {len(pairs)} / {n_all_pairs}（held-out の対 {len(pairs_ho)}"
              f"{'、先頭 ' + str(args.dpo_max_pairs) + ' 対' if args.dpo_max_pairs else ''}）"
              f" 型 {dict(_c.Counter(p['type'] for p in pairs))}、β={args.dpo_beta}、bc_weight={args.dpo_bc_weight}、"
              f"ref={args.dpo_ref}、対/step={args.dpo_batch}、行動次元={args.dpo_action_dims}、"
              f"負けの窓 接触前 {args.dpo_lose_pre_contact} / timeout 末尾 {args.dpo_lose_timeout_tail}、"
              f"条件の重み {'有' if args.dpo_cond_weight else '無'}", flush=True)
        dkw = dict(lose_pre=args.dpo_lose_pre_contact, lose_tail=args.dpo_lose_timeout_tail)
        ds_dpo = DPOPairDataset(args.dpo_pairs, model_cfg.action_horizon, tf, pairs, **dkw)
        if not args.dpo_cond_weight:
            ds_dpo.weight = [1.0] * len(pairs)
        wts = np.asarray(ds_dpo.weight)
        print(f"[dpo] 学習の条件 {ds_dpo.n_cond}、重み 平均 {wts.mean():.3f} 最小 {wts.min():.3f} 最大 {wts.max():.3f}、"
              f"窓の frame 数 {ds_dpo.n_win_frames}", flush=True)
        assert abs(wts.mean() - 1.0) < 1e-6, wts.mean()
        assert len(ds_dpo) >= args.dpo_batch, (len(ds_dpo), args.dpo_batch)
        dl_dpo = DataLoader(ds_dpo, batch_size=args.dpo_batch, shuffle=True, num_workers=args.dpo_workers,
                            collate_fn=collate_pairs, drop_last=True,
                            persistent_workers=args.dpo_workers > 0, pin_memory=True)

        def fixed_items(ps: list[dict], seed: int) -> list:
            """条件を一様に引き、その中の対を一様に引き、時刻 u を決め打つ（評価の試料、決定的）。"""
            r = random.Random(seed)
            by = _c.defaultdict(list)
            for i, p in enumerate(ps):
                by[cond_of(p)].append(i)
            cs = sorted(by)
            return [(r.choice(by[r.choice(cs)]), r.random(), r.random()) for _ in range(args.dpo_eval_n)]

        if pairs_ho:
            dpo_eval_sets["heldout"] = DPOPairDataset(args.dpo_pairs, model_cfg.action_horizon, tf, pairs_ho,
                                                      fixed=fixed_items(pairs_ho, 1), **dkw)
        dpo_eval_sets["train"] = DPOPairDataset(args.dpo_pairs, model_cfg.action_horizon, tf, pairs,
                                                fixed=fixed_items(pairs, 2), **dkw)

    # ---- モデル -----------------------------------------------------------
    # ★`.to(device)` のみ。dtype を外から一律に指定すると小さい射影まで bf16 になって落ちる
    model = pi0_pytorch.PI0Pytorch(model_cfg).to(device)
    safetensors.torch.load_model(model, str(args.ckpt / "model.safetensors"), strict=False)
    if hasattr(model, "gradient_checkpointing_enable"):
        model.gradient_checkpointing_enable()

    if args.targets == "attn-vision":
        # ★既存の ATTN_PROJ は include/exclude 無指定で vision_tower にも一致するので、
        #   「attn に画像エンコーダの注意射影を足す」は素の ATTN_PROJ 適用と同じ集合になり
        #   パラメータ数が増えない。ここでは attn(全層, vision の注意含む) に加えて、
        #   画像エンコーダの MLP（fc1/fc2, vision_tower 限定）を足して視覚側の学習容量を増やす。
        info_attn = parc_lora.inject(model, rank=args.rank, alpha=args.alpha, targets=parc_lora.ATTN_PROJ)
        info_vis_mlp = parc_lora.inject(model, rank=args.rank, alpha=args.alpha,
                                        targets=("fc1", "fc2"), include="vision_tower")
        vision_lora_names = [n for n, m in model.named_modules()
                             if isinstance(m, parc_lora.LoRALinear) and "vision_tower" in n]
        n_vision_lora = sum(p.numel() for n, p in model.named_parameters()
                            if ".lora_" in n and "vision_tower" in n)
        # ★注意: InjectInfo.n_lora_params はその時点のモデル全体の累積値（inject() 内部で
        #   `model.named_parameters()` を毎回スキャンして数える）。したがって2回目の呼び出し
        #   （info_vis_mlp）の n_lora_params は「1回目 + 2回目」の**合計**であり、
        #   info_attn.n_lora_params を足すと二重計上になる。合計は info_vis_mlp の値をそのまま使う。
        n_total_lora = info_vis_mlp.n_lora_params
        print(f"[lora] attn(全層, vision の注意含む): {info_attn}")
        print(f"[lora] +vision-mlp(画像エンコーダの fc1/fc2 のみ): {info_vis_mlp}")
        print(f"[lora] vision_tower 配下の LoRA 層 {len(vision_lora_names)} 件、例: {vision_lora_names[:5]}")
        print(f"[lora] 学習パラメータ合計 {n_total_lora/1e6:.2f}M (うち vision_tower {n_vision_lora/1e6:.2f}M)")
        assert n_total_lora > info_attn.n_lora_params, (
            f"★操作チェック失敗: attn-vision の合計({n_total_lora})が attn 単体"
            f"({info_attn.n_lora_params})より多くない")
        _ATTN_ONLY_REF_M = 10.17  # 実測基準値（rank16/alpha32, scripts/run_train_curriculum.sh 系列の実測ログより）
        if abs(args.rank - 16) < 1e-9 and abs(args.alpha - 32.0) < 1e-9:
            assert n_total_lora / 1e6 > _ATTN_ONLY_REF_M, (
                f"★操作チェック失敗: attn のみ({_ATTN_ONLY_REF_M}M)より多いはずが "
                f"{n_total_lora/1e6:.2f}M しかない")
        info = None
    else:
        targets = parc_lora.ATTN_MLP if args.targets == "attn-mlp" else parc_lora.ATTN_PROJ
        info = parc_lora.inject(model, rank=args.rank, alpha=args.alpha, targets=targets)
    n_train = parc_lora.freeze_base(model)
    if info is not None:
        print(f"[lora] {info}")
    if resume:
        sd_r = torch.load(args.resume_lora, map_location="cpu")
        cur = parc_lora.lora_state_dict(model)
        # ★操作チェック: キー集合と形が一致（rank・targets が元の run と同じ）、B が 0 でない（＝学習済みの重みである）
        assert set(sd_r) == set(cur), (f"★resume: LoRA のキー集合が違う（rank/targets の取り違え）: "
                                       f"file {len(sd_r)} / model {len(cur)}")
        bad = [k for k in sd_r if sd_r[k].shape != cur[k].shape]
        assert not bad, f"★resume: 形が違う {bad[:3]}"
        b_abs = sum(float(v.float().abs().sum()) for k, v in sd_r.items() if ".lora_b." in k)
        assert b_abs > 0, "★resume: lora_b が全て 0（未学習の重み）"
        parc_lora.load_lora_state_dict(model, sd_r)
        print(f"[resume] {args.resume_lora} を読み込み（{len(sd_r)} テンソル、Σ|B|={b_abs:.1f}）。"
              f"step {args.start_step} の続きから {args.steps} まで", flush=True)

    # ---- 操作チェック -----------------------------------------------------
    n_all = sum(p.numel() for p in model.parameters())
    n_req = sum(p.numel() for p in model.parameters() if p.requires_grad)
    assert n_req == n_train and n_req < n_all * 0.05, (
        f"凍結が効いていない: requires_grad {n_req} / 全 {n_all}")
    print(f"[操作チェック] 学習 {n_req/1e6:.2f}M / 全 {n_all/1e9:.3f}B ({100*n_req/n_all:.3f}%)")

    params = [p for p in model.parameters() if p.requires_grad]
    named_train_params = [(n, p) for n, p in model.named_parameters() if p.requires_grad]

    # ---- ★EMA（既定 None = 無効。無効時は以降の分岐が一切実行されず、旧コードパスと同一）----
    ema_shadow: dict[str, torch.Tensor] | None = None
    if args.ema_decay is not None:
        ema_shadow = {n: p.detach().clone() for n, p in named_train_params}
        print(f"[ema] decay={args.ema_decay} を有効化（shadow {len(ema_shadow)} テンソル）", flush=True)

    optim = torch.optim.AdamW(params, lr=args.lr, weight_decay=0.0)
    if not resume:
        sched = torch.optim.lr_scheduler.LambdaLR(
            optim,
            lambda s: (s + 1) / max(args.warmup, 1) if s < args.warmup
            else 0.5 * (1 + np.cos(np.pi * (s - args.warmup) / max(args.steps - args.warmup, 1))),
        )
    else:
        _off = args.start_step
        _rw = 0 if args.resume_optim is not None else args.resume_warmup

        def _lr_lambda(s: int) -> float:
            g = s + _off   # 元の run での scheduler の位置
            m = ((g + 1) / max(args.warmup, 1) if g < args.warmup
                 else 0.5 * (1 + np.cos(np.pi * (g - args.warmup) / max(args.steps - args.warmup, 1))))
            return m * min(1.0, (s + 1) / _rw) if _rw > 0 else m

        sched = torch.optim.lr_scheduler.LambdaLR(optim, _lr_lambda)
        if args.resume_optim is not None:
            st = torch.load(args.resume_optim, map_location="cpu")
            assert int(st["step"]) == args.start_step, f"★resume: optim.pt は step {st['step']}、--start-step は {args.start_step}"
            optim.load_state_dict(st["optim"])
            for g_ in optim.param_groups:   # load_state_dict が lr を保存時の値に戻すので、scheduler の現在値に合わせ直す
                g_["lr"] = args.lr * _lr_lambda(0)
            print(f"[resume] AdamW の状態を復元（{args.resume_optim}）", flush=True)
        print(f"[resume] LR: 最初の step {args.lr * _lr_lambda(0):.3e} → 立ち上げ後 {args.lr * _lr_lambda(max(_rw, 1)):.3e}"
              f"（元の run の step {_off + 1} の LR は "
              f"{args.lr * 0.5 * (1 + np.cos(np.pi * (_off - args.warmup) / max(args.steps - args.warmup, 1))):.3e}）", flush=True)

    @torch.no_grad()
    def evaluate(n_batches: int = 20) -> float:
        model.eval()
        tot, n = 0.0, 0
        for i, b in enumerate(dl_te):
            if i >= n_batches:
                break
            obs, act = to_observation(b, device)
            tot += model(obs, act).mean().item()
            n += 1
        model.train()
        if dpo_eval_sets:
            dpo_eval()
        return tot / max(n, 1)

    dpo_eval_last: dict = {}

    @torch.no_grad()
    def dpo_eval() -> None:
        """held-out 対（と学習対）の margin を、決め打ちの試料・固定の乱数（ε, t, 画像拡張）で測る（§15 (iii)）。
        学習の乱数の流れを変えないよう、乱数の状態を保存して戻す。"""
        rng_cpu, rng_cuda = torch.get_rng_state(), torch.cuda.get_rng_state(device)
        torch.manual_seed(12345)
        model.eval()
        parts = []
        for name, ds in dpo_eval_sets.items():
            dl = DataLoader(ds, batch_size=args.dpo_batch, shuffle=False, num_workers=2, collate_fn=collate_pairs)
            ms, gaps = [], []
            for bw, bl, _w in dl:
                _l, m, g = dpo_loss_fn(model, parc_lora, bw, bl, device, args.dpo_beta, args.dpo_action_dims)
                ms.append(m.float().cpu())
                gaps.append(g)
            m = torch.cat(ms)
            se = float(m.std() / max(len(m), 1) ** 0.5) if len(m) > 1 else float("nan")
            dpo_eval_last[name] = dict(margin=float(m.mean()), pos=float((m > 0).float().mean()), se=se, n=len(m))
            parts.append(f"{name} margin={float(m.mean()):+.3e} pos={float((m > 0).float().mean()):.3f} se={se:.1e} n={len(m)}")
        model.train()
        torch.set_rng_state(rng_cpu)
        torch.cuda.set_rng_state(rng_cuda, device)
        print("[dpo-eval] " + " | ".join(parts), flush=True)

    # ---- ★学習前のベースライン損失（LoRA は B=0 で恒等なので素の π0.5 の値）----
    base_loss = evaluate()
    if not resume:
        print(f"[baseline] 未学習（LoRA は恒等）の test loss = {base_loss:.4f}")
    else:
        print(f"[baseline] resume 時点（step {args.start_step}）の test loss = {base_loss:.4f}")

    hist = {"step": [], "train_loss": [], "test_loss": [], "lr": []}
    model.train()
    it = iter(dl_tr)
    t0 = time.perf_counter()
    run_loss, run_n = 0.0, 0

    step0 = args.start_step if resume else 0   # resume でなければ 0（従来と同一）
    if dl_dpo is not None:
        it_dpo = iter(dl_dpo)
        dpo_acc = {"bc": 0.0, "dpo": 0.0, "margin": 0.0, "acc": 0.0, "ref_gap": 0.0, "n": 0}
        torch.cuda.reset_peak_memory_stats()
    for step in range(step0 + 1, args.steps + 1):
        try:
            batch = next(it)
        except StopIteration:
            it = iter(dl_tr)
            batch = next(it)

        obs, act = to_observation(batch, device)
        if args.wrist_mask_closed_p > 0.0:
            # 操作チェック: 手首カメラがマスクされたサンプルの割合を積算して定期的に出す
            _wm = batch["image_mask"]["left_wrist_0_rgb"]
            wm_masked = locals().get("wm_masked", 0) + int((~_wm.bool()).sum())
            wm_total = locals().get("wm_total", 0) + int(_wm.numel())
            if step in (50, 500) or step % 1000 == 0:
                print(f"[wristmask] step {step}: 手首マスク率 {wm_masked}/{wm_total} = {wm_masked / wm_total:.3f}", flush=True)
        loss = model(obs, act).mean()
        if dl_dpo is None:
            loss.backward()
        else:
            # BC と DPO は別々に backward して勾配を足す（活性化を同時に持たない）
            (args.dpo_bc_weight * loss).backward()
            try:
                bw, bl, bwt = next(it_dpo)
            except StopIteration:
                it_dpo = iter(dl_dpo)
                bw, bl, bwt = next(it_dpo)
            d_loss, margin, ref_gap = dpo_loss_fn(model, parc_lora, bw, bl, device, args.dpo_beta,
                                                  args.dpo_action_dims, weight=bwt)
            if step == step0 + 1 and not resume:
                # 操作チェック: 学習前は LoRA の B=0 なので方策 = 参照、margin は厳密に 0（参照の作りと乱数の固定の確認）
                assert float(margin.abs().max()) < 1e-6, f"★step 1 の margin が 0 でない: {margin.tolist()}"
            d_loss.backward()
            dpo_acc["bc"] += loss.item()
            dpo_acc["dpo"] += d_loss.item()
            dpo_acc["margin"] += float(margin.mean())
            dpo_acc["acc"] += float((margin > 0).float().mean())
            dpo_acc["ref_gap"] += ref_gap
            dpo_acc["n"] += 1
            if step % args.log_every == 0:
                k = max(dpo_acc["n"], 1)
                print(f"[dpo] step {step} bc={dpo_acc['bc']/k:.4f} dpo={dpo_acc['dpo']/k:.4f} "
                      f"margin={dpo_acc['margin']/k:+.3e} acc={dpo_acc['acc']/k:.2f} "
                      f"ref(Lw-Ll)={dpo_acc['ref_gap']/k:+.3e} "
                      f"peak_alloc {torch.cuda.max_memory_allocated()/2**30:.1f}GB", flush=True)
                dpo_acc = {kk: 0.0 for kk in dpo_acc}
                dpo_acc["n"] = 0
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        optim.step()
        sched.step()
        optim.zero_grad(set_to_none=True)

        if ema_shadow is not None:
            with torch.no_grad():
                d = args.ema_decay
                for n, p in named_train_params:
                    ema_shadow[n].mul_(d).add_(p.detach(), alpha=1.0 - d)

        run_loss += loss.item()
        run_n += 1

        if step % args.log_every == 0:
            el = time.perf_counter() - t0
            print(f"  step {step:5d}/{args.steps}  loss {run_loss/run_n:.4f}  "
                  f"lr {sched.get_last_lr()[0]:.2e}  {el/(step-step0):.2f}s/step  "
                  f"残り {(args.steps-step)*el/(step-step0)/60:.0f}分  "
                  f"peak {torch.cuda.max_memory_reserved()/2**30:.1f}GB", flush=True)
            run_loss, run_n = 0.0, 0

        if step % args.eval_every == 0 or step == args.steps:
            te = evaluate()
            hist["step"].append(step)
            hist["test_loss"].append(te)
            hist["lr"].append(sched.get_last_lr()[0])
            if dpo_eval_sets:
                hist.setdefault("dpo_eval", []).append({"step": step, **dpo_eval_last})
            print(f"  [eval] step {step}: test loss {te:.4f} "
                  f"(baseline {base_loss:.4f}, {100*(te-base_loss)/base_loss:+.1f}%)", flush=True)
            (run_dir / "history.json").write_text(
                json.dumps({"baseline_test_loss": base_loss, **hist,
                            "args": {k: str(v) for k, v in vars(args).items()}},
                           indent=2, ensure_ascii=False))

        stop_now = args.stop_after > 0 and (step - step0) >= args.stop_after   # 既定 0 では常に False
        if step % args.save_every == 0 or step == args.steps or stop_now:
            sd = ({n: t.cpu() for n, t in ema_shadow.items()} if ema_shadow is not None
                  else parc_lora.lora_state_dict(model))
            torch.save(sd, run_dir / "lora.pt")
            if args.save_optim:
                torch.save({"optim": optim.state_dict(), "step": step}, run_dir / "optim.pt")
            if args.snapshot_every and step % args.snapshot_every == 0:
                torch.save(sd, run_dir / f"lora_s{step}.pt")  # 途中点を残す（step 数の効果を後から測るため）
        if stop_now:
            print(f"[stop-after] この起動で {args.stop_after} step 進めたので停止（step {step}）", flush=True)
            break

    print(f"\n学習完了 {(time.perf_counter()-t0)/60:.1f} 分。LoRA: {run_dir/'lora.pt'}"
          f"{'（EMA 側）' if ema_shadow is not None else ''}")

    # ---- 提出用に畳み込む -------------------------------------------------
    if args.merge_out:
        if ema_shadow is not None:
            # ★保存/マージは EMA 側の重みを使う。ここで生パラメータを EMA へ置き換える
            #   （以降は学習を続けない前提。merge_() 直後に LoRA 層は素の Linear へ戻るので
            #   「生パラメータに戻す」操作はもう出来ない＝この時点で置き換えるのが最後の機会）。
            with torch.no_grad():
                raw_sq = sum(p.detach().float().pow(2).sum() for _n, p in named_train_params)
                for n, p in named_train_params:
                    p.data.copy_(ema_shadow[n].to(p.device, p.dtype))
                ema_sq = sum(p.detach().float().pow(2).sum() for _n, p in named_train_params)
            raw_norm, ema_norm = float(raw_sq) ** 0.5, float(ema_sq) ** 0.5
            print(f"[ema] merge-out は EMA 側の重みを使用。"
                  f"||raw_last_step||={raw_norm:.4f} ||ema||={ema_norm:.4f} "
                  f"diff={abs(raw_norm-ema_norm):.4f}（0 でなければ EMA が実際に効いている証拠）",
                  flush=True)
        n = parc_lora.merge_(model)
        args.merge_out.mkdir(parents=True, exist_ok=True)
        safetensors.torch.save_model(model, str(args.merge_out / "model.safetensors"))
        # assets（norm_stats）と config を丸ごと持ってくる。★これを忘れると推論が壊れる
        import shutil
        for name in ("assets", "config.json"):
            src = args.ckpt / name
            dst = args.merge_out / name
            if src.is_dir():
                shutil.copytree(src, dst, dirs_exist_ok=True)
            elif src.is_file():
                shutil.copy2(src, dst)
        print(f"[merge] {n} 層を畳み込み → {args.merge_out}")
        print("  ★推論側は無改造で使える。submission/config.json の checkpoint_dir を差し替えるだけ")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
