"""腕 C（計画 §2-3）の誘導サーバ: 結果条件付きで学習した方策に「負の誘導」（classifier-free guidance の負例版）を掛ける。openpi のソースは変えない。

要求 obs に次があれば、成功 prompt（obs["prompt"] のまま）と失敗 prompt（obs["_cfg_neg_prompt"]）の 2 本を入力変換（tokenizer 込み）に通し、
prefix を 2 本埋め込み、**同じ初期ノイズ**から Euler の各 step で  v = v_s + w · (v_s − v_f)  で積分する（openpi の PI0Pytorch.sample_actions のループを写したもの）。
  _cfg_w          float。w = 0 は v = v_s（通常の推論と同じ。ただし v_f も計算して比を記録する）。
  _cfg_neg_prompt str。失敗側の prompt（例 "close the microwave (outcome: failure)"）。
  _noise_seed     int。初期ノイズを再現可能にする（serve_policy_noise.py と同じ: np.random.default_rng(seed).standard_normal((H, D))）。
  _noise_scale    float。初期ノイズのスケール（serve_policy_noise.py と同じ）。
_cfg_w / _cfg_neg_prompt が無い要求は serve_policy_noise.py と同じ通常の経路（Policy.infer）。
応答に cfg_ratio（step 別の ‖v_s − v_f‖/‖v_s‖、長さ num_steps）を足す。サーバのログに step 別中央値を 100 要求ごとに出す（増幅の実効量。計画 §3-4）。
使い方: serve_policy_cfg.py --port 18042 --config pi05_libero --dir <merge 済みの方策ディレクトリ>
"""
import argparse, logging, socket
import numpy as np


class CfgPolicy:
    def __init__(self, policy, num_steps=10, log_every=100):
        self._p = policy
        m = policy._model
        self._H, self._D = int(m.config.action_horizon), int(m.config.action_dim)
        self._rng = np.random.default_rng(0)
        self._num_steps = num_steps
        self._ratios = []
        self._n = 0
        self._log_every = log_every

    @property
    def metadata(self):
        return self._p.metadata

    # ---- 入力 → Observation（Policy.infer の前半と同じ）----
    def _observation(self, obs):
        import jax, torch
        from openpi.models import model as _model
        inputs = jax.tree.map(lambda x: x, obs)
        inputs = self._p._input_transform(inputs)
        inputs = jax.tree.map(lambda x: torch.from_numpy(np.array(x)).to(self._p._pytorch_device)[None, ...], inputs)
        return inputs, _model.Observation.from_dict(inputs)

    def _prefix(self, observation):
        # PI0Pytorch.sample_actions の前半（prefix の埋め込みとキャッシュ）
        import torch
        from openpi.models_pytorch.pi0_pytorch import make_att_2d_masks
        m = self._p._model
        images, img_masks, lang_tokens, lang_masks, state = m._preprocess_observation(observation, train=False)
        pe, pm, pa = m.embed_prefix(images, img_masks, lang_tokens, lang_masks)
        att4 = m._prepare_attention_masks_4d(make_att_2d_masks(pm, pa))
        pos = torch.cumsum(pm, dim=1) - 1
        m.paligemma_with_expert.paligemma.language_model.config._attn_implementation = "eager"  # noqa: SLF001
        _, pkv = m.paligemma_with_expert.forward(attention_mask=att4, position_ids=pos, past_key_values=None, inputs_embeds=[pe, None], use_cache=True)
        return state, pm, pkv

    def _guided(self, obs, neg_prompt, w, noise_np):
        import torch
        m = self._p._model; dev = self._p._pytorch_device
        obs_f = dict(obs); obs_f["prompt"] = neg_prompt
        inputs, o_s = self._observation(obs)
        _, o_f = self._observation(obs_f)
        with torch.no_grad():
            state, pm_s, kv_s = self._prefix(o_s)
            _, pm_f, kv_f = self._prefix(o_f)
            x = torch.from_numpy(noise_np).to(dev)[None, ...]
            bsize = x.shape[0]
            dt = torch.tensor(-1.0 / self._num_steps, dtype=torch.float32, device=dev)
            t = torch.tensor(1.0, dtype=torch.float32, device=dev)
            ratios = []
            while t >= -dt / 2:
                te = t.expand(bsize)
                v_s = m.denoise_step(state, pm_s, kv_s, x, te)
                v_f = m.denoise_step(state, pm_f, kv_f, x, te)
                ratios.append(float((v_s - v_f).norm() / v_s.norm().clamp_min(1e-12)))
                v = v_s + w * (v_s - v_f) if w != 0.0 else v_s
                x = x + dt * v
                t += dt
        out = {"state": inputs["state"], "actions": x}
        import jax
        out = jax.tree.map(lambda a: np.asarray(a[0, ...].detach().cpu()), out)
        return out, np.asarray(ratios, np.float32)

    def infer(self, obs):
        obs = dict(obs)
        w = obs.pop("_cfg_w", None)
        neg = obs.pop("_cfg_neg_prompt", None)
        scale = obs.pop("_noise_scale", None)
        seed = obs.pop("_noise_seed", None)
        rng = np.random.default_rng(int(seed)) if seed is not None else self._rng
        noise = (float(scale) if scale is not None else 1.0) * rng.standard_normal((self._H, self._D)).astype(np.float32)
        if w is None or neg is None:
            out = self._p.infer(obs, noise=noise) if (scale is not None or seed is not None) else self._p.infer(obs)
            out["cfg_ratio"] = np.zeros(0, np.float32)
            return out
        import time as _t
        t0 = _t.monotonic()
        raw, ratios = self._guided(obs, str(neg), float(w), noise)
        out = self._p._output_transform(raw)
        out["policy_timing"] = {"infer_ms": (_t.monotonic() - t0) * 1000}
        out["cfg_ratio"] = ratios
        out["cfg_w_used"] = np.float32(w)
        self._ratios.append(ratios); self._n += 1
        if self._n % self._log_every == 0:
            r = np.median(np.stack(self._ratios[-self._log_every:]), axis=0)
            logging.info("cfg ‖v_s−v_f‖/‖v_s‖ step別中央値（直近 %d 要求）: %s", self._log_every, np.round(r, 3).tolist())
        return out


def build_parser():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--port", type=int, default=18042)
    ap.add_argument("--config", default="pi05_libero")
    ap.add_argument("--dir", required=True, help="merge 済みの方策ディレクトリ（model.safetensors + assets）")
    return ap


def main():
    a = build_parser().parse_args()
    from openpi.policies import policy_config as _policy_config
    from openpi.serving import websocket_policy_server
    from openpi.training import config as _config
    policy = _policy_config.create_trained_policy(_config.get_config(a.config), a.dir)
    logging.info("Creating server (host: %s, ip: %s)", socket.gethostname(), socket.gethostbyname(socket.gethostname()))
    websocket_policy_server.WebsocketPolicyServer(policy=CfgPolicy(policy), host="0.0.0.0", port=a.port, metadata=policy.metadata).serve_forever()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, force=True)
    main()
