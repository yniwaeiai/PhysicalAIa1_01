"""openpi の serve_policy.py と同じ方策サーバに、要求ごとの「初期ノイズのスケール」を足したラッパ（openpi のソースは変えない）。

フローマッチングの行動は、初期ノイズ x0 ~ N(0, I)（形状 (action_horizon, action_dim) = (10, 32)）から流れで積分して作る。
クライアントが obs["_noise_scale"] = τ を付けると、x0 = τ · N(0, I) をサーバ側で生成して Policy.infer(obs, noise=x0) に渡す。
  τ = 1（または未指定）: 通常のサンプリングと同じ分布（ただし乱数はこのラッパが引く）
  τ > 1: 初期ノイズを大きくする（探索を増やす = 停留した方策の別の解を引く狙い）
  τ < 1: 小さくする（より決定的な出力に近づく）
obs["_noise_seed"] を付けると x0 を再現可能にする。サーバは入力から _noise_* を取り除いてから推論する。
使い方: serve_policy_noise.py --port 18011 --config pi05_libero --dir <方策ディレクトリ>
"""
import argparse, logging, socket
import numpy as np
from openpi.policies import policy_config as _policy_config
from openpi.serving import websocket_policy_server
from openpi.training import config as _config


class NoiseScaledPolicy:
    def __init__(self, policy, action_horizon=10, action_dim=32):
        self._p, self._H, self._D = policy, action_horizon, action_dim
        self._rng = np.random.default_rng(0)

    @property
    def metadata(self):
        return self._p.metadata

    def infer(self, obs):
        obs = dict(obs)
        scale = obs.pop("_noise_scale", None)
        seed = obs.pop("_noise_seed", None)
        noise = None
        if scale is not None or seed is not None:
            rng = np.random.default_rng(int(seed)) if seed is not None else self._rng
            noise = (float(scale) if scale is not None else 1.0) * rng.standard_normal((self._H, self._D)).astype(np.float32)
        out = self._p.infer(obs, noise=noise) if noise is not None else self._p.infer(obs)
        out["noise_scale_used"] = np.float32(1.0 if scale is None else scale)
        return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=18011)
    ap.add_argument("--config", default="pi05_libero")
    ap.add_argument("--dir", required=True)
    a = ap.parse_args()
    policy = _policy_config.create_trained_policy(_config.get_config(a.config), a.dir)
    logging.info("Creating server (host: %s, ip: %s)", socket.gethostname(), socket.gethostbyname(socket.gethostname()))
    websocket_policy_server.WebsocketPolicyServer(policy=NoiseScaledPolicy(policy), host="0.0.0.0", port=a.port, metadata=policy.metadata).serve_forever()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, force=True)
    main()
