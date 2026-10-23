"""eval_kadai_closedloop.py の薄い派生（腕 C）: 既存スクリプトを import し、make_element（方策への入力を作る関数）だけ差し替える。元のファイルは変更しない。
差分（これだけ）:
  1. prompt の末尾に結果トークンを付ける（既定 " (outcome: success)"。環境変数 KADAI_PROMPT_SUFFIX で変更、"" で無し）
  2. obs に _cfg_w（環境変数 KADAI_CFG_W、既定 0）と _cfg_neg_prompt（= 元の prompt + " (outcome: failure)"。KADAI_NEG_SUFFIX で変更）を足す
     → 誘導サーバ serve_policy_cfg.py（port 18042）が使う。通常サーバに当てても余計なキーは無視されない場合があるので、必ず誘導サーバに向けること。
引数・出力・記録は eval_kadai_closedloop.py と完全に同じ（--trace, --ref-pose, --seed-list, --shard-start など）。
使い方: KADAI_CFG_W=1.5 python eval_kadai_closedloop_cfg.py --tasks drawer --port 18042 ...（以降は元のスクリプトと同じ）
"""
import os, sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import eval_kadai_closedloop as E   # noqa: E402

SUFFIX = os.environ.get("KADAI_PROMPT_SUFFIX", " (outcome: success)")
NEG_SUFFIX = os.environ.get("KADAI_NEG_SUFFIX", " (outcome: failure)")
CFG_W = float(os.environ.get("KADAI_CFG_W", "0"))

_orig_make_element = E.make_element


def make_element(obs, prompt):
    el = _orig_make_element(obs, prompt + SUFFIX)
    el["_cfg_w"] = CFG_W
    el["_cfg_neg_prompt"] = prompt + NEG_SUFFIX
    return el


E.make_element = make_element   # run_episode / branch が参照するモジュール大域を差し替える

if __name__ == "__main__":
    E.main()
