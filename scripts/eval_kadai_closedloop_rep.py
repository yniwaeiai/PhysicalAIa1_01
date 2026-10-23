"""eval_kadai_closedloop.py の薄い派生（腕 C の rollout 用）: rep 番号を環境変数 KADAI_REP_OFFSET だけずらす。元のファイルは変更しない。
理由: 元のスクリプトは --seed-list × --repeat の rep を 0..repeat-1 で固定する。--save-frames の保存名 {task}_s{seed}.npz には rep が入らないため、
rep ごとに別プロセス（--repeat 1、--save-frames を rep 別ディレクトリ）で回す必要があり、そのとき trace 名 {task}_s{seed}_r{rep}.npz と jsonl の rep を
正しい番号（OFFSET + 0）にするための差し替え。差分はこの 1 関数のラップだけ（方策への入力・記録・判定は元のまま）。
使い方: KADAI_REP_OFFSET=3 python eval_kadai_closedloop_rep.py --seed-list 8000,8001 --repeat 1 --save-frames .../r3 --frame-every 1 --trace ... （他は元と同じ）
"""
import os, sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import eval_kadai_closedloop as E   # noqa: E402

OFFSET = int(os.environ.get("KADAI_REP_OFFSET", "0"))
_orig = E.run_episode


def run_episode(env, name, spec, args, client, rows, prompt, seed, rep, state):
    return _orig(env, name, spec, args, client, rows, prompt, seed, rep + OFFSET, state)


E.run_episode = run_episode   # run_task が参照するモジュール大域を差し替える

if __name__ == "__main__":
    E.main()
