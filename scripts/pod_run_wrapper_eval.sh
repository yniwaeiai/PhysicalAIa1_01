#!/usr/bin/env bash
# W 本番（計画 v2 §11）。24 ワーカー: drawer 7 / k4 5 / k6 12。完了フラグ: /workspace/logs/WRAPPER_EVAL_ALL_DONE。再実行すると済みの (seed, rep, arm) はスキップ。
source /workspace/Kadai/kadai/scripts/pod_env_branches.sh
ulimit -n 1048576 2>/dev/null || ulimit -n 65535 2>/dev/null
OUT=data/wrapper_eval; mkdir -p $OUT
launch() { # task workers
  for w in $(seq 0 $(($2-1))); do
    .venv-plus/bin/python kadai/scripts/run_wrapper_eval.py --task $1 --worker $w --workers $2 --port 18040 --ref-pose kadai/state/ref_pose_20261015.json --out-dir $OUT 2>&1 \
      | grep --line-buffered -vE "WARNING|Gym|gym|migration|EGL|OpenGL|File \"|^\s+(raise|EGL)" >> /workspace/logs/wrapper_eval_$1_w$w.log &
  done
}
launch drawer 7; launch k4 5; launch k6 12
wait
echo "WRAPPER_EVAL_ALL_DONE $(date '+%F %T %Z')" >> /workspace/logs/WRAPPER_EVAL_ALL_DONE
