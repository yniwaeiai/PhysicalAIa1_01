#!/usr/bin/env bash
# I6 本番（計画 v2 §17）。使い方: bash pod_run_expert_eval.sh。完了フラグ: /workspace/logs/EXPERT_EVAL_ALL_DONE
source /workspace/Kadai/kadai/scripts/pod_env_branches.sh
mkdir -p data/expert_eval
run() { mode=$1; n=$2; for s in $(seq 0 $((n-1))); do
  .venv-plus/bin/python kadai/scripts/run_expert_eval.py --mode $mode --shard $s --shards $n --out data/expert_eval/$mode 2>&1 | grep --line-buffered -vE "WARNING|Gym|gym|migration|EGL|OpenGL|File \"|^\s+(raise|EGL)" >> /workspace/logs/expert_eval_${mode}_s$s.log &
done; }
run i 8; run ii 10; run cal 6
wait
echo "EXPERT_EVAL_ALL_DONE $(date "+%F %T %Z")" >> /workspace/logs/EXPERT_EVAL_ALL_DONE
