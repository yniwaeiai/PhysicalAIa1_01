#!/usr/bin/env bash
# I3 本番（計画 v2 §10）。使い方: bash pod_run_branches_i3.sh <シャード数>。完了フラグ: /workspace/logs/BRANCHES_I3_ALL_DONE
source /workspace/Kadai/kadai/scripts/pod_env_branches.sh
N=$1; mkdir -p data/branches_i3
for s in $(seq 0 $((N-1))); do
  .venv-plus/bin/python kadai/scripts/run_branches.py --i3 --shard $s --shards $N --out data/branches_i3 2>&1 | grep --line-buffered -vE "WARNING|Gym|gym|migration|EGL|OpenGL|File \"|^\s+(raise|EGL)" >> /workspace/logs/branches_i3_s$s.log &
done
wait
echo "BRANCHES_I3_ALL_DONE $(date "+%F %T %Z")" >> /workspace/logs/BRANCHES_I3_ALL_DONE
