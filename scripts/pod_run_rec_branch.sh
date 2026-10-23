#!/usr/bin/env bash
# R-a 収集の分岐（trace の t ∈ {5..40} で復元 → expert）。使い方: bash pod_run_rec_branch.sh <シャード数>。完了フラグ: /workspace/logs/REC_COLLECT_ALL_DONE
source /workspace/Kadai/kadai/scripts/pod_env_branches.sh
N=$1; mkdir -p data/rec_collect
for s in $(seq 0 $((N-1))); do
  .venv-plus/bin/python kadai/scripts/collect_rec.py --mode branch --traces data/trace_rec --out data/rec_collect --parts data/kadai_demos_rec_parts --shard $s --shards $N 2>&1 | grep --line-buffered -vE "WARNING|Gym|gym|migration|EGL|OpenGL|File \"|^\s+(raise|EGL)" >> /workspace/logs/rec_branch_s$s.log &
done
wait
echo "REC_COLLECT_ALL_DONE $(date '+%F %T %Z')" >> /workspace/logs/REC_COLLECT_ALL_DONE
