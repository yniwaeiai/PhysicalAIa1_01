#!/usr/bin/env bash
# 既知の失敗 seed を、同じ初期条件のまま方策のノイズだけ変えて反復実行する（共有サーバを使う。サーバは止めない）。
# 使い方: bash pod_run_targeted.sh <名前> <ポート> <タスク> <seed カンマ区切り> <反復数> <シャード数> <摂動水準> [EXTRA の引数...]
# 出力: data/eval_runs/<名前>_<タスク>_s<シャード>.json(.jsonl)
set -u
NAME=$1; PORT=$2; TASK=$3; SEEDS=$4; REPEAT=$5; SHARDS=$6; LEVEL=$7; shift 7; EXTRA="$*"
cd /workspace/Kadai
ulimit -n 1048576 2>/dev/null || ulimit -n 65535 2>/dev/null   # ★環境の再生成で fd がリークする（1 プロセス約 20 エピソードで "Too many open files"）
export HOME=/root LP_NUM_THREADS=2 MUJOCO_GL=osmesa PYOPENGL_PLATFORM=osmesa LIBERO_ROOT=/workspace/Kadai/third_party/LIBERO-plus LIBERO_CONFIG_PATH=/root/.libero PYTHONPATH=/workspace/Kadai/third_party/LIBERO-plus
mkdir -p data/eval_runs; rm -f data/eval_runs/${NAME}_${TASK}_*
N=$(python3 -c "print(len('$SEEDS'.split(',')) * $REPEAT)")
PER=$(( (N + SHARDS - 1) / SHARDS ))
pids=()
for s in $(seq 0 $((SHARDS-1))); do
  .venv-plus/bin/python kadai/scripts/eval_kadai_closedloop.py --tasks $TASK --port $PORT --level $LEVEL --seed-list "$SEEDS" --repeat $REPEAT --shard-start $((s*PER)) --episodes $PER \
     --out data/eval_runs/${NAME}_${TASK}_s$s.json $EXTRA 2>&1 | grep --line-buffered -vE "WARNING|Gym|gym|migration|EGL|OpenGL|File \"|^\s+(raise|EGL)" | sed "s/^/[$NAME $TASK s$s] /" >> /workspace/logs/targeted_$NAME.log &
  pids+=($!)
done
wait "${pids[@]}"
echo "[$NAME $TASK] TARGETED_DONE $(date '+%F %T %Z')" >> /workspace/logs/targeted_$NAME.log
