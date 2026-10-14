#!/usr/bin/env bash
# Pod 上で 方策サーバ（openpi venv, GPU）1 つ + 閉ループ評価クライアント（LIBERO venv, OSMesa）をタスク×シャードで並列に実行する。
# 使い方: bash pod_run_eval.sh <名前> <方策ディレクトリ(重み+assets)> <ポート> <エピソード数/タスク> <タスク,...> <摂動水準> [replan] [seed-offset] [シャード数/タスク]
# 出力: /workspace/Kadai/data/eval_runs/<名前>_<タスク>_s<シャード>.json(.jsonl)、ログ /workspace/logs/eval_<名前>.log
# 集計: python kadai/scripts/summarize_eval.py data/eval_runs/<名前>
set -u
NAME=$1; PDIR=$2; PORT=$3; EPS=$4; TASKS=$5; LEVEL=$6; REPLAN=${7:-5}; SOFF=${8:-1000}; SHARDS=${9:-4}
cd /workspace/Kadai
ulimit -n 1048576 2>/dev/null || ulimit -n 65535 2>/dev/null   # ★環境の再生成で fd がリークする（1 プロセス約 20 エピソードで "Too many open files"）
mkdir -p data/eval_runs; rm -f data/eval_runs/${NAME}_*
export HOME=/root HF_HUB_DISABLE_IMPLICIT_TOKEN=1 TORCHINDUCTOR_CACHE_DIR=/workspace/cache/torchinductor
LOG=/workspace/logs/eval_$NAME.log; SLOG=/workspace/logs/serve_$NAME.log
echo "[eval $NAME] start $(date '+%F %T %Z') policy=$PDIR tasks=$TASKS eps=$EPS level=$LEVEL replan=$REPLAN seed_off=$SOFF shards=$SHARDS" > $LOG
# REUSE_SERVER=1: 既に同じポートで起動済みの方策サーバを使う（複数条件を 1 つのサーバに並列で当てる）。終了時も止めない
if [ -z "${REUSE_SERVER:-}" ]; then
(cd third_party/openpi && setsid .venv/bin/python scripts/serve_policy.py --port $PORT policy:checkpoint --policy.config pi05_libero --policy.dir /workspace/Kadai/$PDIR > $SLOG 2>&1 < /dev/null) &
fi
# ★ポートが開いているだけでは不十分（8001 は nginx が占有していた）。サーバ自身のログに "server listening on 0.0.0.0:<PORT>" が出るのを待つ
# REUSE_SERVER のときは共有サーバのログ（SERVER_LOG）を見る。★以前は存在しない serve_<名前>.log を 20 分待ち続けた
WAITLOG=${SERVER_LOG:-$SLOG}
for i in $(seq 1 240); do
  grep -q "server listening on 0.0.0.0:$PORT" $WAITLOG 2>/dev/null && break; sleep 5
done
grep -q "server listening on 0.0.0.0:$PORT" $WAITLOG 2>/dev/null || { echo "[eval $NAME] SERVER_NOT_UP（$WAITLOG に起動メッセージが無い）" >> $LOG; exit 3; }
echo "[eval $NAME] server up after ~$((i*5))s" >> $LOG
export LP_NUM_THREADS=${LP_NUM_THREADS:-2} MUJOCO_GL=osmesa PYOPENGL_PLATFORM=osmesa LIBERO_ROOT=/workspace/Kadai/third_party/LIBERO-plus LIBERO_CONFIG_PATH=/root/.libero PYTHONPATH=/workspace/Kadai/third_party/LIBERO-plus
PER=$(( (EPS + SHARDS - 1) / SHARDS ))
# ★初回推論は torch.compile で数分かかる。全クライアントが同時に最初の推論を投げて待つだけなので問題ない
pids=()
for t in ${TASKS//,/ }; do
  for s in $(seq 0 $((SHARDS-1))); do
    .venv-plus/bin/python kadai/scripts/eval_kadai_closedloop.py --tasks $t --episodes $PER --level $LEVEL --replan $REPLAN --seed-offset $((SOFF + s*PER)) --port $PORT --out data/eval_runs/${NAME}_${t}_s$s.json ${EXTRA:-} 2>&1 | grep --line-buffered -vE "WARNING|Gym|gym|migration|EGL|OpenGL|File \"|^\s+(raise|EGL)" | sed "s/^/[$t s$s] /" >> $LOG &
    pids+=($!)
  done
done
wait "${pids[@]}"
echo "[eval $NAME] EVAL_DONE $(date '+%F %T %Z')" >> $LOG
[ -z "${REUSE_SERVER:-}" ] && for p in $(pgrep -f "serve_policy.py --port $PORT"); do kill $p 2>/dev/null; done
