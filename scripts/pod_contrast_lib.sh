#!/usr/bin/env bash
# 腕 C の Pod スクリプト共通の関数（source して使う。単体では何もしない）。既存の pod_run_eval.sh / pod_run_targeted.sh には依存せず（rm -f の範囲が違うため）、
# 出力を /workspace/Kadai/data/contrast/ と /workspace/logs/contrast_* に閉じる。ロジックは両スクリプトの写し（クライアント起動部）。
# 関数:
#   ctr_env                                    クライアント共通の環境変数と ulimit
#   ctr_eval NAME PORT CLIENT EPS TASKS LEVEL SOFF SHARDS [extra...]       seed-offset 方式（pod_run_eval.sh と同じ分割）
#   ctr_targeted NAME PORT CLIENT TASK SEEDS REPEAT SHARDS LEVEL [extra...] seed-list 方式（pod_run_targeted.sh と同じ分割）
#   ctr_rollout_rep NAME PORT TASK SEEDS REP TRACE_DIR FRAMES_ROOT   rollout 用（rep 別プロセス・毎 step の画像つき）
#   ctr_wait_listen LOGFILE PORT [TIMEOUT_S]   サーバのログに "server listening on 0.0.0.0:<PORT>" が出るのを待つ（ポートが開くだけでは不十分）
# CLIENT は kadai/scripts/ 配下のクライアント名（eval_kadai_closedloop.py または eval_kadai_closedloop_cfg.py）。
# 出力: data/contrast/eval_runs/<NAME>_<task>_s<k>.json(.jsonl)、ログ /workspace/logs/contrast_eval_<NAME>.log
K=/workspace/Kadai
CDIR=$K/data/contrast
LOGD=/workspace/logs
REF=kadai/state/ref_pose_20261015.json

ctr_env() {
  ulimit -n 65536 2>/dev/null || ulimit -n 65535 2>/dev/null   # DataLoader と hdf5 / 環境再生成の fd 枯渇対策
  export HOME=/root LP_NUM_THREADS=${LP_NUM_THREADS:-2} MUJOCO_GL=osmesa PYOPENGL_PLATFORM=osmesa LIBERO_ROOT=$K/third_party/LIBERO-plus LIBERO_CONFIG_PATH=/root/.libero PYTHONPATH=$K/third_party/LIBERO-plus
  mkdir -p $CDIR/eval_runs
}

_ctr_filter() { grep --line-buffered -vE "WARNING|Gym|gym|migration|EGL|OpenGL|File \"|^\s+(raise|EGL)"; }

ctr_eval() {
  local NAME=$1 PORT=$2 CLIENT=$3 EPS=$4 TASKS=$5 LEVEL=$6 SOFF=$7 SHARDS=$8; shift 8
  cd $K; ctr_env
  rm -f $CDIR/eval_runs/${NAME}_*   # ★NAME は c1 系の接頭辞のみ（他腕の出力に触れない）
  local PER=$(( (EPS + SHARDS - 1) / SHARDS )) pids=() t s
  for t in ${TASKS//,/ }; do for s in $(seq 0 $((SHARDS-1))); do
    .venv-plus/bin/python kadai/scripts/$CLIENT --tasks $t --episodes $PER --level $LEVEL --replan 5 --max-steps 250 --seed-offset $((SOFF + s*PER)) --port $PORT \
       --ref-pose $REF --out $CDIR/eval_runs/${NAME}_${t}_s$s.json "$@" 2>&1 | _ctr_filter | sed "s/^/[$NAME $t s$s] /" >> $LOGD/contrast_eval_$NAME.log &
    pids+=($!)
  done; done
  wait "${pids[@]}"
  echo "[$NAME] EVAL_DONE $(date '+%F %T %Z')" >> $LOGD/contrast_eval_$NAME.log
}

ctr_targeted() {
  local NAME=$1 PORT=$2 CLIENT=$3 TASK=$4 SEEDS=$5 REPEAT=$6 SHARDS=$7 LEVEL=$8; shift 8
  cd $K; ctr_env
  rm -f $CDIR/eval_runs/${NAME}_${TASK}_*
  local N PER pids=() s
  N=$(python3 -c "print(len('$SEEDS'.split(',')) * $REPEAT)"); PER=$(( (N + SHARDS - 1) / SHARDS ))
  for s in $(seq 0 $((SHARDS-1))); do
    .venv-plus/bin/python kadai/scripts/$CLIENT --tasks $TASK --port $PORT --level $LEVEL --replan 5 --max-steps 250 --ref-pose $REF --seed-list "$SEEDS" --repeat $REPEAT \
       --shard-start $((s*PER)) --episodes $PER --out $CDIR/eval_runs/${NAME}_${TASK}_s$s.json "$@" 2>&1 | _ctr_filter | sed "s/^/[$NAME $TASK s$s] /" >> $LOGD/contrast_eval_$NAME.log &
    pids+=($!)
  done
  wait "${pids[@]}"
  echo "[$NAME $TASK] TARGETED_DONE $(date '+%F %T %Z')" >> $LOGD/contrast_eval_$NAME.log
}

# ctr_rollout_rep NAME PORT TASK SEEDS REP TRACE_DIR FRAMES_ROOT   1 本の (task, rep) を 1 プロセスで（seed-list × repeat 1）。毎 step の画像を FRAMES_ROOT/r<REP>/ に残す
ctr_rollout_rep() {
  local NAME=$1 PORT=$2 TASK=$3 SEEDS=$4 REP=$5 TR=$6 FR=$7
  cd $K; ctr_env; mkdir -p $FR/r$REP $TR
  # ★冪等: trace と frames の両方がある seed は飛ばし、残りだけ回す（途中で止めて並列数を変えて再実行できる）。jsonl は追記（再実行で消さない）
  local REM; REM=$(python3 - "$SEEDS" "$TASK" "$REP" "$TR" "$FR" <<'PY'
import sys, os
seeds, task, rep, tr, fr = sys.argv[1:]
print(",".join(s for s in seeds.split(",") if not (os.path.exists(f"{tr}/{task}_s{s}_r{rep}.npz") and os.path.exists(f"{fr}/r{rep}/{task}_s{s}.npz"))))
PY
)
  [ -z "$REM" ] && { echo "[$NAME $TASK r$REP] 全 seed 済み" >> $LOGD/contrast_eval_$NAME.log; return 0; }
  local N; N=$(python3 -c "print(len('$REM'.split(',')))")
  KADAI_REP_OFFSET=$REP .venv-plus/bin/python kadai/scripts/eval_kadai_closedloop_rep.py --tasks $TASK --port $PORT --level strong --replan 5 --max-steps 250 --ref-pose $REF \
     --seed-list "$REM" --repeat 1 --shard-start 0 --episodes $N --trace $TR --save-frames $FR/r$REP --frame-every 1 \
     --out $CDIR/eval_runs/${NAME}_${TASK}_r$REP.json 2>&1 | _ctr_filter | sed "s/^/[$NAME $TASK r$REP] /" >> $LOGD/contrast_eval_$NAME.log
  echo "[$NAME $TASK r$REP] ROLLOUT_DONE $(date '+%F %T %Z')" >> $LOGD/contrast_eval_$NAME.log
}

ctr_wait_listen() {
  local LOG=$1 PORT=$2 T=${3:-1200} i
  for i in $(seq 1 $((T/5))); do grep -q "server listening on 0.0.0.0:$PORT" $LOG 2>/dev/null && return 0; sleep 5; done
  return 1
}
