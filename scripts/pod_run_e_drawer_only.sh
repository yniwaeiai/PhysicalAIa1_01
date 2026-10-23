#!/usr/bin/env bash
# E1 のみ（v3 の再開損傷の切り分け）。使い方: bash pod_run_e1_only.sh <名前(_なし)> <ポート> [--smoke]  SERVER_LOG 必須
set -u
MODEL=$1; PORT=$2; MODE=${3:-}
cd /workspace/Kadai
export REUSE_SERVER=1 HOME=/root
REF=kadai/state/ref_pose_20261015.json
export EXTRA="--ref-pose $REF --max-steps 250"
LOG=/workspace/logs/edrawer_${MODEL}.log
for i in $(seq 1 240); do grep -q "server listening on 0.0.0.0:$PORT" $SERVER_LOG 2>/dev/null && break; sleep 5; done
grep -q "server listening on 0.0.0.0:$PORT" $SERVER_LOG || { echo SERVER_NOT_UP > $LOG; exit 3; }
ulimit -n 1048576 2>/dev/null
mkdir -p data/eval_runs_warm
echo "warmup $(date +%T)" > $LOG
LP_NUM_THREADS=2 MUJOCO_GL=osmesa PYOPENGL_PLATFORM=osmesa LIBERO_ROOT=/workspace/Kadai/third_party/LIBERO-plus LIBERO_CONFIG_PATH=/root/.libero PYTHONPATH=/workspace/Kadai/third_party/LIBERO-plus \
  .venv-plus/bin/python kadai/scripts/eval_kadai_closedloop.py --tasks drawer --episodes 1 --seed-offset 9990 --level none --port $PORT --ref-pose $REF --max-steps 250 --out data/eval_runs_warm/${MODEL}_warm.json 2>&1 | grep -E "SUMMARY|success=" >> $LOG
echo "warmup done $(date +%T)" >> $LOG
if [ "$MODE" = "--smoke" ]; then
  bash kadai/scripts/pod_run_eval.sh ${MODEL}smoke_E1 none $PORT 1 drawer strong 5 3000 1
else
  for C in "E2 1000" "E3m 6000"; do set -- $C; bash kadai/scripts/pod_run_eval.sh ${MODEL}_$1 none $PORT 48 drawer mix 5 $2 4; done
fi
echo "E1 done $(date +%T)" >> $LOG
touch /workspace/logs/${MODEL}_EDRAWER_DONE${MODE:+_smoke}
