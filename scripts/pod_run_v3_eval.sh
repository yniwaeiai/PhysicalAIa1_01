#!/usr/bin/env bash
# v3 評価（計画 v2 §18 + GPT-6 第 2 回 採用 (6)〜(8)）: 1 モデル分の 5 条件を順に流す。共有の方策サーバ（起動済み）に当てる。サーバは止めない。
# 使い方: bash pod_run_v3_eval.sh <名前 v1r|v3|...> <ポート> [--smoke]
#   例（後の担当）: bash pod_run_v3_eval.sh v1r 18040 &  bash pod_run_v3_eval.sh v3 18041 &   （同時に起動する。クライアント合計 約 12〜24、LP_NUM_THREADS=2）
# 前提: そのポートの方策サーバが起動済みで、サーバのログが /workspace/logs/serve_<名前>.log（SERVER_LOG で変更可）。
#       ★新しい重みのサーバは初回推論で torch.compile が数分かかる。多クライアントを当てる前にこのスクリプトが 1 本だけウォームアップを通す（WARMUP=0 で省略）。
#       （v2 の評価で、起動直後に 24 クライアントを当てて keepalive timeout で 4 シャードが落ちた。履歴 §22）
# 条件（各条件とも 250 step, --ref-pose, replan 5, OSMesa）:
#   E1  strong 3000〜3047 各タスク 48        → 出力 <名前>_E1_<task>_s<k>
#   E2  mix    1000〜1047 各タスク 48        → <名前>_E2_*
#   E3s strong 7000〜7047 各タスク 48        → <名前>_E3s_*
#   E3m mix    6000〜6047 各タスク 48        → <名前>_E3m_*
#   T   標的 34 seed × 8 反復（272 本）。mix の 20 seed → <名前>_T_<task>_s<k>、strong（保留）の 14 seed → <名前>_Ts_<task>_s<k>
#       （水準は 1 回の呼び出しに 1 つしか渡せないので mix と strong を別の名前にした。解析は両方を T として束ねる）
# 完了フラグ: /workspace/logs/<名前>_EVAL_ALL_DONE（--smoke のときは <名前>_SMOKE_DONE、出力名は <名前>_smoke_* で本番と混ざらない）
# 並列度: 既定は条件を順に（各 6 クライアント）、T は最大 10。E_PARALLEL=1 なら E 系 4 条件を同時に（24 クライアント）。
# ★ rm -f の範囲: pod_run_eval.sh は data/eval_runs/<NAME>_* を消す。NAME は <名前>_E1 等なので、v1r_E1_* と v3_E1_* は互いに触れない。
#   ただし <名前> 同士が前方一致（例 v3 と v3c ではなく "v3_" の後ろが違う）しないこと。名前に '_' を含めないこと。
set -u
MODEL=${1:?名前}; PORT=${2:?ポート}; MODE=${3:-}
cd /workspace/Kadai
SMOKE=0; [ "$MODE" = "--smoke" ] && SMOKE=1
case "$MODEL" in *_*) echo "名前に _ を含めない: $MODEL"; exit 2;; esac
M=$MODEL; [ $SMOKE = 1 ] && M=${MODEL}_smoke
export SERVER_LOG=${SERVER_LOG:-/workspace/logs/serve_${MODEL}.log}
export REUSE_SERVER=1
REF=kadai/state/ref_pose_20261015.json
export EXTRA="--ref-pose $REF --max-steps 250"
export HOME=/root
FLAG=/workspace/logs/${MODEL}_EVAL_ALL_DONE; [ $SMOKE = 1 ] && FLAG=/workspace/logs/${MODEL}_SMOKE_DONE
rm -f $FLAG
LOG=/workspace/logs/v3eval_${M}.log
echo "[v3eval $M] start $(date '+%F %T %Z') port=$PORT smoke=$SMOKE server_log=$SERVER_LOG" > $LOG

# --- サーバ起動待ち + ウォームアップ 1 本 ---
for i in $(seq 1 240); do grep -q "server listening on 0.0.0.0:$PORT" $SERVER_LOG 2>/dev/null && break; sleep 5; done
grep -q "server listening on 0.0.0.0:$PORT" $SERVER_LOG 2>/dev/null || { echo "[v3eval $M] SERVER_NOT_UP" >> $LOG; exit 3; }
if [ "${WARMUP:-1}" = 1 ]; then
  echo "[v3eval $M] warmup $(date '+%T')" >> $LOG
  ulimit -n 1048576 2>/dev/null || ulimit -n 65535 2>/dev/null
  mkdir -p data/eval_runs_warm
  LP_NUM_THREADS=2 MUJOCO_GL=osmesa PYOPENGL_PLATFORM=osmesa LIBERO_ROOT=/workspace/Kadai/third_party/LIBERO-plus LIBERO_CONFIG_PATH=/root/.libero PYTHONPATH=/workspace/Kadai/third_party/LIBERO-plus \
    .venv-plus/bin/python kadai/scripts/eval_kadai_closedloop.py --tasks drawer --episodes 1 --seed-offset 9990 --level none --port $PORT --ref-pose $REF --max-steps 250 --out data/eval_runs_warm/${M}_warm.json 2>&1 \
    | grep -vE "WARNING|Gym|gym|migration|EGL|OpenGL|File \"|^\s+(raise|EGL)" | tail -3 >> $LOG
  echo "[v3eval $M] warmup done $(date '+%T')" >> $LOG
fi

# --- E 系 ---
if [ $SMOKE = 1 ]; then EPS=1; SH=1; else EPS=48; SH=2; fi
runE() { # 条件名 水準 seed_offset
  bash kadai/scripts/pod_run_eval.sh ${M}_$1 none $PORT $EPS drawer,k4,k6 $2 5 $3 $SH
  echo "[v3eval $M] $1 done $(date '+%T')" >> $LOG
}
if [ "${E_PARALLEL:-0}" = 1 ]; then
  runE E1 strong 3000 & runE E2 mix 1000 & runE E3s strong 7000 & runE E3m mix 6000 & wait
else
  runE E1 strong 3000; runE E2 mix 1000; runE E3s strong 7000; runE E3m mix 6000
fi

# --- T（標的 34 seed。群ラベルは kadai/state/expert_eval_summary_20261016.csv の mode=i 行）---
if [ $SMOKE = 1 ]; then
  T_drawer_mix=1045; T_k4_mix=1043; T_k6_mix=1113; T_drawer_strong=3100; T_k4_strong=3103; T_k6_strong=3103; REP=1; SHD=1; SHK4=1; SHK6=1
else
  T_drawer_mix=1014,1016,1027,1030,1037,1039,1045; T_k4_mix=1043; T_k6_mix=1014,1113,1193,1200,1253,1300,1337,1341,1390,1413,1442,1492
  T_drawer_strong=3100,3101,3102,3103,3104,3105,3106,3107; T_k4_strong=3103,3105; T_k6_strong=3103,3104,3105,3107
  REP=8; SHD=4; SHK4=2; SHK6=4
fi
runT() { # task mix_seeds strong_seeds shards
  bash kadai/scripts/pod_run_targeted.sh ${M}_T $PORT $1 $2 $REP $4 mix $EXTRA
  bash kadai/scripts/pod_run_targeted.sh ${M}_Ts $PORT $1 $3 $REP $4 strong $EXTRA
  echo "[v3eval $M] T $1 done $(date '+%T')" >> $LOG
}
runT drawer $T_drawer_mix $T_drawer_strong $SHD &
runT k4 $T_k4_mix $T_k4_strong $SHK4 &
runT k6 $T_k6_mix $T_k6_strong $SHK6 &
wait
echo "EVAL_ALL_DONE $M $(date '+%F %T %Z')" | tee -a $LOG > $FLAG
