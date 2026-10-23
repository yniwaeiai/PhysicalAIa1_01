#!/usr/bin/env bash
# 腕 C 段 1: 対照データの rollout（v1 方策・strong・seed 8000〜8099 × 8 本 × 3 タスク、24 並列 = 1 タスク 8 シャード）。GPU は方策サーバ v1（18040・起動済み）をクライアントとして使うだけ。
# ★起動前に親（research）の合図を待つこと。サーバは起動も停止もしない（18040 / 18041 に触れない）。
# 使い方: bash kadai/scripts/pod_run_contrast_rollouts.sh [--smoke]     （--smoke: 各タスク 2 seed × 2 rep。出力名が c1smoke になり本番と混ざらない）
# ★画像は --save-frames --frame-every 1 で毎 step 残す（rep ごとに別プロセス・別ディレクトリ。rep 番号は eval_kadai_closedloop_rep.py の KADAI_REP_OFFSET）。
#   方策の標本化乱数はサーバ側で呼び出しごとに違う（_noise_seed を付けない）ので、同じ seed の 8 本は別標本になる。builder が act_exec の rep 間一致を assert で検査する。
# 出力: data/contrast/trace_c/{task}_s{seed}_r{rep}.npz（trace v2）、data/contrast/frames_c/r{rep}/{task}_s{seed}.npz（毎 step の画像・ee）、data/contrast/eval_runs/c1roll_{task}_r{rep}.json(.jsonl)（rep・clean・success 入り）
# 完了フラグ: /workspace/logs/contrast_ROLLOUT_DONE   ログ: /workspace/logs/contrast_eval_c1roll.log
set -u
SMOKE=0; [ "${1:-}" = "--smoke" ] && SMOKE=1
source /workspace/Kadai/kadai/scripts/pod_contrast_lib.sh
PORT=${PORT:-18040}
grep -q "server listening on 0.0.0.0:$PORT" /workspace/logs/serve_v1*.log 2>/dev/null || ss -ltn 2>/dev/null | grep -q ":$PORT " || { echo "方策サーバ v1 (port $PORT) が起動していない"; exit 3; }
if [ $SMOKE = 1 ]; then NAME=c1smoke; TR=$CDIR/trace_c_smoke; FR=$CDIR/frames_c_smoke; SEEDS=8000,8001; REP=2; FLAG=$LOGD/contrast_ROLLOUT_SMOKE_DONE
else NAME=c1roll; TR=$CDIR/trace_c; FR=$CDIR/frames_c; SEEDS=$(seq -s, 8000 8099); REP=8; FLAG=$LOGD/contrast_ROLLOUT_DONE; fi
rm -f $FLAG; PAR=${PAR:-12}; mkdir -p $TR $FR $CDIR/eval_runs
echo "[contrast rollouts] start $(date '+%F %T %Z') name=$NAME port=$PORT seeds=${SEEDS:0:30}... reps=$REP（(task,rep) $((REP*3)) ジョブ、PAR=${PAR:-12} 並列）trace=$TR frames=$FR（毎 step の画像、約 15 GB）" | tee -a $LOGD/contrast_eval_$NAME.log
# ★クライアント（= シャード）数は PAR 以下（既定 12。親の制約: 24 にしない）。ジョブ = (task, rep) の 24 本を PAR 本ずつ。
#   途中で並列数を変えるには: 実行中の本スクリプトとクライアントを止め（pkill -f pod_run_contrast_rollouts; pkill -f eval_kadai_closedloop_rep）、PAR=<新しい数> で同じコマンドを再実行。
#   済みの episode（trace + frames がある）は飛ばして残りだけ回す。止めた瞬間に走っていた 1 episode/クライアントは捨てて再実行になる。
PAR=${PAR:-12}
export -f ctr_rollout_rep ctr_env _ctr_filter
export K CDIR LOGD REF NAME PORT TR FR SEEDS
for r in $(seq 0 $((REP-1))); do for t in drawer k4 k6; do echo "$t $r"; done; done | xargs -P $PAR -L1 bash -c 'source /workspace/Kadai/kadai/scripts/pod_contrast_lib.sh; ctr_rollout_rep $NAME $PORT $0 "$SEEDS" $1 $TR $FR' 
echo "CONTRAST_ROLLOUT_DONE $NAME $(date '+%F %T %Z')" | tee -a $LOGD/contrast_eval_$NAME.log > $FLAG
