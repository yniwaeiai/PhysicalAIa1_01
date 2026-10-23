#!/usr/bin/env bash
# 腕 C 段 3: 誘導サーバ（port 18042）・操作チェック・評価。段ごとに別の呼び出し（親が段ごとに合図できる）。v1 の 18040 / v3 の 18041 には触れない（クライアントとして当てるのみ）。
# 使い方:
#   bash kadai/scripts/pod_run_contrast_eval.sh serve [POLICY]      誘導サーバ起動 → 1 クライアントでウォームアップ。POLICY 既定 c1_s2400（data/contrast/policy/ 配下）。ログ contrast_serve_c1.log
#   bash kadai/scripts/pod_run_contrast_eval.sh check               check_cfg_server.py（w=0 が通常経路と median<1e-3 かつ max<1e-2、w=3/w=0 の行動の大きさ、cfg_ratio）。不合格なら exit 1
#   bash kadai/scripts/pod_run_contrast_eval.sh e1t                 E1（strong 3000〜3047 各タスク 48）+ T（標的 34 seed × 8）を {v1(18040), C(w=0), C(1.5), C(3)} で。PAR=2 条件ずつ並列（既定）
#   bash kadai/scripts/pod_run_contrast_eval.sh e3 <w*>             E3s（strong 7000〜）+ E3m（mix 6000〜）各タスク 48 を {v1, C(w*)} で
#   bash kadai/scripts/pod_run_contrast_eval.sh stop                誘導サーバ（18042 を所有する PID）だけ止める
#   末尾に --smoke を付けると各条件 1 ep の煙検査（出力名 c1smoke*）
# 出力: data/contrast/eval_runs/c1_<cond>_<E1|T|Ts|E3s|E3m>_<task>_s<k>.json(.jsonl)  cond = v1 / w0 / w15 / w3 / wstar（C の w*）、v1 は E3 用に v1e3
# 完了フラグ: /workspace/logs/contrast_E1T_DONE、contrast_E3_DONE
# T の標的 seed は pod_run_v3_eval.sh と同じ 34 seed（mix 20 + strong 14）。
set -u
CMD=${1:?serve|check|e1t|e3|stop}; shift
SMOKE=0; ARGS=(); for x in "$@"; do [ "$x" = "--smoke" ] && SMOKE=1 || ARGS+=("$x"); done
source /workspace/Kadai/kadai/scripts/pod_contrast_lib.sh
cd $K
CP=18042; SLOG=$LOGD/contrast_serve_c1.log; V1=${V1_PORT:-18040}; PAR=${PAR:-2}
P=c1; [ $SMOKE = 1 ] && P=c1smoke
owner_pid() { ss -ltnp 2>/dev/null | grep ":$CP " | sed -n 's/.*pid=\([0-9]*\).*/\1/p' | head -1; }

case $CMD in
serve)
  POLICY=${ARGS[0]:-c1_s2400}
  [ -n "$(owner_pid)" ] && { echo "port $CP は既に使われている（PID $(owner_pid)）。stop してから"; exit 3; }
  [ -f $CDIR/policy/$POLICY/model.safetensors ] || { echo "$CDIR/policy/$POLICY が無い"; exit 3; }
  ulimit -n 65536 2>/dev/null; export HOME=/root TORCHINDUCTOR_CACHE_DIR=/workspace/cache/torchinductor
  (cd third_party/openpi && setsid nohup .venv/bin/python $K/kadai/scripts/serve_policy_cfg.py --port $CP --config pi05_libero --dir $CDIR/policy/$POLICY > $SLOG 2>&1 < /dev/null &)
  ctr_wait_listen $SLOG $CP 1200 || { echo "SERVER_NOT_UP: $SLOG"; exit 3; }
  echo "[serve] 起動 PID=$(owner_pid) policy=$POLICY（起動直後の初回推論は遅い。1 クライアントでウォームアップ）"
  mkdir -p $CDIR/eval_runs_warm
  KADAI_CFG_W=0 ctr_eval ${P}_warm $CP eval_kadai_closedloop_cfg.py 1 drawer none 9990 1 >/dev/null 2>&1
  tail -2 $LOGD/contrast_eval_${P}_warm.log
  echo "SERVER_READY policy=$POLICY $(date '+%F %T %Z')" | tee $LOGD/contrast_SERVER_READY ;;
check)
  TF=$(ls $CDIR/trace_c/drawer_s80*_r0.npz 2>/dev/null | head -3 | tr '\n' ' ')
  [ -z "$TF" ] && TF=$(ls $CDIR/trace_c/*.npz | head -3 | tr '\n' ' ')
  .venv-plus/bin/python kadai/scripts/check_cfg_server.py --trace-file $(echo $TF | cut -d" " -f1) --task drawer --n-obs 3 --cfg-port $CP 2>&1 | tee -a $LOGD/contrast_check.log
  exit ${PIPESTATUS[0]} ;;
e1t)
  if [ $SMOKE = 1 ]; then EPS=1; SH=1; REP=1; SD=1; SK4=1; SK6=1
    T_dm=1045; T_k4m=1043; T_k6m=1113; T_ds=3100; T_k4s=3103; T_k6s=3103
  else EPS=48; SH=2; REP=8; SD=4; SK4=2; SK6=4
    T_dm=1014,1016,1027,1030,1037,1039,1045; T_k4m=1043; T_k6m=1014,1113,1193,1200,1253,1300,1337,1341,1390,1413,1442,1492
    T_ds=3100,3101,3102,3103,3104,3105,3106,3107; T_k4s=3103,3105; T_k6s=3103,3104,3105,3107; fi
  cond() { # 名前 ポート クライアント w
    local N=${P}_$1 PORT=$2 CL=$3 W=$4
    export KADAI_CFG_W=$W
    ctr_eval ${N}_E1 $PORT $CL $EPS drawer,k4,k6 strong 3000 $SH
    ctr_targeted ${N}_T $PORT $CL drawer $T_dm $REP $SD mix & ctr_targeted ${N}_T $PORT $CL k4 $T_k4m $REP $SK4 mix & ctr_targeted ${N}_T $PORT $CL k6 $T_k6m $REP $SK6 mix & wait
    ctr_targeted ${N}_Ts $PORT $CL drawer $T_ds $REP $SD strong & ctr_targeted ${N}_Ts $PORT $CL k4 $T_k4s $REP $SK4 strong & ctr_targeted ${N}_Ts $PORT $CL k6 $T_k6s $REP $SK6 strong & wait
    echo "[cond $N] done $(date '+%T')" | tee -a $LOGD/contrast_eval_${P}.log
  }
  # v1 は通常クライアント（prompt に結果トークンを付けない）。C は cfg クライアント（成功トークン付き）。ctr_targeted の名前は <cond>_T なので ${N}_T_<task> に出る
  [ -n "$(owner_pid)" ] || { echo "誘導サーバ $CP が起動していない（serve を先に）"; exit 3; }
  rm -f $LOGD/contrast_E1T_DONE
  conds=("v1 $V1 eval_kadai_closedloop.py 0" "w0 $CP eval_kadai_closedloop_cfg.py 0" "w15 $CP eval_kadai_closedloop_cfg.py 1.5" "w3 $CP eval_kadai_closedloop_cfg.py 3")
  i=0; for c in "${conds[@]}"; do ( cond $c ) & i=$((i+1)); [ $((i % PAR)) -eq 0 ] && wait; done; wait
  echo "E1T_DONE $(date '+%F %T %Z')" | tee $LOGD/contrast_E1T_DONE ;;
e3)
  WS=${ARGS[0]:?w* を渡す（例 1.5）}
  if [ $SMOKE = 1 ]; then EPS=1; SH=1; else EPS=48; SH=2; fi
  [ -n "$(owner_pid)" ] || { echo "誘導サーバ $CP が起動していない"; exit 3; }
  rm -f $LOGD/contrast_E3_DONE
  ( export KADAI_CFG_W=0; ctr_eval ${P}_v1e3s $V1 eval_kadai_closedloop.py $EPS drawer,k4,k6 strong 7000 $SH; ctr_eval ${P}_v1e3m $V1 eval_kadai_closedloop.py $EPS drawer,k4,k6 mix 6000 $SH ) &
  ( export KADAI_CFG_W=$WS; ctr_eval ${P}_wstar_E3s $CP eval_kadai_closedloop_cfg.py $EPS drawer,k4,k6 strong 7000 $SH; ctr_eval ${P}_wstar_E3m $CP eval_kadai_closedloop_cfg.py $EPS drawer,k4,k6 mix 6000 $SH ) &
  wait
  echo "E3_DONE w*=$WS $(date '+%F %T %Z')" | tee $LOGD/contrast_E3_DONE ;;
stop)
  p=$(owner_pid); [ -n "$p" ] && kill $p && echo "killed $p (port $CP owner)" || echo "port $CP は空き" ;;
esac
