#!/usr/bin/env bash
# 第 2 弾の実験を一括で投入（ラッパサーバ 18011 を共有）。全体評価 3 条件 + 標的評価 3 条件。
# 事前: 標的 seed のリスト（/workspace/Kadai/state_exp2/seeds_<タスク>.txt）を作っておく。
cd /workspace/Kadai
export SERVER_LOG=/workspace/logs/serve_noise.log REUSE_SERVER=1
M=data/openpi/pi05_kadai_v1_final
ov() { NAME=$1; shift; EXTRA="$*" bash kadai/scripts/pod_run_eval.sh $NAME $M 18011 48 drawer,k4,k6 mix 5 1000 4 > /workspace/logs/eval_$NAME.out 2>&1 & }
ov v2_F10 --flow-noise 1.0
ov v2_F15 --flow-noise 1.5
ov v2_S25 --qstall-resample 2.5
tg() { NAME=$1; shift; for t in drawer k4 k6; do case $t in drawer) SH=4;; k4) SH=2;; k6) SH=8;; esac
   bash kadai/scripts/pod_run_targeted.sh $NAME 18011 $t "$(cat state_exp2/seeds_$t.txt)" 8 $SH mix "$@" & done; }
tg t2_T0
tg t2_TF15 --flow-noise 1.5
tg t2_TS25 --qstall-resample 2.5
wait
echo "EXP2_ALL_DONE $(date)"
