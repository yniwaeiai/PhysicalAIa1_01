#!/usr/bin/env bash
# 観察研究 A'（計画 v2 §3）: 標的 30 seed × 4 反復（level mix = 元の標的評価 run_exp2.sh の t2_T0 と同じ）+ 保留集合（strong・seed 3100〜3107 × 2 反復）。
# 共有サーバ（port 18040）に当てる。trace v2 を /workspace/Kadai/data/trace_aprime/ へ。完了フラグ: /workspace/logs/targeted_aprime*.log の "TARGETED_DONE" と /workspace/logs/APRIME_ALL_DONE。
cd /workspace/Kadai
TR="--trace /workspace/Kadai/data/trace_aprime"
P=18040
bash kadai/scripts/pod_run_targeted.sh aprime $P drawer 1014,1016,1027,1030,1037,1039,1045 4 4 mix $TR &
bash kadai/scripts/pod_run_targeted.sh aprime $P k4 1000,1024,1043 4 4 mix $TR &
bash kadai/scripts/pod_run_targeted.sh aprime $P k6 1014,1113,1139,1177,1180,1193,1200,1253,1300,1337,1341,1352,1390,1413,1417,1428,1442,1461,1488,1492 4 8 mix $TR &
for t in drawer k4 k6; do
  bash kadai/scripts/pod_run_targeted.sh aprime_hold $P $t 3100,3101,3102,3103,3104,3105,3106,3107 2 2 strong $TR &
done
wait
echo "APRIME_ALL_DONE $(date '+%F %T %Z')" >> /workspace/logs/APRIME_ALL_DONE
