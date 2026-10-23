#!/usr/bin/env bash
# R-a 収集の rollout（v1 方策・seed 5000〜5099 × 3 タスク、24 並列）。完了フラグ: /workspace/logs/REC_ROLLOUT_DONE
cd /workspace/Kadai
SEEDS=$(seq -s, 5000 5099)
TR="--trace /workspace/Kadai/data/trace_rec --ref-pose kadai/state/ref_pose_20261015.json --max-steps 250 --replan 5"
for t in drawer k4 k6; do
  bash kadai/scripts/pod_run_targeted.sh rec 18040 $t $SEEDS 1 8 mix $TR &
done
wait
echo "REC_ROLLOUT_DONE $(date '+%F %T %Z')" >> /workspace/logs/REC_ROLLOUT_DONE
