#!/usr/bin/env bash
# 腕 C 段 2: builder（CPU・GO 不要）→ [親の開始フラグ待ち] → 学習（GPU）→ merge（2,400 と 3,600）。冪等（DONE フラグ・出力の存在で飛ばす）。
# 流れ: contrast_ROLLOUT_DONE を待つ → build_contrast_dataset.py（--check 込み。失敗したら止まる）→ contrast_BUILD_DONE
#       → /workspace/logs/contrast_TRAIN_GO（親が置く）を待つ → train_lora_kadai.py（$EXP）→ merge_lora.py ×2 → contrast_TRAIN_DONE
# 使い方: bash kadai/scripts/pod_run_contrast_train.sh          （環境変数: PER_WINDOW=2, STEPS=2400, SNAP=0, MERGE_AT=$STEPS。v1・v3 と同一条件の 2,400 step 1 本）
# 出力: data/contrast/ds_c1(/_holdout)、data/contrast/lora_runs/kadai_c1/lora.pt、data/contrast/policy/c1_s2400/（merge 済み: model.safetensors + assets）
# 3,600 step は前提試験が境界（第 1 チャンク AUROC 0.55〜0.65）のときだけ別 run: EXP=kadai_c1_3600 STEPS=3600 MERGE_AT=3600 で再実行（builder・GO は済みなら飛ばす。policy の出力名は c1_s<N>）。
# ログ: /workspace/logs/contrast_train.log
set -u
cd /workspace/Kadai
ulimit -n 65536 2>/dev/null || ulimit -n 65535 2>/dev/null
export HOME=/root
L=/workspace/logs; LOG=$L/contrast_train.log; C=data/contrast; PY=third_party/openpi/.venv/bin/python
EXP=${EXP:-kadai_c1}; PER_WINDOW=${PER_WINDOW:-2}; STEPS=${STEPS:-2400}; SNAP=${SNAP:-0}; MERGE_AT=${MERGE_AT:-"$STEPS"}
say() { echo "[contrast train $(date '+%F %T')] $*" | tee -a $LOG; }
say "start per_window=$PER_WINDOW steps=$STEPS snap=$SNAP"
until [ -f $L/contrast_ROLLOUT_DONE ]; do sleep 30; done
if [ ! -f $L/contrast_BUILD_DONE ]; then
  rm -rf $C/ds_c1 $C/ds_c1_holdout
  $PY kadai/scripts/build_contrast_dataset.py --traces $C/trace_c --frames-root $C/frames_c --images frames --results "$C/eval_runs/c1roll_*.jsonl" --teacher data/kadai_demos_v1 --out $C/ds_c1 \
      --per-window $PER_WINDOW --holdout-seeds 8080-8099 --ckpt data/openpi/pi05_libero_pytorch --check 2>&1 | grep -v Warning | tee -a $LOG
  [ ${PIPESTATUS[0]} -eq 0 ] || { say "BUILD_FAILED（操作チェック失敗を含む）→ 止める"; exit 4; }
  echo "BUILD_DONE $(date '+%F %T %Z')" > $L/contrast_BUILD_DONE
fi
say "builder 完了。summary: $C/ds_c1/summary.json。親の開始フラグ $L/contrast_TRAIN_GO を待つ"
until [ -f $L/contrast_TRAIN_GO ]; do sleep 30; done
if [ ! -f $C/lora_runs/$EXP/TRAIN_OK ]; then
  say "学習開始"
  mkdir -p $C/lora_runs
  $PY scripts/train_lora_kadai.py --data $C/ds_c1 --ckpt data/openpi/pi05_libero_pytorch --exp-name $EXP --out-dir $C/lora_runs \
      --steps $STEPS --batch 16 --lr 1e-4 --warmup 200 --rank 16 --alpha 32 --targets attn-mlp --seed 0 --workers 16 --n-test-tasks 0 \
      --eval-every 400 --save-every 400 --snapshot-every $SNAP >> $L/contrast_train_$EXP.log 2>&1
  grep -q "学習完了" $L/contrast_train_$EXP.log && [ -f $C/lora_runs/$EXP/lora.pt ] || { say "TRAIN_FAILED（学習完了の行か lora.pt が無い）→ $L/contrast_train_$EXP.log"; exit 5; }
  touch $C/lora_runs/$EXP/TRAIN_OK
fi
for N in $MERGE_AT; do
  OUT=$C/policy/c1_s$N
  if [ ! -f $OUT/model.safetensors ]; then
    say "merge s$N → $OUT"
    LP=$C/lora_runs/$EXP/lora_s$N.pt; [ -f $LP ] || { [ $N = $STEPS ] && LP=$C/lora_runs/$EXP/lora.pt; }   # SNAP=0 のときは最終の lora.pt
    $PY scripts/merge_lora.py --lora $LP --out $OUT --ckpt data/openpi/pi05_libero_pytorch --rank 16 --alpha 32 --targets attn-mlp >> $LOG 2>&1 \
      || { say "MERGE_FAILED s$N"; exit 6; }
  fi
done
echo "TRAIN_DONE $(date '+%F %T %Z')" > $L/contrast_TRAIN_DONE
say "完了"
