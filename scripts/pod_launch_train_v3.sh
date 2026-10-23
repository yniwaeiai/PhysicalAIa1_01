#!/usr/bin/env bash
cd /workspace/Kadai
export HOME=/root HF_HUB_DISABLE_IMPLICIT_TOKEN=1 TORCHINDUCTOR_CACHE_DIR=/workspace/cache/torchinductor HF_HOME=/workspace/cache/hf
echo "[kadai_v3] train start $(date -u "+%F %T") UTC"
third_party/openpi/.venv/bin/python scripts/train_lora_kadai.py --data data/kadai_demos_v3 --ckpt data/openpi/pi05_libero_pytorch --exp-name kadai_v3 --steps 2400 --batch 16 --lr 1e-4 --warmup 200 --rank 16 --alpha 32 --targets attn-mlp --workers 16 --n-test-tasks 0 --eval-every 400 --save-every 400 --snapshot-every 800 --seed 0 --out-dir /workspace/Kadai/data/lora_runs
echo "[kadai_v3] TRAIN_RC=$? $(date -u "+%F %T") UTC"
