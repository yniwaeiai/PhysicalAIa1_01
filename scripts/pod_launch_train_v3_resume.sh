#!/usr/bin/env bash
# v3 の再開: 元の run が step 2000 で死んだ（ディスククォータ超過、10/14 22:21-22:31 UTC）。
# step 2000 の lora.pt から残り 400 step を同じレシピで続ける（AdamW の状態は無いので --resume-warmup 100）。
cd /workspace/Kadai
export HOME=/root HF_HUB_DISABLE_IMPLICIT_TOKEN=1 TORCHINDUCTOR_CACHE_DIR=/workspace/cache/torchinductor HF_HOME=/workspace/cache/hf
echo "[kadai_v3r] resume start $(date -u "+%F %T") UTC"
third_party/openpi/.venv/bin/python scripts/train_lora_kadai.py --data data/kadai_demos_v3 --ckpt data/openpi/pi05_libero_pytorch --exp-name kadai_v3r --steps 2400 --batch 16 --lr 1e-4 --warmup 200 --rank 16 --alpha 32 --targets attn-mlp --workers 16 --n-test-tasks 0 --eval-every 400 --save-every 400 --snapshot-every 800 --seed 0 --out-dir /workspace/Kadai/data/lora_runs --resume-lora data/lora_runs/kadai_v3/lora.pt --start-step 2000 --resume-warmup 100
echo "[kadai_v3r] TRAIN_RC=$? $(date -u "+%F %T") UTC"
