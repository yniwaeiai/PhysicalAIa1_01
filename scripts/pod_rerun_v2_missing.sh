#!/usr/bin/env bash
# v2 評価で、サーバの初回コンパイル中に keepalive で切断されて落ちた 4 シャードを再実行する。
# ★先にウォームアップ（1 クライアント、1 エピソード）を通して torch.compile を終わらせてから、残りを並列に投入する（以前は全クライアントを同時に当てて一部が切断された）。
# 落ちたシャード: v2_strong drawer s0・s3、v2_mix drawer s1、v2_mix k6 s0（各 12 エピソード、seed = オフセット + シャード × 12）。
cd /workspace/Kadai
export HOME=/root TORCHINDUCTOR_CACHE_DIR=/workspace/cache/torchinductor LP_NUM_THREADS=2 MUJOCO_GL=osmesa PYOPENGL_PLATFORM=osmesa LIBERO_ROOT=/workspace/Kadai/third_party/LIBERO-plus LIBERO_CONFIG_PATH=/root/.libero PYTHONPATH=/workspace/Kadai/third_party/LIBERO-plus
ulimit -n 1048576 2>/dev/null
(cd third_party/openpi && setsid .venv/bin/python scripts/serve_policy.py --port 18020 policy:checkpoint --policy.config pi05_libero --policy.dir /workspace/Kadai/data/openpi/pi05_kadai_v2 > /workspace/logs/serve_v2_rerun.log 2>&1 < /dev/null) &
for i in $(seq 1 240); do grep -q "server listening on 0.0.0.0:18020" /workspace/logs/serve_v2_rerun.log 2>/dev/null && break; sleep 5; done
grep -q "server listening on 0.0.0.0:18020" /workspace/logs/serve_v2_rerun.log || { echo "SERVER_NOT_UP"; exit 3; }
C=".venv-plus/bin/python kadai/scripts/eval_kadai_closedloop.py --port 18020 --replan 5"
echo "[rerun] warmup start $(date)"
$C --tasks drawer --episodes 1 --seed-offset 5000 --level none --out data/eval_runs/warmup_v2.json > /workspace/logs/rerun_warmup.log 2>&1
echo "[rerun] warmup done $(date)"
run() { name=$1; task=$2; level=$3; soff=$4; $C --tasks $task --episodes 12 --level $level --seed-offset $soff --out data/eval_runs/${name}_${task}_s$5.json > /workspace/logs/rerun_${name}_${task}_s$5.log 2>&1 & }
run v2_strong drawer strong 3000 0
run v2_strong drawer strong $((3000+36)) 3
run v2_mix drawer mix $((1000+12)) 1
run v2_mix k6 mix 1000 0
wait
echo "RERUN_ALL_DONE $(date)"
for p in $(pgrep -f "[s]erve_policy.*18020"); do kill $p; done
