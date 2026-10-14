#!/usr/bin/env bash
# 動画用: 素の π0.5-LIBERO と LoRA v1 を同じ seed・同じ摂動で走らせ、毎ステップのコマを保存する（--frame-every 1）。
# 先に 1 クライアントでウォームアップ（torch.compile 中の keepalive 切断を避ける）。出力 data/frames/video_{base,v1}/<task>_s<seed>.npz
cd /workspace/Kadai
export HOME=/root TORCHINDUCTOR_CACHE_DIR=/workspace/cache/torchinductor LP_NUM_THREADS=2 MUJOCO_GL=osmesa PYOPENGL_PLATFORM=osmesa LIBERO_ROOT=/workspace/Kadai/third_party/LIBERO-plus LIBERO_CONFIG_PATH=/root/.libero PYTHONPATH=/workspace/Kadai/third_party/LIBERO-plus
ulimit -n 1048576 2>/dev/null
serve() { (cd third_party/openpi && setsid .venv/bin/python scripts/serve_policy.py --port $1 policy:checkpoint --policy.config pi05_libero --policy.dir $2 > /workspace/logs/serve_video_$1.log 2>&1 < /dev/null) & }
serve 18030 /workspace/Kadai/data/openpi/pi05_libero_pytorch
serve 18031 /workspace/Kadai/data/openpi/pi05_kadai_v1_final
for p in 18030 18031; do
  for i in $(seq 1 240); do grep -q "server listening on 0.0.0.0:$p" /workspace/logs/serve_video_$p.log 2>/dev/null && break; sleep 5; done
  grep -q "server listening on 0.0.0.0:$p" /workspace/logs/serve_video_$p.log || { echo "SERVER_NOT_UP $p"; exit 3; }
done
echo "servers up $(date)"
C=".venv-plus/bin/python kadai/scripts/eval_kadai_closedloop.py --replan 5 --level mix --frame-every 1"
for pol in base:18030 v1:18031; do n=${pol%%:*}; p=${pol##*:}
  $C --port $p --tasks k6 --seed-list 1000 --episodes 1 --save-frames data/frames/video_warm_$n --out data/eval_runs/video_warm_$n.json > /workspace/logs/video_warm_$n.log 2>&1 &
  [ $n = base ] && WP1=$! || WP2=$!
done; wait $WP1 $WP2; echo "warmup done $(date)"
for pol in base:18030 v1:18031; do n=${pol%%:*}; p=${pol##*:}
  for ts in drawer:1002,1004,1006 k4:1004,1014,1021 k6:1000,1001,1002; do t=${ts%%:*}; sl=${ts##*:}
    $C --port $p --tasks $t --seed-list $sl --episodes 3 --save-frames data/frames/video_$n --out data/eval_runs/video_${n}_$t.json > /workspace/logs/video_${n}_$t.log 2>&1 &
  done
  done
done; wait $(jobs -p | grep -v "^$(pgrep -f serve_policy | tr '\n' '|')" ) 2>/dev/null
echo "VIDEO_ALL_DONE $(date)"
for p in $(pgrep -f "[s]erve_policy.*1803"); do kill $p; done
