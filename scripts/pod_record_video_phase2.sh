cd /workspace/Kadai
export HOME=/root TORCHINDUCTOR_CACHE_DIR=/workspace/cache/torchinductor LP_NUM_THREADS=2 MUJOCO_GL=osmesa PYOPENGL_PLATFORM=osmesa LIBERO_ROOT=/workspace/Kadai/third_party/LIBERO-plus LIBERO_CONFIG_PATH=/root/.libero PYTHONPATH=/workspace/Kadai/third_party/LIBERO-plus
ulimit -n 1048576 2>/dev/null
C=".venv-plus/bin/python kadai/scripts/eval_kadai_closedloop.py --replan 5 --level mix --frame-every 1"
pids=()
for pol in base:18030 v1:18031; do n=${pol%%:*}; p=${pol##*:}
  for ts in drawer:1002,1004,1006 k4:1004,1014,1021 k6:1000,1001,1002; do t=${ts%%:*}; sl=${ts##*:}
    $C --port $p --tasks $t --seed-list $sl --episodes 3 --save-frames data/frames/video_$n --out data/eval_runs/video_${n}_$t.json > /workspace/logs/video_${n}_$t.log 2>&1 &
    pids+=($!)
  done
done
wait "${pids[@]}"
echo "VIDEO_ALL_DONE $(date)"
for p in $(pgrep -f "[s]erve_policy.*1803"); do kill $p; done
