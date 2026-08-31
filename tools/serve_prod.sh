#!/usr/bin/env bash
# knivesysl production server under a restart wrapper.
# - restarts on ANY exit (native crash, engine 8-strike exit(70), OOM kill)
# - core dumps enabled so a native death leaves an artifact for gdb
cd "$(dirname "$0")/.."
ulimit -c unlimited
export CUDA_VISIBLE_DEVICES=0 TQ_KV_Q4=1 TQ_CTX=262144 TQ_EMBED_FP8=2 TQ_W_NVFP4=all
while true; do
  .venv/bin/python -u tools/serve_batched.py \
    --tqf /home/shooting-brake007/models/knivesysl/qwen3_8-27b-e2m3-mtp.tqf \
    --model-dir /home/shooting-brake007/models/knivesysl \
    --model-name knivesysl-axe-28b --max-slots 2 --num-blocks 2100 --port 8000
  rc=$?
  echo "[wrapper] server exited rc=$rc at $(date -Is); restarting in 3s" >&2
  sleep 3
done
