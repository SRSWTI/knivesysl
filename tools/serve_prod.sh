#!/usr/bin/env bash
# knivesysl production server under a restart wrapper.
# - restarts on ANY exit (native crash, engine 8-strike exit(70), OOM kill)
# - core dumps enabled so a native death leaves an artifact for gdb
cd "$(dirname "$0")/.."
ulimit -c unlimited
export CUDA_VISIBLE_DEVICES=0 TQ_KV_Q4=1 TQ_CTX=262144 TQ_EMBED_FP8=2 TQ_W_NVFP4=all
# n-gram verify archive: the agentic/coding workload is its measured win zone
# (+37-60% single-stream through 32k, cost-gated at depth). 4-node archive
# (606 MB): 8 nodes left too little headroom for deep-prefill transients
# (26K clients hit rc=-94 at the tail wave with only 4.1 GB free). The engine
# now claims the archive eagerly at init; TQ_PAGED_ATTN_V3 rides its in-code
# auto default.
export TQ_PAGED_SPEC=${TQ_PAGED_SPEC:-1} TQ_PG_SPEC_NODES=${TQ_PG_SPEC_NODES:-4}
# Host-tier checkpoint demotion is opt-in only until Track D validates it under
# long-context concurrency. The 2026-09-03 prod wedge hit this untested path.
export TQ_CKPT_HOST_GB=${TQ_CKPT_HOST_GB:-0}
while true; do
  .venv/bin/python -u tools/serve_batched.py \
    --tqf /home/shooting-brake007/models/knivesysl/qwen3_8-27b-e2m3-mtp.tqf \
    --model-dir /home/shooting-brake007/models/knivesysl \
    --model-name knivesysl-axe-28b --max-slots 2 --num-blocks 2100 --port 8000
  rc=$?
  echo "[wrapper] server exited rc=$rc at $(date -Is); restarting in 3s" >&2
  sleep 3
done
