#!/usr/bin/env bash
# Group D: sglang reference on the SAME card, measured with the FIXED bench tool.
#
# Reproduction notes that cost five failed boots (see docs/spec-decode-findings):
#   * PATH must carry the venv (jit shells out to `ninja`) and CUDA.
#   * MAX_JOBS caps the jit fan-out -- unbounded ninja OOMs 59 GB of HOST ram
#     (victim is `cicc`, and it dies with no python traceback).
#   * --max-mamba-cache-size must be pinned per concurrency rung (= n * S, S=4
#     under extra_buffer_lazy). The balanced --mamba-full-memory-ratio grossly
#     over-provisions the gdn state pool here: 31,219 -> 168,487 kv tokens at n=4.
#   * DSpark needs the pinned commit 1cf2b8c (quantized target lm_head in the
#     dflash2 selector) -- the pypi wheel crashes projecting through an fp4 head.
set -u
cd /home/shooting-brake007/srswti/qwen38/knivesysl
PY=.venv/bin/python
SG=/tmp/sglang-env/bin/sglang
TGT=/home/shooting-brake007/models/qwen38-27b-nvfp4-radixark
DRAFT=/home/shooting-brake007/models/dspark-draft
LOG=/tmp/gembench

boot() {   # name extra-args...
  local name=$1; shift
  pkill -f "sglang serve"; pkill -f serve_batched; pkill -f serve_prod.sh; sleep 5
  ( cd /tmp/sglang-src && PATH=/tmp/sglang-env/bin:/usr/local/cuda/bin:$PATH \
      MAX_JOBS=6 NVCC_THREADS=1 nohup $SG serve --trust-remote-code \
      --model-path $TGT --served-model-name ref --kv-cache-dtype fp8_e4m3 \
      --attention-backend flashinfer --reasoning-parser qwen3 \
      --tool-call-parser qwen3_coder --chunked-prefill-size 2048 \
      --mamba-radix-cache-strategy extra_buffer_lazy --mamba-ssm-dtype bfloat16 \
      --host 127.0.0.1 --port 30000 "$@" > $LOG/sgl_$name.log 2>&1 & )
  for _ in $(seq 1 40); do
    sleep 10
    curl -s -m 3 http://127.0.0.1:30000/v1/models >/dev/null 2>&1 && break
  done
  curl -s -m 3 http://127.0.0.1:30000/v1/models >/dev/null || { echo "!! $name did not start"; tail -4 $LOG/sgl_$name.log; return 1; }
  grep -oE "max_total_num_tokens=[0-9]+" $LOG/sgl_$name.log | tail -1
}

cell() {   # label ns only
  echo "=== $1 ==="
  timeout 2400 $PY tools/bench_spec_matrix.py --url http://127.0.0.1:30000 \
    --model ref --label "$1" --ns "$2" --only "$3" --slots 8 \
    --pool-tokens 160000 --gen 192 2>&1 | grep -E "^ctx|wrote"
}

boot plain-n1 --mem-fraction-static 0.90 --max-running-requests 1 --cuda-graph-max-bs 1 --max-mamba-cache-size 4  && cell sgl-plain-n1 1 2048:1,8192:1,32768:1,65536:1,131072:1
boot plain-n2 --mem-fraction-static 0.90 --max-running-requests 2 --cuda-graph-max-bs 2 --max-mamba-cache-size 8  && cell sgl-plain-n2 2 2048:2,8192:2,32768:2
boot plain-n4 --mem-fraction-static 0.90 --max-running-requests 4 --cuda-graph-max-bs 4 --max-mamba-cache-size 16 && cell sgl-plain-n4 4 2048:4,8192:4,32768:4
boot dspark-n1 --mem-fraction-static 0.88 --max-running-requests 1 --cuda-graph-max-bs 1 --mamba-full-memory-ratio 5.61 \
     --speculative-algorithm DSPARK --speculative-draft-model-path $DRAFT \
     --speculative-draft-attention-backend flashinfer && cell sgl-dspark-n1 1 2048:1,8192:1

pkill -f "sglang serve"; sleep 4
nohup tools/serve_prod.sh >> /tmp/knivesysl_serve.log 2>&1 &
sleep 14
curl -s -m 3 http://127.0.0.1:8000/v1/models >/dev/null && echo "PRODUCTION-RESTORED" || echo "!! production did not restore"
