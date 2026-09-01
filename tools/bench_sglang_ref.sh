#!/usr/bin/env bash
# SGLang reference on the same GPU, using the same repeated/raw benchmark schema
# and the same feasible context/concurrency rungs as knivesysl.
set -euo pipefail
cd /home/shooting-brake007/srswti/qwen38/knivesysl
PY=.venv/bin/python
SG=/tmp/sglang-env/bin/sglang
TGT=/home/shooting-brake007/models/qwen38-27b-nvfp4-radixark
DRAFT=/home/shooting-brake007/models/dspark-draft
LOG=/tmp/gembench
RAW=$LOG/raw/sglang
mkdir -p "$LOG" "$RAW"
LIVE_TOKENS=0

wait_http() {
  local url=$1 pid=$2 name=$3 logfile=$4
  for _ in $(seq 1 240); do
    curl -s -m 3 "$url" >/dev/null 2>&1 && return 0
    if [ -n "$pid" ] && ! kill -0 "$pid" 2>/dev/null; then
      echo "!! $name exited before becoming ready"
      tail -12 "$logfile"
      return 1
    fi
    sleep 2
  done
  echo "!! timed out waiting for $name"
  tail -12 "$logfile"
  return 1
}

wait_gpu_idle() {
  for _ in $(seq 1 120); do
    if ! nvidia-smi --query-compute-apps=process_name --format=csv,noheader 2>/dev/null |
         grep -Eqi 'python|sglang|knivesysl|EngineCore'; then
      return 0
    fi
    sleep 1
  done
  echo "!! model GPU process did not exit"
  nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv,noheader
  return 1
}

stop_servers() {
  pkill -f serve_prod.sh 2>/dev/null || true
  pkill -f serve_batched.py 2>/dev/null || true
  pkill -f serve_openai.py 2>/dev/null || true
  pkill -f "sglang serve" 2>/dev/null || true
  wait_gpu_idle
}

restore_prod() {
  pkill -f "sglang serve" 2>/dev/null || true
  if curl -s -m 3 http://127.0.0.1:8000/v1/models >/dev/null 2>&1; then
    echo "PRODUCTION-RESTORED"
    return 0
  fi
  wait_gpu_idle
  nohup tools/serve_prod.sh >> /tmp/knivesysl_serve.log 2>&1 &
  wait_http http://127.0.0.1:8000/v1/models "" production /tmp/knivesysl_serve.log
  echo "PRODUCTION-RESTORED"
}

on_exit() {
  local rc=$?
  trap - EXIT
  restore_prod || true
  exit "$rc"
}
trap on_exit EXIT

boot() {   # name extra-args...
  local name=$1
  shift
  local logfile=$LOG/sgl_$name.log pidfile=$LOG/sgl_$name.pid
  stop_servers
  (
    cd /tmp/sglang-src
    env PATH=/tmp/sglang-env/bin:/usr/local/cuda/bin:$PATH MAX_JOBS=6 NVCC_THREADS=1 \
      nohup "$SG" serve --trust-remote-code \
        --model-path "$TGT" --served-model-name ref --kv-cache-dtype fp8_e4m3 \
        --attention-backend flashinfer --reasoning-parser qwen3 \
        --tool-call-parser qwen3_coder --chunked-prefill-size 2048 \
        --mamba-radix-cache-strategy extra_buffer_lazy --mamba-ssm-dtype bfloat16 \
        --host 127.0.0.1 --port 30000 "$@" > "$logfile" 2>&1 &
    echo $! > "$pidfile"
  )
  local pid
  pid=$(cat "$pidfile")
  wait_http http://127.0.0.1:30000/v1/models "$pid" "$name" "$logfile"
  LIVE_TOKENS=$(grep -oE "max_total_num_tokens=[0-9]+" "$logfile" | tail -1 | cut -d= -f2)
  if [ -z "$LIVE_TOKENS" ] || [ "$LIVE_TOKENS" -le 0 ]; then
    echo "!! could not parse max_total_num_tokens for $name"
    tail -12 "$logfile"
    return 1
  fi
  echo "CONFIG-ASSERTED sglang name=$name capacity=$LIVE_TOKENS"
}

frontier_context() {
  local n=$1
  local ctx=$((LIVE_TOKENS / n - 256))
  [ "$ctx" -lt 256 ] && ctx=256
  echo $((ctx / 128 * 128))
}

cell() {   # label n contexts only kind repeats
  local label=$1 n=$2 contexts=$3 only=$4 kind=$5 repeats=$6
  local extra=()
  [ -n "$only" ] && extra+=(--only "$only")
  echo "=== $label capacity=$LIVE_TOKENS ==="
  timeout 14400 "$PY" tools/bench_spec_matrix.py \
    --url http://127.0.0.1:30000 --model ref \
    --engine sglang --output-dir "$RAW" --label "$label" \
    --contexts "$contexts" --ns "$n" "${extra[@]}" \
    --slots "$n" --pool-tokens "$LIVE_TOKENS" --gen 192 \
    --repeats "$repeats" --spec-kind "$kind" \
    2>&1 | grep -E "^ctx|^wrote"
}

run_plain() {   # concurrency
  local n=$1
  local cache=$((n * 4)) name=sgl-plain-n$n
  if ! boot "$name" --mem-fraction-static 0.90 --max-running-requests "$n" \
      --cuda-graph-max-bs "$n" --max-mamba-cache-size "$cache"; then
    echo "GRAPH-BOOT-FAILED n=$n; retrying eager"
    name=sgl-plain-n$n-eager
    boot "$name" --mem-fraction-static 0.90 --max-running-requests "$n" \
      --disable-cuda-graph --max-mamba-cache-size "$cache"
  fi
  local frontier
  frontier=$(frontier_context "$n")
  local contexts="2048,8192,16384,32768,65536,94208,131072,196608,240000,261120,$frontier"
  cell "$name" "$n" "$contexts" "" plain 3
}

# Same concurrency rungs as knivesysl; pool clipping records impossible cells.
run_plain 1
run_plain 2
run_plain 4
run_plain 8
run_plain 12
run_plain 15
run_plain 16

# External speculative reference. Its 11,666-token pool only admits 2k and 8k.
boot sgl-dspark-n1 --mem-fraction-static 0.88 --max-running-requests 1 \
  --cuda-graph-max-bs 1 --mamba-full-memory-ratio 5.61 \
  --speculative-algorithm DSPARK --speculative-draft-model-path "$DRAFT" \
  --speculative-draft-attention-backend flashinfer
cell sgl-dspark-n1 1 "2048,8192" "2048:1,8192:1" dspark 3
