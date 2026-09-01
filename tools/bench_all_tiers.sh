#!/usr/bin/env bash
# Capacity-aware native measurement campaign. Every server is config-asserted,
# every cell is repeated, raw samples are written atomically, and production is
# restored on success, failure, or interruption.
set -euo pipefail
cd /home/shooting-brake007/srswti/qwen38/knivesysl
PY=.venv/bin/python
TQF=/home/shooting-brake007/models/knivesysl/qwen3_8-27b-e2m3-mtp.tqf
MD=/home/shooting-brake007/models/knivesysl
LOG=/tmp/gembench
RAW=$LOG/raw/knivesysl
mkdir -p "$LOG" "$RAW"
LIVE_BLOCKS=0
LIVE_LOG=

wait_http() {   # url pid-or-empty name logfile
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

wait_model_gpu_idle() {
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

teardown() {
  pkill -f serve_prod.sh 2>/dev/null || true
  pkill -f serve_batched.py 2>/dev/null || true
  pkill -f serve_openai.py 2>/dev/null || true
  wait_model_gpu_idle
}

restore_prod() {
  teardown
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

assert_cfg() {   # logfile tier spec slots requested-blocks
  local logfile=$1 tier=$2 spec=$3 slots=$4 requested=$5
  if [ -n "$tier" ]; then
    grep -q "TQ_W_NVFP4=$tier " "$logfile" ||
      { echo "!! tier mismatch: wanted TQ_W_NVFP4=$tier"; return 1; }
  else
    ! grep -q "TQ_W_NVFP4=" "$logfile" ||
      { echo "!! tier mismatch: wanted FP6"; return 1; }
  fi
  local live_spec
  live_spec=$($PY -c "import json,urllib.request; print(json.loads(urllib.request.urlopen('http://127.0.0.1:8000/health',timeout=5).read())['spec']['enabled'])")
  local wanted=False
  [ "$spec" = "1" ] && wanted=True
  [ "$live_spec" = "$wanted" ] ||
    { echo "!! spec mismatch: live=$live_spec wanted=$wanted"; return 1; }
  LIVE_BLOCKS=$($PY -c "import json,urllib.request; print(json.loads(urllib.request.urlopen('http://127.0.0.1:8000/health',timeout=5).read())['total_blocks'])")
  [ "$LIVE_BLOCKS" = "$requested" ] ||
    { echo "!! pool mismatch: live=$LIVE_BLOCKS requested=$requested"; return 1; }
  grep -E "pool blocks=" "$logfile" | tail -1
  echo "CONFIG-ASSERTED slots=$slots blocks=$LIVE_BLOCKS tokens=$((LIVE_BLOCKS * 128))"
}

boot_once() {   # name tier spec ctx slots blocks spec-nodes
  local name=$1 tier=$2 spec=$3 ctx=$4 slots=$5 blocks=$6 nodes=$7
  teardown
  local weight_env="" logfile=$LOG/grid_${name}_b${blocks}.log
  [ -n "$tier" ] && weight_env="TQ_W_NVFP4=$tier"
  env CUDA_VISIBLE_DEVICES=0 TQ_KV_Q4=1 TQ_CTX=$ctx TQ_EMBED_FP8=2 $weight_env \
      TQ_PAGED_SPEC=$spec TQ_PAGED_SPEC_MAXPOS=100000000 \
      TQ_PG_SPEC_SLOTS=$slots TQ_PG_SPEC_NODES=$nodes \
      nohup $PY -u tools/serve_batched.py --tqf "$TQF" --model-dir "$MD" \
        --model-name ksl --max-slots "$slots" --num-blocks "$blocks" \
        --no-prefix-cache --port 8000 > "$logfile" 2>&1 &
  local pid
  pid=$!
  wait_http http://127.0.0.1:8000/v1/models "$pid" "$name" "$logfile" || return 1
  assert_cfg "$logfile" "$tier" "$spec" "$slots" "$blocks" || return 1
  LIVE_LOG=$logfile
}

boot_auto() {   # name tier spec ctx slots start min step nodes
  local name=$1 tier=$2 spec=$3 ctx=$4 slots=$5 blocks=$6 min=$7 step=$8 nodes=$9
  while [ "$blocks" -ge "$min" ]; do
    echo "BOOT-TRY name=$name slots=$slots blocks=$blocks ctx=$ctx spec=$spec nodes=$nodes"
    if boot_once "$name" "$tier" "$spec" "$ctx" "$slots" "$blocks" "$nodes"; then
      return 0
    fi
    blocks=$((blocks - step))
  done
  echo "!! no serviceable pool for $name (min blocks=$min)"
  return 1
}

run_bench() {   # label contexts ns only repeats kind nodes
  local label=$1 contexts=$2 ns=$3 only=$4 repeats=$5 kind=$6 nodes=$7
  local extra=()
  [ -n "$only" ] && extra+=(--only "$only")
  echo "=== $label blocks=$LIVE_BLOCKS capacity=$((LIVE_BLOCKS * 128)) ==="
  timeout 14400 "$PY" tools/bench_spec_matrix.py \
    --engine knivesysl --output-dir "$RAW" --label "$label" \
    --contexts "$contexts" --ns "$ns" "${extra[@]}" \
    --slots "${LIVE_SLOTS}" --pool-tokens "$((LIVE_BLOCKS * 128))" \
    --gen 192 --repeats "$repeats" --spec-kind "$kind" --spec-nodes "$nodes" \
    2>&1 | grep -E "^ctx|^wrote"
}

run_main() {   # name tier spec blocks
  local name=$1 tier=$2 spec=$3 blocks=$4 kind=plain
  [ "$spec" = "1" ] && kind=ngram
  LIVE_SLOTS=4
  boot_auto "$name" "$tier" "$spec" 140288 4 "$blocks" "$blocks" 64 8
  run_bench "$name-b$LIVE_BLOCKS" \
    "2048,8192,16384,32768,65536,94208,131072" "1,2,4" "" 3 "$kind" 8
}

run_deep_plain() {   # name tier ctx blocks min contexts only
  local name=$1 tier=$2 ctx=$3 blocks=$4 min=$5 contexts=$6 only=$7
  LIVE_SLOTS=2
  boot_auto "$name" "$tier" 0 "$ctx" 2 "$blocks" "$min" 64 8
  run_bench "$name-b$LIVE_BLOCKS" "$contexts" "1,2" "$only" 2 plain 8
}

run_deep_ngram_nvfp4() {
  LIVE_SLOTS=2
  # Eight archive nodes cost about 1.2 GB. The lower pool is the actual
  # serviceable n-gram capacity tier, not the plain 2100-block projection.
  boot_auto nvfp4-ngram-deep all 1 262144 2 1740 1400 64 8
  run_bench "nvfp4-ngram-deep-b$LIVE_BLOCKS" \
    "131072,196608,240000,261120" "1,2" "" 2 ngram 8
}

frontier_context() {   # total-token rows divided across N, with gen+guard
  local total=$1 n=$2
  local ctx=$((total / n - 256))
  [ "$ctx" -lt 256 ] && ctx=256
  echo $((ctx / 128 * 128))
}

run_wide_plain() {   # name tier ctx start min
  local name=$1 tier=$2 ctx=$3 start=$4 min=$5
  LIVE_SLOTS=16
  boot_auto "$name" "$tier" 0 "$ctx" 16 "$start" "$min" 64 16
  local total=$((LIVE_BLOCKS * 128))
  local c4 c8 c16
  c4=$(frontier_context "$total" 4)
  c8=$(frontier_context "$total" 8)
  c16=$(frontier_context "$total" 16)
  local contexts="2048,8192,$c4,$c8,$c16"
  local only="2048:12,2048:15,2048:16,8192:12,8192:15,8192:16,$c4:4,$c8:8,$c16:16"
  run_bench "$name-b$LIVE_BLOCKS" "$contexts" "4,8,12,15,16" "$only" 3 plain 16
}

run_wide_fallback() {   # name tier start min
  local name=$1 tier=$2 start=$3 min=$4
  LIVE_SLOTS=16
  # n>=12 has depth zero at the 16-node cap, so no archive is allocated.
  boot_auto "$name" "$tier" 1 16384 16 "$start" "$min" 64 16
  local total=$((LIVE_BLOCKS * 128))
  local c16
  c16=$(frontier_context "$total" 16)
  run_bench "$name-b$LIVE_BLOCKS" "2048,8192,$c16" "12,15,16" \
    "2048:12,2048:15,2048:16,8192:12,8192:15,8192:16,$c16:16" \
    3 ngram 16
}

run_wide_ngram() {   # name tier ctx start min -- active at n=4/8
  local name=$1 tier=$2 ctx=$3 start=$4 min=$5
  LIVE_SLOTS=8
  boot_auto "$name" "$tier" 1 "$ctx" 8 "$start" "$min" 64 16
  local total=$((LIVE_BLOCKS * 128))
  local c4 c8
  c4=$(frontier_context "$total" 4)
  c8=$(frontier_context "$total" 8)
  run_bench "$name-b$LIVE_BLOCKS" "2048,8192,$c4,$c8" "4,8" \
    "2048:8,8192:8,$c4:4,$c8:8" 3 ngram 16
}

run_single_stream() {
  teardown
  local logfile=$LOG/grid_fp6-mtp.log
  env CUDA_VISIBLE_DEVICES=0 TQ_KV_Q4=1 TQ_CTX=140288 TQ_EMBED_FP8=2 \
      nohup "$PY" -u tools/serve_openai.py --tqf "$TQF" --model-dir "$MD" \
        --lib build-qwen/libforward_qwen.so --model-name ksl --no-prefix-cache \
        --port 8000 > "$logfile" 2>&1 &
  local pid
  pid=$!
  wait_http http://127.0.0.1:8000/v1/models "$pid" fp6-mtp "$logfile"
  LIVE_BLOCKS=1100
  LIVE_SLOTS=1
  run_bench fp6-mtp "2048,8192,16384,32768,65536,94208,131072" "1" "" 3 mtp 8
}

# A: primary service grids, every feasible ctx x {1,2,4}.
run_main nvfp4-off all 0 1800
run_main nvfp4-ngram all 1 1800
run_main nvmlp-off mlp 0 1500
run_main nvmlp-ngram mlp 1 1500
run_main fp6-off "" 0 1200
run_main fp6-ngram "" 1 1200

# B: true per-tier single/deep capacity frontiers.
run_deep_plain nvfp4-off-deep all 262144 2100 1700 \
  "131072,196608,240000,261120" \
  "196608:1,240000:1,261120:1,131072:2"
run_deep_plain nvmlp-off-deep mlp 196608 1500 1200 "190000" "190000:1"
run_deep_plain fp6-off-deep "" 153600 1200 900 "150000" "150000:1"
run_deep_ngram_nvfp4

# C: 16-client scheduler scaling plus each tier's actual wide capacity frontier.
run_wide_plain nvfp4-off-wide all 70000 2100 1200
run_wide_plain nvmlp-off-wide mlp 50000 1500 900
run_wide_plain fp6-off-wide "" 40000 1200 640

# D: n>=12 n-gram-enabled depth-zero fallback, explicitly labelled as fallback.
run_wide_fallback nvfp4-ngram-fallback all 2100 1100
run_wide_fallback nvmlp-ngram-fallback mlp 1500 800
run_wide_fallback fp6-ngram-fallback "" 1200 512

# E: real n-gram at n=8 (depth 1) and n=4 (depth 3), 16-node archive.
run_wide_ngram nvfp4-ngram-wide all 70000 1800 900
run_wide_ngram nvmlp-ngram-wide mlp 50000 1200 640
run_wide_ngram fp6-ngram-wide "" 40000 800 512

# F: current single-stream MTP ceiling.
run_single_stream

