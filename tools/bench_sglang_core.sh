#!/usr/bin/env bash
# Matched SGLang reference for the reduced knivesysl core matrix.
# The exhaustive concurrency/capacity reference remains in bench_sglang_ref.sh.
set -euo pipefail
cd /home/shooting-brake007/srswti/qwen38/knivesysl

TQ_CAMPAIGN_LOG=${TQ_CAMPAIGN_LOG:-${TQ_BENCH_ROOT:-$PWD/results}/sglang_core.log}
source tools/bench_sglang_ref.sh

run_core_plain() {   # concurrency
  local n=$1
  local cache=$((n * 4)) name=sgl-core-nocache-plain-n$n
  if ! boot "$name" --mem-fraction-static 0.90 --max-running-requests "$n" \
      --cuda-graph-max-bs "$n" --max-mamba-cache-size "$cache"; then
    echo "GRAPH-BOOT-FAILED n=$n; retrying eager"
    name=sgl-core-nocache-plain-n$n-eager
    boot "$name" --mem-fraction-static 0.90 --max-running-requests "$n" \
      --disable-cuda-graph --max-mamba-cache-size "$cache"
  fi
  cell "$name" "$n" "2048,8192,32768,65536,131072" "" plain 3
}

echo "SGLANG-CORE cache-disabled n=1/2/4 plain plus DSpark n=1"
run_core_plain 1
run_core_plain 2
run_core_plain 4

boot sgl-core-nocache-dspark-n1 --mem-fraction-static 0.88 --max-running-requests 1 \
  --cuda-graph-max-bs 1 --mamba-full-memory-ratio 5.61 \
  --speculative-algorithm DSPARK --speculative-draft-model-path "$DRAFT" \
  --speculative-draft-attention-backend flashinfer
cell sgl-core-nocache-dspark-n1 1 "2048,8192" "2048:1,8192:1" dspark 3
