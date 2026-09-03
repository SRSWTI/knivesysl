#!/usr/bin/env bash
# Reduced decision matrix for kernel selection and matched SGLang comparison.
# The exhaustive capacity/concurrency campaign remains in bench_all_tiers.sh.
set -euo pipefail
cd /home/shooting-brake007/srswti/qwen38/knivesysl

TQ_CAMPAIGN_LOG=${TQ_CAMPAIGN_LOG:-${TQ_BENCH_ROOT:-$PWD/results}/core_matrix.log}
source tools/bench_all_tiers.sh

run_core_paged() {   # name tier spec blocks contexts ns only
  local name=$1 tier=$2 spec=$3 blocks=$4 contexts=$5 ns=$6 only=$7 kind=plain
  [ "$spec" = "1" ] && kind=ngram
  LIVE_SLOTS=4
  boot_auto "$name" "$tier" "$spec" 140288 4 "$blocks" "$blocks" 64 8
  run_bench "$name-b$LIVE_BLOCKS" "$contexts" "$ns" "$only" 3 "$kind" 8
}

echo "CORE-MATRIX 35 native cells maximum; n=1/2/4 only; capacity frontiers deferred"

# Main production baseline: short, ordinary-agent, long, and deep contexts.
run_core_paged core-nvfp4-off all 0 1800 \
  "2048,8192,32768,65536,131072" "1,2,4" ""

# Speculative decision grid: enough to measure acceptance, verify cost, and batching.
run_core_paged core-nvfp4-ngram all 1 1800 \
  "2048,8192,32768,65536" "1,2,4" ""

# Format sentinels: only Pareto-competitive tiers earn a future full grid.
run_core_paged core-nvmlp-off mlp 0 1500 \
  "8192,32768,65536" "1,4" ""
run_core_paged core-fp6-off "" 0 1200 \
  "8192,32768,65536" "1,4" ""

# Existing single-stream MTP control point for the paged-MTP port.
run_single_stream core-fp6-mtp "2048,8192"
