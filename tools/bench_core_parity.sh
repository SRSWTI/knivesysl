#!/usr/bin/env bash
# Fill the deliberately omitted cells from the reduced core matrix without
# rerunning any completed sample. Together, the base and parity artifacts give
# every plain weight tier the same feasible context/concurrency grid.
set -euo pipefail
cd /home/shooting-brake007/srswti/qwen38/knivesysl

TQ_CAMPAIGN_LOG=${TQ_CAMPAIGN_LOG:-${TQ_BENCH_ROOT:-$PWD/results}/core_matrix.log}
source tools/bench_all_tiers.sh

run_parity_paged() {   # name tier spec blocks contexts ns only
  local name=$1 tier=$2 spec=$3 blocks=$4 contexts=$5 ns=$6 only=$7 kind=plain
  [ "$spec" = "1" ] && kind=ngram
  LIVE_SLOTS=4
  boot_auto "$name" "$tier" "$spec" 140288 4 "$blocks" "$blocks" 64 8
  run_bench "$name-b$LIVE_BLOCKS" "$contexts" "$ns" "$only" 3 "$kind" 8
}

echo "CORE-PARITY 18 missing cells; completed core samples are not rerun"

# Complete the cell-for-cell plain-format comparison. These seven pairs were
# omitted from each sentinel artifact for runtime, not for capacity.
PLAIN_MISSING="2048:1,2048:2,2048:4,8192:2,32768:2,65536:2,131072:1"
run_parity_paged core-nvmlp-off-parity mlp 0 1500 \
  "2048,8192,32768,65536,131072" "1,2,4" "$PLAIN_MISSING"
run_parity_paged core-fp6-off-parity "" 0 1200 \
  "2048,8192,32768,65536,131072" "1,2,4" "$PLAIN_MISSING"

# Complete the feasible NVFP4 n-gram depth ladder.
run_parity_paged core-nvfp4-ngram-parity all 1 1800 \
  "131072" "1" "131072:1"

# Match the same context ladder for the existing learned-MTP control. The
# current MTP server is structurally single-stream, so n=2/4 are not valid.
run_single_stream core-fp6-mtp-parity "32768,65536,131072"
