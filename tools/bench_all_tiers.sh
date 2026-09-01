#!/usr/bin/env bash
# Full re-measurement on the FIXED bench tool (token counts from the server's usage
# block, not SSE event counts -- under speculation one event carries accept_len
# tokens, which silently undercounted every prior spec cell by 2-4x).
#
# Two failure modes this script now defends against, both hit on the first attempt:
#   1. `$w` cannot expand into an environment assignment -- bash parses assignment
#      prefixes BEFORE expansion, so an expanded VAR=VAL becomes a command name.
#      Use `env VAR=VAL ...`, which takes them as runtime arguments.
#   2. tools/serve_prod.sh is a RESTART WRAPPER. Killing serve_batched alone lets it
#      resurrect production, and the bench then measures PRODUCTION while reporting
#      it under the intended config's label. Kill the wrapper first, and assert the
#      live server's tier/spec match what was asked for before trusting a cell.
set -u
cd /home/shooting-brake007/srswti/qwen38/knivesysl
PY=.venv/bin/python
TQF=/home/shooting-brake007/models/knivesysl/qwen3_8-27b-e2m3-mtp.tqf
MD=/home/shooting-brake007/models/knivesysl
LOG=/tmp/gembench
mkdir -p $LOG

teardown() { pkill -f serve_prod.sh; sleep 1; pkill -f serve_batched; pkill -f serve_openai; sleep 4; }

# assert_cfg <logfile> <tier> <spec> -- refuses to bench a mislabelled server
assert_cfg() {
  local lg=$1 tier=$2 sp=$3
  if [ -n "$tier" ]; then
    grep -q "TQ_W_NVFP4=$tier " $lg || { echo "!! tier mismatch: expected TQ_W_NVFP4=$tier"; return 1; }
  else
    grep -q "TQ_W_NVFP4=" $lg && { echo "!! tier mismatch: expected FP6, found an NVFP4 repack"; return 1; }
  fi
  local en; en=$($PY -c "import json,urllib.request;print(json.loads(urllib.request.urlopen('http://127.0.0.1:8000/health',timeout=5).read())['spec']['enabled'])" 2>/dev/null)
  local want=False; [ "$sp" = "1" ] && want=True
  [ "$en" = "$want" ] || { echo "!! spec mismatch: server says enabled=$en, wanted $want"; return 1; }
  grep -E "pool blocks=" $lg | tail -1
}

run_paged() {   # name tier("" fp6 | mlp | all) spec(0|1) blocks
  local name=$1 tier=$2 sp=$3 blk=$4
  teardown
  local w=""; [ -n "$tier" ] && w="TQ_W_NVFP4=$tier"
  env CUDA_VISIBLE_DEVICES=0 TQ_KV_Q4=1 TQ_CTX=140288 TQ_EMBED_FP8=2 $w \
      TQ_PAGED_SPEC=$sp TQ_PAGED_SPEC_MAXPOS=100000000 \
      nohup $PY -u tools/serve_batched.py --tqf $TQF --model-dir $MD \
        --model-name ksl --max-slots 4 --num-blocks $blk --no-prefix-cache \
        --port 8000 > $LOG/grid_$name.log 2>&1 &
  sleep 18
  curl -s -m 3 http://127.0.0.1:8000/v1/models >/dev/null || { echo "!! $name did not start"; tail -3 $LOG/grid_$name.log; return 1; }
  echo "=== $name ==="
  assert_cfg $LOG/grid_$name.log "$tier" "$sp" || return 1
  timeout 3000 $PY tools/bench_spec_matrix.py --label "$name" --ns 1,2,4 \
    --slots 4 --pool-tokens $((blk * 128)) --gen 192 2>&1 | grep -E "^ctx|wrote"
}

run_n8() {      # n=8 needs --max-slots 8, whose per-slot state only fits at low ctx
  local name=$1 tier=$2 sp=$3
  teardown
  local w=""; [ -n "$tier" ] && w="TQ_W_NVFP4=$tier"
  env CUDA_VISIBLE_DEVICES=0 TQ_KV_Q4=1 TQ_CTX=32768 TQ_EMBED_FP8=2 $w \
      TQ_PAGED_SPEC=$sp TQ_PAGED_SPEC_MAXPOS=100000000 TQ_PAGED_SPEC_SLOTS=8 \
      nohup $PY -u tools/serve_batched.py --tqf $TQF --model-dir $MD \
        --model-name ksl --max-slots 8 --num-blocks 800 --no-prefix-cache \
        --port 8000 > $LOG/grid_$name.log 2>&1 &
  sleep 18
  curl -s -m 3 http://127.0.0.1:8000/v1/models >/dev/null || { echo "!! $name did not start"; tail -3 $LOG/grid_$name.log; return 1; }
  echo "=== $name ==="
  assert_cfg $LOG/grid_$name.log "$tier" "$sp" || return 1
  timeout 1800 $PY tools/bench_spec_matrix.py --label "$name" --ns 8 \
    --only 2048:8,8192:8 --slots 8 --pool-tokens 102400 --gen 192 2>&1 | grep -E "^ctx|wrote"
}

run_single_stream() {   # fp6 + MTP, n=1 only (single-stream server serialises)
  teardown
  env CUDA_VISIBLE_DEVICES=0 TQ_KV_Q4=1 TQ_CTX=140288 TQ_EMBED_FP8=2 \
      nohup $PY -u tools/serve_openai.py --tqf $TQF --model-dir $MD \
        --lib $(pwd)/build-qwen/libforward_qwen.so --model-name ksl \
        --port 8000 > $LOG/grid_fp6-mtp.log 2>&1 &
  sleep 22
  curl -s -m 3 http://127.0.0.1:8000/v1/models >/dev/null || { echo "!! fp6-mtp did not start"; tail -3 $LOG/grid_fp6-mtp.log; return 1; }
  echo "=== fp6-mtp (single-stream) ==="
  timeout 3000 $PY tools/bench_spec_matrix.py --label fp6-mtp --ns 1 \
    --slots 1 --pool-tokens 230000 --gen 192 2>&1 | grep -E "^ctx|wrote"
}

run_paged nvfp4-off   all 0 1800
run_paged nvfp4-ngram all 1 1800
run_paged nvmlp-off   mlp 0 1500
run_paged nvmlp-ngram mlp 1 1500
run_paged fp6-off     ""  0 1200
run_paged fp6-ngram   ""  1 1200
run_n8 nvfp4-off-n8   all 0
run_n8 nvfp4-ngram-n8 all 1
run_n8 fp6-off-n8     ""  0
run_n8 fp6-ngram-n8   ""  1
run_single_stream

teardown
nohup tools/serve_prod.sh >> /tmp/knivesysl_serve.log 2>&1 &
sleep 14
curl -s -m 3 http://127.0.0.1:8000/v1/models >/dev/null && echo "PRODUCTION-RESTORED" || echo "!! production did not restore"
