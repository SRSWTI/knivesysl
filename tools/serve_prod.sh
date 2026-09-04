#!/usr/bin/env bash
# knivesysl production supervisor.
# - exactly one wrapper/child pair per host (the lock fd is inherited by child)
# - forwards termination, backs off crash loops, and enables core dumps
cd "$(dirname "$0")/.."
exec 9>/tmp/knivesysl-prod.lock
if ! flock -n 9; then
  echo "[wrapper] another production instance owns /tmp/knivesysl-prod.lock" >&2
  exit 73
fi
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
# Health reports a stalled engine after 60s; the independent watchdog exits
# after 120s so the wrapper can restart even when no external monitor is present.
export TQ_HEALTH_STALL_S=${TQ_HEALTH_STALL_S:-60}
export TQ_ENGINE_WATCHDOG_S=${TQ_ENGINE_WATCHDOG_S:-120}
stopping=0
child=
stop_reaper=
shutdown_grace=${KSL_SHUTDOWN_GRACE_S:-15}
if ! [[ "$shutdown_grace" =~ ^[0-9]+([.][0-9]+)?$ ]]; then
  echo "[wrapper] KSL_SHUTDOWN_GRACE_S must be a non-negative number" >&2
  exit 64
fi
stop_child() {
  local target=$child
  if [[ -n "$target" ]] && kill -0 "$target" 2>/dev/null; then
    kill -TERM "$target" 2>/dev/null || true
    if [[ -z "$stop_reaper" ]]; then
      (
        # The reaper is not an owner. Do not let an orphaned sleep retain the
        # singleton lock after a fast child exit.
        exec 9>&-
        sleep "$shutdown_grace"
        if kill -0 "$target" 2>/dev/null; then
          echo "[wrapper] shutdown exceeded ${shutdown_grace}s; killing child $target" >&2
          kill -KILL "$target" 2>/dev/null || true
        fi
      ) &
      stop_reaper=$!
    fi
  fi
}
forward_stop() {
  stopping=1
  stop_child
}
trap forward_stop INT TERM HUP

failures=0
restarts=0
last_exit_rc=0
last_exit_unix=0
while true; do
  (( stopping )) && exit 0
  started=$SECONDS
  export KSL_SUPERVISOR_RESTARTS=$restarts
  export KSL_SUPERVISOR_LAST_EXIT_CODE=$last_exit_rc
  export KSL_SUPERVISOR_LAST_EXIT_TIME_SECONDS=$last_exit_unix
  .venv/bin/python -u tools/serve_batched.py \
    --tqf /home/shooting-brake007/models/knivesysl/qwen3_8-27b-e2m3-mtp.tqf \
    --model-dir /home/shooting-brake007/models/knivesysl \
    --model-name knivesysl-axe-28b --max-slots 2 --num-blocks 2100 \
    --max-queue 128 --max-http-concurrency 128 --http-io-timeout 30 \
    --queue-timeout 300 --request-timeout 900 --port 8000 &
  child=$!
  # A signal may have arrived after the loop guard but before child=$!.
  (( stopping )) && stop_child
  wait "$child"
  rc=$?
  if (( stopping )); then
    if kill -0 "$child" 2>/dev/null; then
      wait "$child"
      rc=$?
    fi
    if [[ -n "$stop_reaper" ]]; then
      kill "$stop_reaper" 2>/dev/null || true
      wait "$stop_reaper" 2>/dev/null || true
    fi
    exit "$rc"
  fi
  child=
  ((restarts += 1))
  last_exit_rc=$rc
  last_exit_unix=$EPOCHSECONDS

  runtime=$((SECONDS - started))
  if (( runtime >= 60 )); then
    failures=0
  else
    # Six failures already reach the 60-second cap. Saturating the exponent
    # prevents signed arithmetic overflow during a persistent crash loop.
    if (( failures < 6 )); then
      ((failures += 1))
    fi
  fi
  delay=$((3 << (failures > 0 ? failures - 1 : 0)))
  (( delay > 60 )) && delay=60
  echo "[wrapper] server exited rc=$rc after ${runtime}s at $(date -Is); restarting in ${delay}s" >&2
  sleep "$delay"
done
