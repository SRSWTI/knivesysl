#!/usr/bin/env bash
# knivesysl production supervisor.
# - exactly one wrapper/child pair per host (the lock fd is inherited by child)
# - forwards termination, backs off crash loops, and enables core dumps
cd "$(dirname "$0")/.." || exit 1

usage() {
  cat <<'USAGE'
Usage: bash tools/serve_prod.sh [--variant nvfp4|mixed|fp6] [--dry-run]
       bash tools/serve_prod.sh --list-variants

Without --variant, a terminal launch asks which profile to run. Enter keeps
NVFP4, or the profile selected by TQ_W_NVFP4. Non-interactive launches use
that default without prompting. --variant takes precedence over the environment.

  --variant NAME   Select weights without the interactive menu
  --dry-run        Show the resolved environment and command; do not launch
  --list-variants  Show measured benefits and drawbacks, then exit
  -h, --help       Show this help

Optimized defaults: plain decoding, CTA-local NVFP4 TMA, unused MTP head
disabled, and N1 attention rows=64. Explicit tuning environment overrides
remain supported. Pool geometry stays at 2 slots / 2100 blocks.
USAGE
}

show_variants() {
  cat <<'PROFILES'

knivesysl - measured serving profiles (RTX 5090, 2026-09-15 UTC)

  1) nvfp4 - speed / memory default
     Main projection weights: NVFP4.
     One-user decode at 2k / 32k / 128k: 70.14 / 65.20 / 52.68 tok/s.
     Two-user 2k decode: 137.50 aggregate tok/s. 2k prefill: 10,758 tok/s.
     Peak process VRAM: native bench 25,056 MiB; HTTP + cache 25,966 MiB.
     Strict tasks 11/16; coding 8/8. Wrong-content / format-only misses: 4 / 1.
     Existing prefix cache, 32k first content: 4.58 s cold -> 0.072 s cached.
     Benefit: fastest single-user decode tested; smallest memory footprint.
     Drawback: more wrong-content answers in this small suite than FP6.

  2) mixed - higher-precision compromise
     MLP weights: NVFP4; attention and DeltaNet weights: FP6.
     One-user decode at 2k / 32k / 128k: 68.16 / 61.27 / 51.46 tok/s.
     Two-user 2k decode: 130.25 aggregate tok/s. 2k prefill: 8,222 tok/s.
     Peak process VRAM: native bench 26,336 MiB; HTTP + cache 27,246 MiB.
     Strict tasks 11/16; coding 8/8. Wrong-content / format-only misses: 3 / 2.
     Existing prefix cache, 32k first content: 5.62 s cold -> 0.073 s cached.
     Benefit: higher-precision attention/DeltaNet while keeping FP4 MLP memory.
     Drawback: slower prefill and more memory than NVFP4; not full FP6.

  3) fp6 - highest weight precision tested here
     Main projection weights: FP6 (no runtime NVFP4 conversion).
     One-user decode at 2k / 32k / 128k: 62.01 / 57.46 / 48.82 tok/s.
     Two-user 2k decode: 120.49 aggregate tok/s. 2k prefill: 5,803 tok/s.
     Peak process VRAM: native bench 29,272 MiB; HTTP + cache 30,182 MiB.
     Strict tasks 11/16; coding 8/8. Wrong-content / format-only misses: 2 / 3.
     Existing prefix cache, 32k first content: 7.22 s cold -> 0.076 s cached.
     Benefit: highest weight precision; fewest wrong-content tasks here.
     Drawback: largest footprint and slowest prefill/decode in this comparison.

CURRENT KV CONFIGURATION (all profiles): 2 slots, 2100 blocks x 128 tokens
= 268,800 SHARED token slots; maximum 262,144 tokens PER CONVERSATION.
The earlier 425,984-token shared pool used 3328 blocks. It is NOT enabled here.
All three passed native execution at position 262,144 and two simultaneous
128k contexts. All passed 4/4 synthetic retrieval prompts, up to 261,888 tokens.

Stats: same two-slot pool; plain inference; 2k=2048 tokens. Speed is from two
96-step teacher-forced repeats; clocks were not locked. VRAM is a 50ms sampled
process peak, not a guaranteed maximum. Answers use temperature0, thinking off.
The 16-task suite includes eight code tasks with 37 functional checks, not a
broad accuracy benchmark. Format-only misses were classified after strict
grading; original scores were not changed. Equal scores do not prove equal
quality. Cache figures compare existing cold/warm reuse, not a new algorithm.
KV/embedding formats remain unchanged; tuning overrides can change results.
Source: results/5090-profiles-20260915-025945/summary.json
PROFILES
}

usage_error() {
  printf '[wrapper] %s\nUse --help for launch options.\n' "$1" >&2
  exit 64
}

variant=
dry_run=0
list_variants=0
while (( $# )); do
  case "$1" in
    --variant|--variant=*)
      [[ -z "$variant" ]] || usage_error "Specify --variant only once"
      if [[ "$1" == --variant ]]; then
        (( $# >= 2 )) || usage_error "--variant requires nvfp4, mixed, or fp6"
        variant=$2
        shift 2
      else
        variant=${1#*=}
        shift
      fi
      case "$variant" in
        nvfp4|mixed|fp6) ;;
        *) usage_error "Unknown variant: $variant (choose nvfp4, mixed, or fp6)" ;;
      esac
      ;;
    --dry-run) dry_run=1; shift ;;
    --list-variants) list_variants=1; shift ;;
    -h|--help) usage; exit 0 ;;
    *) usage_error "Unknown argument: $1" ;;
  esac
done

show_variants
(( list_variants )) && exit 0
if [[ -z "$variant" ]]; then
  case "${TQ_W_NVFP4:-all}" in
    all) variant=nvfp4 ;;
    mlp) variant=mixed ;;
    0) variant=fp6 ;;
    *) usage_error "TQ_W_NVFP4 must be all, mlp, or 0; use --variant to override it" ;;
  esac
  if [[ -t 0 ]]; then
    while true; do
      if ! read -r -p "Choose 1/nvfp4, 2/mixed, or 3/fp6 [${variant}]: " choice; then
        printf '\n[wrapper] Selection cancelled: no input; server not started.\n' >&2
        exit 64
      fi
      case "$choice" in
        '') break ;;
        1|nvfp4) variant=nvfp4; break ;;
        2|mixed) variant=mixed; break ;;
        3|fp6) variant=fp6; break ;;
        *) printf 'Choose 1, 2, 3, or a profile name.\n' >&2 ;;
      esac
    done
  fi
fi
case "$variant" in
  nvfp4) export TQ_W_NVFP4=all ;;
  mixed) export TQ_W_NVFP4=mlp ;;
  fp6) export TQ_W_NVFP4=0 ;;
esac

export CUDA_VISIBLE_DEVICES=0 TQ_KV_Q4=1 TQ_CTX=262144 TQ_EMBED_FP8=2
# Measured plain-inference profiles; explicit tuning overrides stay available.
export TQ_NVFP4_TMA_CTA=${TQ_NVFP4_TMA_CTA:-1}
export TQ_LOAD_MTP=${TQ_LOAD_MTP:-0}
export TQ_PAGED_N1_ROWS=${TQ_PAGED_N1_ROWS:-64}
# Extra GEMM tuning changed reference agreement; it is not a profile default.
export TQ_NVF4_DECODE_AUTOTUNE_COLS=${TQ_NVF4_DECODE_AUTOTUNE_COLS:-0}
export TQ_PAGED_SPEC=${TQ_PAGED_SPEC:-0} TQ_PG_SPEC_NODES=${TQ_PG_SPEC_NODES:-4}
# Host-tier checkpoint demotion is opt-in only until Track D validates it under
# long-context concurrency. The 2026-09-03 prod wedge hit this untested path.
export TQ_CKPT_HOST_GB=${TQ_CKPT_HOST_GB:-0}
# Health reports a stalled engine after 60s; the independent watchdog exits
# after 120s so the wrapper can restart even when no external monitor is present.
export TQ_HEALTH_STALL_S=${TQ_HEALTH_STALL_S:-60}
export TQ_ENGINE_WATCHDOG_S=${TQ_ENGINE_WATCHDOG_S:-120}

server_cmd=(
  .venv/bin/python -u tools/serve_batched.py
  --tqf /home/shooting-brake007/models/knivesysl/qwen3_8-27b-e2m3-mtp.tqf
  --model-dir /home/shooting-brake007/models/knivesysl
  --model-name knivesysl-axe-28b --max-slots 2 --num-blocks 2100
  --max-queue 128 --max-http-concurrency 128 --http-io-timeout 30
  --queue-timeout 300 --request-timeout 900 --port 8000
)
printf '\n[wrapper] Selected variant: %s\n[wrapper] Effective environment:\n' "$variant"
for setting in CUDA_VISIBLE_DEVICES TQ_W_NVFP4 TQ_KV_Q4 TQ_CTX TQ_EMBED_FP8 \
  TQ_NVFP4_TMA_CTA TQ_LOAD_MTP TQ_PAGED_N1_ROWS TQ_NVF4_DECODE_AUTOTUNE_COLS \
  TQ_PAGED_SPEC TQ_PG_SPEC_NODES TQ_CKPT_HOST_GB; do
  printf '  %s=%s\n' "$setting" "${!setting}"
done
printf '[wrapper] Command:'
printf ' %q' "${server_cmd[@]}"
printf '\n'
if (( dry_run )); then
  printf '[wrapper] Dry run: no lock acquired, no model loaded, no server started.\n'
  exit 0
fi

exec 9>/tmp/knivesysl-prod.lock
if ! flock -n 9; then
  echo "[wrapper] another production instance owns /tmp/knivesysl-prod.lock" >&2
  exit 73
fi
ulimit -c unlimited

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
  "${server_cmd[@]}" &
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
