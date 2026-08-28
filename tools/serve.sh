#!/usr/bin/env bash
# Host the knivesysl single-stream OpenAI server (tools/serve_openai.py).
#
#   tools/serve.sh                 # foreground
#   tools/serve.sh daemon          # detached (setsid), waits for readiness
#   tools/serve.sh status | logs | stop
#   tools/serve.sh daemon --depth 8 --dogs-accept-min 1.2     # extra args pass through
#
# Every knob is an env var override, e.g.
#   PORT=8001 TQ_CTX=131072 THINKING=off tools/serve.sh daemon
#   TQF=/path/model.tqf MODEL_DIR=/path/hf-checkpoint tools/serve.sh
set -euo pipefail

REPO="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"

cmd=start
case "${1:-}" in
    start|daemon|stop|status|logs) cmd=$1; shift ;;
    -h|--help) sed -n '2,11p' "${BASH_SOURCE[0]}"; exit 0 ;;
esac

die() { printf 'serve.sh: %s\n' "$*" >&2; exit 1; }

first_match() {
    local p
    for p in "$@"; do
        if [ -e "$p" ]; then printf '%s\n' "$p"; return 0; fi
    done
    return 1
}

: "${PORT:=8000}"
: "${LOG:=/tmp/knivesysl-serve-$PORT.log}"
: "${PIDFILE:=/tmp/knivesysl-serve-$PORT.pid}"
: "${LIB:=$REPO/build-qwen/libforward_qwen.so}"
: "${TQF:=$(first_match "$REPO"/*.tqf "$HOME"/models/knivesysl/*.tqf || true)}"
: "${MODEL_DIR:=$(first_match "$HOME"/.cache/huggingface/hub/models--Qwen--Qwen3.8-27B/snapshots/*/ \
                              "$HOME"/.cache/huggingface/hub/models--Qwen--Qwen3.6-27B/snapshots/*/ || true)}"
: "${THINKING:=on}"                 # off -> --no-thinking (agent / prefix-cache mode)
: "${READY_TIMEOUT:=180}"

# ship config for one 32 GB SM120 card: FP6 weights + 4-bit KV + 6-bit embed table
export TQ_KV_Q4="${TQ_KV_Q4:-1}"
export TQ_EMBED_FP8="${TQ_EMBED_FP8:-2}"
export TQ_MTP_VOCAB_CAP="${TQ_MTP_VOCAB_CAP:-32768}"
export TQ_NGRAM_DRAFT="${TQ_NGRAM_DRAFT:-1}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

server_pid() {
    local pid=""
    if [ -f "$PIDFILE" ]; then pid=$(cat "$PIDFILE" 2>/dev/null || true); fi
    if [ -n "$pid" ] && kill -0 "$pid" 2>/dev/null; then
        printf '%s\n' "$pid"; return 0
    fi
    pid=$(pgrep -f "serve_openai\.py .*--port $PORT( |\$)" | head -1 || true)
    if [ -n "$pid" ]; then printf '%s\n' "$pid"; return 0; fi
    return 1
}

# The KV cache is TQ_CTX-sized and allocated on the FIRST request, so a context
# that does not fit dies mid-request (spec-buffer alloc -> wide_advance_trunk -4)
# rather than at load. Size it from free VRAM instead. Measured on a 5090: the FP6
# weights land at ~22.0 GiB and KV+spec cost ~40.5 KiB/token (196608 ctx -> ~29.6
# GiB steady state); the 44 KiB default keeps ~10% margin for allocator slack.
auto_ctx() {
    local free w slack per_tok budget ctx
    free=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits 2>/dev/null | head -1 || true)
    [ -n "$free" ] || return 1
    w=${WEIGHTS_MIB:-22528}; slack=${SLACK_MIB:-1024}; per_tok=${KV_KIB_PER_TOKEN:-44}
    budget=$(( free - w - slack ))
    [ "$budget" -gt 0 ] || return 1
    ctx=$(( budget * 1024 / per_tok / 8192 * 8192 ))
    if [ "$ctx" -gt 262144 ]; then ctx=262144; fi
    [ "$ctx" -ge 8192 ] || return 1
    printf '%s\n' "$ctx"
}

pick_python() {
    local c
    for c in ${TQ_PYTHON:-} "$REPO/.venv/bin/python" "${VIRTUAL_ENV:-}/bin/python" \
             "$HOME/srswti/shooting-brake/.venv/bin/python" python3; do
        [ -n "$c" ] || continue
        if "$c" -c 'import transformers' >/dev/null 2>&1; then printf '%s\n' "$c"; return 0; fi
    done
    return 1
}

case "$cmd" in
stop)
    pid=$(server_pid) || die "no server on :$PORT"
    kill "$pid" 2>/dev/null || true
    for _ in $(seq 40); do
        if ! kill -0 "$pid" 2>/dev/null; then break; fi
        sleep 0.25
    done
    if kill -0 "$pid" 2>/dev/null; then kill -9 "$pid" 2>/dev/null || true; sleep 1; fi
    rm -f "$PIDFILE"
    echo "stopped $pid (:$PORT)"
    exit 0 ;;
status)
    if pid=$(server_pid); then
        echo "pid $pid on :$PORT  vram: $(nvidia-smi --query-gpu=memory.used,memory.total \
              --format=csv,noheader 2>/dev/null | head -1 || echo n/a)"
        curl -sf -m 10 "http://127.0.0.1:$PORT/v1/models" || echo "(no /v1/models response)"
        echo
        grep -a '^\[serve\] ready' "$LOG" 2>/dev/null | tail -1 || true
        exit 0
    fi
    echo "not running (:$PORT)"; exit 1 ;;
logs)
    exec tail -n "${TAIL_LINES:-40}" -f "$LOG" ;;
esac

# ---------------------------------- start ----------------------------------
[ -f "$LIB" ] || die "missing $LIB -- build it first:
  cmake -B build-qwen -DCMAKE_CUDA_ARCHITECTURES=120
  cmake --build build-qwen --target knivesysl-forward-qwen -j"
if [ -z "${TQF:-}" ] || [ ! -f "$TQF" ]; then
    die "no .tqf found (set TQF=/path/model.tqf) -- convert with:
  TQ_EMIT_MTP=1 TQ_GPU_PACK=1 python3 tools/convert_qwen_tqf.py /path/to/hf-checkpoint \\
      -o model.tqf --block-scaled always --block-layout qmma-e2m3 --block-scale-policy pow2"
fi
if [ -z "${MODEL_DIR:-}" ] || [ ! -d "$MODEL_DIR" ]; then
    die "no HF checkpoint dir (set MODEL_DIR=...) -- needed for the tokenizer + chat template"
fi

PY=$(pick_python) || die "no python with transformers (set TQ_PYTHON=/path/to/python)"

if pid=$(server_pid); then die "already running: pid $pid on :$PORT (tools/serve.sh stop)"; fi

if [ -z "${TQ_CTX:-}" ]; then
    if TQ_CTX=$(auto_ctx); then
        echo "serve.sh: TQ_CTX=$TQ_CTX (auto-sized from free VRAM; set TQ_CTX to override)"
    else
        TQ_CTX=131072
        echo "serve.sh: TQ_CTX=$TQ_CTX (fallback -- free VRAM too small to size from;" \
             "another process may be holding the card)" >&2
    fi
fi
export TQ_CTX

if ! "$PY" "$REPO/tools/inspect_tqf.py" "$TQF" 2>/dev/null | grep -q 'has_mtp_section: True'; then
    echo "serve.sh: WARNING $TQF has no MTP section -- no spec-decode, expect ~40% of the tok/s" >&2
fi

flags=(--port "$PORT" --tqf "$TQF" --model-dir "$MODEL_DIR" --lib "$LIB")
if [ "$THINKING" = off ]; then flags+=(--no-thinking); fi
set -- "${flags[@]}" "$@"

if [ "$cmd" = daemon ]; then
    # own session: a Ctrl-C, a closed terminal or a killed parent shell must not
    # take the engine down mid-request
    setsid nohup "$PY" "$REPO/tools/serve_openai.py" "$@" >"$LOG" 2>&1 </dev/null &
    sleep 1
    pid=$(server_pid) || { tail -n 20 "$LOG" >&2; die "server exited immediately"; }
    echo "$pid" >"$PIDFILE"
    echo "serve.sh: pid $pid, log $LOG -- waiting for readiness (<= ${READY_TIMEOUT}s)"
    for _ in $(seq "$READY_TIMEOUT"); do
        if ! kill -0 "$pid" 2>/dev/null; then
            tail -n 20 "$LOG" >&2; rm -f "$PIDFILE"; die "server died during load"
        fi
        if grep -qa '^\[serve\] ready' "$LOG"; then
            grep -a '^\[serve\] ' "$LOG" | tail -2
            exit 0
        fi
        sleep 1
    done
    die "not ready after ${READY_TIMEOUT}s (see $LOG)"
fi

echo "serve.sh: $PY tools/serve_openai.py $*"
exec "$PY" "$REPO/tools/serve_openai.py" "$@"
