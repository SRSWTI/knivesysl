#!/usr/bin/env python3
"""NVFP4 tier quality gate, on the path the tier actually serves.

tools/tf_agreement.py drives single-stream `qwn_decode`, which is NOT NVFP4-aware (the
repack frees the FP6 payload), so it cannot measure this tier. This gate instead uses
the WIDE prefill entry `qwn_prefill_wide`, which goes through wide_proj and therefore
through the NVFP4 GEMM for every converted weight: it prefills a long prompt in `--chunk`
sized waves and records the greedy argmax the engine emits after each wave.

Each argmax is a full 64-layer forward reduced to one decision, so agreement between two
configs is a real end-to-end signal for the wide path. Run once per config and diff the
`ARGMAX` lines position by position.

(`qwn_prefill_chunk` is NOT usable here -- it caps at TQ_SPEC_MAX_N = 16 tokens, being
the speculative-tree ABI rather than the wide one.)
"""
from __future__ import annotations
import argparse, ctypes, os, sys

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_PINNED = os.path.join(HERE, "build-qwen", "tf_corpus_e3cdb42.txt")
# Default to the PINNED corpus: the live source file shifts between builds and
# turns every cross-build comparison into prompt drift (see CHANGELOG, twice).
BIGTEXT = os.environ.get("TQ_BENCH_TEXT", _PINNED if os.path.exists(_PINNED) else os.path.join(HERE, "src", "forward_qwen.cu"))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tqf", default=os.environ.get("TQ_MODEL_TQF")
                    or "/home/shooting-brake007/models/knivesysl/qwen3_8-27b-e2m3-mtp.tqf")
    ap.add_argument("--model-dir", default=os.environ.get("TQ_MODEL_DIR")
                    or "/home/shooting-brake007/models/knivesysl")
    ap.add_argument("--lib", default=os.environ.get("TQ_LIB")
                    or os.path.join(HERE, "build-qwen", "libforward_qwen.so"))
    ap.add_argument("--prompt-tokens", type=int, default=8192)
    ap.add_argument("--chunk", type=int, default=256)
    args = ap.parse_args()

    from transformers import AutoTokenizer  # noqa: E402
    tok = AutoTokenizer.from_pretrained(args.model_dir, trust_remote_code=True)
    ids = tok(open(BIGTEXT).read(), add_special_tokens=False).input_ids
    if len(ids) < args.prompt_tokens:
        ids = ids * (args.prompt_tokens // max(len(ids), 1) + 2)
    ids = ids[: args.prompt_tokens]

    L = ctypes.CDLL(args.lib)
    L.qwn_init.argtypes = [ctypes.c_char_p]
    L.qwn_init.restype = ctypes.c_int
    L.qwn_reset_state.restype = ctypes.c_int
    L.qwn_prefill_wide.argtypes = [ctypes.POINTER(ctypes.c_int), ctypes.c_int,
                                   ctypes.c_int, ctypes.POINTER(ctypes.c_int)]
    L.qwn_prefill_wide.restype = ctypes.c_int
    L.qwn_free.restype = ctypes.c_int

    if L.qwn_init(args.tqf.encode()) != 0:
        print("qwn_init failed", file=sys.stderr)
        return 1
    if L.qwn_reset_state() != 0:
        print("qwn_reset_state failed", file=sys.stderr)
        return 1

    out, pos = [], 0
    while pos < len(ids):
        n = min(args.chunk, len(ids) - pos)
        buf = (ctypes.c_int * n)(*ids[pos:pos + n])
        am = ctypes.c_int(-1)
        rc = L.qwn_prefill_wide(buf, pos, n, ctypes.byref(am))
        if rc != 0:
            print(f"prefill_wide failed at pos={pos}: {rc}", file=sys.stderr)
            L.qwn_free()
            return 1
        out.append(am.value)
        pos += n

    print("ARGMAX " + " ".join(str(x) for x in out), flush=True)
    print(f"# waves={len(out)} prompt={len(ids)} chunk={args.chunk} "
          f"nvfp4={os.environ.get('TQ_W_NVFP4', '0')}", flush=True)
    L.qwn_free()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
