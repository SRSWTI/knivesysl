#!/usr/bin/env python3
"""Dump REAL activations from the engine, for quantizer studies.

Gaussian synthetic activations cannot evaluate a rotation: Hadamard exists to flatten
OUTLIER channels and a Gaussian has none, so it always reports ~no gain. LLM activations
are heavy-tailed in a few fixed channels -- that is the premise of QuaRot / SmoothQuant --
so the comparison has to run on real ones.

--layer 0  : post-input-layernorm of the embedding (well-conditioned)
--layer N>0: residual stream after N decoder layers, where the "massive activation"
             channels live; RMSNorm keeps the per-channel weight so they survive
             into the GEMM input.

Writes an .npy of shape [n_tokens][H].
"""
from __future__ import annotations
import argparse, ctypes, os, sys
import numpy as np

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tqf", default=os.environ.get("TQ_MODEL_TQF") or "/workspace/models/knivesysl/model.tqf")
    ap.add_argument("--lib", default=os.environ.get("TQ_LIB")
                    or os.path.join(HERE, "build-qwen", "libforward_qwen.so"))
    ap.add_argument("--layer", type=int, default=0)
    ap.add_argument("--tokens", type=int, default=128)
    ap.add_argument("--out", default="/tmp/act.npy")
    args = ap.parse_args()

    L = ctypes.CDLL(args.lib)
    L.qwn_init.argtypes = [ctypes.c_char_p]; L.qwn_init.restype = ctypes.c_int
    L.qwn_hidden_size.restype = ctypes.c_int
    L.qwn_reset_state.restype = ctypes.c_int
    L.qwn_free.restype = ctypes.c_int
    L.qwn_debug_embed_input_norm.argtypes = [ctypes.c_int, ctypes.c_int,
                                             ctypes.POINTER(ctypes.c_float), ctypes.c_int]
    L.qwn_debug_embed_input_norm.restype = ctypes.c_int
    L.qwn_debug_decode_layers.argtypes = [ctypes.c_int, ctypes.c_int, ctypes.c_int,
                                          ctypes.POINTER(ctypes.c_float), ctypes.c_int]
    L.qwn_debug_decode_layers.restype = ctypes.c_int

    if L.qwn_init(args.tqf.encode()) != 0:
        print("qwn_init failed", file=sys.stderr); return 1
    H = L.qwn_hidden_size()
    buf = (ctypes.c_float * H)()
    rows = []
    rng = np.random.default_rng(11)
    ids = rng.integers(1000, 140000, size=args.tokens)
    for t in ids:
        if args.layer == 0:
            n = L.qwn_debug_embed_input_norm(int(t), 0, buf, H)
        else:
            L.qwn_reset_state()
            n = L.qwn_debug_decode_layers(int(t), 0, args.layer, buf, H)
        if n < H:
            print(f"dump returned {n} for token {t}", file=sys.stderr)
            L.qwn_free(); return 1
        rows.append(np.frombuffer(buf, dtype=np.float32, count=H).copy())

    A = np.stack(rows)
    np.save(args.out, A)
    a = np.abs(A)
    ch = a.max(axis=0)                      # per-channel peak over tokens
    print(f"saved {A.shape} -> {args.out}")
    print(f"  |x| mean={a.mean():.4f} p99={np.percentile(a, 99):.4f} max={a.max():.4f}")
    print(f"  per-channel max: median={np.median(ch):.4f} max={ch.max():.4f} "
          f"ratio={ch.max() / np.median(ch):.1f}x  <- outlier severity")
    L.qwn_free()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
