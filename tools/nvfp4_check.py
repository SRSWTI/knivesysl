#!/usr/bin/env python3
"""NVFP4 tier numeric gate.

Loads the engine with TQ_W_NVFP4 set, then for each converted weight kind compares the
shipping nvf4_proj path against an fp64 reference that decodes the SAME packed bytes.
This validates the fragment layout, the per-16 ue4m3 scale-group mapping and the MMA
together -- it is what catches a mis-mapped scale group, which shows up as a uniform
few-percent error rather than garbage.

Passing here does NOT say the tier is accurate vs bf16; it says the kernel computes
exactly what the packed format encodes. Quantization error is measured separately by
tools/tf_agreement.py.
"""
from __future__ import annotations
import argparse, ctypes, os, sys

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

KINDS = ["mlp_gate", "mlp_up", "mlp_down", "q_proj", "o_proj", "linear_in_qkv", "linear_in_z"]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tqf", default=os.environ.get("TQ_MODEL_TQF")
                    or "/home/shooting-brake007/models/knivesysl/qwen3_8-27b-e2m3-mtp.tqf")
    ap.add_argument("--lib", default=os.environ.get("TQ_LIB")
                    or os.path.join(HERE, "build-qwen", "libforward_qwen.so"))
    ap.add_argument("--layers", default="0,1,4", help="comma-separated layer indices")
    ap.add_argument("--ns", default="8,64,128,256", help="comma-separated column counts")
    ap.add_argument("--tol", type=float, default=2e-3,
                    help="max allowed |diff| / max|ref| (fp32 accumulation noise only)")
    args = ap.parse_args()

    if not os.environ.get("TQ_W_NVFP4"):
        print("TQ_W_NVFP4 is not set -- nothing is converted, the gate would be vacuous.")
        print("  e.g. TQ_W_NVFP4=all  or  TQ_W_NVFP4=mlp")
        return 2

    L = ctypes.CDLL(args.lib)
    L.qwn_init.argtypes = [ctypes.c_char_p]
    L.qwn_init.restype = ctypes.c_int
    L.qwn_nvf4_check.argtypes = [ctypes.c_int, ctypes.c_int, ctypes.c_int,
                                 ctypes.POINTER(ctypes.c_float)]
    L.qwn_nvf4_check.restype = ctypes.c_int
    L.qwn_free.restype = ctypes.c_int

    rc = L.qwn_init(args.tqf.encode())
    if rc != 0:
        print(f"qwn_init failed: {rc}", file=sys.stderr)
        return 1

    layers = [int(x) for x in args.layers.split(",")]
    ns = [int(x) for x in args.ns.split(",")]
    worst, checked, failed, skipped = 0.0, 0, 0, 0
    print(f"{'layer':>5} {'kind':<14} {'N':>5} {'max rel err':>12}  verdict")
    print("-" * 52)
    for li in layers:
        for wi, kind in enumerate(KINDS):
            for n in ns:
                mr = ctypes.c_float(0.0)
                rc = L.qwn_nvf4_check(li, wi, n, ctypes.byref(mr))
                if rc == -3:
                    skipped += 1
                    continue          # this kind was not converted by the selector
                if rc != 0:
                    print(f"{li:5d} {kind:<14} {n:5d} {'-':>12}  ERROR rc={rc}")
                    failed += 1
                    continue
                ok = mr.value <= args.tol
                checked += 1
                worst = max(worst, mr.value)
                if not ok:
                    failed += 1
                print(f"{li:5d} {kind:<14} {n:5d} {mr.value:12.3e}  {'OK' if ok else 'FAIL'}")
    print("-" * 52)
    print(f"checked={checked} skipped(not converted)={skipped} failed={failed} "
          f"worst={worst:.3e} tol={args.tol:.1e}")
    L.qwn_free()
    return 1 if failed or checked == 0 else 0


if __name__ == "__main__":
    raise SystemExit(main())
