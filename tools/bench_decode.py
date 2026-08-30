#!/usr/bin/env python3
"""Paged continuous-batch decode benchmark: ms/step and tok/s across concurrency
and context depth. Prefills N slots to depth P with ragged batched prefill waves,
then times `steps` paged decode steps.

Pool sizing matters: per-slot DeltaNet state is ~145 MiB (48 linear layers x a
[48,128,128] fp32 recurrent matrix + conv) and one page=128 block is ~3.44 MiB
(16 full-attention layers of Q4 K + E4M3 V), so both are sized to the case.

Run:
    CUDA_VISIBLE_DEVICES=0 TQ_KV_Q4=1 TQ_EMBED_FP8=2 TQ_CTX=196608 \\
        python3 tools/bench_decode.py --cases 1:2048,32:2048,1:131072

Emits:  DEC N=<n> P=<p> ms_step=<f> tok_s=<f> pf_tok_s=<f>
"""
from __future__ import annotations
import argparse, ctypes, glob, os, time

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.environ.setdefault("TQ_WIDE_PREFILL", "1")
os.environ.setdefault("TQ_WIDE_ATTN_MMA", "1")
from transformers import AutoTokenizer  # noqa: E402


def _first(*globs):
    for g in globs:
        hits = sorted(glob.glob(os.path.expanduser(g)))
        if hits:
            return hits[0]
    return ""


ap = argparse.ArgumentParser()
ap.add_argument("--tqf", default=os.environ.get("TQ_MODEL_TQF") or
                _first(HERE + "/*.tqf", "~/models/knivesysl/*.tqf"))
ap.add_argument("--model-dir", default=os.environ.get("TQ_MODEL_DIR") or
                _first("~/models/knivesysl"))
ap.add_argument("--lib", default=os.environ.get("TQ_LIB") or
                HERE + "/build-qwen/libforward_qwen.so")
ap.add_argument("--page", type=int, default=128)
ap.add_argument("--steps", type=int, default=30)
ap.add_argument("--cases", default="1:2048,4:2048,8:2048,16:2048,32:2048",
                help="comma-separated <concurrency>:<context> cases")
ap.add_argument("--wave", type=int, default=256, help="prefill columns per wave")
ap.add_argument("--profile", action="store_true")
ap.add_argument("--slots", type=int, default=0, help="force pool slots (0 = size to the case)")
ap.add_argument("--blocks", type=int, default=0, help="force pool blocks (0 = size to the case)")
args = ap.parse_args()


def ck(r, what):
    if isinstance(r, int) and r < 0:
        raise RuntimeError(f"{what} failed: {r}")
    return r


L = ctypes.CDLL(args.lib)
L.qwn_init.argtypes = [ctypes.c_char_p]; L.qwn_init.restype = ctypes.c_int
L.qwn_paged_init.argtypes = [ctypes.c_int] * 3; L.qwn_paged_init.restype = ctypes.c_int
L.qwn_paged_free.restype = ctypes.c_int
L.qwn_paged_reset_slot.argtypes = [ctypes.c_int]; L.qwn_paged_reset_slot.restype = ctypes.c_int
L.qwn_paged_decode_step.argtypes = [ctypes.POINTER(ctypes.c_int)] * 3 + [ctypes.c_int, ctypes.POINTER(ctypes.c_int)]
L.qwn_paged_decode_step.restype = ctypes.c_int
L.qwn_paged_prefill_batch.argtypes = [ctypes.POINTER(ctypes.c_int)] * 7 + [ctypes.c_int, ctypes.c_int, ctypes.POINTER(ctypes.c_int)]
L.qwn_paged_prefill_batch.restype = ctypes.c_int
L.qwn_paged_stats.argtypes = [ctypes.POINTER(ctypes.c_int)] * 4; L.qwn_paged_stats.restype = ctypes.c_int
L.qwn_free.restype = ctypes.c_int

ck(L.qwn_init(args.tqf.encode()), "init")
tok = AutoTokenizer.from_pretrained(args.model_dir, trust_remote_code=True)
ids = tok(open(HERE + "/src/forward_qwen.cu").read(), add_special_tokens=False).input_ids
cT = lambda a: (ctypes.c_int * len(a))(*a)

cases = []
for c in args.cases.split(","):
    n, p = c.split(":")
    cases.append((int(n), int(p)))

print(f"{'N':>4s} {'P':>7s} {'prefill s':>10s} {'pf tok/s':>9s} {'ms/step':>9s} {'tok/s':>9s} {'blocks':>11s}")
for (N, P) in cases:
    # Pool geometry is an independent variable, not a function of the case: the server
    # runs a big pool (many slots, many blocks) while this harness used to size one
    # exactly to the case, which made the two incomparable.
    slots = args.slots or N
    blocks = args.blocks or N * ((P + args.page - 1) // args.page + 2)
    r = L.qwn_paged_init(slots, blocks, args.page)
    if r != 0:
        print(f"{N:4d} {P:7d}   paged_init rc={r} (slots={slots} blocks={blocks} "
              f"~{slots * 145 + blocks * 3.44:.0f} MiB) -- skip", flush=True)
        continue
    fb = ctypes.c_int(); tb = ctypes.c_int(); pg = ctypes.c_int(); mb = ctypes.c_int()
    L.qwn_paged_stats(ctypes.byref(fb), ctypes.byref(tb), ctypes.byref(pg), ctypes.byref(mb))
    for c in range(N):
        ck(L.qwn_paged_reset_slot(c), "reset_slot")
    per = max(1, args.wave // N)
    seed = [0] * N
    t0 = time.time()
    w = 0
    while w < P:
        cn = min(per, P - w)
        last = (w + cn >= P)
        toks_c, cslot, cpos, soff, slen, sslot, sfin = [], [], [], [], [], [], []
        off = 0
        for c in range(N):
            soff.append(off); slen.append(cn); sslot.append(c); sfin.append(1 if last else 0)
            for p in range(cn):
                toks_c.append(ids[(c * 7919 + w + p) % (len(ids) - 1)])
                cslot.append(c); cpos.append(w + p)
            off += cn
        oseed = (ctypes.c_int * N)()
        ck(L.qwn_paged_prefill_batch(cT(toks_c), cT(cslot), cT(cpos), cT(sslot), cT(soff),
                                     cT(slen), cT(sfin), N, off, oseed), f"prefill_batch@{w}")
        if last:
            seed = [oseed[c] for c in range(N)]
        w += cn
    pf = time.time() - t0

    sd = (ctypes.c_int * N)(*seed)
    ssl = (ctypes.c_int * N)(*list(range(N)))
    spos = (ctypes.c_int * N)(*([P] * N))
    out = (ctypes.c_int * N)()
    for _ in range(3):
        ck(L.qwn_paged_decode_step(sd, ssl, spos, N, out), "warm")
        for c in range(N):
            sd[c] = out[c]; spos[c] = spos[c] + 1
    rt = ctypes.CDLL("libcudart.so") if args.profile else None
    if rt: rt.cudaProfilerStart()
    t0 = time.time()
    for _ in range(args.steps):
        ck(L.qwn_paged_decode_step(sd, ssl, spos, N, out), "step")
        for c in range(N):
            sd[c] = out[c]; spos[c] = spos[c] + 1
    dt = (time.time() - t0) / args.steps
    if rt: rt.cudaProfilerStop()
    L.qwn_paged_stats(ctypes.byref(fb), ctypes.byref(tb), ctypes.byref(pg), ctypes.byref(mb))
    print(f"{N:4d} {P:7d} {pf:10.2f} {N*P/pf:9.0f} {dt * 1e3:9.2f} {N / dt:9.1f} "
          f"{tb.value - fb.value:5d}/{tb.value:5d}", flush=True)
    print(f"DEC N={N} P={P} ms_step={dt * 1e3:.4f} tok_s={N / dt:.2f} pf_tok_s={N*P/pf:.1f}", flush=True)
    L.qwn_paged_free()
L.qwn_free()
