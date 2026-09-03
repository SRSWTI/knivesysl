#!/usr/bin/env python3
"""Bracket one current-build paged prefill with an NVTX capture range.

This is a profiling harness, not a production path. It loads the shipping shared library,
warms lazy allocation/autotuning before capture, then runs the same paged wide-prefill ABI
used by serve_batched.py. nsys should use `--capture-range=nvtx --nvtx-capture=prefill`.
"""
from __future__ import annotations

import argparse
import ctypes
import os
import time

from transformers import AutoTokenizer
import nvtx


ap = argparse.ArgumentParser()
ap.add_argument("--tqf", required=True)
ap.add_argument("--model-dir", required=True)
ap.add_argument("--lib", default="build-qwen/libforward_qwen.so")
ap.add_argument("--context", type=int, default=8192)
ap.add_argument("--concurrency", type=int, default=1)
ap.add_argument("--wave", type=int, default=2048)
ap.add_argument("--slots", type=int, default=4)
ap.add_argument("--blocks", type=int, default=2100)
ap.add_argument("--page", type=int, default=128)
args = ap.parse_args()


def ci(values):
    return (ctypes.c_int * len(values))(*values)


def ck(rc, label):
    if rc < 0:
        raise RuntimeError(f"{label} failed: {rc}")


lib = ctypes.CDLL(args.lib)
lib.qwn_init.argtypes = [ctypes.c_char_p]
lib.qwn_init.restype = ctypes.c_int
lib.qwn_paged_init.argtypes = [ctypes.c_int] * 3
lib.qwn_paged_init.restype = ctypes.c_int
lib.qwn_paged_reset_slot.argtypes = [ctypes.c_int]
lib.qwn_paged_reset_slot.restype = ctypes.c_int
lib.qwn_paged_prefill_batch.argtypes = [ctypes.POINTER(ctypes.c_int)] * 7 + [
    ctypes.c_int,
    ctypes.c_int,
    ctypes.POINTER(ctypes.c_int),
]
lib.qwn_paged_prefill_batch.restype = ctypes.c_int
lib.qwn_paged_free.restype = ctypes.c_int
lib.qwn_free.restype = ctypes.c_int

ck(lib.qwn_init(args.tqf.encode()), "qwn_init")
ck(lib.qwn_paged_init(args.slots, args.blocks, args.page), "qwn_paged_init")
tokenizer = AutoTokenizer.from_pretrained(args.model_dir, trust_remote_code=True)
source = open(os.path.join(os.path.dirname(os.path.dirname(__file__)), "src", "forward_qwen.cu")).read()
base_ids = tokenizer(source, add_special_tokens=False).input_ids
ids = (base_ids * ((args.context * args.concurrency + len(base_ids) - 1) // len(base_ids) + 1))


def run_prefill(context, concurrency):
    for slot in range(concurrency):
        ck(lib.qwn_paged_reset_slot(slot), f"reset slot {slot}")
    per_wave = max(1, args.wave // concurrency)
    offset = 0
    seed = (ctypes.c_int * concurrency)()
    while offset < context:
        width = min(per_wave, context - offset)
        tokens = []
        col_slot = []
        col_pos = []
        seg_offset = []
        seg_length = []
        seg_slot = []
        seg_final = []
        column = 0
        for slot in range(concurrency):
            seg_offset.append(column)
            seg_length.append(width)
            seg_slot.append(slot)
            seg_final.append(1 if offset + width == context else 0)
            start = slot * context + offset
            tokens.extend(ids[start:start + width])
            col_slot.extend([slot] * width)
            col_pos.extend(range(offset, offset + width))
            column += width
        ck(
            lib.qwn_paged_prefill_batch(
                ci(tokens), ci(col_slot), ci(col_pos), ci(seg_slot), ci(seg_offset),
                ci(seg_length), ci(seg_final), concurrency, column, seed
            ),
            f"paged prefill at {offset}",
        )
        offset += width


# Warm the exact wave shape while keeping capture free of load-time autotuning/allocation.
run_prefill(min(args.context, args.wave // args.concurrency), args.concurrency)
for slot in range(args.concurrency):
    ck(lib.qwn_paged_reset_slot(slot), f"post-warm reset slot {slot}")

nvtx.push_range("prefill")
start = time.perf_counter()
run_prefill(args.context, args.concurrency)
elapsed = time.perf_counter() - start
nvtx.pop_range()

print(
    f"PREFILL context={args.context} n={args.concurrency} wall_s={elapsed:.6f} "
    f"tok_s={args.context * args.concurrency / elapsed:.3f} wave={args.wave}",
    flush=True,
)
lib.qwn_paged_free()
lib.qwn_free()
