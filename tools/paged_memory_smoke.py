#!/usr/bin/env python3
"""Real-model regression for late allocation failures in paged prefill/spec/APC.

Run with the production quantization flags, TQ_CTX=262144, and an idle GPU:
  .venv/bin/python tools/paged_memory_smoke.py --tqf /path/to/model.tqf

The original library fails the 51,581-token checkpoint plus eight-token tail
with rc=-94. This also checks mixed waves, checkpoint churn, speculative/plain
parity across attention split thresholds, optional-cache denial, and KV ownership.
"""
from __future__ import annotations

import argparse
import ctypes
import os
from pathlib import Path

from serve_batched import ck, load_lib


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lib", default=str(Path(__file__).resolve().parents[1] / "build-qwen/libforward_qwen.so"))
    parser.add_argument("--tqf", required=True)
    parser.add_argument("--blocks", type=int, default=2100)
    args = parser.parse_args()
    lib = load_lib(args.lib)
    array = lambda values: (ctypes.c_int * len(values))(*values)
    ck(lib.qwn_init(args.tqf.encode()), "model init")
    cap = lib.qwn_wave_cap()

    def wave(slot, base, count, final=False):
        out = array([-1])
        tokens = [100 + (i % 97) for i in range(base, base + count)]
        rc = lib.qwn_paged_prefill_batch(
            array(tokens), array([slot] * count), array(list(range(base, base + count))),
            array([slot]), array([0]), array([count]), array([int(final)]), 1, count, out,
        )
        assert rc == 0, f"prefill slot={slot} base={base} count={count}: rc={rc}"
        return out[0]

    def prefill_to(slot, base, end):
        while base < end:
            count = min(cap, end - base)
            wave(slot, base, count)
            base += count

    def decode(slot, seed, pos, count=8):
        result = []
        for position in range(pos, pos + count):
            out = array([-1])
            ck(lib.qwn_paged_decode_step(array([seed]), array([slot]), array([position]), 1, out), "decode")
            seed = out[0]
            result.append(seed)
        return result

    def assert_free():
        free, total, page, maximum = (ctypes.c_int() for _ in range(4))
        ck(lib.qwn_paged_stats(ctypes.byref(free), ctypes.byref(total),
                               ctypes.byref(page), ctypes.byref(maximum)), "stats")
        assert free.value == total.value == args.blocks, (free.value, total.value)

    try:
        ck(lib.qwn_paged_init(2, args.blocks, 128), "paged init")
        prefill_to(0, 0, 51581)
        checkpoint = ck(lib.qwn_paged_ckpt_save(0, 51581), "51k checkpoint")
        seed = wave(0, 51581, 8, True)
        assert lib.qwn_paged_ckpt_adopt(1, checkpoint) == 51581
        restored_seed = wave(1, 51581, 8, True)
        assert restored_seed == seed, (seed, restored_seed)
        continuation = decode(0, seed, 51589)
        assert continuation == decode(1, restored_seed, 51589)
        print("PASS: 51k checkpoint, eight-token tail, adopt and decode parity", flush=True)

        expected_decode = decode(1, continuation[-1], 51597, 1)[0]
        ck(lib.qwn_paged_reset_slot(1), "reset mixed-wave slot")
        assert lib.qwn_paged_ckpt_adopt(1, checkpoint) == 51581
        tokens = [continuation[-1]] + [100 + (i % 97) for i in range(51581, 51588)]
        out = array([-1, -1])
        ck(lib.qwn_paged_prefill_batch(
            array(tokens), array([0] + [1] * 7), array([51597] + list(range(51581, 51588))),
            array([0, 1]), array([0, 1]), array([1, 7]), array([1, 1]), 2, 8, out,
        ), "mixed decode/prefill wave")
        assert out[0] == expected_decode
        ck(lib.qwn_paged_reset_slot(0), "reset mixed-wave oracle")
        assert lib.qwn_paged_ckpt_adopt(0, checkpoint) == 51581
        assert out[1] == wave(0, 51581, 7, True)
        print("PASS: mixed decode/prefill wave matches independent executions", flush=True)
        ck(lib.qwn_paged_ckpt_free(checkpoint), "checkpoint free")
        for slot in (0, 1):
            ck(lib.qwn_paged_reset_slot(slot), "reset")

        checkpoints = []
        pos = 0
        for end in (264, 10643, 10862, 10969, 11623, 11807, 11934, 14277, 17532, 28704, 32769, 65537):
            prefill_to(0, pos, end)
            pos = end
            checkpoint = lib.qwn_paged_ckpt_save(0, pos)
            if checkpoint == -3 and checkpoints:
                ck(lib.qwn_paged_ckpt_free(checkpoints.pop(0)), "evict")
                checkpoint = lib.qwn_paged_ckpt_save(0, pos)
            assert checkpoint >= 0, f"checkpoint at {pos}: {checkpoint}"
            checkpoints.append(checkpoint)
            ck(lib.qwn_paged_reset_slot(1), "reset comparison slot")
            assert lib.qwn_paged_ckpt_adopt(1, checkpoint) == pos
            out, emitted = array([-1] * 4), array([0])
            rc = lib.qwn_paged_spec_round(array([0]), array([197]), array([pos]),
                                          array([198, 199, 200]), array([3]), 1, 3, out, emitted)
            assert rc == 0, f"spec at {pos}: rc={rc}"
            assert 1 <= emitted[0] <= 4
            expected = decode(1, 197, pos, emitted[0])
            assert list(out)[:emitted[0]] == expected, (pos, list(out), expected)
            ck(lib.qwn_paged_reset_slot(0), "reset test slot")
            assert lib.qwn_paged_ckpt_adopt(0, checkpoint) == pos
            print(f"PASS: checkpoint churn + speculative/plain parity at {pos}", flush=True)
        for slot in (0, 1):
            ck(lib.qwn_paged_reset_slot(slot), "final reset")
        for checkpoint in checkpoints:
            ck(lib.qwn_paged_ckpt_free(checkpoint), "final checkpoint free")
        assert_free()
        print("PASS: all KV blocks returned; no checkpoint references leaked", flush=True)

        # Deny OPTIONAL caches after startup without injecting a fake allocation
        # failure. Their budget is deliberately exhausted; committed state and
        # ordinary decoding must remain usable. Re-init also checks archive cleanup.
        ck(lib.qwn_paged_free(), "reinit cleanup")
        old_spec = os.environ.get("TQ_PAGED_SPEC")
        old_headroom = os.environ.get("TQ_VRAM_HEADROOM_MB")
        os.environ["TQ_PAGED_SPEC"] = "0"
        try:
            ck(lib.qwn_paged_init(2, args.blocks, 128), "reinit without archive")
            runtime = ctypes.CDLL(str(Path(os.environ.get("CUDA_HOME", "/usr/local/cuda")) / "lib64/libcudart.so"))
            available, total = ctypes.c_size_t(), ctypes.c_size_t()
            runtime.cudaMemGetInfo.argtypes = [ctypes.POINTER(ctypes.c_size_t)] * 2
            assert runtime.cudaMemGetInfo(ctypes.byref(available), ctypes.byref(total)) == 0
            os.environ["TQ_VRAM_HEADROOM_MB"] = str(total.value // 1048576 + 1)
            seed = wave(0, 0, 264, True)
            assert wave(1, 0, 264, True) == seed
            assert lib.qwn_paged_ckpt_save(0, 264) == -3
            out, emitted = array([-1] * 4), array([0])
            rc = lib.qwn_paged_spec_round(array([0]), array([seed]), array([264]),
                                          array([198, 199, 200]), array([3]), 1, 3, out, emitted)
            assert rc == -111, rc
            assert decode(0, seed, 264) == decode(1, seed, 264)
            for slot in (0, 1):
                ck(lib.qwn_paged_reset_slot(slot), "reset budget test")
            assert_free()
            print("PASS: denied checkpoint/spec budgets preserve plain-decode state", flush=True)
            ck(lib.qwn_paged_free(), "budget-test cleanup")
            assert lib.qwn_paged_init(2, args.blocks, 128) == -8
            print("PASS: impossible startup headroom is rejected before readiness", flush=True)
        finally:
            for key, previous in (("TQ_PAGED_SPEC", old_spec), ("TQ_VRAM_HEADROOM_MB", old_headroom)):
                if previous is None:
                    os.environ.pop(key, None)
                else:
                    os.environ[key] = previous
    finally:
        lib.qwn_paged_free()
        lib.qwn_free()


if __name__ == "__main__":
    main()
