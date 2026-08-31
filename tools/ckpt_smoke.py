#!/usr/bin/env python3
"""Checkpoint/fork/refcount parity gate (APC campaign, Phase 1).

Gates, all REQUIRED:
  1. adopt parity   : ckpt_save(A) -> adopt -> prefill S  ==  full prefill A+S (argmax
                      seed AND a decode continuation, token-for-token bit-exact)
  2. fork parity    : fork(live slot) -> prefill S  ==  the same control
  3. donor integrity: the donor slot keeps working after being shared from
  4. refcount cycle : N x (save/adopt/reset/free) returns free_blocks to baseline

A is deliberately NOT block-aligned (exercises the private tail-block copy).
"""
from __future__ import annotations
import argparse, ctypes, os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from mtp_spec_smoke import load_lib, ck  # noqa: E402
from transformers import AutoTokenizer  # noqa: E402

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

ap = argparse.ArgumentParser()
ap.add_argument("--tqf", default=os.environ.get("TQ_MODEL_TQF"))
ap.add_argument("--model-dir", default=os.environ.get("TQ_MODEL_DIR"))
ap.add_argument("--lib", default=os.path.join(HERE, "build-qwen", "libforward_qwen.so"))
ap.add_argument("--prefix-tokens", type=int, default=777)    # NOT a multiple of 128
ap.add_argument("--suffix-tokens", type=int, default=301)
ap.add_argument("--decode-steps", type=int, default=24)
ap.add_argument("--page", type=int, default=128)
args = ap.parse_args()

L = load_lib(args.lib)
for name, argt, rest in [
    ("qwn_paged_init", [ctypes.c_int] * 3, ctypes.c_int),
    ("qwn_paged_reset_slot", [ctypes.c_int], ctypes.c_int),
    ("qwn_paged_ckpt_save", [ctypes.c_int, ctypes.c_int], ctypes.c_int),
    ("qwn_paged_ckpt_adopt", [ctypes.c_int, ctypes.c_int], ctypes.c_int),
    ("qwn_paged_ckpt_free", [ctypes.c_int], ctypes.c_int),
    ("qwn_paged_fork", [ctypes.c_int] * 3, ctypes.c_int),
]:
    fn = getattr(L, name)
    fn.argtypes = argt
    fn.restype = rest
L.qwn_paged_prefill_batch.argtypes = [ctypes.POINTER(ctypes.c_int)] * 7 + [ctypes.c_int, ctypes.c_int, ctypes.POINTER(ctypes.c_int)]
L.qwn_paged_prefill_batch.restype = ctypes.c_int
L.qwn_paged_decode_step.argtypes = [ctypes.POINTER(ctypes.c_int)] * 3 + [ctypes.c_int, ctypes.POINTER(ctypes.c_int)]
L.qwn_paged_decode_step.restype = ctypes.c_int
L.qwn_paged_stats.argtypes = [ctypes.POINTER(ctypes.c_int)] * 4
L.qwn_paged_stats.restype = ctypes.c_int

ck(L.qwn_init(args.tqf.encode()), "init")
tok = AutoTokenizer.from_pretrained(args.model_dir, trust_remote_code=True)
corpus = os.path.join(HERE, "build-qwen", "tf_corpus_e3cdb42.txt")
ids = tok(open(corpus).read(), add_special_tokens=False).input_ids
cT = lambda a: (ctypes.c_int * len(a))(*a)

P, S, page = args.prefix_tokens, args.suffix_tokens, args.page
A = [ids[i] for i in range(P)]
SFX = [ids[10_000 + i] for i in range(S)]

ck(L.qwn_paged_init(6, 96, page), "paged_init")


def free_blocks():
    fb = ctypes.c_int(); tb = ctypes.c_int(); pg = ctypes.c_int(); mb = ctypes.c_int()
    L.qwn_paged_stats(ctypes.byref(fb), ctypes.byref(tb), ctypes.byref(pg), ctypes.byref(mb))
    return fb.value


def prefill(slot, toks, base, final):
    """One wave carrying `toks` at positions [base, base+len)."""
    n = len(toks)
    oseed = (ctypes.c_int * 1)()
    ck(L.qwn_paged_prefill_batch(cT(toks), cT([slot] * n), cT(list(range(base, base + n))),
                                 cT([slot]), cT([0]), cT([n]), cT([1 if final else 0]),
                                 1, n, oseed), f"prefill s{slot}@{base}")
    return oseed[0]


def decode(slot, seed, base, steps):
    sd = (ctypes.c_int * 1)(seed)
    ssl = (ctypes.c_int * 1)(slot)
    spos = (ctypes.c_int * 1)(base)
    out = (ctypes.c_int * 1)()
    got = []
    for _ in range(steps):
        ck(L.qwn_paged_decode_step(sd, ssl, spos, 1, out), "decode")
        got.append(out[0])
        sd[0] = out[0]; spos[0] += 1
    return got


base_free = free_blocks()

# --- control: full prefill A+S on slot 2 ---
ck(L.qwn_paged_reset_slot(2), "reset2")
prefill(2, A, 0, False)
seed_ctl = prefill(2, SFX, P, True)
dec_ctl = decode(2, seed_ctl, P + S, args.decode_steps)

# --- donor: prefill A on slot 0, checkpoint at P ---
ck(L.qwn_paged_reset_slot(0), "reset0")
prefill(0, A, 0, False)
ckid = L.qwn_paged_ckpt_save(0, P)
assert ckid >= 0, f"ckpt_save rc={ckid}"

# --- gate 1: adopt into slot 1, prefill suffix, compare ---
ck(L.qwn_paged_reset_slot(1), "reset1")
pos = L.qwn_paged_ckpt_adopt(1, ckid)
assert pos == P, f"adopt pos={pos} want {P}"
seed_ad = prefill(1, SFX, P, True)
dec_ad = decode(1, seed_ad, P + S, args.decode_steps)
g1 = (seed_ad == seed_ctl) and (dec_ad == dec_ctl)
print(f"GATE1 adopt-parity : seed {seed_ad}=={seed_ctl} decode match "
      f"{sum(a == b for a, b in zip(dec_ad, dec_ctl))}/{args.decode_steps} -> "
      f"{'PASS' if g1 else 'FAIL'}")

# --- gate 2: fork the LIVE donor into slot 3, prefill suffix, compare ---
ck(L.qwn_paged_reset_slot(3), "reset3")
rc = L.qwn_paged_fork(0, 3, P)
assert rc == 0, f"fork rc={rc}"
seed_fk = prefill(3, SFX, P, True)
dec_fk = decode(3, seed_fk, P + S, args.decode_steps)
g2 = (seed_fk == seed_ctl) and (dec_fk == dec_ctl)
print(f"GATE2 fork-parity  : seed {seed_fk}=={seed_ctl} decode match "
      f"{sum(a == b for a, b in zip(dec_fk, dec_ctl))}/{args.decode_steps} -> "
      f"{'PASS' if g2 else 'FAIL'}")

# --- gate 3: donor unharmed by being shared from ---
seed_dn = prefill(0, SFX, P, True)
dec_dn = decode(0, seed_dn, P + S, args.decode_steps)
g3 = (seed_dn == seed_ctl) and (dec_dn == dec_ctl)
print(f"GATE3 donor-integr : seed {seed_dn}=={seed_ctl} decode match "
      f"{sum(a == b for a, b in zip(dec_dn, dec_ctl))}/{args.decode_steps} -> "
      f"{'PASS' if g3 else 'FAIL'}")

# --- gate 4: refcount cycling returns the pool to baseline ---
for slot in (0, 1, 2, 3):
    ck(L.qwn_paged_reset_slot(slot), f"reset{slot}")
ck(L.qwn_paged_ckpt_free(ckid), "ckpt_free")
mid_free = free_blocks()
for it in range(10):
    ck(L.qwn_paged_reset_slot(0), "r0")
    prefill(0, A, 0, False)
    cid = L.qwn_paged_ckpt_save(0, P)
    assert cid >= 0, f"cycle save rc={cid}"
    ck(L.qwn_paged_reset_slot(1), "r1")
    apos = L.qwn_paged_ckpt_adopt(1, cid)
    assert apos == P
    ck(L.qwn_paged_reset_slot(1), "r1b")
    ck(L.qwn_paged_reset_slot(0), "r0b")
    ck(L.qwn_paged_ckpt_free(cid), "cf")
end_free = free_blocks()
g4 = (mid_free == base_free) and (end_free == base_free)
print(f"GATE4 refcount     : baseline {base_free} after-gates {mid_free} "
      f"after-cycles {end_free} -> {'PASS' if g4 else 'FAIL'}")

ok = g1 and g2 and g3 and g4
print(f"SUMMARY ckpt_smoke: {'ALL PASS' if ok else 'FAILURES PRESENT'}")
L.qwn_free()
sys.exit(0 if ok else 1)
