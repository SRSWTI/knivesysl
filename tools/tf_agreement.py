#!/usr/bin/env python3
"""Teacher-forced agreement gate.

Prefill the same prompt, then feed the TRUE continuation token at every step and
record the argmax. Two libs driven identically diverge only through numerics, so
the flip rate is the honest measure of "equally-valid forward" vs "wrong".
Emits one `TF <ids...>` line so two runs (two libs, or two configs of one lib) can be
diffed position by position. The control that makes the number meaningful: run the SAME
lib at two prefill wave widths -- that is the engine's own float-eps sensitivity with no
code change at all, and any real regression has to be worse than it.

Run:
    CUDA_VISIBLE_DEVICES=0 TQ_KV_Q4=1 TQ_EMBED_FP8=2 TQ_CTX=16384 \\
        python3 tools/tf_agreement.py --prompt-tokens 2048 --steps 256 --chunk 256
"""
from __future__ import annotations
import argparse, ctypes, glob, os, sys, time

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(HERE, "tools"))
os.environ.setdefault("TQ_WIDE_PREFILL", "1")
os.environ.setdefault("TQ_WIDE_ATTN_MMA", "1")
from mtp_spec_smoke import load_lib, Eng, prefill, ck, BIGTEXT  # noqa: E402
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
ap.add_argument("--prompt-tokens", type=int, default=2048)
ap.add_argument("--steps", type=int, default=256)
ap.add_argument("--chunk", type=int, default=256)
args = ap.parse_args()

os.environ["TQ_PREFILL_CHUNK"] = str(args.chunk)
L = load_lib(args.lib)
ck(L.qwn_init(args.tqf.encode()), "init")
e = Eng(L)
tok = AutoTokenizer.from_pretrained(args.model_dir, trust_remote_code=True)
ids = tok(open(BIGTEXT).read(), add_special_tokens=False).input_ids
P = args.prompt_tokens
assert len(ids) >= P + args.steps + 8

t0 = time.time()
seed = prefill(e, ids, P - 1)
dt = time.time() - t0
print(f"PREFILL tokens={P} chunk={args.chunk} secs={dt:.4f} tok_s={P/dt:.1f} seed={seed}", flush=True)

# teacher forcing: at position p the engine has consumed ids[0..p]; feeding ids[p+1]
# advances to p+1 and returns argmax at p+1.
out = [int(seed)]
pos = P - 1
for i in range(args.steps):
    t = ids[P + i]                      # the TRUE next token
    a = e.decode(int(t), int(pos) + 1)
    out.append(int(a))
    pos += 1
print("TF " + " ".join(str(x) for x in out), flush=True)
L.qwn_free()
