#!/usr/bin/env python3
"""Measure any OpenAI-compatible endpoint the same way, so two engines are comparable.

  PREFILL - one request, P-token prompt, max_tokens=1. TTFT is essentially pure prefill,
            so P / ttft is prefill tok/s. `cold` is the first rep, `warm` the best.
  DECODE  - N concurrent requests, G output tokens each. Steady-state throughput is
            (N*G) / (wall - median ttft). Also reports per-stream ms/token.

Two prompt regimes, because they answer different questions:
  default          - a distinct random P-token window per request, so NOTHING can be
                     reused. This is raw kernel speed.
  --shared-prefix  - one prompt reused by every request, so a prefix cache can serve it.
                     This is what real traffic looks like (multi-turn, system prompts,
                     retries) and it is a different number, not a better-measured one.

Prompt lengths are exact: built from real text with the model's own tokenizer, because
max_model_len is a hard limit and an over-long prompt is a 400, not a slower run.
"""
from __future__ import annotations
import argparse, json, os, random, statistics, sys, threading, time, urllib.request

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

def stream(url, body, timeout=600):
    """Return (ttft, t_done, n_tokens)."""
    body = dict(body, stream=True)
    req = urllib.request.Request(
        url, data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"})
    t0 = time.time()
    ttft, n = None, 0
    with urllib.request.urlopen(req, timeout=timeout) as r:
        for raw in r:
            line = raw.decode().strip()
            if not line.startswith("data: "):
                continue
            if line == "data: [DONE]":
                break
            try:
                d = json.loads(line[6:])
            except Exception:
                continue
            ch = d.get("choices") or [{}]
            txt = ch[0].get("text") or (ch[0].get("delta") or {}).get("content") or ""
            if txt:
                if ttft is None:
                    ttft = time.time() - t0
                n += 1
    return ttft or (time.time() - t0), time.time() - t0, n


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="http://127.0.0.1:8000/v1")
    ap.add_argument("--model", default="nvfp4")
    ap.add_argument("--prompt-tokens", type=int, default=4096)
    ap.add_argument("--gen", type=int, default=64)
    ap.add_argument("--conc", default="1,8,32")
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--shared-prefix", action="store_true",
                    help="reuse one prompt so a prefix cache can serve it (real traffic); "
                         "default is a distinct window per request (raw prefill speed)")
    args = ap.parse_args()
    url = args.base + "/completions"

    # Exact token lengths matter: max_model_len is a hard limit and a prompt that is
    # 3x longer than intended is a 400, not a slower run. Build from real text with the
    # model's own tokenizer, then make each prompt unique so no prefix cache can serve it.
    from transformers import AutoTokenizer
    md = os.environ.get("TQ_MODEL_DIR", "/home/shooting-brake007/models/knivesysl")
    tk = AutoTokenizer.from_pretrained(md, trust_remote_code=True)
    corpus = open(os.path.join(HERE, "src", "forward_qwen.cu")).read()
    ids_all = tk(corpus, add_special_tokens=False).input_ids
    rnd = random.Random(1234)

    def prompt(p):
        off = rnd.randrange(0, max(1, len(ids_all) - p - 8))
        ids = ids_all[off:off + p]
        while len(ids) < p:
            ids = ids + ids_all[: p - len(ids)]
        return tk.decode(ids[:p])

    # PREFILL. --shared-prefix reuses ONE prompt across reps so a prefix cache can serve
    # it; the default uses a distinct random window per rep so nothing can. Real traffic
    # is the former (multi-turn, system prompts, retries), raw kernel speed is the latter,
    # and they are different questions -- report both, never one labelled as the other.
    fixed = prompt(args.prompt_tokens)
    best, first = 1e30, None
    for _ in range(args.reps):
        p = fixed if args.shared_prefix else prompt(args.prompt_tokens)
        t, _, _ = stream(url, {"model": args.model, "prompt": p,
                               "max_tokens": 1, "temperature": 0.0})
        if first is None: first = t
        best = min(best, t)
    print(f"PREFILL prompt={args.prompt_tokens} shared={int(args.shared_prefix)} "
          f"cold={first:.4f}s best={best:.4f}s "
          f"cold_tok_s={args.prompt_tokens / first:.1f} "
          f"best_tok_s={args.prompt_tokens / best:.1f}", flush=True)

    # DECODE: N concurrent streams
    for N in [int(x) for x in args.conc.split(",")]:
        res = [None] * N
        # under --shared-prefix every stream sends the SAME 2048-token prompt, which is
        # the multi-client shared-system-prompt case both engines optimise for.
        shared2k = prompt(2048)
        def run(i):
            res[i] = stream(url, {"model": args.model,
                                  "prompt": shared2k if args.shared_prefix else prompt(2048),
                                  "max_tokens": args.gen, "temperature": 0.0,
                                  "ignore_eos": True})
        ths = [threading.Thread(target=run, args=(i,)) for i in range(N)]
        t0 = time.time()
        for t in ths: t.start()
        for t in ths: t.join()
        wall = time.time() - t0
        ok = [r for r in res if r]
        ttfts = [r[0] for r in ok]
        toks = sum(r[2] for r in ok)
        dec_wall = wall - statistics.median(ttfts)
        print(f"DECODE N={N:3d} gen={args.gen} wall={wall:.2f}s "
              f"ttft_p50={statistics.median(ttfts):.3f}s "
              f"decode_tok_s={toks / max(dec_wall, 1e-9):.1f} "
              f"ms_per_tok_per_stream={1000.0 * dec_wall / args.gen:.2f}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
