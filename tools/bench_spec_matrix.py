#!/usr/bin/env python3
"""Paged spec-decode matrix bench: decode tok/s at (context x concurrency),
spec on vs off, on the SAME server config (restart with TQ_PAGED_SPEC toggled).

Cells are clipped to the KV pool: n*ctx must fit --num-blocks * page. Each
client gets a DISTINCT corpus slice (no shared-prefix effects); generation
continues the context (realistic repetition -> honest n-gram accept rates,
which GROW with depth as the drafter's index covers more history).

Measures the DECODE window only (first streamed token -> last): per-request
tok/s, cell aggregate, plus /health spec-round deltas (tokens per round).

Usage: bench_spec_matrix.py --label on|off [--url ...] [--gen 256]
Writes /tmp/gembench/specmatrix_<label>.json
"""
from __future__ import annotations
import argparse, json, os, threading, time, urllib.request

ap = argparse.ArgumentParser()
ap.add_argument("--url", default="http://127.0.0.1:8000")
ap.add_argument("--label", required=True)
ap.add_argument("--gen", type=int, default=256)
ap.add_argument("--temp", type=float, default=0.0, help="sampled-verify probe (seeded per client)")
ap.add_argument("--workload", choices=["repetitive", "prose"], default="repetitive",
                help="prose = ngram-hostile (probes the EMA/depth gating cost floor)")
ap.add_argument("--pool-tokens", type=int, default=230000, help="num_blocks*page budget")
ap.add_argument("--slots", type=int, default=4)
args = ap.parse_args()

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
corpus = open(os.path.join(HERE, "build-qwen", "tf_corpus_e3cdb42.txt")).read()
CHR = 2.87                       # measured chars/token for this corpus+tokenizer

CTXS = [2048, 8192, 32768, 65536, 94208, 131072]
NS = [1, 2, 4]

def prompt_for(ctx_tok, ci):
    need = int(ctx_tok * CHR)
    off = (ci * 49999 + ctx_tok * 7) % max(1, len(corpus) - need - 1)
    if args.workload == "prose":
        return ("Background reading:\n" + corpus[off:off + need]
                + "\nNow write an ORIGINAL essay (not quoting the text above) about the"
                  " long-term maintainability of large software systems. At least 400 words.")
    return ("Repository file contents:\n" + corpus[off:off + need]
            + "\nContinue writing the file content above, staying consistent with its style.")


def stream_one(prompt, out, ci=0):
    body = {"model": "ksl", "max_tokens": args.gen, "temperature": args.temp, "stream": True,
            "ignore_eos": True, "messages": [{"role": "user", "content": prompt}],
            "chat_template_kwargs": {"enable_thinking": False}}
    if args.temp > 0:
        body["seed"] = 42 + ci
    req = urllib.request.Request(args.url + "/v1/chat/completions",
                                 json.dumps(body).encode(), {"Content-Type": "application/json"})
    t0 = time.time(); times = []
    with urllib.request.urlopen(req, timeout=2400) as r:
        for raw in r:
            ln = raw.decode("utf-8", "replace").strip()
            if not ln.startswith("data: ") or ln == "data: [DONE]":
                continue
            d = json.loads(ln[6:])
            for c in d.get("choices", []):
                if c.get("delta", {}).get("content") or c.get("delta", {}).get("reasoning_content"):
                    times.append(time.time())
    itl = sorted(times[i] - times[i - 1] for i in range(1, len(times)))
    out.append({"ttft": times[0] - t0 if times else None,
                "n_tok": len(times),
                "dec_s": (times[-1] - times[0]) if len(times) > 1 else 0.0,
                "itl_p50": itl[len(itl) // 2] * 1e3 if itl else None,
                "itl_p99": itl[int(len(itl) * 0.99)] * 1e3 if itl else None})


def health():
    return json.loads(urllib.request.urlopen(args.url + "/health", timeout=10).read())


def run_cell(ctx, n):
    h0 = health()
    outs = []
    th = [threading.Thread(target=stream_one, args=(prompt_for(ctx, i), outs, i)) for i in range(n)]
    t0 = time.time()
    for t in th: t.start()
    for t in th: t.join()
    wall = time.time() - t0
    h1 = health()
    toks = sum(o["n_tok"] for o in outs)
    dec = [o["n_tok"] / o["dec_s"] for o in outs if o["dec_s"] > 0]
    rounds = h1["spec"]["rounds"] - h0["spec"]["rounds"]
    committed = h1["spec"]["committed"] - h0["spec"]["committed"]
    return {"ctx": ctx, "n": n, "wall": wall,
            "ttft_max": max((o["ttft"] or 0) for o in outs),
            "dec_toks": toks,
            "per_req_tok_s": sum(dec) / len(dec) if dec else 0.0,
            "agg_tok_s": toks / max(1e-9, wall - max((o["ttft"] or 0) for o in outs)),
            "spec_rounds": rounds,
            "tok_per_round": committed / rounds if rounds else None,
            "itl_p50_ms": max((o["itl_p50"] or 0) for o in outs),
            "itl_p99_ms": max((o["itl_p99"] or 0) for o in outs)}


def main():
    # warmup: autotune + allocator paths
    w = []; stream_one(prompt_for(2048, 99), w)
    results = []
    for ctx in CTXS:
        for n in NS:
            if n * (ctx + args.gen + 64) > args.pool_tokens or n > args.slots:
                continue
            r = run_cell(ctx, n)
            results.append(r)
            tpr = f" tok/round={r['tok_per_round']:.2f}" if r["tok_per_round"] else ""
            print(f"ctx={ctx:6d} n={n} per-req={r['per_req_tok_s']:6.1f} tok/s "
                  f"agg={r['agg_tok_s']:6.1f} ttft={r['ttft_max']:6.2f}s{tpr}", flush=True)
    os.makedirs("/tmp/gembench", exist_ok=True)
    with open(f"/tmp/gembench/specmatrix_{args.label}.json", "w") as f:
        json.dump(results, f, indent=1)
    print(f"wrote /tmp/gembench/specmatrix_{args.label}.json")


if __name__ == "__main__":
    main()
