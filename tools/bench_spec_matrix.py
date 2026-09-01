#!/usr/bin/env python3
"""Repeated context x concurrency benchmark for knivesysl and reference servers.

The artifact is intentionally self-contained: launch/config metadata, feasibility
skips, every raw request/sample, and median/min/max summaries live together. Under
speculative decoding SSE deltas are bursts, not tokens; completion token counts
always come from the server's usage block.
"""
from __future__ import annotations
import argparse
import datetime
import json
import os
import statistics
import threading
import time
import urllib.request

ap = argparse.ArgumentParser()
ap.add_argument("--url", default="http://127.0.0.1:8000")
ap.add_argument("--label", required=True)
ap.add_argument("--engine", default="knivesysl")
ap.add_argument("--output-dir", default="/tmp/gembench")
ap.add_argument("--gen", type=int, default=256)
ap.add_argument("--temp", type=float, default=0.0, help="sampled probe, seeded per client")
ap.add_argument("--workload", choices=["repetitive", "prose"], default="repetitive")
ap.add_argument("--pool-tokens", type=int, default=230000, help="actual num_blocks*page capacity")
ap.add_argument("--slots", type=int, default=4)
ap.add_argument("--only", default="", help="cell filter: ctx:n,ctx:n")
ap.add_argument("--ns", default="1,2,4", help="concurrency rungs")
ap.add_argument("--contexts", default="2048,8192,16384,32768,65536,94208,131072")
ap.add_argument("--model", default="ksl")
ap.add_argument("--repeats", type=int, default=3)
ap.add_argument("--spec-kind", choices=["plain", "ngram", "mtp", "dspark"], default="plain")
ap.add_argument("--spec-nodes", type=int, default=8,
                help="global n-gram verify archive nodes; labels depth-zero fallback")
args = ap.parse_args()

if args.repeats < 1:
    ap.error("--repeats must be >= 1")

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
with open(os.path.join(HERE, "build-qwen", "tf_corpus_e3cdb42.txt")) as f:
    corpus = f.read()
CHR = 2.87
CTXS = [int(x) for x in args.contexts.split(",") if x]
NS = [int(x) for x in args.ns.split(",") if x]
NUMERIC_SAMPLE_KEYS = (
    "wall_s", "ttft_max_s", "ttft_median_s", "dec_toks",
    "per_req_tok_s", "agg_decode_tok_s", "e2e_agg_tok_s",
    "prefill_tok_s_est", "spec_rounds", "spec_rounds_target_n", "spec_committed",
    "tok_per_round", "tok_per_delta", "itl_p50_ms", "itl_p99_ms",
)


def prompt_for(ctx_tok, ci):
    need = int(ctx_tok * CHR)
    if need > len(corpus):
        text = (corpus * ((need + len(corpus) - 1) // len(corpus)))[:need]
        off = 0
    else:
        off = (ci * 49999 + ctx_tok * 7) % max(1, len(corpus) - need - 1)
        text = corpus[off:off + need]
    if args.workload == "prose":
        return ("Background reading:\n" + text
                + "\nNow write an ORIGINAL essay (not quoting the text above) about the"
                  " long-term maintainability of large software systems. At least 400 words.")
    return ("Repository file contents:\n" + text
            + "\nContinue writing the file content above, staying consistent with its style.")


def stream_one(prompt, out, ci=0, barrier=None):
    body = {"model": args.model, "max_tokens": args.gen, "temperature": args.temp,
            "stream": True, "ignore_eos": True,
            "messages": [{"role": "user", "content": prompt}],
            "chat_template_kwargs": {"enable_thinking": False},
            "stream_options": {"include_usage": True}}
    if args.temp > 0:
        body["seed"] = 42 + ci
    req = urllib.request.Request(args.url + "/v1/chat/completions",
                                 json.dumps(body).encode(), {"Content-Type": "application/json"})
    try:
        if barrier is not None:
            barrier.wait()
        t0 = time.perf_counter()
        times = []
        usage_tok = None
        with urllib.request.urlopen(req, timeout=2400) as response:
            for raw in response:
                line = raw.decode("utf-8", "replace").strip()
                if not line.startswith("data: ") or line == "data: [DONE]":
                    continue
                event = json.loads(line[6:])
                usage = event.get("usage") or {}
                if usage.get("completion_tokens") is not None:
                    usage_tok = int(usage["completion_tokens"])
                for choice in event.get("choices", []):
                    delta = choice.get("delta", {})
                    if delta.get("content") or delta.get("reasoning_content"):
                        times.append(time.perf_counter())
        tend = time.perf_counter()
        if usage_tok is None:
            raise RuntimeError("stream ended without completion_tokens usage")
        if not times:
            raise RuntimeError("stream returned no content events")
        itl = sorted(times[i] - times[i - 1] for i in range(1, len(times)))
        dec_s = times[-1] - times[0] if len(times) > 1 else 0.0
        out.append({
            "client": ci,
            "ttft_s": times[0] - t0,
            "e2e_s": tend - t0,
            "first_at": times[0],
            "last_at": times[-1],
            "n_tok": usage_tok,
            "n_delta": len(times),
            "tok_per_delta": usage_tok / len(times),
            "decode_s": dec_s,
            "decode_tok_s": usage_tok / dec_s if dec_s > 0 else None,
            "itl_p50_ms": itl[len(itl) // 2] * 1e3 if itl else None,
            "itl_p99_ms": itl[min(len(itl) - 1, int(len(itl) * 0.99))] * 1e3 if itl else None,
        })
    except Exception as exc:
        out.append({"client": ci, "error": f"{type(exc).__name__}: {exc}"})


def health():
    try:
        with urllib.request.urlopen(args.url + "/health", timeout=10) as response:
            return json.loads(response.read())
    except Exception:
        return {}


def run_once(ctx, n, repeat):
    before = health()
    clients = []
    barrier = threading.Barrier(n + 1)
    threads = [threading.Thread(target=stream_one,
                                args=(prompt_for(ctx, i), clients, i, barrier),
                                daemon=True)
               for i in range(n)]
    for thread in threads:
        thread.start()
    t0 = time.perf_counter()
    barrier.wait()
    for thread in threads:
        thread.join()
    wall = time.perf_counter() - t0
    errors = [client for client in clients if "error" in client]
    if errors:
        raise RuntimeError(f"{len(errors)}/{n} clients failed: {errors[:2]}")
    if len(clients) != n:
        raise RuntimeError(f"expected {n} client results, received {len(clients)}")
    clients.sort(key=lambda client: client["client"])
    after = health()
    toks = sum(client["n_tok"] for client in clients)
    deltas = sum(client["n_delta"] for client in clients)
    decode_rates = [client["decode_tok_s"] for client in clients
                    if client["decode_tok_s"] is not None]
    decode_window = max(client["last_at"] for client in clients) - min(
        client["first_at"] for client in clients)
    ttfts = [client["ttft_s"] for client in clients]
    s0, s1 = before.get("spec", {}), after.get("spec", {})
    rounds = int(s1.get("rounds", 0)) - int(s0.get("rounds", 0))
    committed = int(s1.get("committed", 0)) - int(s0.get("committed", 0))
    by0 = {int(key): int(value) for key, value in (s0.get("rounds_by_n") or {}).items()}
    by1 = {int(key): int(value) for key, value in (s1.get("rounds_by_n") or {}).items()}
    rounds_by_n = {key: by1.get(key, 0) - by0.get(key, 0)
                   for key in sorted(set(by0) | set(by1))
                   if by1.get(key, 0) - by0.get(key, 0)}
    target_rounds = rounds_by_n.get(n, 0)
    expected_depth = max(0, min(15, args.spec_nodes // n - 1))
    if args.spec_kind == "plain":
        behavior = "off"
    elif args.spec_kind == "ngram" and expected_depth == 0:
        behavior = "fallback-tail-active" if rounds > 0 else "fallback"
    elif args.spec_kind == "ngram":
        behavior = ("active" if target_rounds > 0 else
                    "tail-active" if rounds > 0 else "fallback")
    else:
        behavior = "active-uninstrumented" if rounds <= 0 else "active"
    return {
        "repeat": repeat,
        "wall_s": wall,
        "ttft_max_s": max(ttfts),
        "ttft_median_s": statistics.median(ttfts),
        "dec_toks": toks,
        "per_req_tok_s": statistics.mean(decode_rates) if decode_rates else 0.0,
        "agg_decode_tok_s": toks / max(1e-9, decode_window),
        "e2e_agg_tok_s": toks / max(1e-9, wall),
        "prefill_tok_s_est": n * ctx / max(1e-9, max(ttfts)),
        "spec_rounds": rounds,
        "spec_rounds_target_n": target_rounds,
        "spec_rounds_by_n": rounds_by_n,
        "spec_committed": committed,
        "tok_per_round": committed / rounds if rounds else None,
        "tok_per_delta": toks / max(1, deltas),
        "spec_behavior": behavior,
        "itl_p50_ms": max((client["itl_p50_ms"] or 0.0) for client in clients),
        "itl_p99_ms": max((client["itl_p99_ms"] or 0.0) for client in clients),
        "clients": clients,
    }


def summarize(samples):
    summary = {}
    for key in NUMERIC_SAMPLE_KEYS:
        values = [sample[key] for sample in samples if sample.get(key) is not None]
        if values:
            summary[key] = {
                "median": statistics.median(values),
                "min": min(values),
                "max": max(values),
            }
    behaviors = sorted({sample["spec_behavior"] for sample in samples})
    summary["spec_behavior"] = behaviors[0] if len(behaviors) == 1 else "mixed:" + ",".join(behaviors)
    summary["repeats"] = len(samples)
    return summary


def save_artifact(artifact):
    os.makedirs(args.output_dir, exist_ok=True)
    path = os.path.join(args.output_dir, f"specmatrix_{args.label}.json")
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(artifact, f, indent=1)
    os.replace(tmp, path)
    return path


def main():
    artifact = {
        "schema": 2,
        "status": "running",
        "created_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "engine": args.engine,
        "label": args.label,
        "config": {
            "url": args.url, "model": args.model, "gen": args.gen, "temperature": args.temp,
            "workload": args.workload, "pool_tokens": args.pool_tokens, "slots": args.slots,
            "contexts": CTXS, "concurrency": NS, "repeats": args.repeats,
            "spec_kind": args.spec_kind, "spec_nodes": args.spec_nodes,
        },
        "server_start": health(),
        "cells": [],
        "skipped": [],
    }
    only = {tuple(map(int, cell.split(":"))) for cell in args.only.split(",") if cell}
    try:
        warm = []
        stream_one(prompt_for(2048, 99), warm, 99)
        if not warm or "error" in warm[0]:
            raise RuntimeError(f"warmup failed: {warm}")
        for ctx in CTXS:
            for n in NS:
                if only and (ctx, n) not in only:
                    continue
                required = n * (ctx + args.gen + 64)
                if n > args.slots or required > args.pool_tokens:
                    artifact["skipped"].append({
                        "ctx": ctx, "n": n, "required_pool_tokens": required,
                        "reason": "slot-limit" if n > args.slots else "kv-pool",
                    })
                    continue
                expected_depth = (max(0, min(15, args.spec_nodes // n - 1))
                                  if args.spec_kind == "ngram" else None)
                cell = {
                    "ctx": ctx,
                    "n": n,
                    "required_pool_tokens": required,
                    "expected_spec_depth": expected_depth,
                    "expected_spec_behavior": (
                        "fallback" if expected_depth == 0 else
                        "active" if expected_depth is not None else
                        "off" if args.spec_kind == "plain" else "active"
                    ),
                    "samples": [],
                }
                artifact["cells"].append(cell)
                for repeat in range(args.repeats):
                    cell["samples"].append(run_once(ctx, n, repeat))
                    cell["summary"] = summarize(cell["samples"])
                    save_artifact(artifact)
                summary = cell["summary"]
                med = lambda key: summary[key]["median"]
                tpr = (f" tok/round={med('tok_per_round'):.2f}"
                       if "tok_per_round" in summary else "")
                print(f"ctx={ctx:6d} n={n:2d} repeats={args.repeats} "
                      f"per-req={med('per_req_tok_s'):6.1f} tok/s "
                      f"agg-dec={med('agg_decode_tok_s'):7.1f} "
                      f"e2e={med('e2e_agg_tok_s'):7.1f} "
                      f"ttft={med('ttft_max_s'):7.2f}s "
                      f"spec={summary['spec_behavior']}{tpr}", flush=True)
        artifact["status"] = "complete"
        artifact["server_end"] = health()
        artifact["completed_utc"] = datetime.datetime.now(datetime.timezone.utc).isoformat()
        path = save_artifact(artifact)
        print(f"wrote {path}", flush=True)
    except Exception as exc:
        artifact["status"] = "failed"
        artifact["error"] = f"{type(exc).__name__}: {exc}"
        artifact["server_end"] = health()
        save_artifact(artifact)
        raise


if __name__ == "__main__":
    main()
