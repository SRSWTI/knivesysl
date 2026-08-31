#!/usr/bin/env python3
"""APC proof benchmark (phase 3): TTFT-focused, streaming.

Modes:
  turns : append-only conversation -- BASE-token system context, ~TURN tokens of
          new material per turn. Reports per-turn TTFT (time to first streamed
          delta). The APC claim: TTFT stays ~flat instead of growing with depth.
  fleet : FAN concurrent requests sharing a BASE-token prefix with distinct
          assignments (the subagent fan-out). Reports per-request TTFT + wall.

Works against any OpenAI endpoint (ours or vLLM) -- same client, same corpus.
"""
from __future__ import annotations
import argparse, json, os, sys, threading, time, urllib.request

ap = argparse.ArgumentParser()
ap.add_argument("--url", default="http://127.0.0.1:8000/v1/chat/completions")
ap.add_argument("--model", default="ksl")
ap.add_argument("--mode", choices=["turns", "fleet"], default="turns")
ap.add_argument("--base", type=int, default=24000, help="shared context tokens (approx)")
ap.add_argument("--turn", type=int, default=2000, help="new tokens per turn (approx)")
ap.add_argument("--turns", type=int, default=12)
ap.add_argument("--fan", type=int, default=6)
ap.add_argument("--gen", type=int, default=16)
ap.add_argument("--think", type=int, default=0, help="enable_thinking (0 for clean TTFT)")
args = ap.parse_args()

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
corpus = open(os.path.join(HERE, "build-qwen", "tf_corpus_e3cdb42.txt")).read()
# ~4 chars/token for code text: cut by chars, sized generously
CHR = 4


def chunk(i, ntok):
    off = (i * 7919 * CHR) % max(1, len(corpus) - ntok * CHR - 1)
    return corpus[off:off + ntok * CHR]


def stream_ttft(messages, tag):
    body = {"model": args.model, "messages": messages, "max_tokens": args.gen,
            "temperature": 0, "stream": True,
            "chat_template_kwargs": {"enable_thinking": bool(args.think)}}
    req = urllib.request.Request(args.url, json.dumps(body).encode(),
                                 {"Content-Type": "application/json"})
    t0 = time.time()
    ttft = None
    ptoks = 0
    with urllib.request.urlopen(req, timeout=1200) as r:
        for raw in r:
            ln = raw.decode("utf-8", "replace").strip()
            if not ln.startswith("data: ") or ln == "data: [DONE]":
                continue
            d = json.loads(ln[6:])
            for c in d.get("choices", []):
                dl = c.get("delta", {})
                if ttft is None and (dl.get("content") or dl.get("reasoning_content")
                                     or dl.get("tool_calls")):
                    ttft = time.time() - t0
            if d.get("usage"):
                ptoks = d["usage"].get("prompt_tokens", 0)
    return ttft, time.time() - t0, ptoks


if args.mode == "turns":
    hist = [{"role": "system", "content": "You are a coding assistant. Repo context:\n" + chunk(0, args.base)}]
    print(f"# turns mode: base~{args.base} +~{args.turn}/turn x {args.turns} (gen={args.gen})")
    for t in range(1, args.turns + 1):
        hist.append({"role": "user",
                     "content": f"Turn {t}. New file contents:\n" + chunk(t, args.turn)
                                + "\nReply with just: ok"})
        ttft, wall, pt = stream_ttft(hist, f"t{t}")
        tag = f"{ttft:6.3f}s" if ttft is not None else "  FAIL "
        print(f"TURN {t:2d} ttft={tag} wall={wall:6.3f}s ptoks={pt}", flush=True)
        hist.append({"role": "assistant", "content": "ok"})
elif args.mode == "fleet":
    sysmsg = {"role": "system", "content": "You are a coding subagent. Shared context:\n" + chunk(0, args.base)}
    print(f"# fleet mode: {args.fan} subagents x {args.base}-token shared prefix (gen={args.gen})")
    results = [None] * args.fan
    t_all = time.time()

    def worker(i):
        msgs = [sysmsg, {"role": "user", "content": f"Subagent {i}: assignment:\n" + chunk(100 + i, 400)
                                                    + "\nReply with just: ok"}]
        results[i] = stream_ttft(msgs, f"f{i}")

    # warm ONE to build the shared-prefix checkpoint, then fan out concurrently
    worker(0)
    _w = results[0]
    _tag = f"{_w[0]:6.3f}s" if _w[0] is not None else "  FAIL "
    print(f"WARM  0 ttft={_tag} wall={_w[1]:6.3f}s ptoks={_w[2]}", flush=True)
    t_fan = time.time()
    th = [threading.Thread(target=worker, args=(i,)) for i in range(1, args.fan)]
    for x in th: x.start()
    for x in th: x.join()
    for i in range(1, args.fan):
        ttft, wall, pt = results[i]
        tag = f"{ttft:6.3f}s" if ttft is not None else "  FAIL "
        print(f"FLEET {i} ttft={tag} wall={wall:6.3f}s ptoks={pt}", flush=True)
    print(f"FANOUT wall={time.time()-t_fan:6.3f}s total={time.time()-t_all:6.3f}s", flush=True)
