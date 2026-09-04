#!/usr/bin/env python3
"""Black-box reliability checks for a running knivesysl OpenAI server.

This test intentionally uses only the public HTTP surface. It does not mock the
engine or import serve_batched.py. Start tools/serve_prod.sh first, then run:

    python3 tools/test_server_reliability.py

KSL_TEST_URL and KSL_API_KEY select a non-default endpoint and bearer token.
"""
from __future__ import annotations

import concurrent.futures
import hashlib
import http.client
import json
import os
import socket
import sys
import threading
import time
import urllib.error
import urllib.parse
import urllib.request

BASE = os.environ.get("KSL_TEST_URL", "http://127.0.0.1:8000").rstrip("/")
API_KEY = os.environ.get("KSL_API_KEY")
TIMEOUT = float(os.environ.get("KSL_TEST_TIMEOUT", "120"))


def fail(message: str):
    raise AssertionError(message)


def auth_headers() -> dict[str, str]:
    return {"Authorization": f"Bearer {API_KEY}"} if API_KEY else {}


def call(method: str, path: str, body=None, headers=None, timeout=TIMEOUT):
    hdr = dict(auth_headers())
    if headers:
        hdr.update(headers)
    data = None
    if body is not None:
        if isinstance(body, bytes):
            data = body
        else:
            data = json.dumps(body, separators=(",", ":")).encode()
            hdr.setdefault("Content-Type", "application/json")
    req = urllib.request.Request(BASE + path, data=data, headers=hdr,
                                 method=method)
    try:
        with urllib.request.urlopen(req, timeout=timeout) as response:
            status = response.status
            content_type = response.headers.get("Content-Type", "")
            raw = response.read()
    except urllib.error.HTTPError as error:
        status = error.code
        content_type = error.headers.get("Content-Type", "")
        raw = error.read()
    if content_type.startswith("application/json"):
        parsed = json.loads(raw) if raw else None
    else:
        parsed = raw.decode("utf-8", "replace")
    return status, content_type, parsed


def error_code(result):
    obj = result[2]
    return obj.get("error", {}).get("code") if isinstance(obj, dict) else None


def wait_idle(timeout=30.0):
    deadline = time.monotonic() + timeout
    last = None
    while time.monotonic() < deadline:
        status, _, last = call("GET", "/healthz", timeout=5)
        if (status == 200 and last.get("active") == 0 and
                last.get("prefilling") == 0 and last.get("queued") == 0 and
                last.get("engine_phase") == "idle"):
            return last
        time.sleep(0.05)
    fail(f"server did not become idle: {last}")


def parse_sse(response, close_after=None):
    events = []
    done = False
    while True:
        line = response.readline()
        if not line:
            break
        if not line.startswith(b"data: "):
            continue
        payload = line[6:].strip()
        if payload == b"[DONE]":
            done = True
            break
        events.append(json.loads(payload))
        if close_after is not None and len(events) >= close_after:
            response.close()
            break
    return events, done


def open_sse(payload):
    headers = auth_headers()
    headers["Content-Type"] = "application/json"
    req = urllib.request.Request(
        BASE + "/v1/completions",
        data=json.dumps(payload, separators=(",", ":")).encode(),
        headers=headers,
        method="POST",
    )
    return urllib.request.urlopen(req, timeout=TIMEOUT)


def raw_missing_content_length():
    url = urllib.parse.urlsplit(BASE)
    host = url.hostname or "127.0.0.1"
    port = url.port or (443 if url.scheme == "https" else 80)
    if url.scheme != "http":
        fail("raw framing check currently requires an http:// KSL_TEST_URL")
    auth = f"Authorization: Bearer {API_KEY}\r\n" if API_KEY else ""
    wire = (
        f"POST /v1/completions HTTP/1.1\r\nHost: {host}\r\n"
        f"{auth}Content-Type: application/json\r\nConnection: close\r\n\r\n"
        "{}"
    ).encode()
    with socket.create_connection((host, port), timeout=10) as sock:
        sock.sendall(wire)
        raw = b""
        while True:
            chunk = sock.recv(4096)
            if not chunk:
                break
            raw += chunk
    first = raw.split(b"\r\n", 1)[0]
    if b" 411 " not in first:
        fail(f"missing Content-Length returned {first!r}")


def main():
    results = {}

    for path, expected_type in (
        ("/healthz", "application/json"),
        ("/readyz", "application/json"),
        ("/livez", "application/json"),
        ("/v1/models", "application/json"),
        ("/metrics", "text/plain"),
    ):
        result = call("GET", path)
        if result[0] != 200 or not result[1].startswith(expected_type):
            fail(f"GET {path}: {result[:2]}")
        results[path] = result[0]

    model_result = call("GET", "/v1/models")
    models = model_result[2].get("data", [])
    if not models:
        fail("/v1/models returned no model")
    model = models[0]["id"]
    base = {"model": model, "prompt": "Count upward:", "max_tokens": 8,
            "temperature": 0.0}

    wrong_model = model + "-does-not-exist"
    cases = {
        "malformed_json": (call("POST", "/v1/completions", b"{" ,
                                {"Content-Type": "application/json"}), 400,
                           "invalid_request"),
        "missing_model": (call("POST", "/v1/completions",
                               {"prompt": "x", "max_tokens": 1}), 400,
                          "model_required"),
        "wrong_model": (call("POST", "/v1/completions",
                             {**base, "model": wrong_model}), 404,
                        "model_not_found"),
        "empty_prompt": (call("POST", "/v1/completions",
                              {**base, "prompt": ""}), 400, None),
        "zero_tokens": (call("POST", "/v1/completions",
                             {**base, "max_tokens": 0}), 400,
                        "invalid_max_tokens"),
        "bad_temperature": (call("POST", "/v1/completions",
                                 {**base, "temperature": -1}), 400,
                            "invalid_temperature"),
        "unsupported_logprobs": (call("POST", "/v1/completions",
                                      {**base, "logprobs": 1}), 400,
                                 "unsupported_parameter"),
        "bad_stream_options": (call("POST", "/v1/completions",
                                    {**base, "stream_options": []}), 400,
                               "invalid_stream_options"),
    }
    for name, (result, status, code) in cases.items():
        if result[0] != status or (code is not None and error_code(result) != code):
            fail(f"{name}: status/code={result[0]}/{error_code(result)}")
        results[name] = [result[0], error_code(result)]
    raw_missing_content_length()
    results["missing_content_length"] = 411

    normal = call("POST", "/v1/completions", base)
    usage = normal[2].get("usage", {}) if isinstance(normal[2], dict) else {}
    choices = normal[2].get("choices", []) if isinstance(normal[2], dict) else []
    if normal[0] != 200 or usage.get("completion_tokens") != 8 or not choices:
        fail(f"normal completion failed: {normal}")
    results["completion"] = {"status": 200, "usage": usage}

    stream_payload = {**base, "max_tokens": 32, "stream": True,
                      "ignore_eos": True,
                      "stream_options": {"include_usage": True}}
    with open_sse(stream_payload) as response:
        if response.status != 200 or not response.headers.get(
                "Content-Type", "").startswith("text/event-stream"):
            fail("SSE response headers are invalid")
        events, done = parse_sse(response)
    usage_events = [e for e in events if e.get("usage")]
    terminal = [e for e in events if e.get("choices") and
                e["choices"][0].get("finish_reason")]
    if not done or len(usage_events) != 1 or not terminal:
        fail("SSE terminal, usage, or [DONE] event is missing")
    results["sse"] = {"events": len(events),
                      "usage": usage_events[0]["usage"]}

    response = open_sse({**stream_payload, "max_tokens": 256})
    disconnected, _ = parse_sse(response, close_after=1)
    idle = wait_idle()
    if not disconnected or not idle.get("engine_thread_alive"):
        fail(f"disconnect recovery failed: {idle}")
    results["disconnect"] = {"events_before_close": len(disconnected),
                             "free_blocks": idle.get("free_blocks")}

    long_prompt = ("north south east west " * 700).strip()
    barrier = threading.Barrier(4)

    def concurrent_request(index):
        barrier.wait()
        payload = {**base, "prompt": long_prompt, "max_tokens": 128,
                   "ignore_eos": True}
        started = time.monotonic()
        result = call("POST", "/v1/completions", payload)
        text = result[2].get("choices", [{}])[0].get("text", "")
        return result[0], time.monotonic() - started, hashlib.sha256(
            text.encode()).hexdigest()[:12]

    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
        futures = [executor.submit(concurrent_request, i) for i in range(4)]
        observed = []
        while not all(f.done() for f in futures):
            health = call("GET", "/healthz", timeout=5)[2]
            observed.append((health.get("active", 0),
                             health.get("prefilling", 0),
                             health.get("queued", 0)))
            time.sleep(0.02)
        completed = [future.result() for future in futures]
    if any(status != 200 for status, _, _ in completed):
        fail(f"concurrent requests failed: {completed}")
    if not observed or max(sum(row) for row in observed) < 2:
        fail(f"concurrent ownership was not observable: {observed[-10:]}")
    results["concurrency"] = {
        "statuses": [row[0] for row in completed],
        "seconds": [round(row[1], 3) for row in completed],
        "max_active_prefill_queued": max(sum(row) for row in observed),
    }

    metrics = call("GET", "/metrics")
    required_metrics = (
        "knivesysl_engine_up ",
        "knivesysl_engine_busy ",
        "knivesysl_engine_phase{phase=",
        "knivesysl_requests_total{state=\"completed\"}",
        "knivesysl_requests_total{state=\"cancelled\"}",
        "knivesysl_scheduler_events_total{kind=\"native_failure\"}",
        "knivesysl_supervisor_restarts_total ",
    )
    missing = [name for name in required_metrics if name not in metrics[2]]
    if missing:
        fail(f"missing Prometheus metrics: {missing}")
    results["metrics"] = "ok"

    final = wait_idle()
    if final.get("free_blocks", -1) < 0 or final.get("free_blocks", 0) > final.get(
            "total_blocks", 0):
        fail(f"invalid final block accounting: {final}")
    results["final_health"] = {
        key: final.get(key) for key in
        ("status", "active", "prefilling", "queued", "free_blocks",
         "total_blocks", "engine_phase", "engine_thread_alive")
    }
    print(json.dumps(results, indent=2, sort_keys=True))


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        print(f"FAIL: {type(error).__name__}: {error}", file=sys.stderr)
        raise
