#!/usr/bin/env python3
"""Small agentic coding client for a local OpenAI-compatible vLLM server.

The default endpoint is http://127.0.0.1:8000/v1. The served model is
resolved from /v1/models, so the client does not hard-code a Hugging Face ID.
"""

from __future__ import annotations

import argparse
import codecs
import glob as globlib
import json
import os
from pathlib import Path
import re
import selectors
import shlex
import shutil
import signal
import subprocess
import sys
import tempfile
import time
import urllib.request
from typing import Any

try:
    import tiktoken
    from dotenv import load_dotenv
    from openai import OpenAI
except ImportError as exc:
    raise SystemExit(
        "Missing client dependencies. Run with the vLLM environment, for example:\n"
        "  /home/shooting-brake007/srswti/shooting-brake/.venv/bin/python "
        "tools/axe_vllm.py\n"
        f"Import error: {exc}"
    ) from exc

# Optional web-search dependencies: the web_search tool degrades to a clear
# error when either is missing, but the rest of the client still works.
try:
    from ddgs import DDGS
except ImportError:
    DDGS = None
try:
    import trafilatura
except ImportError:
    trafilatura = None


load_dotenv()

DEFAULT_BASE_URL = os.getenv("VLLM_BASE_URL", "http://127.0.0.1:8000/v1").rstrip("/")
DEFAULT_API_KEY = os.getenv("VLLM_API_KEY", "not-needed")
DEFAULT_MODEL = os.getenv("VLLM_MODEL")
DEFAULT_MAX_COMPLETION_TOKENS = int(os.getenv("VLLM_MAX_COMPLETION_TOKENS", "8192"))
DEFAULT_MAX_TOOL_ROUNDS = int(os.getenv("VLLM_MAX_TOOL_ROUNDS", "32"))

_USE_COLOR = sys.stdout.isatty() and "NO_COLOR" not in os.environ


def _ansi(code: str) -> str:
    return code if _USE_COLOR else ""


RESET, BOLD, DIM = _ansi("\033[0m"), _ansi("\033[1m"), _ansi("\033[2m")
BLUE, CYAN, GREEN, YELLOW, RED, GRAY = (
    _ansi("\033[34m"),
    _ansi("\033[36m"),
    _ansi("\033[32m"),
    _ansi("\033[33m"),
    _ansi("\033[31m"),
    _ansi("\033[90m"),
)

TOTAL_INPUT_TOKENS = 0
TOTAL_OUTPUT_TOKENS = 0


def read_file(args: dict[str, Any]) -> str:
    try:
        path = Path(args["path"])
        lines = path.read_text(errors="replace").splitlines(keepends=True)
        offset = max(0, int(args.get("offset", 0)))
        limit = max(0, int(args.get("limit", len(lines))))
        selected = lines[offset : offset + limit]
        return "".join(
            f"{offset + index + 1:4}| {line}"
            for index, line in enumerate(selected)
        )
    except Exception as exc:
        return f"error: {exc}"


def _atomic_write(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    mode = path.stat().st_mode if path.exists() else None
    with tempfile.NamedTemporaryFile(
        mode="w", encoding="utf-8", dir=path.parent, delete=False
    ) as handle:
        handle.write(content)
        temporary = Path(handle.name)
    if mode is not None:
        temporary.chmod(mode)
    os.replace(temporary, path)


def write_file(args: dict[str, Any]) -> str:
    try:
        _atomic_write(Path(args["path"]), args["content"])
        return "ok"
    except Exception as exc:
        return f"error: {exc}"


def edit_file(args: dict[str, Any]) -> str:
    try:
        path = Path(args["path"])
        text = path.read_text(errors="strict")
        old, new = args["old"], args["new"]
        if old not in text:
            return "error: old_string not found"
        count = text.count(old)
        if not args.get("all") and count > 1:
            return (
                f"error: old_string appears {count} times; "
                "it must be unique unless all=true"
            )
        replacement = text.replace(old, new) if args.get("all") else text.replace(old, new, 1)
        _atomic_write(path, replacement)
        return "ok"
    except Exception as exc:
        return f"error: {exc}"


def glob_files(args: dict[str, Any]) -> str:
    try:
        pattern = str(Path(args.get("path", ".")) / args["pat"])
        paths = globlib.glob(pattern, recursive=True)
        paths.sort(
            key=lambda item: os.path.getmtime(item) if os.path.exists(item) else 0,
            reverse=True,
        )
        limit = max(1, int(args.get("limit", 200)))
        selected = paths[:limit]
        result = "\n".join(selected) or "none"
        if len(paths) > limit:
            result += f"\n[truncated: {len(paths) - limit} additional paths]"
        return result
    except Exception as exc:
        return f"error: {exc}"


def grep_files(args: dict[str, Any]) -> str:
    try:
        pattern = re.compile(args["pat"])
        hits: list[str] = []
        root = args.get("path", ".")
        limit = max(1, int(args.get("limit", 50)))
        for filename in globlib.iglob(str(Path(root) / "**"), recursive=True):
            path = Path(filename)
            if not path.is_file():
                continue
            try:
                with path.open(errors="replace") as handle:
                    for line_number, line in enumerate(handle, 1):
                        if pattern.search(line):
                            hits.append(f"{path}:{line_number}:{line.rstrip()}")
                            if len(hits) >= limit:
                                return "\n".join(hits) + "\n[truncated]"
            except (OSError, UnicodeError):
                continue
        return "\n".join(hits) or "none"
    except Exception as exc:
        return f"error: {exc}"


def shell_command(args: dict[str, Any]) -> str:
    command = args["cmd"]
    timeout = max(1, int(args.get("timeout", 30)))
    process = subprocess.Popen(
        command,
        shell=True,
        executable="/bin/bash",
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        start_new_session=True,
    )
    try:
        output, _ = process.communicate(timeout=timeout)
    except subprocess.TimeoutExpired:
        os.killpg(process.pid, signal.SIGKILL)
        output, _ = process.communicate()
        return (output.rstrip() + f"\n[timeout after {timeout}s]").lstrip()
    result = output.strip() or "(empty)"
    if process.returncode:
        result += f"\n[exit status {process.returncode}]"
    return result


def run_streaming_command(
    command: list[str], timeout: int | None = None, return_summary: bool = False
) -> str:
    """Run a command while continuously draining output and enforcing timeout."""
    try:
        process = subprocess.Popen(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
    except Exception as exc:
        return f"error: {exc}"

    assert process.stdout is not None
    selector = selectors.DefaultSelector()
    selector.register(process.stdout, selectors.EVENT_READ)
    decoder = codecs.getincrementaldecoder("utf-8")(errors="replace")
    pending = ""
    output_lines: list[str] = []
    started = time.monotonic()
    timed_out = False

    def consume(text: str, final: bool = False) -> None:
        nonlocal pending
        pending += text
        pieces = pending.splitlines(keepends=True)
        pending = ""
        if pieces and not pieces[-1].endswith(("\n", "\r")) and not final:
            pending = pieces.pop()
        for piece in pieces:
            line = piece.rstrip("\r\n")
            if any(
                marker in line
                for marker in ("Loading weights", "%|", "BertModel LOAD REPORT", "Key | Status")
            ):
                continue
            output_lines.append(line)
            display_line = line if len(line) <= 100 else f"{line[:48]} ... {line[-48:]}"
            print(f"{GRAY}  |  {display_line}{RESET}", flush=True)

    while True:
        if timeout and time.monotonic() - started > timeout:
            os.killpg(process.pid, signal.SIGKILL)
            timed_out = True
        events = selector.select(timeout=0.1)
        for key, _ in events:
            chunk = os.read(key.fileobj.fileno(), 65536)
            if chunk:
                consume(decoder.decode(chunk))
            else:
                selector.unregister(key.fileobj)
        if process.poll() is not None and not selector.get_map():
            break

    consume(decoder.decode(b"", final=True), final=True)
    if pending:
        consume("", final=True)
    if timed_out:
        output_lines.append(f"[timeout after {timeout}s]")
    elif process.returncode:
        output_lines.append(f"[exit status {process.returncode}]")

    if return_summary:
        summary = output_lines[-20:]
        return "[Command finished. Last 20 lines:]\n" + "\n".join(summary)
    return "\n".join(output_lines) or "[Command finished with no output]"


def _chop_command(*parts: str) -> str | list[str]:
    executable = shutil.which("chop")
    if not executable:
        return "error: chop is not installed or not on PATH"
    return [executable, *parts]


def code_search(args: dict[str, Any]) -> str:
    command = _chop_command(
        "semantic", "search", args["query"], "--path", args.get("path", ".")
    )
    if isinstance(command, str):
        return command
    output = run_streaming_command(command, timeout=60)
    try:
        start, end = output.find("["), output.rfind("]") + 1
        if start >= 0 and end > start:
            data = json.loads(output[start:end])
            if isinstance(data, list):
                return json.dumps(data[:5], indent=2)
    except (ValueError, json.JSONDecodeError):
        pass
    return output


def code_impact(args: dict[str, Any]) -> str:
    command = _chop_command("impact", args["symbol"], args.get("path", "."))
    if isinstance(command, str):
        return command
    return run_streaming_command(command, timeout=60)


# The web_search tool summarises scraped pages with the same model the agent
# is running on. run_turn() refreshes this context at the start of every turn
# so /model switches are honoured.
_LLM_CONTEXT: dict[str, Any] = {}


def _fetch_page(url: str, timeout: int = 15) -> str:
    request = urllib.request.Request(
        url, headers={"User-Agent": "Mozilla/5.0 (compatible; axe-vllm web_search)"}
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        raw = response.read(2_000_000)
    return raw.decode("utf-8", errors="replace")


def _llm_summarize(system: str, user: str) -> str:
    """One streaming completion on the same client/model as the main loop,
    with the same max-completion-token budget (no separate cap). The nested
    answer is streamed to the terminal as it arrives, bracketed by start and
    completion events, so the user can watch the summarisation happen."""
    client = _LLM_CONTEXT.get("client")
    model = _LLM_CONTEXT.get("model")
    max_tokens = _LLM_CONTEXT.get("max_completion_tokens") or 8192
    if client is None or not model:
        return "error: no LLM context available for summarisation"
    try:
        stream = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            temperature=0,
            max_completion_tokens=max_tokens,
            stream=True,
            stream_options={"include_usage": True},
            extra_body={"enable_thinking": False},
        )
    except Exception as exc:
        return f"error: {type(exc).__name__}: {exc}"

    parts: list[str] = []
    output_tokens: int | None = None
    started = False
    try:
        for chunk in stream:
            usage = getattr(chunk, "usage", None)
            if usage is not None and getattr(usage, "completion_tokens", None):
                output_tokens = usage.completion_tokens
            if not chunk.choices:
                continue
            delta = chunk.choices[0].delta
            if not delta.content:
                continue
            if not started:
                print(f"{GRAY}  | {delta.content}{RESET}", end="", flush=True)
                started = True
            else:
                print(delta.content, end="", flush=True)
            parts.append(delta.content)
    except Exception as exc:
        if started:
            print(flush=True)
        return f"error: {type(exc).__name__}: {exc}"
    if started:
        print(flush=True)

    text = "".join(parts).strip()
    if not text:
        return "(empty completion)"
    if output_tokens is not None:
        print(f"{DIM}  | summarisation complete: {output_tokens} output tokens{RESET}")
    else:
        print(f"{DIM}  | summarisation complete: {len(text)} chars{RESET}")
    return text


def web_search(args: dict[str, Any]) -> str:
    if DDGS is None or trafilatura is None:
        missing = [
            name
            for name, module in (("ddgs", DDGS), ("trafilatura", trafilatura))
            if module is None
        ]
        return f"error: missing optional dependencies: {', '.join(missing)}"
    query = str(args["query"]).strip()
    max_results = max(1, min(10, int(args.get("max_results", 5))))
    max_chars = max(500, int(args.get("max_chars", 6000)))
    print(f"{GRAY}  | searching DuckDuckGo for {query!r}...{RESET}")
    try:
        hits = DDGS().text(query, max_results=max_results)
    except Exception as exc:
        return f"error: ddgs search failed: {type(exc).__name__}: {exc}"
    if not hits:
        return f"no results for {query!r}"

    sections: list[str] = []
    for index, hit in enumerate(hits, 1):
        url = hit.get("href") or ""
        title = hit.get("title") or url
        snippet = (hit.get("body") or "").strip()
        body = ""
        if url:
            print(f"{GRAY}  | [{index}/{len(hits)}] fetching {url}{RESET}")
            try:
                html = _fetch_page(url)
                body = (
                    trafilatura.extract(html, url=url, include_comments=False) or ""
                ).strip()
            except Exception as exc:
                body = f"[fetch failed: {type(exc).__name__}: {exc}]"
        if len(body) > max_chars:
            body = body[:max_chars] + " ...[truncated]"
        sections.append(
            f"### [{index}] {title}\nURL: {url}\nSnippet: {snippet}\n\n"
            f"{body or '[no extractable text]'}"
        )

    digest = "\n\n".join(sections)
    print(f"{GRAY}  |  summarising {len(hits)} scraped page(s) with the model...{RESET}")
    summary = _llm_summarize(
        "You are a careful research assistant. You are given a web search query and, "
        "for each of the top results, the URL plus the main text scraped from that "
        "page. Produce a summary for the coding agent that called you:\n"
        "1. For each source, in order, give its [n] number, title, and URL, then "
        "2-4 sentences on what the page actually says and the concrete facts, "
        "numbers, or code it contains. Stay faithful to the scraped text; do not "
        "invent details that are not present.\n"
        "2. Note where sources agree or disagree.\n"
        "3. End with a short synthesis of the most useful, actionable takeaways for "
        "the original query.\n"
        "If a page could not be fetched or has no extractable text, say so in one "
        "line and move on. Be concise but complete; prefer specific facts over "
        "generic description.",
        f"Search query: {query}\n\nScraped results:\n\n{digest}",
    )
    if summary.startswith("error:"):
        return f"error: LLM summarisation failed ({summary}); raw digest follows:\n\n{digest}"
    return summary


def _trigger_doc_update(file_path: str, summary: str) -> str:
    if Path(file_path).name == Path(__file__).name:
        return ""
    builder = Path("axe_knowledge_builder.py")
    uv = shutil.which("uv")
    if not builder.is_file() or not uv:
        return ""
    print(f"{GRAY}  |  Updating knowledge base for {Path(file_path).name}...{RESET}")
    result = run_streaming_command(
        [uv, "run", str(builder), "--update", file_path, "--summary", summary],
        timeout=60,
    )
    return f"\n[Knowledge base update: {result}]"


def wrapped_write(args: dict[str, Any]) -> str:
    result = write_file(args)
    if result == "ok":
        result += _trigger_doc_update(args["path"], "Rewrote file content via write tool")
    return result


def wrapped_edit(args: dict[str, Any]) -> str:
    result = edit_file(args)
    if result == "ok":
        result += _trigger_doc_update(args["path"], "Edited file content via edit tool")
    return result


TOOLS: dict[str, tuple[str, dict[str, str], Any]] = {
    "read": (
        "Read a text file with one-based line numbers; path must name a file",
        {"path": "string", "offset": "number?", "limit": "number?"},
        read_file,
    ),
    "write": (
        "Atomically create or replace a text file",
        {"path": "string", "content": "string"},
        wrapped_write,
    ),
    "edit": (
        "Replace exact text in a file; old must be unique unless all=true",
        {"path": "string", "old": "string", "new": "string", "all": "boolean?"},
        wrapped_edit,
    ),
    "glob": (
        "Find paths by glob pattern, sorted by modification time",
        {"pat": "string", "path": "string?", "limit": "number?"},
        glob_files,
    ),
    "grep": (
        "Search text files using a regular expression",
        {"pat": "string", "path": "string?", "limit": "number?"},
        grep_files,
    ),
    "bash": (
        "Run a Bash command in the current working directory",
        {"cmd": "string", "timeout": "number?"},
        shell_command,
    ),
    "code_search": (
        "Find code by behavior or meaning using Chop semantic search",
        {"query": "string", "path": "string?"},
        code_search,
    ),
    "code_impact": (
        "Find callers and dependents of a symbol using Chop",
        {"symbol": "string", "path": "string?"},
        code_impact,
    ),
    "web_search": (
        "Search the web via DuckDuckGo (ddgs), scrape the top result pages with "
        "trafilatura, and return a model-written summary of the links and their content",
        {"query": "string", "max_results": "number?", "max_chars": "number?"},
        web_search,
    ),
}


def run_tool(name: str, args: dict[str, Any]) -> str:
    tool = TOOLS.get(name)
    if tool is None:
        return f"error: unknown tool {name!r}"
    try:
        result = tool[2](args)
        return result if isinstance(result, str) else json.dumps(result, ensure_ascii=False)
    except Exception as exc:
        return f"error: {type(exc).__name__}: {exc}"


def make_openai_tools() -> list[dict[str, Any]]:
    tools = []
    for name, (description, params, _function) in TOOLS.items():
        properties: dict[str, dict[str, str]] = {}
        required = []
        for param_name, param_type in params.items():
            optional = param_type.endswith("?")
            base_type = param_type.rstrip("?")
            json_type = "integer" if base_type == "number" else base_type
            properties[param_name] = {"type": json_type}
            if not optional:
                required.append(param_name)
        tools.append(
            {
                "type": "function",
                "function": {
                    "name": name,
                    "description": description,
                    "parameters": {
                        "type": "object",
                        "properties": properties,
                        "required": required,
                        "additionalProperties": False,
                    },
                },
            }
        )
    return tools


def count_tokens(text: str, model: str) -> int:
    try:
        encoding = tiktoken.encoding_for_model(model)
    except KeyError:
        encoding = tiktoken.get_encoding("cl100k_base")
    return len(encoding.encode(text))


def message_text(messages: list[dict[str, Any]]) -> str:
    parts = []
    for message in messages:
        content = message.get("content")
        if isinstance(content, str):
            parts.append(content)
        elif content is not None:
            parts.append(json.dumps(content, ensure_ascii=False))
    return "\n".join(parts)


def _smooth_print(text: str, delay: float) -> None:
    if delay <= 0:
        print(text, end="", flush=True)
        return
    for character in text:
        print(character, end="", flush=True)
        time.sleep(delay)


def call_api(
    client: OpenAI,
    messages: list[dict[str, Any]],
    model: str,
    max_completion_tokens: int,
    enable_thinking: bool,
    print_delay: float,
) -> dict[str, Any] | None:
    global TOTAL_INPUT_TOKENS, TOTAL_OUTPUT_TOKENS

    estimated_input = count_tokens(message_text(messages), model)
    print(f"{DIM}Estimated input tokens: {estimated_input}{RESET}")

    try:
        stream = client.chat.completions.create(
            model=model,
            messages=messages,
            tools=make_openai_tools(),
            tool_choice="auto",
            stream=True,
            stream_options={"include_usage": True},
            temperature=0,
            max_completion_tokens=max_completion_tokens,
            extra_body={"enable_thinking": enable_thinking},
        )

        print(f"\n{CYAN}assistant>{RESET} ", end="", flush=True)
        content_parts: list[str] = []
        reasoning_parts: list[str] = []
        collected_tool_calls: dict[int, dict[str, Any]] = {}
        exact_input: int | None = None
        exact_output: int | None = None
        in_reasoning = False

        for chunk in stream:
            usage = getattr(chunk, "usage", None)
            if usage is not None:
                exact_input = usage.prompt_tokens
                exact_output = usage.completion_tokens
            if not chunk.choices:
                continue
            delta = chunk.choices[0].delta
            delta_data = delta.model_dump()
            reasoning = delta_data.get("reasoning") or delta_data.get("reasoning_content")
            if reasoning:
                if not in_reasoning:
                    in_reasoning = True
                    print(f"\n{GRAY}thinking>{RESET} {GRAY}", end="", flush=True)
                _smooth_print(reasoning, print_delay)
                reasoning_parts.append(reasoning)
            if delta.content:
                if in_reasoning:
                    in_reasoning = False
                    print(f"{RESET}\n\n{CYAN}response>{RESET} ", end="", flush=True)
                _smooth_print(delta.content, print_delay)
                content_parts.append(delta.content)
            for tool_call in delta.tool_calls or []:
                index = tool_call.index
                current = collected_tool_calls.setdefault(
                    index,
                    {"id": "", "type": "function", "function": {"name": "", "arguments": ""}},
                )
                if tool_call.id:
                    current["id"] = tool_call.id
                if tool_call.function:
                    if tool_call.function.name:
                        current["function"]["name"] += tool_call.function.name
                    if tool_call.function.arguments:
                        current["function"]["arguments"] += tool_call.function.arguments

        print(RESET)
        full_content = "".join(content_parts) or None
        full_reasoning = "".join(reasoning_parts) or None
        tool_calls = [collected_tool_calls[index] for index in sorted(collected_tool_calls)]

        input_tokens = exact_input if exact_input is not None else estimated_input
        output_tokens = exact_output
        if output_tokens is None:
            output_tokens = count_tokens((full_reasoning or "") + (full_content or ""), model)
        TOTAL_INPUT_TOKENS += input_tokens
        TOTAL_OUTPUT_TOKENS += output_tokens
        print(
            f"{DIM}Tokens: {input_tokens} input, {output_tokens} output; "
            f"session {TOTAL_INPUT_TOKENS + TOTAL_OUTPUT_TOKENS}{RESET}"
        )

        assistant_message: dict[str, Any] = {
            "role": "assistant",
            "content": full_content,
        }
        if full_reasoning:
            assistant_message["reasoning_content"] = full_reasoning
        if tool_calls:
            assistant_message["tool_calls"] = tool_calls
        return assistant_message
    except Exception as exc:
        print(f"{RED}API error: {type(exc).__name__}: {exc}{RESET}")
        return None


def separator() -> str:
    width = min(shutil.get_terminal_size(fallback=(80, 24)).columns, 100)
    return f"{DIM}{'-' * width}{RESET}"


def select_model(client: OpenAI, current: str) -> str:
    models = [model.id for model in client.models.list().data]
    if current not in models:
        models.insert(0, current)
    print(f"\n{BOLD}Select model:{RESET}")
    for index, model in enumerate(models, 1):
        print(f" {index}) {model}")
    choice = input(f"\n{BOLD}Choice (default 1):{RESET} ").strip()
    if not choice:
        return models[0]
    try:
        return models[int(choice) - 1]
    except (ValueError, IndexError):
        print(f"{YELLOW}Invalid choice; keeping {current}{RESET}")
        return current


def discover_server(client: OpenAI, requested_model: str | None) -> tuple[str, int | None]:
    listing = client.models.list()
    if not listing.data:
        raise RuntimeError("the vLLM server returned no models")
    available = [model.id for model in listing.data]
    model = requested_model or available[0]
    if model not in available:
        raise RuntimeError(f"model {model!r} is not served; available: {', '.join(available)}")
    selected = next(item for item in listing.data if item.id == model)
    metadata = selected.model_dump()
    context = metadata.get("max_model_len")
    return model, int(context) if context is not None else None


def system_prompt() -> str:
    tool_list = ", ".join(TOOLS)
    return f"""You are Axe, an expert agentic coding assistant operating in the current directory.

Workflow:
1. Use grep first when you know an exact name or text fragment.
2. Use code_search for behavioral or semantic questions when Chop is available.
3. Use code_impact before changing a public symbol or call graph.
4. Read the relevant section before editing it.
5. Use edit for precise replacements and write only for new files or complete rewrites.
6. Verify every behavioral change with a focused command or scenario.
7. Use web_search for external or up-to-date information; it returns a
   model-written summary of the scraped pages, not raw HTML.

Available tools: {tool_list}
Do not invent tool results. If a tool fails, inspect the failure and choose a grounded alternative.
"""


def run_turn(
    client: OpenAI,
    messages: list[dict[str, Any]],
    model: str,
    user_input: str,
    max_completion_tokens: int,
    max_tool_rounds: int,
    enable_thinking: bool,
    print_delay: float,
) -> bool:
    _LLM_CONTEXT["client"] = client
    _LLM_CONTEXT["model"] = model
    _LLM_CONTEXT["max_completion_tokens"] = max_completion_tokens
    messages.append({"role": "user", "content": user_input})
    for _round in range(max_tool_rounds + 1):
        response = call_api(
            client,
            messages,
            model,
            max_completion_tokens,
            enable_thinking,
            print_delay,
        )
        if response is None:
            return False
        messages.append(response)
        tool_calls = response.get("tool_calls") or []
        if not tool_calls:
            return True
        if _round >= max_tool_rounds:
            print(f"{RED}Stopped after {max_tool_rounds} tool rounds{RESET}")
            return False
        for tool_call in tool_calls:
            function = tool_call.get("function") or {}
            tool_name = function.get("name", "")
            try:
                tool_args = json.loads(function.get("arguments") or "{}")
                if not isinstance(tool_args, dict):
                    raise ValueError("arguments must decode to a JSON object")
            except (json.JSONDecodeError, ValueError) as exc:
                result = f"error: invalid arguments for {tool_name!r}: {exc}"
                tool_args = {}
            else:
                preview = json.dumps(tool_args, ensure_ascii=False)[:100]
                print(f"\n{GREEN}tool>{RESET} {tool_name}({DIM}{preview}{RESET})")
                result = run_tool(tool_name, tool_args)
            result_lines = result.splitlines() or [""]
            summary = result_lines[0][:100]
            if len(result_lines) > 1:
                summary += f" ... +{len(result_lines) - 1} lines"
            print(f"{DIM}  | {summary}{RESET}")
            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": tool_call.get("id") or f"call_{_round}",
                    "content": result,
                }
            )
    return False


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", nargs="?", help="working directory")
    parser.add_argument("--base-url", default=DEFAULT_BASE_URL)
    parser.add_argument("--api-key", default=DEFAULT_API_KEY)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument(
        "--max-completion-tokens",
        type=int,
        default=DEFAULT_MAX_COMPLETION_TOKENS,
    )
    parser.add_argument("--max-tool-rounds", type=int, default=DEFAULT_MAX_TOOL_ROUNDS)
    parser.add_argument("--timeout", type=float, default=300.0)
    parser.add_argument(
        "--no-thinking",
        action="store_true",
        help="disable model thinking; thinking is streamed to the terminal by default",
    )
    parser.add_argument("--prompt", help="run one prompt non-interactively and exit")
    parser.add_argument(
        "--print-delay",
        type=float,
        default=0.002 if sys.stdout.isatty() else 0.0,
        help="delay between streamed characters",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.max_completion_tokens <= 0:
        print("error: --max-completion-tokens must be positive", file=sys.stderr)
        return 2
    if args.max_tool_rounds < 0:
        print("error: --max-tool-rounds cannot be negative", file=sys.stderr)
        return 2
    if args.directory:
        try:
            os.chdir(args.directory)
        except OSError as exc:
            print(f"error: cannot enter {args.directory}: {exc}", file=sys.stderr)
            return 2

    client = OpenAI(base_url=args.base_url.rstrip("/"), api_key=args.api_key, timeout=args.timeout)
    try:
        current_model, context_length = discover_server(client, args.model)
    except Exception as exc:
        print(
            f"{RED}Cannot connect to vLLM at {args.base_url}: "
            f"{type(exc).__name__}: {exc}{RESET}",
            file=sys.stderr,
        )
        return 1

    context_label = f" | context {context_length:,}" if context_length else ""
    thinking_label = "off" if args.no_thinking else "streamed"
    print(
        f"{BOLD}axe-vllm{RESET} | {DIM}{current_model} | {os.getcwd()}"
        f"{context_label} | max output {args.max_completion_tokens:,} | thinking {thinking_label}{RESET}\n"
    )
    messages: list[dict[str, Any]] = [{"role": "system", "content": system_prompt()}]

    if args.prompt is not None:
        ok = run_turn(
            client,
            messages,
            current_model,
            args.prompt,
            args.max_completion_tokens,
            args.max_tool_rounds,
            not args.no_thinking,
            args.print_delay,
        )
        return 0 if ok else 1

    print(f"{DIM}Commands: /model, /clear, /quit{RESET}")
    while True:
        try:
            print(separator())
            user_input = input(f"{BOLD}{BLUE}>{RESET} ").strip()
            print(separator())
        except (KeyboardInterrupt, EOFError):
            print()
            return 0
        if not user_input:
            continue
        if user_input.lower() in ("/q", "/quit", "exit", "quit"):
            return 0
        if user_input.lower() in ("/c", "/clear"):
            messages = [{"role": "system", "content": system_prompt()}]
            print(f"{GREEN}Conversation cleared{RESET}")
            continue
        if user_input.lower() == "/model":
            try:
                current_model = select_model(client, current_model)
            except Exception as exc:
                print(f"{RED}Could not list models: {exc}{RESET}")
            continue
        run_turn(
            client,
            messages,
            current_model,
            user_input,
            args.max_completion_tokens,
            args.max_tool_rounds,
            not args.no_thinking,
            args.print_delay,
        )
        print()


if __name__ == "__main__":
    raise SystemExit(main())
