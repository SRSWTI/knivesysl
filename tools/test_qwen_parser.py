#!/usr/bin/env python3
"""Qwen parsing regressions using the real tokenizer, vLLM, and live server.

Run after starting serve_prod.sh:
    .venv/bin/python tools/test_qwen_parser.py

KSL_TEST_MODEL_DIR selects tokenizer assets; KSL_TEST_URL/KSL_API_KEY select
an HTTP server. No inference engine or parser is mocked.
"""
import json
import os
import unittest

from transformers import AutoTokenizer

from qwen_parser import QwenChatParser
from test_server_reliability import call

MODEL = os.environ.get("KSL_TEST_MODEL", "knivesysl-axe-28b")
MODEL_DIR = os.environ.get(
    "KSL_TEST_MODEL_DIR", "/home/shooting-brake007/models/knivesysl")
ADD_TOOL = {"type": "function", "function": {
    "name": "add", "description": "Add two integers.", "parameters": {
        "type": "object", "properties": {
            "a": {"type": "integer"}, "b": {"type": "integer"}},
        "required": ["a", "b"]}}}
DIRECTORY_TOOL = {"type": "function", "function": {
    "name": "list_directory", "parameters": {
        "type": "object", "properties": {"path": {"type": "string"}}}}}
SCREENSHOT_CALL = (
    "<tool_call>\n<function=list_directory>\n<parameter=path>\n"
    "programming\n</parameter>\n</function>\n</tool_call>")


def combine_deltas(deltas):
    message = {"role": "assistant", "content": ""}
    calls = {}
    for delta in deltas:
        for field in ("content", "reasoning_content"):
            if delta.get(field):
                message[field] = message.get(field, "") + delta[field]
        for part in delta.get("tool_calls", []):
            target = calls.setdefault(part["index"], {
                "function": {"name": "", "arguments": ""}})
            for key in ("id", "type"):
                if key in part:
                    target[key] = part[key]
            for key, value in part.get("function", {}).items():
                target["function"][key] += value
    message["content"] = message["content"] or None
    if calls:
        assert sorted(calls) == list(range(len(calls)))
        message["tool_calls"] = [calls[i] for i in sorted(calls)]
    return message


def without_ids(message):
    result = dict(message)
    if result.get("tool_calls"):
        result["tool_calls"] = [
            {"name": tc["function"]["name"],
             "arguments": json.loads(tc["function"]["arguments"])}
            for tc in result["tool_calls"]]
    return result


class ParserTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tokenizer = AutoTokenizer.from_pretrained(
            MODEL_DIR, trust_remote_code=True)

    def parser(self, tools=None, thinking=True, tool_choice=None):
        return QwenChatParser(
            self.tokenizer, MODEL, [{"role": "user", "content": "Do the task."}],
            tools, thinking, tool_choice)

    def assert_chunk_parity(self, text, tools, thinking=True):
        expected = self.parser(tools, thinking).parse(text)
        for width in (1, 2, 7, 31, len(text)):
            with self.subTest(width=width, thinking=thinking):
                parser = self.parser(tools, thinking)
                deltas = []
                for offset in range(0, len(text), width):
                    delta = parser.feed(text[offset:offset + width])
                    if delta:
                        deltas.append(delta)
                final = parser.feed("", finished=True)
                if final:
                    deltas.append(final)
                self.assertEqual(without_ids(combine_deltas(deltas)),
                                 without_ids(expected))
        return expected

    def test_tool_call_implicitly_ends_thinking(self):
        result = self.assert_chunk_parity(
            "Let me check the directory.\n" + SCREENSHOT_CALL,
            [DIRECTORY_TOOL])
        self.assertEqual(result["reasoning_content"],
                         "Let me check the directory.\n")
        self.assertIsNone(result["content"])
        self.assertEqual(json.loads(result["tool_calls"][0]["function"]["arguments"]),
                         {"path": "programming"})

    def test_tools_with_and_without_explicit_thinking(self):
        for thinking, prefix in ((True, "Check.\n</think>"), (False, "")):
            result = self.assert_chunk_parity(
                prefix + SCREENSHOT_CALL, [DIRECTORY_TOOL], thinking)
            self.assertEqual(len(result["tool_calls"]), 1)

    def test_multiple_calls_and_duplicate_thinking_end(self):
        result = self.assert_chunk_parity(
            "Check.</think></think>" + SCREENSHOT_CALL + SCREENSHOT_CALL,
            [DIRECTORY_TOOL])
        self.assertEqual(len(result["tool_calls"]), 2)
        self.assertNotEqual(result["tool_calls"][0]["id"],
                            result["tool_calls"][1]["id"])
        self.assertNotIn("</think>", result.get("content") or "")

    def test_missing_outer_tool_wrapper(self):
        text = "<function=list_directory><parameter=path>programming</parameter></function>"
        result = self.assert_chunk_parity(text, [DIRECTORY_TOOL], False)
        self.assertEqual(len(result["tool_calls"]), 1)

    def test_schema_types_unicode_and_multiline_arguments(self):
        properties = {
            "count": {"type": "integer"},
            "enabled": {"type": "boolean"},
            "items": {"type": "array", "items": {"type": "integer"}},
            "options": {"type": "object", "properties": {"n": {"type": "integer"}}},
            "code": {"type": "string"},
        }
        tool = {"type": "function", "function": {
            "name": "submit", "parameters": {"type": "object", "properties": properties}}}
        code = '    print("caf\u00e9 \u65e5\u672c\u8a9e")\n    return "<tag>"'
        values = {"count": "42", "enabled": "true", "items": "[1, 2]",
                  "options": '{"n": "3"}', "code": code}
        text = "<tool_call><function=submit>" + "".join(
            f"<parameter={key}>\n{value}\n</parameter>"
            for key, value in values.items()) + "</function></tool_call>"
        result = self.assert_chunk_parity(text, [tool], False)
        arguments = json.loads(result["tool_calls"][0]["function"]["arguments"])
        self.assertEqual(arguments, {"count": 42, "enabled": True, "items": [1, 2],
                                     "options": {"n": 3}, "code": code})

    def test_no_tools_keeps_literal_xml_as_content(self):
        result = self.assert_chunk_parity(SCREENSHOT_CALL, None, False)
        self.assertNotIn("tool_calls", result)
        self.assertEqual(result["content"], SCREENSHOT_CALL)

    def test_tool_choice_none_suppresses_calls(self):
        result = self.parser([DIRECTORY_TOOL], False, "none").parse(SCREENSHOT_CALL)
        self.assertNotIn("tool_calls", result)
        parser = self.parser([DIRECTORY_TOOL], False, "none")
        deltas = [delta for char in SCREENSHOT_CALL if (delta := parser.feed(char))]
        final = parser.feed("", finished=True)
        if final:
            deltas.append(final)
        self.assertNotIn("tool_calls", combine_deltas(deltas))

    def test_thinking_exhaustion_stays_reasoning(self):
        result = self.assert_chunk_parity("Let me reason through this.", [DIRECTORY_TOOL])
        self.assertIsNone(result["content"])
        self.assertEqual(result["reasoning_content"], "Let me reason through this.")

    def test_partial_delimiter_flushes_without_truncating_content(self):
        result = self.assert_chunk_parity("Literal tail <tool_", [DIRECTORY_TOOL], False)
        self.assertEqual(result["content"], "Literal tail <tool_")

    def test_arguments_stream_before_call_is_closed(self):
        parser = self.parser([DIRECTORY_TOOL], False)
        prefix = "<tool_call><function=list_directory><parameter=path>programming"
        deltas = [delta for char in prefix if (delta := parser.feed(char))]
        partial = combine_deltas(deltas)["tool_calls"][0]
        self.assertEqual(partial["function"]["name"], "list_directory")
        self.assertIn("programming", partial["function"]["arguments"])
        self.assertTrue(parser.has_tool_calls)


class LiveServerTests(unittest.TestCase):
    def chat(self, body):
        status, content_type, response = call("POST", "/v1/chat/completions", {
            "model": MODEL, "temperature": 0, "max_tokens": 512, **body})
        self.assertEqual(status, 200, response)
        if not body.get("stream"):
            return response["choices"][0]["message"], response
        self.assertTrue(content_type.startswith("text/event-stream"))
        lines = response.splitlines()
        self.assertIn("data: [DONE]", lines)
        events = [json.loads(line[6:]) for line in lines
                  if line.startswith("data: ") and line != "data: [DONE]"]
        self.assertFalse(any("error" in event for event in events), events)
        deltas = [choice["delta"] for event in events for choice in event.get("choices", [])]
        return combine_deltas(deltas), events

    def test_parallel_typed_tools_stream_and_real_result_roundtrip(self):
        messages = [{"role": "user", "content":
                     "Use the add tool to calculate 17 + 25 and 31 + 11. "
                     "Call it twice in this response."}]
        for thinking in (False, True):
            body = {"messages": messages, "tools": [ADD_TOOL], "parallel_tool_calls": True,
                    "chat_template_kwargs": {"enable_thinking": thinking}}
            full, response = self.chat(body)
            streamed, events = self.chat({**body, "stream": True,
                                          "stream_options": {"include_usage": True}})
            self.assertEqual(without_ids(streamed), without_ids(full))
            calls = full["tool_calls"]
            self.assertEqual(len(calls), 2)
            self.assertEqual(response["choices"][0]["finish_reason"], "tool_calls")
            self.assertTrue(any(choice.get("finish_reason") == "tool_calls"
                                for event in events for choice in event.get("choices", [])))
            self.assertTrue(any(event.get("usage") for event in events))
            self.assertNotIn("<tool_call>", streamed.get("reasoning_content", ""))
            self.assertGreater(sum(bool(choice["delta"].get("tool_calls"))
                                   for event in events for choice in event.get("choices", [])), 2)
            results = []
            for tool_call in calls:
                self.assertEqual(tool_call["function"]["name"], "add")
                arguments = json.loads(tool_call["function"]["arguments"])
                self.assertIs(type(arguments["a"]), int)
                self.assertIs(type(arguments["b"]), int)
                results.append({"role": "tool", "tool_call_id": tool_call["id"],
                                "content": str(arguments["a"] + arguments["b"])})
            answer, _ = self.chat({**body, "messages": messages + [full] + results})
            self.assertIn("42", answer["content"])

    def test_chat_output_limits_and_stop_strings(self):
        for limit in ("max_tokens", "max_completion_tokens"):
            _, response = self.chat({
                "messages": [{"role": "user", "content": "Count upwards from one."}],
                "chat_template_kwargs": {"enable_thinking": False},
                "max_tokens": None, limit: 8, "ignore_eos": True})
            self.assertEqual(response["usage"]["completion_tokens"], 8)
        body = {"messages": [{"role": "user", "content":
                              "Output exactly: alpha STOP omega"}],
                "chat_template_kwargs": {"enable_thinking": False}, "stop": "STOP"}
        full, _ = self.chat(body)
        streamed, _ = self.chat({**body, "stream": True})
        self.assertEqual(streamed, full)
        self.assertNotIn("STOP", full["content"] or "")
        self.assertNotIn("omega", full["content"] or "")

    def test_raw_edit_completion_is_unchanged(self):
        body = {"model": MODEL, "prompt":
                "<|fim_prefix|>def add(a, b):\n    <|fim_suffix|>\n\n"
                "assert add(2, 3) == 5\n<|fim_middle|>",
                "temperature": 0, "max_tokens": 32}
        status, _, full = call("POST", "/v1/completions", body)
        self.assertEqual(status, 200, full)
        self.assertIn("return a + b", full["choices"][0]["text"])
        status, _, stream = call("POST", "/v1/completions", {**body, "stream": True})
        self.assertEqual(status, 200)
        self.assertIn("data: [DONE]", stream)
        events = [json.loads(line[6:]) for line in stream.splitlines()
                  if line.startswith("data: ") and line != "data: [DONE]"]
        self.assertEqual("".join(choice.get("text", "") for event in events
                                 for choice in event.get("choices", [])),
                         full["choices"][0]["text"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
