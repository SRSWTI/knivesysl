"""OpenAI wire adapter for the installed vLLM unified Qwen3 parser."""
from vllm.entrypoints.openai.chat_completion.protocol import ChatCompletionRequest
from vllm.parser.qwen3 import Qwen3Parser


class _SchemaAwareQwen3Parser(Qwen3Parser):
    """Retain vLLM parsing, correcting its 0.28 string-to-container early return."""

    @classmethod
    def _coerce_nested(cls, value, schema):
        changed = False
        if not isinstance(value, (dict, list)):
            value, changed = Qwen3Parser._coerce_value(value, schema)
        if isinstance(value, dict) and isinstance(schema.get("properties"), dict):
            value, nested_changed = cls._coerce_dict(value, schema["properties"])
            changed |= nested_changed
        elif isinstance(value, list) and isinstance(schema.get("items"), dict):
            for index, item in enumerate(value):
                value[index], item_changed = cls._coerce_nested(item, schema["items"])
                changed |= item_changed
        return value, changed

    @classmethod
    def _coerce_dict(cls, args, properties):
        changed = False
        for key, value in args.items():
            schema = properties.get(key)
            if isinstance(schema, dict):
                args[key], item_changed = cls._coerce_nested(value, schema)
                changed |= item_changed
        return args, changed


class QwenChatParser:
    """One request owns one parser; reasoning and tools share its state machine."""

    def __init__(self, tokenizer, model, messages, tools, thinking, tool_choice=None):
        self.request = ChatCompletionRequest(
            model=model,
            messages=messages,
            tools=tools or None,
            tool_choice=tool_choice if tool_choice is not None else (
                "auto" if tools else "none"),
            chat_template_kwargs={"enable_thinking": thinking},
        )
        self.parser = _SchemaAwareQwen3Parser(
            tokenizer,
            tools=self.request.tools,
            chat_template_kwargs=self.request.chat_template_kwargs,
        )
        self.parser.skip_tool_parsing = not bool(tools)
        self.has_tool_calls = False

    def feed(self, text, *, finished=False):
        # Text is already detokenized by the server. Do not pass token IDs from
        # a different boundary: stop holdback may split a token's decoded text.
        delta = self.parser.parse_delta(text, [], self.request, finished=finished)
        if delta is None:
            return None
        result = delta.model_dump(exclude_none=True)
        # vLLM 0.28 calls the internal field `reasoning`; Zed and our public
        # chat API consume the OpenAI-compatible `reasoning_content` field.
        if "reasoning" in result:
            result["reasoning_content"] = result.pop("reasoning")
        if result.get("tool_calls"):
            self.has_tool_calls = True
        return result

    def parse(self, text):
        reasoning, content, calls = self.parser.parse(text, self.request)
        result = {"role": "assistant", "content": content}
        if reasoning:
            result["reasoning_content"] = reasoning
        if calls:
            self.has_tool_calls = True
            result["tool_calls"] = [
                {"id": call.id, "type": "function", "function": {
                    "name": call.name, "arguments": call.arguments}}
                for call in calls
            ]
        return result
