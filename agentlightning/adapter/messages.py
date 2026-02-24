# Copyright (c) Microsoft. All rights reserved.

from __future__ import annotations

import json
from collections import defaultdict
from typing import TYPE_CHECKING, Any, Dict, Generator, Iterable, List, Optional, Sequence, TypedDict, Union, cast

from pydantic import TypeAdapter

from agentlightning.types import Span

from .base import TraceAdapter

if TYPE_CHECKING:
    from openai.types.chat import (
        ChatCompletionFunctionToolParam,
        ChatCompletionMessageFunctionToolCallParam,
        ChatCompletionMessageParam,
    )


class OpenAIMessages(TypedDict):
    """OpenAI-style chat messages with optional tool definitions.

    Attributes:
        messages: Ordered chat messages that describe the conversation.
        tools: Tool specifications available to the assistant, if any.
    """

    messages: List[ChatCompletionMessageParam]
    tools: Optional[List[ChatCompletionFunctionToolParam]]


class _RawSpanInfo(TypedDict):
    """Intermediate representation parsed from a span.

    Attributes:
        prompt: Prompt messages reconstructed from span attributes.
        completion: Assistant completions following tool invocations.
        request: Request payload recorded in the trace.
        response: Response payload recorded in the trace.
        tools: Tool call metadata extracted from child spans.
    """

    prompt: List[Dict[str, Any]]
    completion: List[Dict[str, Any]]
    request: Dict[str, Any]
    response: Dict[str, Any]
    tools: List[Dict[str, Any]]


def _json_loads_maybe(value: Any) -> Any:
    """Best-effort JSON parsing helper.

    Returns the original object for non-string values and `None` for invalid JSON strings.
    """
    if isinstance(value, str):
        try:
            return json.loads(value)
        except json.JSONDecodeError:
            return None
    return value


def _extract_text_from_parts(parts: Any) -> Optional[str]:
    """Extract concatenated text content from OpenAI-style `parts` blocks."""
    if not isinstance(parts, list):
        return None
    chunks: List[str] = []
    for part in parts:
        if not isinstance(part, dict):
            continue
        # LiteLLM can emit text blocks as {"type":"text","content":"..."} or {"text":"..."}.
        candidate = part.get("content")
        if not isinstance(candidate, str):
            candidate = part.get("text")
        if isinstance(candidate, str):
            chunks.append(candidate)
    if not chunks:
        return None
    return "".join(chunks)


def _extract_message_content(message: Dict[str, Any]) -> Optional[str]:
    """Extract a text content string from heterogeneous message payloads."""
    content = message.get("content")
    if isinstance(content, str):
        return content
    if isinstance(content, dict):
        maybe_text = content.get("text")
        if isinstance(maybe_text, str):
            return maybe_text
    if isinstance(content, list):
        extracted = _extract_text_from_parts(content)
        if extracted is not None:
            return extracted
    return _extract_text_from_parts(message.get("parts"))


def _normalize_tool_calls(raw_tool_calls: Any) -> List[Dict[str, str]]:
    """Normalize tool calls to `{id,name,arguments}` with stringified arguments."""
    if not isinstance(raw_tool_calls, list):
        return []
    normalized: List[Dict[str, str]] = []
    for idx, call in enumerate(raw_tool_calls):
        if not isinstance(call, dict):
            continue
        function = call.get("function") if isinstance(call.get("function"), dict) else {}
        name_raw = function.get("name") if isinstance(function, dict) else None
        if not isinstance(name_raw, str) or not name_raw:
            continue
        arguments = function.get("arguments", "{}")
        if isinstance(arguments, (dict, list)):
            arguments_str = json.dumps(arguments, ensure_ascii=False)
        elif arguments is None:
            arguments_str = "{}"
        else:
            arguments_str = str(arguments)
        call_id_raw = call.get("id")
        call_id = str(call_id_raw) if call_id_raw is not None else f"call_{idx}"
        normalized.append({"id": call_id, "name": name_raw, "arguments": arguments_str})
    return normalized


def _normalize_message_dict(raw_message: Any) -> Optional[Dict[str, Any]]:
    """Normalize a message object from `gen_ai.{input,output}.messages` payloads."""
    if not isinstance(raw_message, dict):
        return None
    role = raw_message.get("role")
    if not isinstance(role, str):
        return None

    message: Dict[str, Any] = {"role": role}
    content = _extract_message_content(raw_message)
    tool_calls = _normalize_tool_calls(raw_message.get("tool_calls"))

    if role == "assistant" and tool_calls:
        message["tool_calls"] = tool_calls
        message["content"] = content
    else:
        message["content"] = content if content is not None else ""

    if role == "tool":
        tool_call_id = raw_message.get("tool_call_id")
        if isinstance(tool_call_id, str) and tool_call_id:
            message["tool_call_id"] = tool_call_id

    return message


def _messages_from_blob(raw_blob: Any) -> List[Dict[str, Any]]:
    """Parse and normalize message arrays from JSON string blobs."""
    parsed = _json_loads_maybe(raw_blob)
    if not isinstance(parsed, list):
        return []
    messages: List[Dict[str, Any]] = []
    for item in parsed:
        normalized = _normalize_message_dict(item)
        if normalized is not None:
            messages.append(normalized)
    return messages


def group_genai_dict(data: Dict[str, Any], prefix: str) -> Union[Dict[str, Any], List[Any]]:
    """Convert flattened trace attributes into nested structures.

    Attributes emitted by the tracing pipeline often arrive as dotted paths (for example
    `gen_ai.prompt.0.role`). This helper groups those keys into nested dictionaries or lists so that
    downstream processing can operate on structured data.

    Args:
        data: Flat dictionary whose keys are dotted paths.
        prefix: Top-level key (for example `gen_ai.prompt`) that determines which attributes are
            grouped.

    Returns:
        A nested dictionary (no numeric index detected) or list (numeric indices detected) containing
        the grouped values.
    """
    result: Union[Dict[str, Any], List[Any]] = {}

    # Collect keys that match the prefix
    relevant = {k[len(prefix) + 1 :]: v for k, v in data.items() if k.startswith(prefix + ".")}

    # Detect if we have numeric indices (-> list) or not (-> dict)
    indexed = any(part.split(".")[0].isdigit() for part in relevant.keys())

    if indexed:
        # Group by index
        grouped: Dict[int, Dict[str, Any]] = defaultdict(dict)
        for k, v in relevant.items():
            parts = k.split(".")
            if not parts[0].isdigit():
                continue
            idx, rest = int(parts[0]), ".".join(parts[1:])
            grouped[idx][rest] = v
        # Recursively build
        result = []
        for i in sorted(grouped.keys()):
            result.append(group_genai_dict({f"{prefix}.{rest}": val for rest, val in grouped[i].items()}, prefix))
    else:
        # No indices: build dict
        nested: Dict[str, Any] = defaultdict(dict)
        for k, v in relevant.items():
            if "." in k:
                head, _tail = k.split(".", 1)
                nested[head][f"{prefix}.{k}"] = v
            else:
                result[k] = v
        # Recurse into nested dicts
        for head, subdict in nested.items():
            result[head] = group_genai_dict(subdict, prefix + "." + head)

    return result


def convert_to_openai_messages(prompt_completion_list: List[_RawSpanInfo]) -> Generator[OpenAIMessages, None, None]:
    """Convert raw trace payloads into OpenAI-style chat messages.

    The function consumes an iterable produced by
    [`TraceToMessages.adapt()`][agentlightning.TraceToMessages.adapt] and yields
    structures that match the OpenAI fine-tuning JSONL schema, including tool definitions.

    Args:
        prompt_completion_list: Raw prompt/completion/tool payloads extracted from a trace.

    Returns:
        A generator that yields [`OpenAIMessages`][agentlightning.adapter.messages.OpenAIMessages]
        entries compatible with the OpenAI Functions fine-tuning format.
    """

    # Import locally to avoid legacy OpenAI version type import errors
    from openai.types.chat import (
        ChatCompletionAssistantMessageParam,
        ChatCompletionFunctionToolParam,
        ChatCompletionMessageFunctionToolCallParam,
        ChatCompletionMessageParam,
    )

    def _build_tool_calls(calls: List[Dict[str, Any]]) -> List[ChatCompletionMessageFunctionToolCallParam]:
        built: List[ChatCompletionMessageFunctionToolCallParam] = []
        for call in calls:
            name = call.get("name")
            if not isinstance(name, str) or not name:
                continue
            call_id_raw = call.get("id")
            call_id = str(call_id_raw) if call_id_raw is not None else "call_0"
            arguments = call.get("arguments", "{}")
            if not isinstance(arguments, str):
                arguments = json.dumps(arguments, ensure_ascii=False)
            built.append(
                ChatCompletionMessageFunctionToolCallParam(
                    id=call_id,
                    type="function",
                    function={"name": name, "arguments": arguments},
                )
            )
        return built

    for pc_entry in prompt_completion_list:
        messages: List[ChatCompletionMessageParam] = []

        # Extract messages
        for msg in pc_entry["prompt"]:
            role = msg["role"]

            if role == "assistant" and "tool_calls" in msg:
                # Use the tool_calls directly
                # This branch is usually not used in the wild.
                tool_calls = _build_tool_calls(cast(List[Dict[str, Any]], msg["tool_calls"]))
                messages.append(
                    ChatCompletionAssistantMessageParam(
                        role="assistant",
                        content=msg.get("content"),
                        tool_calls=tool_calls,
                    )
                )
            else:
                # Normal user/system/tool content
                message = cast(
                    ChatCompletionMessageParam,
                    TypeAdapter(ChatCompletionMessageParam).validate_python(
                        dict(role=role, content=msg.get("content", ""), tool_call_id=msg.get("tool_call_id", None))
                    ),
                )
                messages.append(message)

        # Extract completions (assistant outputs after tool responses)
        for comp in pc_entry["completion"]:
            if comp.get("role") == "assistant":
                content = comp.get("content")
                comp_tool_calls_raw = comp.get("tool_calls")
                if isinstance(comp_tool_calls_raw, list) and comp_tool_calls_raw:
                    tool_calls = _build_tool_calls(cast(List[Dict[str, Any]], comp_tool_calls_raw))
                    messages.append(
                        ChatCompletionAssistantMessageParam(role="assistant", content=content, tool_calls=tool_calls)
                    )
                elif pc_entry["tools"]:
                    tool_calls = [
                        ChatCompletionMessageFunctionToolCallParam(
                            id=tool["call"]["id"],
                            type=tool["call"]["type"],
                            function={"name": tool["name"], "arguments": tool["parameters"]},
                        )
                        for tool in pc_entry["tools"]
                    ]
                    messages.append(
                        ChatCompletionAssistantMessageParam(role="assistant", content=content, tool_calls=tool_calls)
                    )
                else:
                    messages.append(ChatCompletionAssistantMessageParam(role="assistant", content=content))

        # Build tools definitions (if available)
        if "functions" in pc_entry["request"]:
            tools = [
                ChatCompletionFunctionToolParam(
                    type="function",
                    function={
                        "name": fn["name"],
                        "description": fn.get("description", ""),
                        "parameters": (
                            json.loads(fn["parameters"]) if isinstance(fn["parameters"], str) else fn["parameters"]
                        ),
                    },
                )
                for fn in pc_entry["request"]["functions"]
            ]
            yield OpenAIMessages(messages=messages, tools=tools)
        else:
            yield OpenAIMessages(messages=messages, tools=None)


class TraceToMessages(TraceAdapter[List[OpenAIMessages]]):
    """Convert trace spans into OpenAI-compatible conversation messages.

    The adapter reconstructs prompts, completions, tool calls, and function definitions from
    `gen_ai.*` span attributes. The resulting objects match the JSONL structure expected by the
    OpenAI fine-tuning pipeline.

    !!! warning
        The adapter assumes all spans share a common trace and that tool call spans are direct
        children of the associated completion span.
    """

    def get_tool_calls(self, completion: Span, all_spans: Sequence[Span], /) -> Iterable[Dict[str, Any]]:
        """Yield tool call payloads for a completion span.

        Args:
            completion: The completion span whose descendants should be inspected.
            all_spans: The complete span list belonging to the trace.

        Yields:
            Dictionaries describing tool calls with identifiers, names, and arguments.

        Raises:
            ValueError: If a candidate tool span cannot be converted into a dictionary.
        """
        # Get all the spans that are children of the completion span
        children = [span for span in all_spans if span.parent_id == completion.span_id]
        # Get the tool calls from the children
        for maybe_tool_call in children:
            tool_call = group_genai_dict(maybe_tool_call.attributes, "tool")
            if not isinstance(tool_call, dict):
                raise ValueError(f"Extracted tool call from trace is not a dict: {tool_call}")
            if tool_call:
                yield tool_call

    def adapt(self, source: Sequence[Span], /) -> List[OpenAIMessages]:
        """Transform trace spans into OpenAI chat payloads.

        Args:
            source: Spans containing `gen_ai.*` attributes emitted by the tracing pipeline.

        Returns:
            A list of [`OpenAIMessages`][agentlightning.adapter.messages.OpenAIMessages] entries that
            capture prompts, completions, tools, and metadata.
        """
        raw_prompt_completions: List[_RawSpanInfo] = []

        for span in source:
            attributes = {k: v for k, v in span.attributes.items()}

            # Get all related information from the trace span
            prompt = group_genai_dict(attributes, "gen_ai.prompt") or []
            completion = group_genai_dict(attributes, "gen_ai.completion") or []
            request = group_genai_dict(attributes, "gen_ai.request") or {}
            response = group_genai_dict(attributes, "gen_ai.response") or {}
            if not prompt:
                prompt = _messages_from_blob(attributes.get("gen_ai.input.messages"))
            if not completion:
                completion = _messages_from_blob(attributes.get("gen_ai.output.messages"))
            if isinstance(request, dict) and "functions" not in request:
                request_functions = group_genai_dict(attributes, "llm.request.functions")
                if isinstance(request_functions, list) and request_functions:
                    request = {**request, "functions": request_functions}
            if not isinstance(prompt, list):
                raise ValueError(f"Extracted prompt from trace is not a list: {prompt}")
            if not isinstance(completion, list):
                raise ValueError(f"Extracted completion from trace is not a list: {completion}")
            if not isinstance(request, dict):
                raise ValueError(f"Extracted request from trace is not a dict: {request}")
            if not isinstance(response, dict):
                raise ValueError(f"Extracted response from trace is not a dict: {response}")
            if prompt or completion or request or response:
                tools = list(self.get_tool_calls(span, source)) or []
                raw_prompt_completions.append(
                    _RawSpanInfo(
                        prompt=prompt or [], completion=completion, request=request, response=response, tools=tools
                    )
                )

        return list(convert_to_openai_messages(raw_prompt_completions))
