# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for the Motif lenient Hermes tool parser.

Malformation cases mirror the measured corpus categories from
steel-browsecomp/analysis/motif_toolcall_parser_improvement.md plus the
GDPval repair rules. A dummy tokenizer is used because the Hermes
non-streaming path never touches the tokenizer.
"""

import json

import pytest

from tests.tool_parsers.utils import (
    run_tool_extraction,
    run_tool_extraction_streaming,
)
from vllm.entrypoints.openai.chat_completion.protocol import (
    ChatCompletionRequest,
)
from vllm.tool_parsers import ToolParserManager
from vllm.tool_parsers.motif_tool_parser import (
    _REPAIR_CACHE_MAX,
    MotifToolParser,
    _repair_block,
    sanitize_model_output,
)


class _DummyTokenizer:
    def get_vocab(self) -> dict[str, int]:
        return {}

    def tokenize(self, text: str) -> list[str]:
        return []


@pytest.fixture(scope="module")
def motif_parser() -> MotifToolParser:
    return MotifToolParser(_DummyTokenizer())


# (malformed block, expected parsed dict) — one per corpus category.
REPAIR_CASES = [
    pytest.param(
        '{"name": "search", "arguments": {"queries": ["a", "b"}}',
        {"name": "search", "arguments": {"queries": ["a", "b"]}},
        id="missing-close-bracket",
    ),
    pytest.param(
        '{"name": "search", "arguments": {"queries": "q one", "q two"}}',
        {"name": "search", "arguments": {"queries": ["q one", "q two"]}},
        id="missing-open-bracket",
    ),
    pytest.param(
        '{"name": "search", "arguments": {"queries": "a", "b"]}}',
        {"name": "search", "arguments": {"queries": ["a", "b"]}},
        id="missing-open-bracket-with-trailing-close",
    ),
    pytest.param(
        '{"name": "search", "arguments": {"queries": ["a"]]}}',
        {"name": "search", "arguments": {"queries": ["a"]}},
        id="duplicated-close-bracket",
    ),
    pytest.param(
        '{"name": "search", "arguments": {"queries": ["the "best" one"]}}',
        {"name": "search", "arguments": {"queries": ['the "best" one']}},
        id="unescaped-inner-quotes",
    ),
    pytest.param(
        '{"name": "run_code", "arguments": {"cmd": "grep -c \\$HOME f"}}',
        {"name": "run_code", "arguments": {"cmd": "grep -c $HOME f"}},
        id="invalid-json-escape",
    ),
    # Regex-heavy cmd mixing invalid escapes (\s) with valid escaped
    # backslashes (\\[): the lone-backslash drop must consume escapes left to
    # right so the \\ pair survives (observed live on the 20260708 snapshots).
    pytest.param(
        '{"name": "run_code", "arguments":'
        ' {"cmd": "m=re.search(r\'\\"x\\"\\s*:\\s*(\\\\[.*?\\\\])\', h)"}}',
        {"name": "run_code", "arguments": {"cmd": 'm=re.search(r\'"x"s*:s*(\\[.*?\\])\', h)'}},
        id="mixed-invalid-escape-and-escaped-backslash",
    ),
    pytest.param(
        '{"name": "fetch", "arguments": {"urls": ["http://x"]}}}}',
        {"name": "fetch", "arguments": {"urls": ["http://x"]}},
        id="extra-trailing-braces",
    ),
    pytest.param(
        '{"name": "fetch", "urls": ["http://x"]}',
        {"name": "fetch", "arguments": {"urls": ["http://x"]}},
        id="flat-arguments-wrapper",
    ),
    pytest.param(
        '{"name": "note", "arguments": {"text": "line1\nline2"}}',
        {"name": "note", "arguments": {"text": "line1\nline2"}},
        id="raw-control-char-in-string",
    ),
    # R-backtrack: JSON-lookalike content inside a string. The local
    # close-on-structural heuristic mis-closes at '"@type":' (quote followed
    # by ':'), so only the backtracking search over close-vs-content
    # interpretations recovers it (observed live on the 20260708 snapshots).
    pytest.param(
        '{"name": "run_code", "arguments": {"cmd": "echo "@type":"x", done"}}',
        {"name": "run_code", "arguments": {"cmd": 'echo "@type":"x", done'}},
        id="json-lookalike-content-in-string",
    ),
]


@pytest.mark.parametrize("block,expected", REPAIR_CASES)
def test_repair_block(block: str, expected: dict):
    repaired = _repair_block(block)
    assert repaired is not None
    assert json.loads(repaired) == expected


def test_repair_block_valid_json_unchanged():
    block = '{"name": "search", "arguments": {"queries": ["안녕", "b"]}}'
    repaired = _repair_block(block)
    assert repaired is not None
    assert json.loads(repaired) == json.loads(block)


def test_repair_block_unrecoverable_returns_none():
    assert _repair_block('{"queries": totally broken') is None
    assert _repair_block("") is None


def test_backtracking_schema_oracle():
    """Backtracking candidates are gated by the request's tool schemas."""
    block = '{"name": "run_code", "arguments": {"cmd": "echo "@type":"x", done"}}'
    run_code = {
        "type": "function",
        "function": {
            "name": "run_code",
            "parameters": {
                "properties": {"cmd": {"type": "string"}},
                "additionalProperties": False,
            },
        },
    }
    fixed = _repair_block(block, [run_code])
    assert fixed is not None
    assert json.loads(fixed) == {
        "name": "run_code",
        "arguments": {"cmd": 'echo "@type":"x", done'},
    }
    # A registered schema set that does not contain the call's name rejects
    # every backtracking candidate -> unrecoverable (stock fallback).
    search_only = {
        "type": "function",
        "function": {
            "name": "search",
            "parameters": {
                "properties": {"queries": {"type": "array"}},
                "additionalProperties": False,
            },
        },
    }
    assert _repair_block(block, [search_only]) is None


def test_sanitize_no_tool_call_passthrough():
    text = "just a plain answer with no tools"
    assert sanitize_model_output(text) is text


def test_sanitize_keeps_unrecoverable_block_verbatim():
    text = '<tool_call>{"queries": totally broken</tool_call>'
    assert sanitize_model_output(text) == text


def test_sanitize_repairs_unclosed_trailing_block():
    text = 'intro <tool_call>{"name": "search", "arguments": {"queries": ["a"}}'
    sanitized = sanitize_model_output(text)
    assert sanitized.startswith("intro <tool_call>")
    assert sanitized.endswith("</tool_call>")
    inner = sanitized.split("<tool_call>")[1].split("</tool_call>")[0]
    assert json.loads(inner) == {
        "name": "search",
        "arguments": {"queries": ["a"]},
    }


def test_extract_tool_calls_repairs_malformed_blocks(motif_parser):
    output = (
        "Let me search.\n"
        '<tool_call>{"name": "search", "arguments": {"queries": ["a", "b"}}'
        "</tool_call>\n"
        '<tool_call>{"name": "fetch", "arguments": {"urls": ["http://x"]}}'
        "</tool_call>"
    )
    content, tool_calls = run_tool_extraction(motif_parser, output)
    assert content == "Let me search.\n"
    assert [call.function.name for call in tool_calls] == ["search", "fetch"]
    assert json.loads(tool_calls[0].function.arguments) == {"queries": ["a", "b"]}
    assert json.loads(tool_calls[1].function.arguments) == {"urls": ["http://x"]}


def test_extract_tool_calls_no_tools(motif_parser):
    content, tool_calls = run_tool_extraction(motif_parser, "plain answer")
    assert content == "plain answer"
    assert tool_calls == []


def test_extract_tool_calls_unrecoverable_falls_back_to_stock(motif_parser):
    output = '<tool_call>{"queries": totally broken</tool_call>'
    content, tool_calls = run_tool_extraction(motif_parser, output)
    assert tool_calls == []
    assert content == output


def test_parser_registered_under_both_names():
    for name in ("motif", "motif_hermes"):
        assert ToolParserManager.get_tool_parser(name) is MotifToolParser


# --- Streaming --------------------------------------------------------------
# Streaming state lives on the parser instance, so each test builds a fresh
# parser. Deltas are passed as pre-split chunks; the Hermes streaming path
# never uses token ids, so the dummy tokenizer suffices.


def _chunks(text: str, size: int) -> list[str]:
    return [text[i : i + size] for i in range(0, len(text), size)]


def _run_streaming(deltas: list[str]):
    parser = MotifToolParser(_DummyTokenizer())
    return run_tool_extraction_streaming(
        parser, deltas, assert_one_tool_per_delta=False
    )


@pytest.mark.parametrize("chunk_size", [1, 7, 1000])
@pytest.mark.parametrize("block,expected", REPAIR_CASES)
def test_streaming_repairs_malformed_block(
    block: str, expected: dict, chunk_size: int
):
    text = f"<tool_call>{block}</tool_call>"
    result = _run_streaming(_chunks(text, chunk_size))
    assert len(result.tool_calls) == 1
    call = result.tool_calls[0].function
    assert call.name == expected["name"]
    assert json.loads(call.arguments) == expected["arguments"]


def test_streaming_valid_block_unchanged():
    text = (
        '<tool_call>{"name": "search", "arguments": {"queries": ["안녕", "b"]}}'
        "</tool_call>"
    )
    result = _run_streaming(_chunks(text, 1))
    assert len(result.tool_calls) == 1
    assert result.tool_calls[0].function.name == "search"
    assert json.loads(result.tool_calls[0].function.arguments) == {
        "queries": ["안녕", "b"]
    }


def test_streaming_content_before_tool_call():
    text = (
        "Let me search.\n"
        '<tool_call>{"name": "search", "arguments": {"queries": ["a", "b"}}'
        "</tool_call>"
    )
    result = _run_streaming(_chunks(text, 3))
    assert result.other_content == "Let me search.\n"
    assert len(result.tool_calls) == 1
    assert json.loads(result.tool_calls[0].function.arguments) == {
        "queries": ["a", "b"]
    }


def test_streaming_multiple_blocks_one_malformed():
    text = (
        '<tool_call>{"name": "search", "arguments": {"queries": ["a", "b"}}'
        "</tool_call>\n"
        '<tool_call>{"name": "fetch", "arguments": {"urls": ["http://x"]}}'
        "</tool_call>"
    )
    result = _run_streaming(_chunks(text, 5))
    assert [call.function.name for call in result.tool_calls] == ["search", "fetch"]
    assert json.loads(result.tool_calls[0].function.arguments) == {
        "queries": ["a", "b"]
    }
    assert json.loads(result.tool_calls[1].function.arguments) == {
        "urls": ["http://x"]
    }


def test_streaming_valid_body_without_end_tag():
    # A body that is already valid JSON counts as complete even before the
    # </tool_call> tokens arrive (stock Hermes semantics).
    text = '<tool_call>{"name": "search", "arguments": {"queries": ["a"]}}'
    result = _run_streaming(_chunks(text, 4))
    assert len(result.tool_calls) == 1
    assert json.loads(result.tool_calls[0].function.arguments) == {"queries": ["a"]}


def test_streaming_unrecoverable_block_emits_no_tool():
    text = '<tool_call>{"queries": totally broken</tool_call>'
    result = _run_streaming(_chunks(text, 6))
    assert result.tool_calls == []
    assert not result.other_content


def test_streaming_holds_back_args_until_block_completes():
    parser = MotifToolParser(_DummyTokenizer())
    request = ChatCompletionRequest(messages=[], model="test-model")
    prefix = '<tool_call>{"name": "search", "arguments": {"queries": ["a", "b"'

    previous = ""
    for delta in _chunks(prefix, 4):
        current = previous + delta
        msg = parser.extract_tool_calls_streaming(
            previous, current, delta, [], [], [], request
        )
        if msg is not None:
            for tool_call in msg.tool_calls:
                assert not tool_call.function.arguments
        previous = current

    # The name streams early, before the block is complete.
    assert parser.prev_tool_call_arr[0]["name"] == "search"

    closing = "}}</tool_call>"
    msg = parser.extract_tool_calls_streaming(
        previous, previous + closing, closing, [], [], [], request
    )
    assert msg is not None
    args = "".join(
        tool_call.function.arguments or "" for tool_call in msg.tool_calls
    )
    assert json.loads(args) == {"queries": ["a", "b"]}


def test_streaming_repair_cache_is_bounded():
    # A bare-number body is valid JSON at every prefix, so every delta
    # attempts a repair under a new (growing) cache key; the cap must keep
    # the cache from growing with output length.
    parser = MotifToolParser(_DummyTokenizer())
    request = ChatCompletionRequest(messages=[], model="test-model")
    previous = "<tool_call>1"
    for _ in range(2 * _REPAIR_CACHE_MAX):
        current = previous + "1"
        parser.extract_tool_calls_streaming(
            previous, current, "1", [], [], [], request
        )
        previous = current
    assert len(parser._repair_cache) <= _REPAIR_CACHE_MAX
