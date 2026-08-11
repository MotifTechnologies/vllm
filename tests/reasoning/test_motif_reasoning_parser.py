# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest

from tests.reasoning.utils import run_reasoning_extraction
from vllm.reasoning import ReasoningParser, ReasoningParserManager

parser_name = "motif"
# The parser this one builds on; with thinking on the two must agree exactly.
base_parser_name = "deepseek_r1"
start_token = "<think>"
end_token = "</think>"
start_token_id = 1
end_token_id = 2


class StubTokenizer:
    """Minimal tokenizer stand-in.

    The parser only reads the vocab, to resolve the think token ids, so no real
    tokenizer download is needed.
    """

    def get_vocab(self) -> dict[str, int]:
        return {start_token: start_token_id, end_token: end_token_id}


@pytest.fixture(scope="module")
def motif_tokenizer():
    return StubTokenizer()


def build_parser(tokenizer, **chat_template_kwargs) -> ReasoningParser:
    """Construct the parser the way the server does, via the manager."""
    parser_cls = ReasoningParserManager.get_reasoning_parser(parser_name)
    if not chat_template_kwargs:
        return parser_cls(tokenizer)
    return parser_cls(tokenizer, chat_template_kwargs=chat_template_kwargs)


# Shapes the parser has to survive: with and without the start token, empty
# reasoning, empty content, and stray end markers sitting in the answer.
OUTPUTS = [
    "This is a reasoning section</think>This is the rest",
    "This is a reasoning section</think>",
    "This is a reasoning section",
    "<think>This is a reasoning section</think>This is the rest",
    "</think>This is the rest",
    "</think>",
    "<think>reasoning</think>answer head </think> answer tail",
    "<think>reasoning</think>the answer</think>",
]


# --- enable_thinking on (the default): nothing may change. ---


@pytest.mark.parametrize("output", OUTPUTS)
@pytest.mark.parametrize(
    "chat_template_kwargs",
    [{}, {"enable_thinking": True}],
    ids=["default", "explicit_enable_thinking"],
)
def test_thinking_enabled_matches_base_parser(
    output, chat_template_kwargs, motif_tokenizer
):
    """Thinking is on unless explicitly disabled, and on that path the parser
    must produce exactly what the parser it builds on produces."""
    motif = build_parser(motif_tokenizer, **chat_template_kwargs)
    base = ReasoningParserManager.get_reasoning_parser(base_parser_name)(
        motif_tokenizer
    )

    assert run_reasoning_extraction(motif, [output]) == (
        run_reasoning_extraction(base, [output])
    )


def test_thinking_enabled_splits_reasoning_from_content(motif_tokenizer):
    """Pins the happy path so a parity-only suite cannot pass while both
    parsers are broken the same way."""
    parser = build_parser(motif_tokenizer)

    assert run_reasoning_extraction(parser, ["<think>why</think>answer"]) == (
        "why",
        "answer",
    )


def test_thinking_enabled_still_gates_on_end_token(motif_tokenizer):
    """Guards that the flag actually switches behaviour rather than being inert."""
    parser = build_parser(motif_tokenizer, enable_thinking=True)

    assert parser.is_reasoning_end([start_token_id, 42]) is False
    assert parser.is_reasoning_end([start_token_id, 42, end_token_id]) is True


# --- enable_thinking=False: the parser must get out of the way entirely. ---


@pytest.mark.parametrize("output", OUTPUTS)
def test_thinking_disabled_treats_everything_as_content(output, motif_tokenizer):
    """The prompt already closed the block, so no output is reasoning and the
    markers stay verbatim -- they are answer text, not structure."""
    parser = build_parser(motif_tokenizer, enable_thinking=False)

    reasoning, content = run_reasoning_extraction(parser, [output])

    assert reasoning is None
    assert content == output


def test_thinking_disabled_does_not_gate_tool_parsing(motif_tokenizer):
    """Tool-call parsing is gated on `is_reasoning_end`; with thinking off no
    `</think>` is coming, so it must never report the block as still open."""
    parser = build_parser(motif_tokenizer, enable_thinking=False)

    assert parser.is_reasoning_end([]) is True
    assert parser.is_reasoning_end([start_token_id, 42]) is True
    assert parser.is_reasoning_end_streaming([], []) is True
    assert parser.extract_content_ids([start_token_id, 42, end_token_id, 43]) == [
        start_token_id,
        42,
        end_token_id,
        43,
    ]


def test_thinking_disabled_streams_content_only(motif_tokenizer):
    parser = build_parser(motif_tokenizer, enable_thinking=False)

    delta = parser.extract_reasoning_streaming(
        previous_text="",
        current_text="answer",
        delta_text="answer",
        previous_token_ids=[],
        current_token_ids=[42],
        delta_token_ids=[42],
    )

    assert delta is not None
    assert delta.content == "answer"
    assert delta.reasoning is None
