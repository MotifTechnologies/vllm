# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Reasoning parser for Motif models.

Motif delimits its thinking block with ``<think>``/``</think>``, so this parser
builds on the stock thinking parser and changes exactly one thing: it honours
``enable_thinking``.

``enable_thinking`` is read from ``chat_template_kwargs`` at construction, which
the OpenAI serving layer does per request, so the flag follows the request. It
defaults to ``True`` -- Motif is a reasoning model, so thinking is the normal
mode and only an explicit opt-out turns it off.

With thinking **on**, nothing is overridden: parsing is the inherited behaviour,
byte for byte.

With thinking **off**, the chat template has already opened *and closed* the
thinking block in the prompt, so the model is answering directly: any
``</think>`` in the output is answer text the model happened to type, not
structure. Nothing is reasoning, so every parsing entry point hands off to an
``IdentityReasoningParser``::

    extract_reasoning           -> (None, model_output)  markers kept verbatim
    extract_reasoning_streaming -> every delta is content
    is_reasoning_end            -> True                  never gates tool calls
    is_reasoning_end_streaming  -> True
    extract_content_ids         -> all ids

The third line is why the hand-off spans all of them rather than just
``extract_reasoning``: the inherited ``is_reasoning_end`` stays ``False`` until
it sees a ``</think>``, and tool-call parsing is gated on it, so a partial
hand-off would silently disable tool calls for the whole request.
``reasoning_start_str``/``reasoning_end_str`` are deliberately left inherited --
only ``ReasoningConfig.initialize_token_ids`` reads them, and it builds the
parser without ``chat_template_kwargs``, i.e. always in thinking mode.
"""

from collections.abc import Iterable, Sequence
from typing import TYPE_CHECKING

from vllm.entrypoints.openai.engine.protocol import DeltaMessage
from vllm.reasoning.abs_reasoning_parsers import ReasoningParserManager
from vllm.reasoning.deepseek_r1_reasoning_parser import DeepSeekR1ReasoningParser
from vllm.reasoning.identity_reasoning_parser import IdentityReasoningParser
from vllm.tokenizers import TokenizerLike

if TYPE_CHECKING:
    from vllm.entrypoints.openai.chat_completion.protocol import ChatCompletionRequest
    from vllm.entrypoints.openai.responses.protocol import ResponsesRequest


class MotifReasoningParser(DeepSeekR1ReasoningParser):
    """Thinking-aware ``<think>``/``</think>`` parser for Motif: hands off to an
    identity parser when thinking is disabled, and is the inherited parser
    otherwise."""

    def __init__(self, tokenizer: TokenizerLike, *args, **kwargs):
        super().__init__(tokenizer, *args, **kwargs)
        chat_kwargs = kwargs.get("chat_template_kwargs", {}) or {}
        self.thinking_enabled = chat_kwargs.get("enable_thinking", True)
        # Nothing to parse when thinking is off: the prompt already closed the
        # reasoning block, so markers in the output are answer text, not
        # structure. Built only on that path so the usual one stays untouched.
        self.passthrough: IdentityReasoningParser | None = (
            None
            if self.thinking_enabled
            else IdentityReasoningParser(tokenizer, *args, **kwargs)
        )

    def extract_reasoning(
        self, model_output: str, request: "ChatCompletionRequest | ResponsesRequest"
    ) -> tuple[str | None, str | None]:
        if self.passthrough:
            return self.passthrough.extract_reasoning(model_output, request)
        return super().extract_reasoning(model_output, request)

    def extract_reasoning_streaming(
        self,
        previous_text: str,
        current_text: str,
        delta_text: str,
        previous_token_ids: Sequence[int],
        current_token_ids: Sequence[int],
        delta_token_ids: Sequence[int],
    ) -> DeltaMessage | None:
        if self.passthrough:
            return self.passthrough.extract_reasoning_streaming(
                previous_text,
                current_text,
                delta_text,
                previous_token_ids,
                current_token_ids,
                delta_token_ids,
            )
        return super().extract_reasoning_streaming(
            previous_text,
            current_text,
            delta_text,
            previous_token_ids,
            current_token_ids,
            delta_token_ids,
        )

    def is_reasoning_end(self, input_ids: Sequence[int]) -> bool:
        if self.passthrough:
            return self.passthrough.is_reasoning_end(input_ids)
        return super().is_reasoning_end(input_ids)

    def is_reasoning_end_streaming(
        self, input_ids: Sequence[int], delta_ids: Iterable[int]
    ) -> bool:
        if self.passthrough:
            return self.passthrough.is_reasoning_end_streaming(input_ids, delta_ids)
        return super().is_reasoning_end_streaming(input_ids, delta_ids)

    def extract_content_ids(self, input_ids: list[int]) -> list[int]:
        if self.passthrough:
            return self.passthrough.extract_content_ids(input_ids)
        return super().extract_content_ids(input_ids)


# In-tree the name is registered lazily from `vllm/reasoning/__init__.py`; this
# eager registration additionally makes the file usable as a drop-in
# `--reasoning-parser-plugin` on a wheel built before it existed (the parser is
# pure Python, so no rebuild is required).
ReasoningParserManager.register_module("motif", module=MotifReasoningParser)
