# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""MOTIF: tests for the think-budget vllm_xargs surface after the split of
ThinkingTokenBudgetLogitsProcessor into think_budget.py.

Priority chain under test: vllm_xargs > env var > default (off).
"""

from types import SimpleNamespace

import pytest
import torch

from vllm.sampling_params import SamplingParams
from vllm.v1.sample.logits_processor.interface import BatchUpdate
from vllm.v1.sample.logits_processor.think_budget import (
    ThinkingTokenBudgetLogitsProcessor,
    resolve_force_str,
    resolve_think_budget_ratio,
    validate_think_budget_xargs,
)

THINK_START, THINK_END = 100, 101
MAX_MODEL_LEN = 20000


class _FakeTokenizer:
    eos_token_id = 0

    def encode(self, text, add_special_tokens=False):
        return [150, 151, THINK_END]


@pytest.fixture
def budget_env(monkeypatch):
    monkeypatch.delenv("VLLM_THINK_BUDGET_RATIO", raising=False)
    monkeypatch.delenv("VLLM_THINK_BUDGET_FORCE_STR", raising=False)
    monkeypatch.delenv("MOTIF_THINK_BUDGET_FORCE_STR", raising=False)
    monkeypatch.setattr(
        "vllm.tokenizers.cached_tokenizer_from_config",
        lambda model_config: _FakeTokenizer(),
    )


def _make_ttblp(reasoning=True, spec_decode=False):
    reasoning_config = (
        SimpleNamespace(
            enabled=True,
            reasoning_start_token_ids=[THINK_START],
            reasoning_end_token_ids=[THINK_END],
        )
        if reasoning
        else None
    )
    vllm_config = SimpleNamespace(
        reasoning_config=reasoning_config,
        scheduler_config=SimpleNamespace(max_num_seqs=8),
        model_config=SimpleNamespace(max_model_len=MAX_MODEL_LEN),
        speculative_config=object() if spec_decode else None,
    )
    return ThinkingTokenBudgetLogitsProcessor(vllm_config, torch.device("cpu"), False)


def _add(ttblp, index, params, prompt):
    out: list[int] = []
    ttblp.update_state(
        BatchUpdate(
            batch_size=index + 1,
            removed=[],
            added=[(index, params, prompt, out)],
            moved=[],
        )
    )
    return out


# ---------------------------------------------------------------------------
# Resolvers (priority: vllm_xargs > env > off)
# ---------------------------------------------------------------------------


def test_ratio_priority(budget_env, monkeypatch):
    assert resolve_think_budget_ratio(SamplingParams()) is None
    monkeypatch.setenv("VLLM_THINK_BUDGET_RATIO", "0.9")
    assert resolve_think_budget_ratio(SamplingParams()) == 0.9
    params = SamplingParams(extra_args={"think_budget_ratio": 0.5})
    assert resolve_think_budget_ratio(params) == 0.5  # xargs beats env


def test_ratio_validation(budget_env):
    with pytest.raises(ValueError):
        resolve_think_budget_ratio(
            SamplingParams(extra_args={"think_budget_ratio": "abc"})
        )
    with pytest.raises(ValueError):
        resolve_think_budget_ratio(
            SamplingParams(extra_args={"think_budget_ratio": 1.5})
        )


def test_force_str_priority(budget_env, monkeypatch):
    assert resolve_force_str(SamplingParams()) is None
    monkeypatch.setenv("VLLM_THINK_BUDGET_FORCE_STR", "env...</think>")
    assert resolve_force_str(SamplingParams()) == "env...</think>"
    params = SamplingParams(extra_args={"think_budget_force_str": "x...</think>"})
    assert resolve_force_str(params) == "x...</think>"  # xargs beats env


def test_legacy_motif_force_str_env(budget_env, monkeypatch):
    monkeypatch.setenv("MOTIF_THINK_BUDGET_FORCE_STR", "legacy...</think>")
    assert resolve_force_str(SamplingParams()) == "legacy...</think>"
    monkeypatch.setenv("VLLM_THINK_BUDGET_FORCE_STR", "new...</think>")
    assert resolve_force_str(SamplingParams()) == "new...</think>"  # new wins


def test_validate_xargs_requires_reasoning(budget_env):
    params = SamplingParams(extra_args={"think_budget_ratio": 0.5})
    validate_think_budget_xargs(params, reasoning_enabled=True)
    with pytest.raises(ValueError):
        validate_think_budget_xargs(params, reasoning_enabled=False)
    # No xargs -> nothing to validate even without reasoning.
    validate_think_budget_xargs(SamplingParams(), reasoning_enabled=False)


# ---------------------------------------------------------------------------
# Processor state (per-request budget and force sequence)
# ---------------------------------------------------------------------------


def test_xargs_ratio_creates_budget_state(budget_env):
    # max_tokens=None -> the ratio applies to the model's post-prompt space
    # (legacy behavior; serving fills max_tokens with the remaining context).
    ttblp = _make_ttblp()
    prompt = [1] * 10
    params = SamplingParams(max_tokens=None, extra_args={"think_budget_ratio": 0.5})
    _add(ttblp, 0, params, prompt)
    state = ttblp._state[0]
    avail = MAX_MODEL_LEN - len(prompt)
    expected = avail - max(4096, int(avail * 0.5))
    assert state["thinking_token_budget"] == expected

    # No ratio anywhere -> no budget state.
    _add(ttblp, 1, SamplingParams(), prompt)
    assert 1 not in ttblp._state


def test_ratio_applies_to_completion_budget(budget_env):
    # The headline semantics: "think may use 60% of my 1000-token response"
    # is ratio 0.6 verbatim -- no back-solving against the 20K context.
    ttblp = _make_ttblp()
    prompt = [1] * 10
    params = SamplingParams(max_tokens=1000, extra_args={"think_budget_ratio": 0.6})
    _add(ttblp, 0, params, prompt)
    # avail=1000, reserve_floor=min(4096, 250)=250,
    # reserve=max(250, 400)=400 -> budget 600.
    assert ttblp._state[0]["thinking_token_budget"] == 600

    # High ratio: the avail/4 floor still guarantees answer room.
    params = SamplingParams(max_tokens=1000, extra_args={"think_budget_ratio": 0.9})
    _add(ttblp, 1, params, prompt)
    assert ttblp._state[1]["thinking_token_budget"] == 750  # 1000 - 250

    # max_tokens beyond the remaining context clamps to the context.
    params = SamplingParams(
        max_tokens=MAX_MODEL_LEN * 2, extra_args={"think_budget_ratio": 0.5}
    )
    _add(ttblp, 2, params, prompt)
    avail = MAX_MODEL_LEN - len(prompt)
    assert ttblp._state[2]["thinking_token_budget"] == avail - max(
        4096, int(avail * 0.5)
    )


def test_xargs_ratio_beats_env_ratio(budget_env, monkeypatch):
    monkeypatch.setenv("VLLM_THINK_BUDGET_RATIO", "0.9")
    ttblp = _make_ttblp()
    prompt = [1] * 10
    _add(
        ttblp,
        0,
        SamplingParams(max_tokens=None, extra_args={"think_budget_ratio": 0.5}),
        prompt,
    )
    avail = MAX_MODEL_LEN - len(prompt)
    assert ttblp._state[0]["thinking_token_budget"] == avail - max(
        4096, int(avail * 0.5)
    )
    # env fallback when xargs silent
    _add(ttblp, 1, SamplingParams(max_tokens=None), prompt)
    assert ttblp._state[1]["thinking_token_budget"] == avail - max(
        4096, int(avail * 0.1)
    )


def test_max_completion_tokens_normalization():
    """The budget ratio consumes SamplingParams.max_tokens, which serving
    fills from the request's completion cap. Pin the protocol-layer
    normalization: max_completion_tokens (the newer OpenAI spelling) and
    the deprecated max_tokens both feed that cap, with the newer spelling
    winning when both are present."""
    from vllm.entrypoints.openai.chat_completion.protocol import (
        ChatCompletionRequest,
    )

    model_config = SimpleNamespace(max_model_len=MAX_MODEL_LEN)
    msgs = [{"role": "user", "content": "hi"}]

    tok = ChatCompletionRequest(
        model="m", messages=msgs, max_completion_tokens=1000
    ).build_tok_params(model_config)
    assert tok.max_output_tokens == 1000
    assert tok.max_output_tokens_param == "max_completion_tokens"

    tok = ChatCompletionRequest(
        model="m", messages=msgs, max_tokens=777
    ).build_tok_params(model_config)
    assert tok.max_output_tokens == 777
    assert tok.max_output_tokens_param == "max_tokens"

    # Both present: the newer spelling wins.
    tok = ChatCompletionRequest(
        model="m", messages=msgs, max_tokens=777, max_completion_tokens=1000
    ).build_tok_params(model_config)
    assert tok.max_output_tokens == 1000


def test_typed_budget_bypasses_ratio(budget_env):
    ttblp = _make_ttblp()
    params = SamplingParams(
        thinking_token_budget=777, extra_args={"think_budget_ratio": 0.5}
    )
    _add(ttblp, 0, params, [1] * 10)
    assert ttblp._state[0]["thinking_token_budget"] == 777


def test_per_request_force_str(budget_env, monkeypatch):
    monkeypatch.setenv("VLLM_THINK_BUDGET_RATIO", "0.5")
    ttblp = _make_ttblp()
    # Request with xargs force_str -> encoded per request.
    params = SamplingParams(extra_args={"think_budget_force_str": "...</think>"})
    _add(ttblp, 0, params, [1] * 10)
    assert ttblp._state[0]["force_end_ids"] == [150, 151, THINK_END]
    # Request without -> server default (bare end ids, no env force str).
    _add(ttblp, 1, SamplingParams(), [1] * 10)
    assert ttblp._state[1]["force_end_ids"] == [THINK_END]
