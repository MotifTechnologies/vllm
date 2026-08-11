# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""MOTIF: thinking-token-budget logits processor (split out of builtin.py).

Limits the number of tokens inside a 'thinking' section, forcing the
reasoning end sequence when the budget is reached. Per-request configuration
resolves with the MOTIF priority chain (vllm_xargs > env var > default):

* budget: ``SamplingParams.thinking_token_budget`` (absolute, most specific)
  > ``vllm_xargs.think_budget_ratio`` > ``VLLM_THINK_BUDGET_RATIO`` env
  > off (no budget by default). The ratio applies to the request's actual
  completion budget ``min(max_tokens, max_model_len - prompt_len)``, so
  "think may use 60% of my response" is ``0.6`` verbatim -- no
  back-solving against the model context. Without ``max_tokens`` this
  degrades to the post-prompt space of the model context.
* force sequence: ``vllm_xargs.think_budget_force_str``
  > ``VLLM_THINK_BUDGET_FORCE_STR`` env > bare reasoning end token ids.
  The string must END with the reasoning end text (e.g. ``</think>``).
"""

import os
from typing import TYPE_CHECKING, Any

import numpy as np
import torch

from vllm import SamplingParams
from vllm.logger import init_logger
from vllm.v1.sample.logits_processor.interface import (
    BatchUpdate,
    LogitsProcessor,
    MoveDirectionality,
)

if TYPE_CHECKING:
    from vllm.config import ModelConfig, VllmConfig

logger = init_logger(__name__)

# vllm_xargs keys (test/experimentation surface).
_XARGS_BUDGET_RATIO = "think_budget_ratio"
_XARGS_FORCE_STR = "think_budget_force_str"

# Env vars are VLLM_-prefixed (upstream-PR friendly); the fork-internal
# MOTIF_-prefixed spellings that predate the unification keep working with
# a one-time deprecation warning.
_warned_legacy_env: set[str] = set()


def env_with_legacy(name: str) -> str:
    """Read env var ``name`` (VLLM_-prefixed), falling back to the legacy
    MOTIF_-prefixed spelling."""
    value = os.environ.get(name, "")
    if value:
        return value
    legacy = "MOTIF_" + name.removeprefix("VLLM_")
    value = os.environ.get(legacy, "")
    if value and legacy not in _warned_legacy_env:
        _warned_legacy_env.add(legacy)
        logger.debug("%s is deprecated; use %s (same semantics).", legacy, name)
    return value


_FORCE_STR_ENCODE_CACHE: dict[str, list[int]] = {}


def encode_force_str(force_str: str, model_config: "ModelConfig") -> list[int]:
    """Tokenize a force string, cached (single tokenizer per server)."""
    ids = _FORCE_STR_ENCODE_CACHE.get(force_str)
    if ids is None:
        from vllm.tokenizers import cached_tokenizer_from_config

        tokenizer = cached_tokenizer_from_config(model_config)
        ids = list(tokenizer.encode(force_str, add_special_tokens=False))
        _FORCE_STR_ENCODE_CACHE[force_str] = ids
    return list(ids)


def resolve_think_budget_ratio(params: SamplingParams) -> float | None:
    """Per-request budget ratio: vllm_xargs > env > None (off).

    Raises ValueError on malformed xargs values (called by the input
    processor during request validation, so clients get a 400).
    """
    extra = params.extra_args or {}
    raw = extra.get(_XARGS_BUDGET_RATIO)
    if raw is not None:
        try:
            ratio = float(raw)
        except (TypeError, ValueError):
            raise ValueError(
                f"vllm_xargs.{_XARGS_BUDGET_RATIO} must be a float, got {raw!r}."
            ) from None
        if not 0.0 < ratio <= 1.0:
            raise ValueError(
                f"vllm_xargs.{_XARGS_BUDGET_RATIO} must be in (0, 1], got {ratio}."
            )
        return ratio
    env_ratio = float(env_with_legacy("VLLM_THINK_BUDGET_RATIO") or 0)
    return env_ratio if 0.0 < env_ratio <= 1.0 else None


def resolve_force_str(params: SamplingParams) -> str | None:
    """Per-request force string: vllm_xargs > env > None (bare end ids).

    Raises ValueError on malformed xargs values.
    """
    extra = params.extra_args or {}
    raw = extra.get(_XARGS_FORCE_STR)
    if raw is not None:
        if not isinstance(raw, str) or not raw:
            raise ValueError(
                f"vllm_xargs.{_XARGS_FORCE_STR} must be a non-empty string."
            )
        return raw
    return env_with_legacy("VLLM_THINK_BUDGET_FORCE_STR") or None


def validate_think_budget_xargs(
    params: SamplingParams, reasoning_enabled: bool
) -> None:
    """Request-admission validation of the think-budget vllm_xargs keys.

    Only xargs-origin values are checked: a server-wide env ratio on a
    server without reasoning stays a startup warning (existing behavior),
    never a per-request error. Raises ValueError (-> HTTP 400).
    """
    extra = params.extra_args or {}
    has_ratio = extra.get(_XARGS_BUDGET_RATIO) is not None
    has_force = extra.get(_XARGS_FORCE_STR) is not None
    if not has_ratio and not has_force:
        return
    if has_ratio:
        resolve_think_budget_ratio(params)
    if has_force:
        resolve_force_str(params)
    if not reasoning_enabled:
        raise ValueError(
            "vllm_xargs think_budget_ratio / think_budget_force_str require "
            "reasoning to be enabled (--reasoning-parser)."
        )


class ThinkingTokenBudgetLogitsProcessor(LogitsProcessor):
    """Limits the number of tokens allowed inside a 'thinking' section."""

    def __init__(
        self, vllm_config: "VllmConfig", device: torch.device, is_pin_memory: bool
    ):
        reasoning_config = vllm_config.reasoning_config
        max_num_reqs = vllm_config.scheduler_config.max_num_seqs

        # Check if thinking is enabled
        self.is_enabled = reasoning_config is not None and reasoning_config.enabled

        self.reasoning_start_token_ids = getattr(
            reasoning_config, "reasoning_start_token_ids", []
        )
        self.reasoning_end_token_ids = getattr(
            reasoning_config, "reasoning_end_token_ids", []
        )

        # MOTIF: server-side default thinking budget from VLLM_THINK_BUDGET_RATIO.
        # Mirrors the old custom ThinkLogitsProcessor: the ratio is applied to the
        # POST-PROMPT space (max_model_len - prompt_len), NOT to max_model_len, and
        # at least `default_answer_reserve` tokens are always kept for the answer.
        # Because it depends on prompt length, the budget is computed per request
        # in update_state. Requests that set `thinking_token_budget` explicitly
        # bypass this. Unset / out of (0, 1] -> no server default (per-request
        # `thinking_token_budget` only).
        _ratio = float(env_with_legacy("VLLM_THINK_BUDGET_RATIO") or 0)
        self.default_budget_ratio: float | None = (
            _ratio if 0.0 < _ratio <= 1.0 else None
        )
        self.default_answer_reserve = 4096
        self.max_model_len = vllm_config.model_config.max_model_len

        # Under speculative decoding, apply_with_spec_decode() (verify positions)
        # is the sole budget enforcer; apply() (the bonus token) must NOT also
        # force, otherwise the terminal </think> is emitted twice in one step
        # (verify forces it, the bonus re-forces it) and leaks into `content`.
        self.has_spec_decode = vllm_config.speculative_config is not None

        # MOTIF: graceful "force" sequence emitted when the budget is hit,
        # SEPARATE from natural-exit detection. Natural exit is still detected on
        # reasoning_end_token_ids (e.g. bare </think>); when the budget forces a
        # stop, emit VLLM_THINK_BUDGET_FORCE_STR instead -- a graceful transition
        # phrase that must END with </think>. Unset -> force = reasoning end token.
        self.model_config = vllm_config.model_config
        _force_str = env_with_legacy("VLLM_THINK_BUDGET_FORCE_STR")
        if _force_str and self.is_enabled:
            self.force_end_ids = encode_force_str(_force_str, self.model_config)
        else:
            self.force_end_ids = list(self.reasoning_end_token_ids or [])

        self.pin_memory = is_pin_memory
        self.device = device
        # Per-request state tracking for thinking token management
        # Key: request_index, Value: state dict containing:
        # "in_think": bool - currently in thinking mode
        # "in_end": bool - currently forcing end tokens output
        # "check_count_down": int - steps remaining until next think
        #                            start/end token parsing
        # "think_count": int - number of thinking tokens generated
        # "end_count": int - number of end tokens forced so far
        # "thinking_token_budget": int - max allowed thinking tokens
        # "output_tok_ids": list[int] - generated output tokens
        # "prev_output_length": int - previous output length for
        #                               incremental processing
        self._state: dict[int, dict[str, Any]] = {}

        # MOTIF: one-shot startup confirmation (the old custom processor logged
        # this) so operators can see the budget is active and on what basis.
        # Also warn loudly on the easy misconfig: ratio/force set but reasoning
        # not enabled (no --reasoning-parser/--reasoning-config) -> silent no-op.
        if self.is_enabled:
            logger.debug(
                "[think-budget] active: default_ratio=%s (of post-prompt space, "
                "answer_reserve>=%d tok), end_token_ids=%s, force_ids=%s%s",
                self.default_budget_ratio,
                self.default_answer_reserve,
                self.reasoning_end_token_ids,
                self.force_end_ids,
                "" if _force_str else " (bare </think>)",
            )
        elif self.default_budget_ratio is not None or _force_str:
            logger.warning(
                "[think-budget] VLLM_THINK_BUDGET_RATIO / "
                "VLLM_THINK_BUDGET_FORCE_STR is set but reasoning is NOT enabled "
                "(pass --reasoning-parser, e.g. deepseek_r1); think-budget is a "
                "no-op."
            )

        # Preallocate reusable tensors
        self.mask = torch.zeros(max_num_reqs, dtype=torch.bool, device=device)
        self.force_token_ids = torch.full(
            (max_num_reqs,), -1, dtype=torch.long, device=device
        )

    @staticmethod
    def _find_last_sequence_index(target_list: list[int], token_ids: list[int]) -> int:
        """
        Returns the index of the last occurrence of token_ids in target_list.

        Args:
          target_list (list[int]): The list of token IDs.
          token_ids (list[int]): The sequence of token IDs to find.
        """
        if not token_ids:
            return -1
        for i in range(len(target_list) - len(token_ids), -1, -1):
            if target_list[i : i + len(token_ids)] == token_ids:
                return i
        return -1

    def _init_state_entry(
        self, prompt_tok_ids: list[int] | None, thinking_token_budget: int
    ) -> dict[str, Any]:
        """Initializes the tracking state for a given sequence index."""
        if prompt_tok_ids is None:
            last_start = -1
            last_end = -1
            in_think = False
            think_count = 0
        else:
            last_start = self._find_last_sequence_index(
                prompt_tok_ids, self.reasoning_start_token_ids
            )
            last_end = self._find_last_sequence_index(
                prompt_tok_ids, self.reasoning_end_token_ids
            )
            in_think = last_start > last_end
            if in_think:
                think_count = len(prompt_tok_ids) - (
                    last_start + len(self.reasoning_start_token_ids)
                )
            else:
                think_count = 0

        return {
            "in_think": in_think,  # Currently in thinking mode
            "in_end": in_think and thinking_token_budget == 0,
            "check_count_down": thinking_token_budget,
            "think_count": think_count,  # Number of tokens in thinking section
            "end_count": 0,  # Number of end tokens forced so far
            "prompt_tok_ids": prompt_tok_ids,
            "output_tok_ids": [],
            "thinking_token_budget": thinking_token_budget,
            "prev_output_length": 0,
            # Track previous output length for incremental updates
        }

    def _update_think_state(self, state: dict[str, Any]):
        """Updates the state based on newly generated output tokens."""
        if not state.get("in_end", False) and state.get("check_count_down", 0) > 0:
            # MOTIF: decrement by tokens generated since the last check, not by 1 per
            # step. Under speculative decoding a step commits >1 token, so a per-step
            # countdown (initialised to the token budget) never reaches 0 before the
            # context fills and the budget never fires. Count tokens.
            _out = state.get("output_tok_ids", [])
            _delta = len(_out) - state.get("_countdown_seen", 0)
            state["_countdown_seen"] = len(_out)
            state["check_count_down"] = max(
                0, state["check_count_down"] - max(1, _delta)
            )
            if state["check_count_down"] > 0:
                return

        output = state.get("output_tok_ids", [])
        if not output:
            return

        # Track previous output length for incremental processing
        prev_length = state.get("prev_output_length", 0)
        current_length = len(output)

        if current_length <= prev_length:
            return

        # Process only newly added tokens
        new_tokens = output[prev_length:]
        state["prev_output_length"] = current_length

        # Check if new tokens contain think start or end sequences
        start_len = len(self.reasoning_start_token_ids)
        end_len = len(self.reasoning_end_token_ids)

        # Look for think sequences in recent tokens (including boundary)
        # Check overlapping regions where sequences might span boundaries
        check_start_idx = max(0, prev_length - max(start_len, end_len) + 1)
        recent_tokens = output[check_start_idx:]

        # Find any think start/end sequences in recent tokens
        recent_start_pos = self._find_last_sequence_index(
            recent_tokens, self.reasoning_start_token_ids
        )
        recent_end_pos = self._find_last_sequence_index(
            recent_tokens, self.reasoning_end_token_ids
        )

        # Update state based on recent sequences
        if not state["in_end"]:
            if recent_start_pos >= 0 and recent_end_pos >= 0:
                if recent_start_pos > recent_end_pos:
                    # Case: ...<end>...<start>... - entering think mode
                    absolute_start_pos = check_start_idx + recent_start_pos
                    new_think_count = current_length - (absolute_start_pos + start_len)
                    state["in_think"] = True
                    state["think_count"] = new_think_count
                else:
                    # Case: ...<start>...<end>... - exiting think mode
                    state["in_think"] = False
                    state["think_count"] = 0
            elif recent_start_pos >= 0:
                # Found think start - entering think mode
                absolute_start_pos = check_start_idx + recent_start_pos
                new_think_count = current_length - (absolute_start_pos + start_len)
                state["in_think"] = True
                state["think_count"] = new_think_count
            elif recent_end_pos >= 0:
                # Found think end - exiting think mode
                state["in_think"] = False
                state["think_count"] = 0
            elif state["in_think"]:
                # Continue thinking mode, increment count by new tokens
                state["think_count"] += len(new_tokens)

            # Set countdown based on current state
            if state["in_think"]:
                remaining_budget = max(
                    0, state["thinking_token_budget"] - state["think_count"]
                )
                state["check_count_down"] = max(0, remaining_budget - 1)
            else:
                state["check_count_down"] = state["thinking_token_budget"]

            # Check if need to transition to end mode
            if (
                state["in_think"]
                and state["think_count"] >= state["thinking_token_budget"]
            ):
                state["in_think"] = False
                state["in_end"] = True
                state["end_count"] = 0
                state["check_count_down"] = state["thinking_token_budget"]
                logger.info(
                    "[think-budget] budget %d reached -> forcing end sequence "
                    "(force token ids %s)",
                    state["thinking_token_budget"],
                    self._force_ids(state),
                )
        else:
            # In end mode (emitting the forced end sequence)
            state["end_count"] += 1
            if state["end_count"] >= len(self._force_ids(state)):
                state.update(
                    {
                        "in_end": False,
                        "end_count": 0,
                        "check_count_down": state["thinking_token_budget"],
                    }
                )

    def _force_ids(self, state: dict[str, Any]) -> list[int]:
        """Per-request force sequence (vllm_xargs > env > bare end ids)."""
        return state.get("force_end_ids") or self.force_end_ids

    def is_argmax_invariant(self) -> bool:
        """This logits processor can change the outcome of
        greedy sampling by forcing that the thinking section
        ends after a certain number of tokens."""
        return False

    def update_state(self, batch_update: BatchUpdate | None):
        if not self.is_enabled:
            return
        if batch_update:
            for index, params, prompt_tok_ids, output_tok_ids in batch_update.added:
                thinking_token_budget = params.thinking_token_budget
                if thinking_token_budget is None:
                    # MOTIF priority chain for the ratio:
                    # vllm_xargs.think_budget_ratio > VLLM_THINK_BUDGET_RATIO
                    # env > off. The explicit thinking_token_budget request
                    # field (absolute token count) bypasses the ratio.
                    try:
                        budget_ratio = resolve_think_budget_ratio(params)
                    except ValueError:
                        # The input processor 400s malformed xargs up front;
                        # fail closed here regardless.
                        budget_ratio = None
                    if budget_ratio is not None:
                        # MOTIF: the ratio applies to the request's actual
                        # completion budget, not the model context --
                        # avail = min(max_tokens, max_model_len - prompt).
                        # A user saying "60% of my 4096-token response may
                        # think" writes ratio 0.6 directly instead of
                        # back-solving against a 262K context. When
                        # max_tokens is absent (offline) or serving already
                        # defaulted it to the remaining context, this
                        # degrades to the old post-prompt-space behavior.
                        # The answer reserve floor shrinks to avail/4 for
                        # small budgets, else the historical 4096 floor
                        # would zero the think budget entirely.
                        avail = self.max_model_len - len(prompt_tok_ids or ())
                        max_toks = getattr(params, "max_tokens", None)
                        if max_toks:
                            avail = min(avail, max_toks)
                        reserve_floor = min(
                            self.default_answer_reserve, max(1, avail // 4)
                        )
                        answer_reserve = max(
                            reserve_floor,
                            int(avail * (1.0 - budget_ratio)),
                        )
                        thinking_token_budget = max(0, avail - answer_reserve)

                if thinking_token_budget is not None:
                    self._state[index] = self._init_state_entry(
                        prompt_tok_ids, thinking_token_budget
                    )
                    self._state[index]["output_tok_ids"] = output_tok_ids
                    # MOTIF: per-request force sequence
                    # (vllm_xargs.think_budget_force_str > env > bare end ids).
                    try:
                        force_str = resolve_force_str(params)
                    except ValueError:
                        force_str = None
                    if force_str:
                        self._state[index]["force_end_ids"] = encode_force_str(
                            force_str, self.model_config
                        )
                    else:
                        self._state[index]["force_end_ids"] = self.force_end_ids
                else:
                    # Remove state if no thinking budget
                    self._state.pop(index, None)

            for index in batch_update.removed:
                self._state.pop(index, {})

            for i1, i2, direction in batch_update.moved:
                if direction == MoveDirectionality.SWAP:
                    state1 = self._state.pop(i1, None)
                    state2 = self._state.pop(i2, None)
                    if state1 is not None:
                        self._state[i2] = state1
                    if state2 is not None:
                        self._state[i1] = state2
                else:
                    state = self._state.pop(i1, None)
                    if state is not None:
                        self._state[i2] = state

        for state in self._state.values():
            self._update_think_state(state)

    def apply(self, logits: torch.Tensor) -> torch.Tensor:
        if not self.is_enabled or not self._state:
            return logits

        # Spec decode: the verify path (apply_with_spec_decode) already forces the
        # budget at the speculative positions. Forcing the bonus token here too
        # would double-emit the terminal </think> when the drafter predicts it
        # (verify accepts </think>, the bonus re-forces </think>), and the extra
        # </think> leaks into `content`. Leave the bonus free under spec decode.
        if self.has_spec_decode:
            return logits

        batch_size = logits.size(0)
        self.mask[:batch_size] = False

        for i in range(batch_size):
            state = self._state.get(i)
            if state and state["in_end"]:
                self.mask[i] = True
                self.force_token_ids[i] = self._force_ids(state)[state["end_count"]]

        # Check in CPU first not to sync with GPU
        has_active_thinking = any(
            state.get("in_end", False) for state in self._state.values()
        )

        if has_active_thinking:
            current_mask = self.mask[:batch_size]
            active_indices = current_mask.nonzero(as_tuple=False).view(-1)
            if len(active_indices) > 0:
                force_tokens = self.force_token_ids[active_indices]
                # Apply a large value for the end thinking token id index
                logits[active_indices, force_tokens] = 1e9

        return logits

    def apply_with_spec_decode(
        self,
        logits: torch.Tensor,
        num_draft_tokens: list[int],
        spec_token_ids: list[list[int]] | None,
    ) -> torch.Tensor:
        """Spec-decode version of apply().

        Forces the reasoning end token at the verify positions whose
        (speculative) think state is over budget, so the budget is enforced even
        when several tokens are verified per step. Mirrors apply() but replays
        each request's draft tokens through the same think state machine on a
        COPY of the committed state (the real accepted tokens are folded into the
        committed state later, in update_state).

        ``logits`` has shape ``[sum(num_draft_tokens), V]``; rows are grouped per
        request via ``cumsum(num_draft_tokens)`` (the layout MinTokens uses).
        ``spec_token_ids[i]`` is request i's draft token ids (CPU list), used to
        replay the think state across the speculative positions.
        """
        if not self.is_enabled or not self._state:
            return logits
        # MOTIF: detection uses reasoning_end_token_ids (e.g. bare </think>);
        # forcing at budget emits the per-request force sequence (the
        # graceful phrase; vllm_xargs > env > bare end ids).
        detect_end_ids = self.reasoning_end_token_ids

        num_draft_arr = np.asarray(num_draft_tokens, dtype=np.int64)
        cumsum = np.concatenate([[0], np.cumsum(num_draft_arr)])
        start_ids = self.reasoning_start_token_ids
        single_start = start_ids[0] if len(start_ids) == 1 else None
        single_end = detect_end_ids[0] if len(detect_end_ids) == 1 else None

        rows: list[int] = []
        toks: list[int] = []
        for i, state in self._state.items():
            if i >= len(num_draft_arr):
                continue
            k = int(num_draft_arr[i])
            if k <= 0:
                continue
            force_ids = self._force_ids(state)
            if not force_ids:
                continue
            drafts = (
                spec_token_ids[i]
                if spec_token_ids is not None and i < len(spec_token_ids)
                else []
            )
            # Copy committed state; replay is speculative so never mutate it.
            in_think = state["in_think"]
            in_end = state["in_end"]
            think_count = state["think_count"]
            end_count = state["end_count"]
            budget = state["thinking_token_budget"]
            base = int(cumsum[i])
            for j in range(k):
                # Decide the force at row j from the state produced by the first
                # j draft tokens (the tokens that would precede this position).
                force_tok = None
                if in_end:
                    force_tok = force_ids[end_count]
                    end_count += 1
                    if end_count >= len(force_ids):
                        in_end = False
                        end_count = 0
                elif in_think and think_count >= budget:
                    in_think = False
                    in_end = True
                    end_count = 0
                    force_tok = force_ids[end_count]
                    end_count += 1
                    if end_count >= len(force_ids):
                        in_end = False
                        end_count = 0
                if force_tok is not None:
                    rows.append(base + j)
                    toks.append(int(force_tok))
                # Advance state by consuming draft token j (single-token
                # delimiters; on a forced position the draft is rejected and spec
                # stops, so later positions this step are discarded anyway).
                if not in_end and j < len(drafts):
                    d = drafts[j]
                    if single_start is not None and d == single_start:
                        in_think = True
                        think_count = 0
                    elif single_end is not None and d == single_end:
                        in_think = False
                        think_count = 0
                    elif in_think:
                        think_count += 1

        if rows:
            logger.info(
                "[think-budget] forced reasoning end at %d spec verify "
                "position(s) (budget reached)",
                len(rows),
            )
            # Built only when something is actually forced (rare: ~once per
            # request, at its budget boundary). from_numpy + non_blocking H2D
            # mirrors MinTokens.apply_with_spec_decode; the common no-force path
            # above creates no tensors at all.
            rows_t = torch.from_numpy(np.asarray(rows, dtype=np.int64)).to(
                logits.device, non_blocking=True
            )
            toks_t = torch.from_numpy(np.asarray(toks, dtype=np.int64)).to(
                logits.device, non_blocking=True
            )
            logits[rows_t, toks_t] = 1e9
        return logits
