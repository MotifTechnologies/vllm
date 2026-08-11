# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""MOTIF: thinking-token-budget enforcement for Model Runner V2.

V2 port of ThinkingTokenBudgetLogitsProcessor (vllm/v1/sample/logits_processor/
builtin.py). The V1 logits-processor framework does not exist in V2, so this is
implemented in the V2 sampler style: per-slot GPU state tensors + Triton
kernels, hooked into the single sampling choke point
(Sampler.apply_sampling_params) that every path shares — non-spec sampling and
all three rejection-sample methods (strict / synthetic / probabilistic).

Semantics mirror the V1 processor:
- Per-request budget: SamplingParams.thinking_token_budget, else a server
  default from VLLM_THINK_BUDGET_RATIO applied to the post-prompt space with
  >= 4096 tokens reserved for the answer.
- Natural exit is detected on reasoning_end_token_ids (e.g. bare </think>);
  hitting the budget instead forces MOTIF_THINK_BUDGET_FORCE_STR (a graceful
  phrase that must end with </think>), token by token.
- Spec decode: the expanded verify rows are replayed against a COPY of the
  committed state (row j's decision sees the j draft tokens that precede it),
  so the budget fires at the right verify position.
- Committed state advances once per step from the accepted tokens
  (commit(), called after [rejection] sampling), so it is exact under both
  async scheduling and speculative decoding.

Restriction (same as the V1 spec-decode replay): reasoning start/end must be
single tokens. Multi-token delimiters disable the feature with a warning.
"""
from typing import TYPE_CHECKING

import numpy as np
import torch

from vllm.logger import init_logger
from vllm.sampling_params import SamplingParams
from vllm.triton_utils import tl, triton
from vllm.v1.worker.gpu.buffer_utils import StagedWriteTensor

if TYPE_CHECKING:
    from vllm.config import VllmConfig

logger = init_logger(__name__)

# think_state columns (int32): [active, in_think, think_count, in_end, end_count]
_ACTIVE, _IN_THINK, _COUNT, _IN_END, _END_COUNT = 0, 1, 2, 3, 4
_NUM_COLS = 5
_FORCE_LOGIT = 1e9


def _find_last_index(target_list: list[int], token_id: int) -> int:
    for i in range(len(target_list) - 1, -1, -1):
        if target_list[i] == token_id:
            return i
    return -1


class ThinkBudgetState:
    def __init__(self, vllm_config: "VllmConfig", device: torch.device):
        import os

        reasoning_config = vllm_config.reasoning_config
        self.is_enabled = reasoning_config is not None and reasoning_config.enabled

        start_ids = list(
            getattr(reasoning_config, "reasoning_start_token_ids", []) or []
        )
        end_ids = list(getattr(reasoning_config, "reasoning_end_token_ids", []) or [])

        _ratio = float(os.environ.get("VLLM_THINK_BUDGET_RATIO", "0") or 0)
        self.default_budget_ratio: float | None = (
            _ratio if 0.0 < _ratio <= 1.0 else None
        )
        self.default_answer_reserve = 4096
        self.max_model_len = vllm_config.model_config.max_model_len

        _force_str = os.environ.get("MOTIF_THINK_BUDGET_FORCE_STR", "")
        if _force_str and self.is_enabled:
            from vllm.tokenizers import cached_tokenizer_from_config

            _tok = cached_tokenizer_from_config(vllm_config.model_config)
            self.force_end_ids = list(_tok.encode(_force_str, add_special_tokens=False))
        else:
            self.force_end_ids = list(end_ids)

        # The GPU state machine consumes tokens one by one; like the V1
        # spec-decode replay, it supports single-token think delimiters only.
        # A missing/multi-token START only disables re-entry detection (think
        # mode is still initialized from the prompt, matching V1 behavior);
        # END must be a single token and the force sequence non-empty.
        if self.is_enabled and (len(end_ids) != 1 or not self.force_end_ids):
            logger.warning(
                "[think-budget] V2 port requires a single-token reasoning "
                "end id and a non-empty force sequence (end=%s force=%s); "
                "disabling think-budget.",
                end_ids,
                self.force_end_ids,
            )
            self.is_enabled = False

        if self.is_enabled:
            logger.info(
                "[think-budget] active (V2 runner): default_ratio=%s (of "
                "post-prompt space, answer_reserve>=%d tok), end_token_ids=%s, "
                "force_ids=%s%s",
                self.default_budget_ratio,
                self.default_answer_reserve,
                end_ids,
                self.force_end_ids,
                "" if _force_str else " (bare </think>)",
            )
        elif self.default_budget_ratio is not None or _force_str:
            logger.warning(
                "[think-budget] VLLM_THINK_BUDGET_RATIO / "
                "MOTIF_THINK_BUDGET_FORCE_STR is set but reasoning is NOT "
                "enabled (pass --reasoning-parser, e.g. deepseek_r1); "
                "think-budget is a no-op."
            )

        if not self.is_enabled:
            return

        # -1 never matches a real token id -> start detection disabled.
        self.start_id = start_ids[0] if len(start_ids) == 1 else -1
        self.end_id = end_ids[0]
        max_num_reqs = vllm_config.scheduler_config.max_num_seqs
        self.think_state = StagedWriteTensor(
            (max_num_reqs, _NUM_COLS), dtype=torch.int32, device=device
        )
        self.budget = StagedWriteTensor(
            max_num_reqs, dtype=torch.int32, device=device
        )
        self.force_ids_gpu = torch.tensor(
            self.force_end_ids, dtype=torch.int32, device=device
        )
        # CPU-side skip mask: True only for slots with an active budget, so
        # steps without any budgeted request launch no kernel at all.
        self.has_budget = np.zeros(max_num_reqs, dtype=bool)

    def add_request(
        self,
        req_idx: int,
        prompt_len: int,
        sampling_params: SamplingParams,
        prompt_token_ids: list[int] | None,
    ) -> None:
        if not self.is_enabled:
            return

        budget = getattr(sampling_params, "thinking_token_budget", None)
        if budget is None and self.default_budget_ratio is not None:
            # Prompt-aware server default: ratio of the post-prompt space,
            # always keeping >= default_answer_reserve tokens for the answer.
            avail = self.max_model_len - prompt_len
            answer_reserve = max(
                self.default_answer_reserve,
                int(avail * (1.0 - self.default_budget_ratio)),
            )
            budget = max(0, avail - answer_reserve)

        if budget is None:
            # No budget for this request: deactivate the slot (slots are
            # reused; a stale active state must not leak into a new request).
            self.has_budget[req_idx] = False
            self.think_state.stage_write(req_idx, 0, [0] * _NUM_COLS)
            return

        # Initialize think state from the prompt tail (the chat template may
        # end the prompt inside <think>, so generation starts in think mode).
        in_think = False
        think_count = 0
        if prompt_token_ids:
            last_start = _find_last_index(prompt_token_ids, self.start_id)
            last_end = _find_last_index(prompt_token_ids, self.end_id)
            in_think = last_start > last_end
            if in_think:
                think_count = len(prompt_token_ids) - (last_start + 1)

        in_end = bool(in_think and budget == 0)
        self.has_budget[req_idx] = True
        self.budget.stage_write_elem(req_idx, int(budget))
        self.think_state.stage_write(
            req_idx,
            0,
            [1, int(in_think), int(think_count), int(in_end), 0],
        )

    def apply_staged_writes(self) -> None:
        if not self.is_enabled:
            return
        self.think_state.apply_write()
        self.budget.apply_write()

    def apply(
        self,
        logits: torch.Tensor,
        expanded_idx_mapping: torch.Tensor,
        idx_mapping_np: np.ndarray,
        input_ids: torch.Tensor,
        expanded_local_pos: torch.Tensor,
    ) -> None:
        """Force the end sequence at over-budget rows (all sampling paths).

        Rows are the (possibly expanded) logits rows. For spec-decode verify,
        request rows are contiguous and expanded_local_pos gives the position
        within the group, so each row replays its own preceding draft tokens
        (inputs of rows base+1..base+j) against a copy of the committed state.
        Committed state itself is only advanced in commit().
        """
        if not self.is_enabled or not np.any(self.has_budget[idx_mapping_np]):
            return
        num_rows = logits.shape[0]
        _replay_force_kernel[(num_rows,)](
            logits,
            logits.stride(0),
            self.think_state.gpu,
            self.think_state.gpu.stride(0),
            self.budget.gpu,
            expanded_idx_mapping,
            expanded_local_pos,
            input_ids,
            self.force_ids_gpu,
            FORCE_LEN=len(self.force_end_ids),
            START_ID=self.start_id,
            END_ID=self.end_id,
            FORCE_LOGIT=_FORCE_LOGIT,
        )

    def commit(
        self,
        sampled_token_ids: torch.Tensor,
        num_sampled: torch.Tensor,
        idx_mapping: torch.Tensor,
        idx_mapping_np: np.ndarray,
    ) -> None:
        """Advance committed state by this step's accepted tokens."""
        if not self.is_enabled or not np.any(self.has_budget[idx_mapping_np]):
            return
        num_reqs = idx_mapping.shape[0]
        _commit_kernel[(num_reqs,)](
            sampled_token_ids,
            sampled_token_ids.stride(0),
            num_sampled,
            self.think_state.gpu,
            self.think_state.gpu.stride(0),
            self.budget.gpu,
            idx_mapping,
            FORCE_LEN=len(self.force_end_ids),
            START_ID=self.start_id,
            END_ID=self.end_id,
        )


@triton.jit
def _replay_force_kernel(
    logits_ptr,
    logits_stride,
    state_ptr,
    state_stride,
    budget_ptr,
    expanded_idx_mapping_ptr,
    local_pos_ptr,
    input_ids_ptr,
    force_ids_ptr,
    FORCE_LEN: tl.constexpr,
    START_ID: tl.constexpr,
    END_ID: tl.constexpr,
    FORCE_LOGIT: tl.constexpr,
):
    row = tl.program_id(0)
    slot = tl.load(expanded_idx_mapping_ptr + row)
    active = tl.load(state_ptr + slot * state_stride + 0)
    if active == 0:
        return

    in_think = tl.load(state_ptr + slot * state_stride + 1)
    count = tl.load(state_ptr + slot * state_stride + 2)
    in_end = tl.load(state_ptr + slot * state_stride + 3)
    end_cnt = tl.load(state_ptr + slot * state_stride + 4)
    budget = tl.load(budget_ptr + slot)

    j = tl.load(local_pos_ptr + row)
    base = row - j

    # Replay the j tokens preceding this row's prediction on a local copy.
    # Row 0's input is the last committed token (already in committed state);
    # rows 1..j consume the draft tokens (inputs of rows base+1..base+j).
    for t in range(1, j + 1):
        if (in_end == 0) and (in_think == 1) and (count >= budget):
            in_think = 0
            in_end = 1
            end_cnt = 0
        if in_end == 1:
            # The preceding position was forced; its token is the forced one.
            end_cnt += 1
            if end_cnt >= FORCE_LEN:
                in_end = 0
                end_cnt = 0
        else:
            tok = tl.load(input_ids_ptr + base + t)
            if tok == START_ID:
                in_think = 1
                count = 0
            elif tok == END_ID:
                in_think = 0
                count = 0
            elif in_think == 1:
                count += 1

    # Decide the force for this row.
    if (in_end == 0) and (in_think == 1) and (count >= budget):
        in_end = 1
        end_cnt = 0
    if in_end == 1:
        force_tok = tl.load(force_ids_ptr + end_cnt)
        tl.store(logits_ptr + row * logits_stride + force_tok, FORCE_LOGIT)


@triton.jit
def _commit_kernel(
    sampled_ptr,
    sampled_stride,
    num_sampled_ptr,
    state_ptr,
    state_stride,
    budget_ptr,
    idx_mapping_ptr,
    FORCE_LEN: tl.constexpr,
    START_ID: tl.constexpr,
    END_ID: tl.constexpr,
):
    req = tl.program_id(0)
    slot = tl.load(idx_mapping_ptr + req)
    active = tl.load(state_ptr + slot * state_stride + 0)
    if active == 0:
        return

    in_think = tl.load(state_ptr + slot * state_stride + 1)
    count = tl.load(state_ptr + slot * state_stride + 2)
    in_end = tl.load(state_ptr + slot * state_stride + 3)
    end_cnt = tl.load(state_ptr + slot * state_stride + 4)
    budget = tl.load(budget_ptr + slot)

    n = tl.load(num_sampled_ptr + req)
    for i in range(n):
        if (in_end == 0) and (in_think == 1) and (count >= budget):
            in_think = 0
            in_end = 1
            end_cnt = 0
        if in_end == 1:
            # While forcing, every accepted token is a forced token (the
            # forced position rejects the draft, so at most the forced token
            # itself commits from that position).
            end_cnt += 1
            if end_cnt >= FORCE_LEN:
                in_end = 0
                end_cnt = 0
        else:
            tok = tl.load(sampled_ptr + req * sampled_stride + i)
            if tok == START_ID:
                in_think = 1
                count = 0
            elif tok == END_ID:
                in_think = 0
                count = 0
            elif in_think == 1:
                count += 1

    tl.store(state_ptr + slot * state_stride + 1, in_think)
    tl.store(state_ptr + slot * state_stride + 2, count)
    tl.store(state_ptr + slot * state_stride + 3, in_end)
    tl.store(state_ptr + slot * state_stride + 4, end_cnt)
