# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""MOTIF: logits-level repetition guard (THINK-ONLY).

Detects repeated token patterns in a recent window of the committed output
(suffix array + LCP scan; variable-length patterns, overlapping/tandem loops
included) and intervenes BEFORE sampling, strictly INSIDE the thinking
section -- the answer section is never touched, by design:

* ``mode == "truncate"``: force the reasoning end sequence
  (``VLLM_THINK_BUDGET_FORCE_STR`` when set, else the bare reasoning end
  token ids -- the machinery shared with ThinkingTokenBudget) so the model
  exits the loop and still writes an answer.

The TRIGGER is coverage-based (see ``find_active_repeats``): a pattern
fires only when the union of its occurrences covers >= ``rep_coverage``
(default 0.5) of the detection region, with a small occurrence floor.
aa-lcr e2e showed count-only thresholds both false-fire on document
quoting / entity re-use AND miss long-period paragraph loops.

This guard STEERS generation. It never terminates a request; the
scheduler-level hard stop (``max_pattern_size > 0`` in ``check_stop``) is a
separate, mutually-exclusive mechanism.

Detection is intentionally stateless per check: every
``VLLM_REP_CHECK_INTERVAL`` committed tokens the detector re-runs over the
recent ``VLLM_REP_WINDOW`` tokens of the current think section (plus its
prompt-side tail for prompt-initiated sections). Rejected
speculative-decoding drafts are never committed and therefore never seen,
so no rollback handling is needed (cf. the stale-incremental-state bug
flagged on vllm-project/vllm#41539).

Configuration follows the MOTIF priority chain everywhere:
``vllm_xargs`` > typed ``repetition_detection`` field > env var > default.
Activation: ``rep_trunc_mode`` > ``repetition_detection.mode`` >
``VLLM_REP_MODE`` (server-wide default; requests can disable with
``rep_trunc_mode: "off"``) > off. Requires reasoning to be enabled --
without a thinking section there is nothing this guard may touch.
"""

from typing import TYPE_CHECKING, Any

import numpy as np
import torch

from vllm import SamplingParams
from vllm.logger import init_logger
from vllm.v1.sample.logits_processor.builtin import process_dict_updates
from vllm.v1.sample.logits_processor.interface import BatchUpdate, LogitsProcessor
from vllm.v1.sample.logits_processor.think_budget import (
    encode_force_str,
    env_with_legacy,
    resolve_force_str,
)

if TYPE_CHECKING:
    from vllm.config import VllmConfig

logger = init_logger(__name__)

# Hard dependency: the guard's detector is SA+LCP and pydivsufsort is its
# only backend. Fail fast at import (= server startup) instead of silently
# degrading -- serving images must install it (deploy.yaml does).
from pydivsufsort import divsufsort as _divsufsort  # noqa: E402
from pydivsufsort import kasai as _kasai  # noqa: E402


def _env_float(name: str, default: float) -> float:
    try:
        return float(env_with_legacy(name) or default)
    except ValueError:
        return default


def _env_int(name: str, default: int) -> int:
    try:
        return int(env_with_legacy(name) or default)
    except ValueError:
        return default


# vllm_xargs keys mirroring (and extending) RepetitionDetectionParams,
# e.g. extra_body={"vllm_xargs": {"rep_trunc_mode": "truncate", ...}}.
_XARGS_MODE = "rep_trunc_mode"
_XARGS_SCOPE = "rep_trunc_scope"
_XARGS_MIN_COUNT = "rep_min_count"
_XARGS_MIN_PATTERN = "rep_min_pattern_size"
_XARGS_WINDOW = "rep_window"
_XARGS_INTERVAL = "rep_check_interval"
_XARGS_COVERAGE = "rep_coverage"
_XARGS_MIN_REGION = "rep_min_region"

# Tail-of-chain defaults, used only once a request opted in via `mode`;
# nothing applies when no surface requests the guard.
# The primary trigger is COVERAGE (aa-lcr e2e: count thresholds false-fire
# on document quoting / entity re-use and simultaneously miss long-period
# paragraph loops, since min_count x period > window is undetectable).
# min_count is only a small floor: even 3 repetitions of a huge paragraph
# are already pathological when they dominate the window.
_DEFAULT_MIN_COUNT = 3
_DEFAULT_MIN_LEN = 5
# window 8192: detection needs floor(3) occurrences visible at once, so the
# largest catchable loop period is ~window/3. aa-lcr id=29 (period ~1.55K
# tokens) sat just outside window 4096 / 3 and burned 541K chars until the
# think-budget backstop; 8192 moves the boundary to ~2.7K-token periods for
# free (overhead measured at noise level up to 2048+, see tc_perf_overhead).
_DEFAULT_WINDOW = 8192
_DEFAULT_INTERVAL = 32
_DEFAULT_COVERAGE = 0.5
# Do not run detection before the region has this many tokens: with a tiny
# region the coverage threshold is meaningless and early structural
# enumerations false-fire (aa-lcr id=213: "Document N: ... not mentioned"
# sweep truncated at ~110 tokens into think).
_DEFAULT_MIN_REGION = 512


def resolve_env_mode() -> str | None:
    """VLLM_REP_MODE env value, normalized. Server-wide default activation:
    'truncate' verbatim; boolean-ish values ('1', 'true', 'on',
    'yes') mean truncate (the documented env default). Unknown values are
    ignored (the processor ctor warns once)."""
    raw = env_with_legacy("VLLM_REP_MODE").strip().lower()
    if not raw:
        return None
    if raw == "truncate":
        return raw
    if raw in ("1", "true", "on", "yes"):
        return "truncate"
    return None


def resolve_rep_guard_request_config(
    params: SamplingParams,
) -> dict[str, Any] | None:
    """Fully-resolved per-request guard config, or None when not active.

    Every knob follows the MOTIF priority chain:
    ``vllm_xargs`` > typed ``repetition_detection`` field > env var >
    default. Activation follows the same chain: ``rep_trunc_mode`` (xargs)
    > ``repetition_detection.mode`` (typed) > ``VLLM_REP_MODE`` (env,
    server-wide default: truncate + scope "think") > off. A request can
    opt OUT of the env default with ``rep_trunc_mode: "off"``.

    The guard is THINK-ONLY: it never touches the answer section.

    The returned config carries ``source`` ("request" | "env") so callers
    can fail requests loudly for explicit opt-ins but degrade silently for
    the server-wide env default (e.g. reasoning not enabled, or the
    request armed the scheduler-level hard stop instead).

    Raises ValueError on malformed xargs values -- the input processor
    calls this during request validation, so clients get a 400 instead of
    a silent no-op (vllm_xargs itself is unvalidated by design).
    """
    rep = params.repetition_detection
    extra = params.extra_args or {}

    mode = extra.get(_XARGS_MODE)
    if mode is not None and mode not in ("truncate", "off"):
        raise ValueError(
            f"vllm_xargs.{_XARGS_MODE} must be 'truncate' or 'off', got {mode!r}."
        )
    source = "request"
    if mode is None and rep is not None:
        mode = rep.mode
    if mode is None:
        env_mode = resolve_env_mode()
        if env_mode is not None:
            mode = env_mode
            source = "env"
    # The guard is THINK-ONLY by design: it never touches the answer
    # section. "think" is accepted for surface compatibility; anything
    # else is rejected. The 'penalize' mode has been removed; only
    # 'truncate' is supported.
    scope = extra.get(_XARGS_SCOPE)
    if scope is not None and scope != "think":
        raise ValueError(
            f"vllm_xargs.{_XARGS_SCOPE} only supports 'think' -- the "
            f"repetition guard never intervenes outside the thinking "
            f"section. Got {scope!r}."
        )
    if mode is None or mode == "off":
        if mode is None and scope is not None:
            raise ValueError(
                f"vllm_xargs.{_XARGS_SCOPE} requires an active mode "
                f"(vllm_xargs.{_XARGS_MODE}, repetition_detection.mode or "
                "VLLM_REP_MODE)."
            )
        return None
    if rep is not None and rep.max_pattern_size > 0:
        if source == "env":
            # The request explicitly armed the scheduler-level hard stop;
            # the server-wide default guard yields instead of conflicting.
            return None
        raise ValueError(
            "The logits-level repetition guard (mode) is mutually exclusive "
            "with scheduler-level repetition detection "
            "(repetition_detection.max_pattern_size > 0)."
        )
        if mode == "penalize":
            raise ValueError(
                "The logits-level repetition guard no longer supports "
                "'penalize' mode; use 'truncate' instead."
            )

    # Coverage threshold: xargs > env > default.
    raw_coverage = extra.get(_XARGS_COVERAGE)
    if raw_coverage is not None:
        try:
            coverage = float(raw_coverage)
        except (TypeError, ValueError):
            raise ValueError(
                f"vllm_xargs.{_XARGS_COVERAGE} must be a float, got {raw_coverage!r}."
            ) from None
        if not 0.0 < coverage <= 1.0:
            raise ValueError(
                f"vllm_xargs.{_XARGS_COVERAGE} must be in (0, 1], got {coverage}."
            )
    else:
        env_coverage = _env_float("VLLM_REP_COVERAGE", 0.0)
        coverage = env_coverage if 0.0 < env_coverage <= 1.0 else _DEFAULT_COVERAGE

    def _xargs_int(key: str) -> int:
        value = extra.get(key)
        if value is None:
            return 0
        try:
            parsed = int(value)
        except (TypeError, ValueError):
            raise ValueError(
                f"vllm_xargs.{key} must be an integer, got {value!r}."
            ) from None
        if parsed < 0:
            raise ValueError(f"vllm_xargs.{key} must be >= 0.")
        return parsed

    # Trigger threshold: xargs > typed field > env ("exceeds VLLM_REP_MAX"
    # -> int + 1) > default.
    min_count = _xargs_int(_XARGS_MIN_COUNT)
    if min_count == 1:
        raise ValueError(
            f"vllm_xargs.{_XARGS_MIN_COUNT} must be 0 (fall through) or >= 2."
        )
    if min_count == 0 and rep is not None and rep.min_count >= 2:
        min_count = rep.min_count
    if min_count == 0:
        rep_max = _env_float("VLLM_REP_MAX", 0.0)
        min_count = int(rep_max) + 1 if rep_max > 0 else _DEFAULT_MIN_COUNT

    min_len = _xargs_int(_XARGS_MIN_PATTERN)
    if min_len == 0 and rep is not None and rep.min_pattern_size > 0:
        min_len = rep.min_pattern_size
    if min_len == 0:
        min_len = _env_int("VLLM_REP_MIN_LEN", _DEFAULT_MIN_LEN)

    window = _xargs_int(_XARGS_WINDOW)
    if window == 0:
        window = _env_int("VLLM_REP_WINDOW", _DEFAULT_WINDOW)

    interval = _xargs_int(_XARGS_INTERVAL)
    if interval == 0:
        interval = _env_int("VLLM_REP_CHECK_INTERVAL", _DEFAULT_INTERVAL)

    min_region = _xargs_int(_XARGS_MIN_REGION)
    if min_region == 0:
        min_region = _env_int("VLLM_REP_MIN_REGION", _DEFAULT_MIN_REGION)

    return {
        "mode": mode,
        "source": source,
        "min_count": min_count,
        "min_len": max(2, min_len),
        "window": max(16, window),
        "interval": max(1, interval),
        "coverage": coverage,
        "min_region": max(1, min_region),
    }


def _find_last_sequence_index(target_list: list[int], token_ids: list[int]) -> int:
    """Index of the last occurrence of token_ids in target_list, else -1."""
    if not token_ids:
        return -1
    for i in range(len(target_list) - len(token_ids), -1, -1):
        if target_list[i : i + len(token_ids)] == token_ids:
            return i
    return -1


def _sa_lcp(tokens: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """(sa, lcp) with lcp[i] = LCP(suffix sa[i], suffix sa[i+1]).

    pydivsufsort's kasai() follows this exact convention (verified: for
    "banana$"-like input lcp == [0,1,3,0,0,2,0]), with a trailing 0 pad.
    """
    sa = _divsufsort(tokens)
    lcp = _kasai(tokens, sa)
    return np.asarray(sa, dtype=np.int64), np.asarray(lcp, dtype=np.int64)


def find_active_repeats(
    tokens: np.ndarray,
    min_len: int,
    min_count: int,
    tail_slack: int,
    min_coverage: float,
) -> list[np.ndarray]:
    """Find repeated patterns that DOMINATE the tail of ``tokens``.

    The degeneracy invariant is COVERAGE, not occurrence count: a pattern
    (>= ``min_len`` tokens, >= ``min_count`` occurrences -- a small floor,
    overlaps allowed since tandem loops necessarily overlap) triggers only
    when the union of its occurrence intervals covers at least
    ``min_coverage`` of the region. Healthy text re-uses entities and
    quotes documents (many occurrences, low coverage: the aa-lcr false
    positives); a degenerate loop fills the region regardless of its
    period, so long-period paragraph loops trigger after 2-3 repetitions
    where any count threshold would need an impossibly large window.

    A pattern must also be "live": its most recent occurrence ends within
    ``tail_slack + pattern_len`` tokens of the end -- old repetition the
    model already escaped must not keep triggering.

    Returns the patterns as int arrays of token ids (deduplication is the
    caller's concern; patterns from distinct suffix-array runs may share
    tokens).
    """
    length = len(tokens)
    if min_count < 2 or length < min_len * 2:
        return []
    tokens = np.ascontiguousarray(tokens, dtype=np.int32)
    sa, lcp = _sa_lcp(tokens)
    good = lcp[: length - 1] >= min_len
    if not good.any():
        return []
    # Maximal runs of consecutive adjacent-suffix LCPs >= min_len: the run
    # good[s:e] groups suffixes sa[s..e] (e-s+1 of them) which all share a
    # prefix of length min(lcp[s:e]) >= min_len.
    padded = np.concatenate(([False], good, [False]))
    edges = np.flatnonzero(padded[1:] != padded[:-1])
    patterns: list[np.ndarray] = []
    for s, e in zip(edges[::2], edges[1::2]):
        count = e - s + 1
        if count < min_count:
            continue
        occ = np.sort(sa[s : e + 1])
        pattern_len = int(lcp[s:e].min())
        latest_end = int(occ[-1]) + pattern_len
        if latest_end < length - (tail_slack + pattern_len):
            continue
        # Union of the occurrence intervals [o, o+pattern_len): consecutive
        # occurrences closer than pattern_len overlap and only contribute
        # their gap.
        if occ.size > 1:
            union = int(np.minimum(np.diff(occ), pattern_len).sum()) + pattern_len
        else:
            union = pattern_len
        if union < min_coverage * length:
            continue
        first = int(occ[0])
        patterns.append(tokens[first : first + pattern_len].astype(np.int64))
    return patterns


class RepetitionGuardLogitsProcessor(LogitsProcessor):
    """Per-request repetition guard driven by SamplingParams
    ``repetition_detection.mode`` / ``.scope`` (see module docstring)."""

    def __init__(
        self, vllm_config: "VllmConfig", device: torch.device, is_pin_memory: bool
    ):
        self.device = device
        self.pin_memory = is_pin_memory

        reasoning_config = vllm_config.reasoning_config
        self.reasoning_enabled = (
            reasoning_config is not None and reasoning_config.enabled
        )
        self.start_ids: list[int] = list(
            getattr(reasoning_config, "reasoning_start_token_ids", None) or []
        )
        self.end_ids: list[int] = list(
            getattr(reasoning_config, "reasoning_end_token_ids", None) or []
        )

        # Under speculative decoding apply_with_spec_decode() (verify
        # positions) is the sole "truncate" enforcer; apply() (bonus token)
        # must not also force (same double-</think> hazard as
        # ThinkingTokenBudgetLogitsProcessor). The truncate force sequence is
        # safe on every path.
        self.has_spec_decode = vllm_config.speculative_config is not None

        self.model_config = vllm_config.model_config

        self._state: dict[int, dict[str, Any]] = {}

        env_defaults = {
            name: env_with_legacy(name)
            for name in (
                "VLLM_REP_MODE",
                "VLLM_REP_MAX",
                "VLLM_REP_COVERAGE",
                "VLLM_REP_WINDOW",
                "VLLM_REP_MIN_LEN",
                "VLLM_REP_CHECK_INTERVAL",
                "VLLM_REP_MIN_REGION",
            )
            if env_with_legacy(name)
        }
        if env_defaults:
            logger.info(
                "[rep-guard] env config %s (priority vllm_xargs > request "
                "field > env > default)",
                env_defaults,
            )

        # VLLM_REP_MODE: server-wide default activation (requests may
        # override or disable via rep_trunc_mode, incl. "off").
        env_mode_raw = env_with_legacy("VLLM_REP_MODE")
        env_mode = resolve_env_mode()
        if env_mode_raw and env_mode is None:
            logger.warning(
                "[rep-guard] VLLM_REP_MODE=%r is not a valid mode "
                "(truncate/1); server-wide activation is OFF.",
                env_mode_raw,
            )
        elif env_mode is not None:
            if not self.reasoning_enabled:
                logger.warning(
                    "[rep-guard] VLLM_REP_MODE=%s is set but reasoning is "
                    "NOT enabled (pass --reasoning-parser); the guard is "
                    "think-only, so the server-wide activation is a no-op.",
                    env_mode,
                )
            else:
                logger.info(
                    "[rep-guard] server-wide activation: mode=%s (think-only) "
                    "for every request (override per request via "
                    "vllm_xargs.rep_trunc_mode, 'off' disables)",
                    env_mode,
                )
        if env_with_legacy("VLLM_REP_SCOPE"):
            logger.warning(
                "[rep-guard] VLLM_REP_SCOPE is ignored: the guard is "
                "think-only and never touches the answer section."
            )

    def is_argmax_invariant(self) -> bool:
        """Masking repeated tokens / forcing the reasoning end changes
        the greedy argmax by design."""
        return False

    def _new_state(
        self,
        params: SamplingParams,
        prompt_tok_ids: list[int] | None,
        output_tok_ids: list[int],
    ) -> dict[str, Any] | None:
        try:
            config = resolve_rep_guard_request_config(params)
        except ValueError:
            # The input processor rejects malformed configs up front; fail
            # closed here regardless.
            return None
        if config is None:
            return None
        mode = config["mode"]
        if not self.reasoning_enabled:
            # Think-only guard: nothing to guard without a thinking section.
            # Request opt-ins are rejected by the input processor; the env
            # default degrades silently (ctor warned once).
            return None
        # Per-request force sequence for truncate, shared with think-budget:
        # vllm_xargs.think_budget_force_str > VLLM_THINK_BUDGET_FORCE_STR
        # env > bare reasoning end ids.
        force_end_ids: list[int] = []
        if mode == "truncate":
            try:
                force_str = resolve_force_str(params)
            except ValueError:
                force_str = None
            if force_str:
                force_end_ids = encode_force_str(force_str, self.model_config)
            else:
                force_end_ids = list(self.end_ids)

        # Recompute the think state over prompt + existing output so that
        # re-added requests (batch reordering, preemption resume) start
        # from the correct section.
        in_think = False
        think_start_len = 0
        prompt_think_tail: list[int] = []
        if self.reasoning_enabled:
            prompt_len = len(prompt_tok_ids or [])
            seq = list(prompt_tok_ids or []) + list(output_tok_ids)
            last_start = _find_last_sequence_index(seq, self.start_ids)
            last_end = _find_last_sequence_index(seq, self.end_ids)
            in_think = last_start > last_end
            if in_think:
                think_start_len = max(0, last_start + len(self.start_ids) - prompt_len)
                # When the current think section begins INSIDE the prompt
                # (chat template opens <think>, continuation/multi-turn),
                # keep its prompt-side tail: a loop that already filled the
                # earlier context must be caught within a few output tokens
                # instead of re-accumulating min_count occurrences (mirrors
                # think-budget counting prompt think tokens).
                think_content_start = last_start + len(self.start_ids)
                if think_content_start < prompt_len:
                    prompt_think_tail = seq[think_content_start:prompt_len][
                        -config["window"] :
                    ]

        return {
            "mode": mode,
            # Fully resolved by resolve_rep_guard_request_config
            # (vllm_xargs > request field > env > default).
            "min_count": config["min_count"],
            "min_len": config["min_len"],
            "window": config["window"],
            "interval": config["interval"],
            "coverage": config["coverage"],
            "min_region": config["min_region"],
            "force_end_ids": force_end_ids,
            "in_think": in_think,
            # Output-list index where the current think section's content
            # begins (0 when thinking started inside the prompt).
            "think_start_len": think_start_len,
            # Prompt-side tail of the current think section (window-bounded;
            # empty when thinking started in the output or not at all).
            "prompt_tail": np.asarray(prompt_think_tail, dtype=np.int32),
            "in_end": False,
            "end_count": 0,
            # Rows actually forced in the last apply pass; consumed by
            # update_state to advance end_count (placeholder-safe).
            "last_forced": 0,
            "out_ids": output_tok_ids,  # live reference (see BatchUpdate)
            "prev_len": len(output_tok_ids),
            "last_check": len(output_tok_ids),
        }

    def _advance_think_state(self, state: dict[str, Any]) -> None:
        """Fold newly committed tokens into the think-section tracking."""
        out = state["out_ids"]
        cur = len(out)
        prev = state["prev_len"]
        if cur <= prev:
            return
        state["prev_len"] = cur
        if not self.reasoning_enabled:
            return
        max_delim = max(len(self.start_ids), len(self.end_ids), 1)
        check_from = max(0, prev - max_delim + 1)
        recent = out[check_from:cur]
        pos_start = _find_last_sequence_index(recent, self.start_ids)
        pos_end = _find_last_sequence_index(recent, self.end_ids)
        if pos_start < 0 and pos_end < 0:
            return
        if pos_start > pos_end:
            state["in_think"] = True
            state["think_start_len"] = check_from + pos_start + len(self.start_ids)
        elif state["in_think"]:
            state["in_think"] = False

    def _maybe_detect(self, state: dict[str, Any]) -> None:
        out = state["out_ids"]
        cur = len(out)
        if cur - state["last_check"] < state["interval"]:
            return
        state["last_check"] = cur

        # THINK-ONLY: outside the thinking section the guard does nothing,
        # ever -- the answer section is off limits by design.
        if not state["in_think"]:
            return
        region_start = max(max(0, cur - state["window"]), state["think_start_len"])
        min_len = state["min_len"]
        region = np.asarray(out[region_start:cur], dtype=np.int32)
        # While still inside the prompt-initiated think section, prepend the
        # prompt-side think tail so loops continuing from the prompt trigger
        # without re-accumulating occurrences in the output.
        prompt_tail = state["prompt_tail"]
        if prompt_tail.size and state["think_start_len"] == 0:
            need = state["window"] - region.size
            if need > 0:
                region = np.concatenate([prompt_tail[-need:], region])
        # Async scheduling backfills the output list later; drop the -1
        # placeholders so they neither break pattern periodicity nor get
        # detected as a repeat themselves.
        region = region[region >= 0]
        # Below min_region the coverage threshold is meaningless (early
        # structural enumerations false-fire); wait for more context.
        if region.size < max(min_len * 2, state["min_region"]):
            return
        patterns = find_active_repeats(
            region,
            min_len=min_len,
            min_count=state["min_count"],
            tail_slack=state["interval"],
            min_coverage=state["coverage"],
        )
        if not patterns:
            return

        # truncate (think section only) -- force the reasoning end sequence
        state["in_end"] = True
        state["end_count"] = 0
        logger.info(
            "[rep-guard] repetition detected in think section -> "
            "forcing reasoning end %s",
            state["force_end_ids"],
        )

    def update_state(self, batch_update: BatchUpdate | None):
        process_dict_updates(self._state, batch_update, self._new_state)
        for state in self._state.values():
            if state["in_end"]:
                # Advance the force sequence by min(committed, forced-last-
                # step). Token-value matching is NOT safe here: under async
                # scheduling the output list holds -1 placeholders when
                # update_state runs (real ids are backfilled later), so a
                # value match stalls at 0 and force_ids[0] gets re-forced
                # forever. Counting is exact instead: rejection sampling
                # commits [accepted forced prefix][forced recovery token]
                # [+1 free bonus only when every row was accepted], so of
                # the `delta` committed tokens at most `last_forced` are
                # ours and any excess is exactly the bonus token.
                out = state["out_ids"]
                cur = len(out)
                prev = state["prev_len"]
                delta = cur - prev
                if delta > 0:
                    advance = min(delta, state["last_forced"])
                    state["last_forced"] = 0
                    if advance > 0:
                        state["end_count"] += advance
                        if state["end_count"] >= len(state["force_end_ids"]):
                            # Force sequence fully committed: the section is
                            # closed by construction (the sequence ends with
                            # </think>). Don't rely on the delimiter scan --
                            # under async scheduling it may only see
                            # placeholders and would miss the transition.
                            state["in_end"] = False
                            state["end_count"] = 0
                            state["in_think"] = False
                self._advance_think_state(state)
                continue
            self._advance_think_state(state)
            self._maybe_detect(state)

    def apply(self, logits: torch.Tensor) -> torch.Tensor:
        if not self._state:
            return logits
        num_rows = logits.shape[0]
        for i, state in self._state.items():
            if i >= num_rows:
                continue
            if state["in_end"] and state["force_end_ids"] and not self.has_spec_decode:
                # Same 1e9 convention as ThinkingTokenBudgetLogitsProcessor.
                logits[i, state["force_end_ids"][state["end_count"]]] = 1e9
                state["last_forced"] = 1
        return logits

    def apply_with_spec_decode(
        self,
        logits: torch.Tensor,
        num_draft_tokens: list[int],
    ) -> torch.Tensor:
        """Spec-decode version of apply() over the verify-position logits.

        ``logits`` has shape ``[sum(num_draft_tokens), V]``; rows are grouped
        per request via ``cumsum(num_draft_tokens)`` (the layout MinTokens
        and ThinkingTokenBudget use). The truncate force sequence is emitted
        row-by-row from the committed ``end_count`` (positions after a
        forced mismatch are discarded by rejection sampling anyway).
        """
        if not self._state:
            return logits
        num_draft_arr = np.asarray(num_draft_tokens, dtype=np.int64)
        cumsum = np.concatenate(([0], np.cumsum(num_draft_arr)))
        force_rows: list[int] = []
        force_toks: list[int] = []
        for i, state in self._state.items():
            if i >= len(num_draft_arr):
                continue
            num_rows = int(num_draft_arr[i])
            if num_rows <= 0:
                continue
            base = int(cumsum[i])
            if state["in_end"] and state["force_end_ids"]:
                force_ids = state["force_end_ids"]
                end_count = state["end_count"]
                forced = 0
                for j in range(num_rows):
                    if end_count >= len(force_ids):
                        break
                    force_rows.append(base + j)
                    force_toks.append(force_ids[end_count])
                    end_count += 1
                    forced += 1
                # Consumed by update_state to advance end_count by exactly
                # how many of the committed tokens were ours (see there).
                state["last_forced"] = forced
        if force_rows:
            rows_t = torch.from_numpy(np.asarray(force_rows, dtype=np.int64)).to(
                logits.device, non_blocking=True
            )
            toks_t = torch.from_numpy(np.asarray(force_toks, dtype=np.int64)).to(
                logits.device, non_blocking=True
            )
            logits[rows_t, toks_t] = 1e9
        return logits
