# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""tests for the logits-level repetition guard.

Pure CPU; the processor is constructed against a duck-typed VllmConfig and a
monkeypatched tokenizer so no model assets are needed.
"""

from types import SimpleNamespace

import numpy as np
import pytest
import torch

from vllm.sampling_params import RepetitionDetectionParams, SamplingParams
from vllm.v1.sample.logits_processor.interface import BatchUpdate
from vllm.v1.sample.logits_processor.repetition import (
    RepetitionGuardLogitsProcessor,
    find_active_repeats,
)

VOCAB = 200
THINK_START, THINK_END = 100, 101
EOS = 0


# ---------------------------------------------------------------------------
# Detector
# ---------------------------------------------------------------------------


def test_detects_tandem_loop():
    tokens = np.array([1, 2, 3] * 20, dtype=np.int32)
    patterns = find_active_repeats(
        tokens, min_len=5, min_count=3, tail_slack=8, min_coverage=0.5
    )
    assert patterns
    assert set(np.concatenate(patterns).tolist()) <= {1, 2, 3}


def test_scattered_phrase_is_below_coverage():
    # aa-lcr FP regression: a phrase re-used a few times between unique
    # text has LOW coverage and must not trigger at the default, even
    # though its occurrence count clears the floor.
    phrase = [7, 8, 9, 10, 11, 12]
    background = iter(range(20, 2000))
    tokens: list[int] = []
    for _ in range(3):
        tokens.extend(next(background) for _ in range(10))
        tokens.extend(phrase)
    arr = np.array(tokens, dtype=np.int32)
    assert not find_active_repeats(
        arr, min_len=5, min_count=3, tail_slack=8, min_coverage=0.5
    )
    # An explicitly lowered coverage knob still catches it.
    assert find_active_repeats(
        arr, min_len=5, min_count=3, tail_slack=8, min_coverage=0.3
    )


def test_scattered_entity_many_occurrences_not_detected():
    # 8-token entity quoted 16 times between unique prose: high count,
    # ~17% coverage -> healthy document-QA reasoning, no trigger.
    entity = list(range(10, 18))
    background = iter(range(1000, 9000))
    tokens: list[int] = []
    for _ in range(16):
        tokens.extend(next(background) for _ in range(40))
        tokens.extend(entity)
    arr = np.array(tokens, dtype=np.int32)
    assert not find_active_repeats(
        arr, min_len=5, min_count=3, tail_slack=16, min_coverage=0.5
    )


def test_long_period_loop_detected_with_small_floor():
    # A 40-token paragraph repeated 3 times dominates the region: count
    # thresholds like 16 could never fire (16 x 40 >> window), coverage
    # fires on the small floor.
    paragraph = list(range(500, 540))
    tokens = list(range(1, 21)) + paragraph * 3
    arr = np.array(tokens, dtype=np.int32)
    patterns = find_active_repeats(
        arr, min_len=5, min_count=3, tail_slack=16, min_coverage=0.5
    )
    assert patterns
    assert set(np.concatenate(patterns).tolist()) <= set(paragraph)


def test_no_repetition_no_patterns():
    tokens = np.arange(500, dtype=np.int32)
    assert not find_active_repeats(
        tokens, min_len=5, min_count=2, tail_slack=8, min_coverage=0.5
    )


def test_stale_repetition_is_not_live():
    # Phrase repeated early, then a long unique tail: the loop was escaped
    # and must not keep triggering.
    phrase = [7, 8, 9, 10, 11, 12]
    tokens = phrase * 4 + list(range(1000, 1200))
    arr = np.array(tokens, dtype=np.int32)
    assert not find_active_repeats(
        arr, min_len=5, min_count=3, tail_slack=8, min_coverage=0.5
    )


def test_min_count_floor():
    # Tandem tripling of a 6-token phrase: coverage ~1.0, so the floor is
    # the only remaining gate (a x2 tandem yields SA-run count 2 and stays
    # below the default floor of 3 -- a single immediate re-quote is not
    # yet treated as a loop).
    phrase = [7, 8, 9, 10, 11, 12]
    arr = np.array(phrase * 3, dtype=np.int32)
    assert find_active_repeats(
        arr, min_len=5, min_count=3, tail_slack=8, min_coverage=0.5
    )
    assert not find_active_repeats(
        arr, min_len=5, min_count=10, tail_slack=8, min_coverage=0.5
    )
    assert not find_active_repeats(
        np.array(phrase * 2, dtype=np.int32),
        min_len=5,
        min_count=3,
        tail_slack=8,
        min_coverage=0.5,
    )


def test_sa_lcp_convention():
    """pydivsufsort (hard dependency) must keep the kasai convention the
    detector assumes: lcp[i] = LCP(suffix sa[i], suffix sa[i+1]).
    'banana$'-style probe with the empirically verified expectation."""
    from vllm.v1.sample.logits_processor.repetition import _sa_lcp

    tokens = np.array([2, 1, 3, 1, 3, 1, 0], dtype=np.int32)  # b a n a n a $
    sa, lcp = _sa_lcp(tokens)
    assert sa.tolist() == [6, 5, 3, 1, 0, 4, 2]
    assert lcp.tolist()[:6] == [0, 1, 3, 0, 0, 2]


# ---------------------------------------------------------------------------
# Processor harness
# ---------------------------------------------------------------------------


class _FakeTokenizer:
    eos_token_id = EOS

    def encode(self, text, add_special_tokens=False):
        # Deterministic multi-token "graceful" force sequence ending with
        # the reasoning end token.
        return [150, 151, THINK_END]


def _fake_vllm_config(reasoning: bool, spec_decode: bool):
    reasoning_config = (
        SimpleNamespace(
            enabled=True,
            reasoning_start_token_ids=[THINK_START],
            reasoning_end_token_ids=[THINK_END],
        )
        if reasoning
        else None
    )
    return SimpleNamespace(
        reasoning_config=reasoning_config,
        speculative_config=object() if spec_decode else None,
        model_config=SimpleNamespace(),
    )


@pytest.fixture
def guard_env(monkeypatch):
    monkeypatch.setenv("VLLM_REP_MAX", "3")  # trigger at count >= 4
    monkeypatch.setenv("VLLM_REP_WINDOW", "256")
    monkeypatch.setenv("VLLM_REP_MIN_LEN", "5")
    monkeypatch.setenv("VLLM_REP_CHECK_INTERVAL", "8")
    # Unit tests drive tiny outputs; keep the early-detection gate low so
    # small synthetic loops still trigger (default 512 is asserted in
    # test_no_env_activation_with_defaults / min-region tests).
    monkeypatch.setenv("VLLM_REP_MIN_REGION", "16")
    monkeypatch.delenv("VLLM_REP_MODE", raising=False)
    monkeypatch.delenv("VLLM_REP_SCOPE", raising=False)
    monkeypatch.delenv("VLLM_REP_COVERAGE", raising=False)
    monkeypatch.delenv("VLLM_THINK_BUDGET_FORCE_STR", raising=False)
    monkeypatch.setattr(
        "vllm.tokenizers.cached_tokenizer_from_config",
        lambda model_config: _FakeTokenizer(),
    )


def _make_guard(reasoning=True, spec_decode=False):
    return RepetitionGuardLogitsProcessor(
        _fake_vllm_config(reasoning, spec_decode), torch.device("cpu"), False
    )


def _add_request(
    guard,
    index,
    mode,
    scope=None,
    prompt=None,
    out=None,
):
    params = SamplingParams(
        repetition_detection=RepetitionDetectionParams(mode=mode, scope=scope)
    )
    out = out if out is not None else []
    guard.update_state(
        BatchUpdate(
            batch_size=index + 1,
            removed=[],
            added=[(index, params, prompt or [], out)],
            moved=[],
        )
    )
    return out


def _step(guard, out, tokens):
    out.extend(tokens)
    guard.update_state(None)


def _drive_loop(guard, out, phrase, steps):
    for _ in range(steps):
        _step(guard, out, phrase)


def _drive_until_in_end(guard, out, phrase, index=0, max_steps=30):
    """Feed the loop until the truncate force engages (like the real system,
    where forcing starts the moment detection fires)."""
    for _ in range(max_steps):
        _step(guard, out, phrase)
        if guard._state[index]["in_end"]:
            return True
    return False


def test_truncate_in_think_forces_end_sequence(guard_env):
    guard = _make_guard()
    out = _add_request(guard, 0, "truncate", prompt=[5, THINK_START])
    assert _drive_until_in_end(guard, out, [7, 8, 9])
    state = guard._state[0]

    # Force step: bare </think> (no VLLM_THINK_BUDGET_FORCE_STR set).
    logits = torch.zeros(1, VOCAB)
    guard.apply(logits)
    assert logits[0, THINK_END] == 1e9

    # The forced token commits; the guard leaves the think section and
    # becomes dormant (scope="think").
    _step(guard, out, [THINK_END])
    state = guard._state[0]
    assert not state["in_end"]
    assert not state["in_think"]
    _drive_loop(guard, out, [7, 8, 9], 12)
    assert not guard._state[0]["in_end"]


def test_truncate_uses_force_str(guard_env, monkeypatch):
    monkeypatch.setenv("VLLM_THINK_BUDGET_FORCE_STR", "...</think>")
    guard = _make_guard()
    out = _add_request(guard, 0, "truncate", prompt=[THINK_START])
    force_end_ids = guard._state[0]["force_end_ids"]
    assert force_end_ids == [150, 151, THINK_END]
    assert _drive_until_in_end(guard, out, [7, 8, 9])
    for expected in force_end_ids:
        logits = torch.zeros(1, VOCAB)
        guard.apply(logits)
        assert logits[0, expected] == 1e9
        _step(guard, out, [expected])
    assert not guard._state[0]["in_end"]
    assert not guard._state[0]["in_think"]


def test_post_think_never_touched(guard_env):
    # THINK-ONLY guarantee: once the think section closed, the guard
    # never intervenes, no matter how degenerate the answer section gets.
    guard = _make_guard()
    out = _add_request(guard, 0, "truncate", prompt=[THINK_START])
    _step(guard, out, [THINK_END])
    _drive_loop(guard, out, [7, 8, 9], 30)
    state = guard._state[0]
    assert not state["in_think"]
    assert not state["in_end"]


def test_without_reasoning_is_inert(guard_env):
    # Think-only guard on a server without reasoning: nothing to guard.
    guard = _make_guard(reasoning=False)
    _add_request(guard, 0, "truncate")
    assert 0 not in guard._state


def test_loop_with_eos_in_pattern_still_truncates(guard_env):
    guard = _make_guard()
    out = _add_request(guard, 0, "truncate", prompt=[THINK_START])
    # In-think loop whose pattern includes EOS (not a think delimiter, so
    # the section stays open); truncate fires just like any other loop.
    _drive_loop(guard, out, [7, 8, EOS, 9, 10], 12)
    state = guard._state[0]
    assert state["in_think"]
    assert state["in_end"]


def test_force_sequence_released_after_loop_escapes(guard_env):
    guard = _make_guard()
    out = _add_request(guard, 0, "truncate", prompt=[THINK_START])
    _drive_loop(guard, out, [7, 8, 9], 12)
    assert guard._state[0]["in_end"]
    # The force sequence is emitted by apply() (which sets last_forced);
    # once the forced </think> commits, the guard leaves the think section
    # and becomes dormant.
    logits = torch.zeros(1, VOCAB)
    guard.apply(logits)
    assert logits[0, THINK_END] == 1e9
    _step(guard, out, [THINK_END])
    assert not guard._state[0]["in_end"]
    assert not guard._state[0]["in_think"]


def test_interval_gating(guard_env):
    guard = _make_guard()
    out = _add_request(guard, 0, "truncate", prompt=[THINK_START])
    # 6 tokens < interval(8): no check yet even though the loop is obvious.
    _step(guard, out, [7, 8, 9, 7, 8, 9])
    assert guard._state[0]["in_end"] is False


def test_no_env_activation_with_defaults(monkeypatch):
    # No env vars at all: a request opting in via mode still works, with the
    # tail-of-chain defaults. Without any opt-in nothing applies.
    for name in (
        "VLLM_REP_MAX",
        "VLLM_REP_COVERAGE",
        "VLLM_REP_MIN_REGION",
        "VLLM_REP_WINDOW",
        "VLLM_REP_MIN_LEN",
        "VLLM_REP_CHECK_INTERVAL",
        "VLLM_THINK_BUDGET_FORCE_STR",
    ):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setattr(
        "vllm.tokenizers.cached_tokenizer_from_config",
        lambda model_config: _FakeTokenizer(),
    )
    guard = _make_guard()
    _add_request(guard, 0, "truncate", prompt=[THINK_START])
    state = guard._state[0]
    assert state["min_count"] == 3  # occurrence floor
    assert state["min_len"] == 5
    assert state["window"] == 8192
    assert state["interval"] == 32
    assert state["coverage"] == 0.5
    assert state["min_region"] == 512

    # No mode anywhere -> no state, regardless of other knobs being set.
    params = SamplingParams(
        repetition_detection=RepetitionDetectionParams(min_count=3),
        extra_args={"rep_min_count": 2},
    )
    guard.update_state(
        BatchUpdate(batch_size=2, removed=[], added=[(1, params, [], [])], moved=[])
    )
    assert 1 not in guard._state


def test_priority_xargs_over_typed_over_env(guard_env):
    # guard_env sets VLLM_REP_MAX=3 (env-derived min_count 4).
    guard = _make_guard()
    params = SamplingParams(
        repetition_detection=RepetitionDetectionParams(mode="truncate", min_count=3),
        extra_args={"rep_min_count": 2, "rep_check_interval": 4, "rep_window": 64},
    )
    out: list[int] = []
    guard.update_state(
        BatchUpdate(
            batch_size=1,
            removed=[],
            added=[(0, params, [THINK_START], out)],
            moved=[],
        )
    )
    state = guard._state[0]
    assert state["min_count"] == 2  # xargs beats typed field (3) and env (4)
    assert state["interval"] == 4  # xargs beats env (8)
    assert state["window"] == 64  # xargs beats env (256)

    # typed field beats env when xargs is silent.
    params2 = SamplingParams(
        repetition_detection=RepetitionDetectionParams(mode="truncate", min_count=3)
    )
    guard.update_state(
        BatchUpdate(
            batch_size=2, removed=[], added=[(1, params2, [THINK_START], [])], moved=[]
        )
    )
    assert guard._state[1]["min_count"] == 3


def test_request_min_count_override(guard_env):
    guard = _make_guard()
    params = SamplingParams(
        repetition_detection=RepetitionDetectionParams(mode="truncate", min_count=2)
    )
    out: list[int] = []
    guard.update_state(
        BatchUpdate(
            batch_size=1,
            removed=[],
            added=[(0, params, [THINK_START], out)],
            moved=[],
        )
    )
    assert guard._state[0]["min_count"] == 2
    # Server default (VLLM_REP_MAX=3 -> exceeds -> 4) applies otherwise.
    out2 = _add_request(guard, 1, "truncate", prompt=[THINK_START])
    assert guard._state[1]["min_count"] == 4
    del out2


# ---------------------------------------------------------------------------
# Spec decode (MTP) path
# ---------------------------------------------------------------------------


def test_spec_decode_truncate_forces_first_draft_row(guard_env):
    guard = _make_guard(spec_decode=True)
    out0 = _add_request(guard, 0, "truncate", prompt=[THINK_START])
    _add_request(guard, 1, "truncate", prompt=[THINK_START])
    _drive_loop(guard, out0, [7, 8, 9], 12)
    guard.update_state(None)
    assert guard._state[0]["in_end"]
    assert guard._state[1]["in_end"] is False

    # Request 0 owns rows 0-2, request 1 rows 3-4. The bare </think> force
    # sequence is a single token, emitted row-by-row from end_count, so only
    # the first draft row of request 0 is forced; request 1 is untouched.
    logits = torch.zeros(5, VOCAB)
    guard.apply_with_spec_decode(logits, [3, 2])
    assert guard._state[0]["force_end_ids"] == [THINK_END]
    assert logits[0, THINK_END] == 1e9
    assert (logits[1:] == 0).all()


def test_spec_decode_truncate_replays_force_sequence(guard_env, monkeypatch):
    monkeypatch.setenv("VLLM_THINK_BUDGET_FORCE_STR", "...</think>")
    guard = _make_guard(spec_decode=True)
    out = _add_request(guard, 0, "truncate", prompt=[THINK_START])
    assert _drive_until_in_end(guard, out, [7, 8, 9])

    force_end_ids = guard._state[0]["force_end_ids"]
    logits = torch.zeros(4, VOCAB)
    guard.apply_with_spec_decode(logits, [4])
    # Rows 0..2 carry the 3-token force sequence; row 3 has nothing left.
    for row, tok in enumerate(force_end_ids):
        assert logits[row, tok] == 1e9
    assert (logits[3] == 0).all()

    # Bonus-token apply() must NOT force under spec decode.
    bonus = torch.zeros(1, VOCAB)
    guard.apply(bonus)
    assert (bonus == 0).all()

    # Two forced tokens accepted in one step -> end_count advances by 2
    # (advance = min(delta=2, forced=3)).
    _step(guard, out, force_end_ids[:2])
    assert guard._state[0]["end_count"] == 2
    # Next verify pass forces the remaining </think>, which then commits.
    guard.apply_with_spec_decode(torch.zeros(1, VOCAB), [1])
    _step(guard, out, force_end_ids[2:])
    assert not guard._state[0]["in_end"]
    assert not guard._state[0]["in_think"]


def test_spec_decode_bonus_token_does_not_break_force_sequence(guard_env, monkeypatch):
    """Regression: under spec decode the free bonus token can interleave an
    arbitrary token mid-force-sequence. end_count must advance by matching
    the expected sequence, not by counting commits, so the terminal
    </think> still gets forced."""
    monkeypatch.setenv("VLLM_THINK_BUDGET_FORCE_STR", "...</think>")
    guard = _make_guard(spec_decode=True)
    out = _add_request(guard, 0, "truncate", prompt=[THINK_START])
    assert _drive_until_in_end(guard, out, [7, 8, 9])

    # Verify pass forces 2 rows (f0, f1); both drafts accepted, so the free
    # bonus token (42) also commits. advance = min(delta=3, forced=2) = 2.
    guard.apply_with_spec_decode(torch.zeros(2, VOCAB), [2])
    _step(guard, out, [*guard._state[0]["force_end_ids"][:2], 42])
    state = guard._state[0]
    assert state["in_end"]
    assert state["end_count"] == 2  # the bonus junk did not count

    # The remaining </think> is still forced at the next verify positions.
    logits = torch.zeros(2, VOCAB)
    guard.apply_with_spec_decode(logits, [2])
    assert logits[0, THINK_END] == 1e9
    _step(guard, out, [THINK_END])
    assert not guard._state[0]["in_end"]
    assert not guard._state[0]["in_think"]


def test_force_sequence_progresses_despite_async_placeholders(guard_env, monkeypatch):
    """Regression for the live-server stall: under async scheduling the
    output list contains -1 placeholders when update_state runs (real ids
    are backfilled later). Token-value matching stalled end_count at 0 and
    re-forced force_ids[0] forever (the 'I'/'Z?' flood). The count-based
    advance (min(committed, last_forced)) must progress regardless of the
    committed token VALUES."""
    monkeypatch.setenv("VLLM_THINK_BUDGET_FORCE_STR", "...</think>")
    guard = _make_guard(spec_decode=True)
    out = _add_request(guard, 0, "truncate", prompt=[THINK_START])
    assert _drive_until_in_end(guard, out, [7, 8, 9])
    force_len = len(guard._state[0]["force_end_ids"])
    assert force_len == 3

    # Each step: verify pass forces 1 row (num_spec=1), then the step
    # commits as placeholders (-1) -- values never match force ids.
    for expected_count in (1, 2):
        guard.apply_with_spec_decode(torch.zeros(1, VOCAB), [1])
        _step(guard, out, [-1])
        assert guard._state[0]["end_count"] == expected_count
        assert guard._state[0]["in_end"]
    guard.apply_with_spec_decode(torch.zeros(1, VOCAB), [1])
    _step(guard, out, [-1])
    assert not guard._state[0]["in_end"]
    # The forced sequence ends with </think>: the guard must close the
    # think section itself (the delimiter scan only saw placeholders).
    assert not guard._state[0]["in_think"]


def test_placeholders_are_not_detected_or_forced(guard_env):
    """-1 placeholders must neither trigger detection as a 'repeat' nor
    end up in the force sequence (logits[:, -1] would mask the last
    vocab entry)."""
    guard = _make_guard()
    out = _add_request(guard, 0, "truncate", prompt=[THINK_START])
    # A flood of placeholders alone must not trigger.
    _drive_loop(guard, out, [-1, -1, -1], 12)
    assert guard._state[0]["in_end"] is False
    # Real loop mixed with trailing placeholders: detected, but -1 never
    # enters the force sequence.
    _drive_loop(guard, out, [7, 8, 9, -1], 12)
    assert guard._state[0]["in_end"]
    assert guard._state[0]["force_end_ids"][-1] >= 0


# ---------------------------------------------------------------------------
# vllm_xargs surface
# ---------------------------------------------------------------------------


def _add_request_xargs(guard, index, xargs, prompt=None):
    params = SamplingParams(extra_args=xargs)
    out: list[int] = []
    guard.update_state(
        BatchUpdate(
            batch_size=index + 1,
            removed=[],
            added=[(index, params, prompt or [], out)],
            moved=[],
        )
    )
    return out


def test_xargs_truncate_drives_force_sequence(guard_env):
    guard = _make_guard()
    out = _add_request_xargs(
        guard,
        0,
        {"rep_trunc_mode": "truncate", "rep_min_count": 2},
        prompt=[THINK_START],
    )
    assert guard._state[0]["min_count"] == 2  # xargs override, not env default
    _drive_loop(guard, out, [7, 8, 9], 12)
    assert guard._state[0]["in_end"]


def test_scope_all_is_rejected(guard_env):
    # The guard is think-only: any non-"think" scope is refused on every
    # surface, and "think" itself is accepted as a no-op.
    from vllm.v1.sample.logits_processor.repetition import (
        resolve_rep_guard_request_config,
    )

    with pytest.raises(ValueError):
        resolve_rep_guard_request_config(
            SamplingParams(
                extra_args={"rep_trunc_mode": "truncate", "rep_trunc_scope": "all"}
            )
        )
    with pytest.raises((ValueError, TypeError)):
        RepetitionDetectionParams(scope="all")
    config = resolve_rep_guard_request_config(
        SamplingParams(
            extra_args={"rep_trunc_mode": "truncate", "rep_trunc_scope": "think"}
        )
    )
    assert config["mode"] == "truncate"


def test_xargs_wins_over_typed_field(guard_env):
    # priority chain: vllm_xargs > typed request field.
    guard = _make_guard()
    params = SamplingParams(
        repetition_detection=RepetitionDetectionParams(mode="truncate"),
        extra_args={"rep_trunc_mode": "truncate"},
    )
    out: list[int] = []
    guard.update_state(
        BatchUpdate(
            batch_size=1,
            removed=[],
            added=[(0, params, [THINK_START], out)],
            moved=[],
        )
    )
    assert guard._state[0]["mode"] == "truncate"


def test_malformed_xargs_fail_closed(guard_env):
    # The input processor 400s these before admission; the processor itself
    # must fail closed if one slips through.
    guard = _make_guard()
    _add_request_xargs(guard, 0, {"rep_trunc_mode": "bogus"}, prompt=[THINK_START])
    assert 0 not in guard._state


def test_resolver_validates_xargs():
    from vllm.v1.sample.logits_processor.repetition import (
        resolve_rep_guard_request_config,
    )

    resolve = resolve_rep_guard_request_config
    assert resolve(SamplingParams()) is None
    # String ints coerce (vllm_xargs values may arrive as strings).
    config = resolve(
        SamplingParams(extra_args={"rep_trunc_mode": "truncate", "rep_min_count": "3"})
    )
    assert config["min_count"] == 3
    with pytest.raises(ValueError):
        resolve(SamplingParams(extra_args={"rep_trunc_mode": "trancate"}))
    with pytest.raises(ValueError):
        resolve(SamplingParams(extra_args={"rep_trunc_scope": "all"}))
    with pytest.raises(ValueError):
        resolve(
            SamplingParams(
                extra_args={"rep_trunc_mode": "truncate", "rep_min_count": 1}
            )
        )
    with pytest.raises(ValueError):
        resolve(
            SamplingParams(
                extra_args={"rep_trunc_mode": "truncate", "rep_min_count": "abc"}
            )
        )
    with pytest.raises(ValueError):
        # Scheduler-level hard stop and the guard stay mutually exclusive
        # across surfaces.
        resolve(
            SamplingParams(
                repetition_detection=RepetitionDetectionParams(
                    max_pattern_size=8, min_count=3
                ),
                extra_args={"rep_trunc_mode": "truncate"},
            )
        )
    # Coverage knob: xargs > env > default, validated.
    config = resolve(
        SamplingParams(extra_args={"rep_trunc_mode": "truncate", "rep_coverage": 0.3})
    )
    assert config["coverage"] == 0.3
    for bad in (1.5, 0, "abc"):
        with pytest.raises(ValueError):
            resolve(
                SamplingParams(
                    extra_args={"rep_trunc_mode": "truncate", "rep_coverage": bad}
                )
            )


def test_min_region_gates_early_detection(guard_env, monkeypatch):
    """aa-lcr id=213 regression: with the default min_region (512), a
    structural enumeration in the first ~hundred think tokens must not
    trigger; the same loop does trigger once the region is large enough."""
    monkeypatch.setenv("VLLM_REP_MIN_REGION", "512")
    monkeypatch.setenv("VLLM_REP_WINDOW", "1024")  # fixture(256) < 512 방지
    guard = _make_guard()
    out = _add_request(guard, 0, "truncate", prompt=[THINK_START])
    _drive_loop(guard, out, [7, 8, 9], 40)  # 120 tokens of pure loop
    assert guard._state[0]["in_end"] is False  # region < 512: gated
    _drive_loop(guard, out, [7, 8, 9], 140)  # ~540 tokens total
    assert guard._state[0]["in_end"]  # gate lifted -> detected


def test_min_region_xargs_override(guard_env):
    guard = _make_guard()
    _add_request_xargs(
        guard,
        0,
        {"rep_trunc_mode": "truncate", "rep_min_region": 640},
        prompt=[THINK_START],
    )
    assert guard._state[0]["min_region"] == 640


def test_coverage_env_chain(guard_env, monkeypatch):
    from vllm.v1.sample.logits_processor.repetition import (
        resolve_rep_guard_request_config,
    )

    monkeypatch.setenv("VLLM_REP_COVERAGE", "0.7")
    config = resolve_rep_guard_request_config(
        SamplingParams(extra_args={"rep_trunc_mode": "truncate"})
    )
    assert config["coverage"] == 0.7  # env beats default
    config = resolve_rep_guard_request_config(
        SamplingParams(extra_args={"rep_trunc_mode": "truncate", "rep_coverage": 0.2})
    )
    assert config["coverage"] == 0.2  # xargs beats env


# ---------------------------------------------------------------------------
# VLLM_REP_MODE: server-wide env activation
# ---------------------------------------------------------------------------


def _add_bare_request(guard, index=0, prompt=None, params=None):
    out: list[int] = []
    guard.update_state(
        BatchUpdate(
            batch_size=index + 1,
            removed=[],
            added=[(index, params or SamplingParams(), prompt or [], out)],
            moved=[],
        )
    )
    return out


def test_env_mode_activates_bare_requests(guard_env, monkeypatch):
    monkeypatch.setenv("VLLM_REP_MODE", "1")  # boolean alias -> truncate
    guard = _make_guard()
    _add_bare_request(guard, prompt=[THINK_START])
    state = guard._state[0]
    assert state["mode"] == "truncate"  # env default mode
    assert state["force_end_ids"] == [THINK_END]

    # VLLM_REP_SCOPE is ignored (think-only guard); explicit env mode
    # still resolves.
    monkeypatch.setenv("VLLM_REP_MODE", "truncate")
    monkeypatch.setenv("VLLM_REP_SCOPE", "all")
    guard2 = _make_guard()
    _add_bare_request(guard2, prompt=[THINK_START])
    assert guard2._state[0]["mode"] == "truncate"
    assert "scope" not in guard2._state[0]


def test_env_mode_request_overrides_and_off(guard_env, monkeypatch):
    monkeypatch.setenv("VLLM_REP_MODE", "truncate")
    guard = _make_guard()
    # xargs beats env.
    _add_request_xargs(guard, 0, {"rep_trunc_mode": "truncate"}, prompt=[THINK_START])
    assert guard._state[0]["mode"] == "truncate"
    # "off" disables the server-wide default for this request.
    _add_request_xargs(guard, 1, {"rep_trunc_mode": "off"}, prompt=[THINK_START])
    assert 1 not in guard._state


def test_env_mode_invalid_is_ignored(guard_env, monkeypatch):
    monkeypatch.setenv("VLLM_REP_MODE", "bogus")
    guard = _make_guard()
    _add_bare_request(guard, prompt=[THINK_START])
    assert 0 not in guard._state


def test_env_mode_yields_to_scheduler_detection(guard_env, monkeypatch):
    from vllm.v1.sample.logits_processor.repetition import (
        resolve_rep_guard_request_config,
    )

    monkeypatch.setenv("VLLM_REP_MODE", "truncate")
    params = SamplingParams(
        repetition_detection=RepetitionDetectionParams(max_pattern_size=8, min_count=3)
    )
    # Request armed the scheduler hard stop: env guard yields, no error.
    assert resolve_rep_guard_request_config(params) is None


def test_env_mode_without_reasoning_is_inert(guard_env, monkeypatch):
    monkeypatch.setenv("VLLM_REP_MODE", "truncate")
    guard = _make_guard(reasoning=False)
    _add_bare_request(guard)
    assert 0 not in guard._state


# ---------------------------------------------------------------------------
# Prompt think-tail (loops continuing from the prompt)
# ---------------------------------------------------------------------------


def test_prompt_think_tail_speeds_up_detection(guard_env):
    loop = [7, 8, 9]
    # Think section starts (and already loops) inside the prompt.
    guard = _make_guard()
    out = _add_request(guard, 0, "truncate", prompt=[THINK_START, *loop * 12])
    assert guard._state[0]["prompt_tail"].size == 36
    _drive_loop(guard, out, loop, 3)  # 9 output tokens -> first check
    assert guard._state[0]["in_end"], "prompt-side loop must trigger quickly"

    # Control: same 9 output tokens without the prompt-side loop history
    # are not enough occurrences on their own.
    guard2 = _make_guard()
    out2 = _add_request(guard2, 0, "truncate", prompt=[THINK_START])
    _drive_loop(guard2, out2, loop, 3)
    assert not guard2._state[0]["in_end"]


def test_prompt_tail_not_used_after_new_think_section(guard_env):
    loop = [7, 8, 9]
    guard = _make_guard()
    out = _add_request(guard, 0, "truncate", prompt=[THINK_START, *loop * 12])
    # Leave the prompt-initiated think section, then re-enter a new one.
    _step(guard, out, [THINK_END, 50, THINK_START])
    state = guard._state[0]
    assert state["in_think"] and state["think_start_len"] > 0
    # 9 output tokens in the NEW section: the prompt tail must not count.
    _drive_loop(guard, out, loop, 3)
    assert guard._state[0]["in_end"] is False


# ---------------------------------------------------------------------------
# Param validation
# ---------------------------------------------------------------------------


def test_scope_requires_mode():
    # "all" is no longer a valid Literal; "think" alone still requires mode.
    with pytest.raises((ValueError, TypeError)):
        RepetitionDetectionParams(scope="all")
    with pytest.raises(ValueError):
        RepetitionDetectionParams(scope="think")


def test_mode_excludes_scheduler_detection():
    with pytest.raises(ValueError):
        RepetitionDetectionParams(mode="truncate", max_pattern_size=8, min_count=3)


def test_mode_min_count_one_rejected():
    with pytest.raises(ValueError):
        RepetitionDetectionParams(mode="truncate", min_count=1)


def test_invalid_mode_value():
    # pydantic dataclass validates the Literal on construction.
    with pytest.raises((ValueError, TypeError)):
        RepetitionDetectionParams(mode="explode")


def test_scheduler_detection_still_validates():
    with pytest.raises(ValueError):
        RepetitionDetectionParams(max_pattern_size=8, min_count=1)
    params = RepetitionDetectionParams(max_pattern_size=8, min_count=3)
    assert params.mode is None
