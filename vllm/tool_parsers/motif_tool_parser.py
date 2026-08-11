# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Lenient Hermes-style tool call parser for Motif models.

Motif emits Hermes-format tool calls (``<tool_call>{json}</tool_call>``) but
frequently produces malformed JSON inside the tags. The stock Hermes parser
drops the *entire* turn when any block fails to parse (HTTP 200 with
``tool_calls=[]`` and the raw ``<tool_call>`` text leaked into ``content``),
which agent harnesses cannot distinguish from a final text answer.

This parser repairs each ``<tool_call>`` block independently with a ladder of
deterministic syntax-level fixes, then delegates to the stock Hermes parser.
Already-valid blocks pass through with identical semantics, and blocks that
remain unparseable are left untouched (stock behavior, no regression).

Streaming reuses the Hermes incremental parser: tool names stream as soon as
they appear, argument streaming is held back while a block is still open
(repairs are not append-only, so raw fragments streamed early could
contradict the repaired result), and each block is repaired once it
completes. A trailing block whose ``</tool_call>`` never arrives (e.g.
length-capped output) is only recovered on the non-streaming path.

"""

import json
import re
from collections.abc import Callable

from vllm.entrypoints.openai.chat_completion.protocol import (
    ChatCompletionRequest,
)
from vllm.entrypoints.openai.engine.protocol import (
    ExtractedToolCallInformation,
)
from vllm.tokenizers import TokenizerLike
from vllm.tool_parsers.abstract_tool_parser import Tool, ToolParserManager
from vllm.tool_parsers.hermes_tool_parser import Hermes2ProToolParser

_TOOL_CALL_START = "<tool_call>"
_TOOL_CALL_END = "</tool_call>"

_TOOL_CALL_BLOCK_RE = re.compile(r"<tool_call>\s*(.*?)\s*</tool_call>", re.DOTALL)

# Valid JSON escapes: \" \\ \/ \b \f \n \r \t \uXXXX. Any other backslash
# (e.g. shell ``\&`` ``\$``, regex ``\s`` ``\[``) is invalid JSON and gets
# dropped. The alternation consumes escapes left to right so the trailing
# backslash of a valid ``\\`` pair is never re-read as the start of the next
# escape: a lookahead-only sub turned ``\\[`` (escaped backslash, then ``[``)
# into the invalid ``\[`` by dropping its second backslash.
_ESCAPE_OR_LONE_BACKSLASH_RE = re.compile(r'(\\["\\/bfnrtu])|\\')

# Keys whose values are ``list[str]`` in the Motif tool schemas. Scoped to
# known keys on purpose: blindly wrapping any string value in ``[`` would
# let the bracket balancer "fix" unrelated breakage into wrong JSON.
_ARRAY_VALUE_KEY_RE = re.compile(r'"(queries|urls)"(\s*:\s*)"')


def _normalize_invalid_escapes(text: str) -> str:
    """Drop backslashes that do not form a valid JSON escape."""
    return _ESCAPE_OR_LONE_BACKSLASH_RE.sub(lambda m: m.group(1) or "", text)


def _coerce_arguments_wrapper(obj: dict) -> dict:
    """Wrap flat calls: ``{"name": .., "x": ..}`` -> ``{"name", "arguments"}``."""
    if isinstance(obj, dict) and "name" in obj and "arguments" not in obj:
        args = {k: v for k, v in obj.items() if k != "name"}
        return {"name": obj["name"], "arguments": args}
    return obj


def _escape_quotes_in_strings(block: str) -> str:
    """R-quote: escape unescaped quotes inside string values.

    A ``"`` inside a string is treated as a closing quote only when the next
    non-whitespace character is a JSON structural character (``,]}:``) or
    end of input; otherwise it is content and becomes ``\\"``.
    """
    out: list[str] = []
    in_str = False
    i, n = 0, len(block)
    while i < n:
        ch = block[i]
        if not in_str:
            out.append(ch)
            if ch == '"':
                in_str = True
            i += 1
            continue
        if ch == "\\" and i + 1 < n:
            out.append(ch)
            out.append(block[i + 1])
            i += 2
            continue
        if ch == '"':
            j = i + 1
            while j < n and block[j] in " \t\r\n":
                j += 1
            next_ch = block[j] if j < n else ""
            if next_ch in ",]}:" or j >= n:
                in_str = False
                out.append(ch)
            else:
                out.append('\\"')
            i += 1
            continue
        out.append(ch)
        i += 1
    return "".join(out)


def _quote_repair_candidates(block: str, budget: int = 64):
    """R-backtrack: enumerate close-vs-content interpretations of quotes.

    :func:`_escape_quotes_in_strings` decides locally: an unescaped ``"``
    inside a string closes it iff the next non-whitespace character is a JSON
    structural character (``,]}:``). That rule is fooled by string content
    that *looks like* JSON (e.g. a comment ``{"@type":"MusicRecording"}``
    inside a ``cmd``), where a mid-content quote is followed by ``:``.

    Such quotes are genuinely ambiguous, so treat each as a choice point and
    DFS over interpretations, yielding fully rendered candidate blocks. The
    close interpretation is explored first, so the first candidate equals the
    output of :func:`_escape_quotes_in_strings`. The caller accepts the first
    candidate that parses (and passes the schema oracle); ``budget`` bounds
    the number of leaves, keeping the worst case (2^choice_points) small.
    """
    n = len(block)
    yielded = 0
    # frame: (index, in_str, rendered-so-far)
    stack: list[tuple[int, bool, str]] = [(0, False, "")]
    while stack and yielded < budget:
        i, in_str, acc = stack.pop()
        buf: list[str] = []
        branched = False
        while i < n:
            ch = block[i]
            if not in_str:
                buf.append(ch)
                if ch == '"':
                    in_str = True
                i += 1
                continue
            if ch == "\\" and i + 1 < n:
                buf.append(ch)
                buf.append(block[i + 1])
                i += 2
                continue
            if ch == '"':
                j = i + 1
                while j < n and block[j] in " \t\r\n":
                    j += 1
                if j >= n:  # end of input: the string must terminate here
                    in_str = False
                    buf.append(ch)
                    i += 1
                    continue
                if block[j] in ",]}:":
                    # Ambiguous: branch. LIFO, so push content-escape first
                    # and the close interpretation (stock heuristic) pops
                    # first.
                    base = acc + "".join(buf)
                    stack.append((i + 1, True, base + '\\"'))
                    stack.append((i + 1, False, base + '"'))
                    branched = True
                    break
                buf.append('\\"')  # mid-content quote: forced escape
                i += 1
                continue
            buf.append(ch)
            i += 1
        if not branched:
            yielded += 1
            yield acc + "".join(buf)


def _tool_arg_specs(tools) -> dict[str, tuple[frozenset[str] | None, bool]]:
    """Extract ``name -> (property names, additionalProperties)`` from tools.

    Tolerates ChatCompletionToolsParam / FunctionTool objects and plain
    dicts (with or without the ``{"function": ...}`` wrapper).
    """
    specs: dict[str, tuple[frozenset[str] | None, bool]] = {}
    for tool in tools or []:
        fn = tool.get("function") if isinstance(tool, dict) else getattr(tool, "function", None)
        src = tool if fn is None else fn
        if isinstance(src, dict):
            name, params = src.get("name"), src.get("parameters")
        else:
            name, params = getattr(src, "name", None), getattr(src, "parameters", None)
        if not name:
            continue
        props: frozenset[str] | None = None
        additional = True
        if isinstance(params, dict):
            if isinstance(params.get("properties"), dict):
                props = frozenset(params["properties"])
            additional = bool(params.get("additionalProperties", True))
        specs[name] = (props, additional)
    return specs


def _schema_oracle(tools):
    """Acceptance predicate for backtracking candidates.

    A repaired candidate is only trusted when its tool name is registered
    and (for closed schemas) its argument keys are a subset of the declared
    properties — a wrong close interpretation that happens to parse tends to
    invent argument keys out of string content (e.g. ``"@type"``), which no
    registered schema declares. Without tool information every parsed
    candidate is accepted (matching the schema-less ladder rungs).
    """
    specs = _tool_arg_specs(tools)
    if not specs:
        return lambda obj: True

    def accept(obj: dict) -> bool:
        name = obj.get("name")
        if name not in specs:
            return False
        props, additional = specs[name]
        args = obj.get("arguments")
        if isinstance(args, dict) and props is not None and not additional:
            return frozenset(args) <= props
        return True

    return accept


def _open_string_array(block: str) -> str:
    """R-array: ``"queries": "a", "b"`` -> ``"queries": ["a", "b"``.

    Only inserts the opening ``[``; the closing ``]`` is supplied by
    :func:`_balance_brackets`.
    """
    return _ARRAY_VALUE_KEY_RE.sub(r'"\1"\2["', block, count=1)


def _balance_brackets(block: str) -> str:
    """R-bracket: balance ``[]`` / ``{}`` counts near the end of the block.

    Handles duplicated ``]]``, excess trailing ``]``, missing ``]`` (inserted
    right before the trailing ``}`` run), and over/under-closed braces.
    """
    t = block.rstrip()
    if "]]" in t and t.count("]") > t.count("["):
        t = t.replace("]]", "]", 1)
    while t.count("]") > t.count("[") and re.search(r"\]\s*\}*\s*$", t):
        t = re.sub(r"\](\s*\}*\s*)$", r"\1", t, count=1)
    missing = t.count("[") - t.count("]")
    if missing > 0:
        t = re.sub(r"(\}+)\s*$", "]" * missing + r"\1", t, count=1)
    open_braces, close_braces = t.count("{"), t.count("}")
    if open_braces > close_braces:
        t = t + "}" * (open_braces - close_braces)
    else:
        while t.count("}") > t.count("{") and t.endswith("}"):
            t = t[:-1].rstrip()
    return t


# Streaming repair-cache bound. Completed blocks add one or two keys each,
# but a trailing body that stays valid JSON while growing (e.g. a bare
# number) would add a new key per delta; past the cap, repairs simply run
# uncached so memory stays bounded.
_REPAIR_CACHE_MAX = 128

# Tried in order; the first variant that yields a dict wins. Ordered by
# measured corpus coverage so already-valid blocks exit on the first rung.
_REPAIR_LADDER: tuple[Callable[[str], str], ...] = (
    lambda block: block,
    _balance_brackets,
    lambda block: _balance_brackets(_open_string_array(block)),
    _escape_quotes_in_strings,
    lambda block: _balance_brackets(_escape_quotes_in_strings(block)),
    lambda block: _balance_brackets(
        _open_string_array(_escape_quotes_in_strings(block))
    ),
)


def _try_load(block: str) -> dict | None:
    """Parse with idiom fixes: invalid escapes, trailing ``}``, control chars."""
    bases = [block, _normalize_invalid_escapes(block)]
    for base in bases:
        trimmed = base.rstrip()
        variants = [trimmed]
        candidate = trimmed
        for _ in range(3):
            if candidate.endswith("}"):
                candidate = candidate[:-1].rstrip()
                variants.append(candidate)
        for variant in variants:
            # strict=False accepts raw control characters inside strings.
            for strict in (True, False):
                try:
                    obj = json.loads(variant, strict=strict)
                except Exception:
                    continue
                if isinstance(obj, dict):
                    return obj
    return None


def _repair_block(block: str, tools=None) -> str | None:
    """Return the block as a valid JSON string, or None if unrecoverable.

    The deterministic ladder runs first (unchanged semantics). If every rung
    fails, R-backtrack searches over ambiguous-quote interpretations; those
    candidates are additionally gated by the schema oracle when ``tools``
    are known, because a wrong interpretation can still parse as JSON.
    """
    for repair in _REPAIR_LADDER:
        obj = _try_load(repair(block))
        if obj is not None:
            return json.dumps(_coerce_arguments_wrapper(obj), ensure_ascii=False)
    accept = _schema_oracle(tools)
    seen: set[str] = set()
    for cand in _quote_repair_candidates(block):
        for variant in (cand, _balance_brackets(cand)):
            if variant in seen:
                continue
            seen.add(variant)
            obj = _try_load(variant)
            if obj is None:
                continue
            coerced = _coerce_arguments_wrapper(obj)
            if accept(coerced):
                return json.dumps(coerced, ensure_ascii=False)
    return None


def sanitize_model_output(text: str, tools=None) -> str:
    """Rewrite each ``<tool_call>`` block as valid JSON where possible.

    Prose around the blocks is left untouched; unrecoverable blocks are kept
    verbatim so the downstream Hermes parser behaves exactly like stock.
    A trailing block whose ``</tool_call>`` never arrived is repaired and
    closed when its body can be made valid. ``tools`` (when known) gate the
    backtracking repair candidates via the schema oracle.
    """
    if _TOOL_CALL_START not in text:
        return text

    def _sub_block(match: re.Match) -> str:
        fixed = _repair_block(match.group(1), tools)
        if fixed is None:
            return match.group(0)
        return f"{_TOOL_CALL_START}\n{fixed}\n{_TOOL_CALL_END}"

    text = _TOOL_CALL_BLOCK_RE.sub(_sub_block, text)

    open_pos = text.rfind(_TOOL_CALL_START)
    if open_pos != -1 and text.find(_TOOL_CALL_END, open_pos) == -1:
        inner = text[open_pos + len(_TOOL_CALL_START) :]
        fixed = _repair_block(inner.strip(), tools)
        if fixed is not None:
            text = text[:open_pos] + f"{_TOOL_CALL_START}\n{fixed}\n{_TOOL_CALL_END}"
    return text


@ToolParserManager.register_module(["motif", "motif_hermes"])
class MotifToolParser(Hermes2ProToolParser):
    """Hermes parser with Motif-specific malformed-JSON repair.

    Non-streaming extraction sanitizes the model output first. Streaming
    reuses ``Hermes2ProToolParser.extract_tool_calls_streaming`` with two
    hook overrides: completed ``<tool_call>`` blocks are repaired before
    Hermes diffs them, and argument streaming is held back until a block
    completes because repairs are not append-only.
    """

    def __init__(self, tokenizer: TokenizerLike, tools: list[Tool] | None = None):
        super().__init__(tokenizer, tools)
        # Streaming re-parses the full text on every delta, so each frozen
        # (completed) block would be repaired once per delta; cache by block
        # content instead. Parser instances are per-request in streaming.
        self._repair_cache: dict[str, str | None] = {}

    def extract_tool_calls(
        self,
        model_output: str,
        request: ChatCompletionRequest,
    ) -> ExtractedToolCallInformation:
        tools = getattr(request, "tools", None) or self.tools
        return super().extract_tool_calls(
            sanitize_model_output(model_output, tools), request
        )

    def _cached_repair(self, block: str) -> str | None:
        # self.tools is fixed per parser instance, so the block-content cache
        # key stays valid for the schema-gated backtracking rung too.
        if block in self._repair_cache:
            return self._repair_cache[block]
        fixed = _repair_block(block, self.tools)
        if len(self._repair_cache) < _REPAIR_CACHE_MAX:
            self._repair_cache[block] = fixed
        return fixed

    def _extract_tool_call_jsons(self, text: str) -> list[tuple[str, bool]]:
        """Repair each completed block before Hermes diffs it (streaming).

        A block is repaired only once complete (its end tag arrived, or its
        body is already valid JSON): completed text is frozen, so the repair
        is identical on every subsequent delta and Hermes's append-only
        argument diffing stays consistent. Unrecoverable blocks pass through
        verbatim (stock behavior).
        """
        repaired: list[tuple[str, bool]] = []
        for tc_json, is_complete in super()._extract_tool_call_jsons(text):
            if is_complete:
                fixed = self._cached_repair(tc_json)
                if fixed is not None:
                    tc_json = fixed
            repaired.append((tc_json, is_complete))
        return repaired

    def _compute_args_diff(
        self, index: int, tc_json: str, is_complete: bool
    ) -> str | None:
        # Hold back arguments while the block is still open: repairs are not
        # append-only, so a raw fragment streamed early could contradict the
        # repaired final arguments. Names still stream as soon as they appear.
        if not is_complete:
            return None
        return super()._compute_args_diff(index, tc_json, is_complete)
