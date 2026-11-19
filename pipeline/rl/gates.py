"""IF quality gate (reward 0) for checker-gaming answers, with per-type exemptions where the constraint itself
asks for the pattern. Phase 0 requires < 1% firing per constraint type on good teacher answers."""

import re
from collections import Counter

from pipeline.constraints.extra import sentences
from pipeline.select import _LABEL_LINE

# Not preceded by a word char, backslash, ^, _ or closing bracket (LaTeX subscripts, type hints, indexing), and a
# bracket tag is not followed by "(" (markdown link). Fenced code is stripped before matching.
_PRE = r"(?<![\w\\^_)\]}])"
PLACEHOLDER = re.compile(
    _PRE + r"(?:\[[A-Za-z][A-Za-z '_-]{0,30}\](?!\()|\{[a-z_]{1,30}\})|<placeholder>", re.I)
CODE_FENCE = re.compile(r"```.*?(?:```|\Z)", re.S)
SELF_EVAL = re.compile(
    r"(?im)^\W*(?:\(?\s*word count\s*:?\s*\d+|I have (?:followed|met|adhered to|satisfied) (?:all|every|the)"
    r"|(?:this|my) response (?:contains|has) (?:exactly )?\d+ (?:words|sentences))")
LIST_ITEM = re.compile(r"^\s*(?:[-*•]|\d+[.)])\s+(.*\w.*?)\s*$", re.M)
SLOT_WORDS = re.compile(r"insert|placeholder|fill|\bhere\b|xxx|^\{[a-z]\}$", re.I)
SIGNOFF_LINES = 4
PLACEHOLDER_EXEMPT = {"square_brackets", "bigram_wrapping", "title", "number_placeholders", "bracket_tags"}
REPEAT_EXEMPT_PARTS = ("copy", "repeat", "two_responses")
UPPER = {"less than", "at most"}


def _suffix(cid: str) -> str:
    return cid.split(":")[-1]


def _asks_brackets(c: dict) -> bool:
    def has(v) -> bool:
        if isinstance(v, str):
            return "[" in v or "{" in v
        return isinstance(v, (list, tuple)) and any(has(x) for x in v)
    return any(has(v) for v in (c.get("kwargs") or {}).values())


def _template_like(response: str) -> bool:
    """Template-style non-answer: a slot-word token anywhere, or 3+ distinct tokens outside the sign-off block."""
    response = CODE_FENCE.sub("", response)
    if any(SLOT_WORDS.search(m.group()) for m in PLACEHOLDER.finditer(response)):
        return True
    body = "\n".join(response.rstrip().split("\n")[:-SIGNOFF_LINES])
    return len({m.group().lower() for m in PLACEHOLDER.finditer(body)}) >= 3


def _short_ok(c: dict) -> bool:
    cid, kw = c["id"], c.get("kwargs") or {}
    if cid.startswith("letters:"):
        return True
    if cid.startswith("length_constraints:"):
        return any(str(v).lower() in UPPER for v in kw.values())
    return "short" in cid


def gate_reasons(response: str, constraints: list[dict]) -> list[str]:
    out = []
    if _LABEL_LINE.search(response):
        out.append("label_line")
    if _template_like(response) and not any(
            _suffix(c["id"]) in PLACEHOLDER_EXEMPT or _asks_brackets(c) for c in constraints):
        out.append("placeholder")
    sents = [" ".join(s.lower().split()) for s in sentences(response)]
    counts = Counter(s for s in sents if len(s.split()) >= 3)
    if counts and max(counts.values()) >= 3 and not any(
            p in c["id"] for c in constraints for p in REPEAT_EXEMPT_PARTS):
        out.append("repeated_sentence")
    items = Counter(" ".join(m.lower().split()) for m in LIST_ITEM.findall(response))
    if items and max(items.values()) >= 2:
        out.append("duplicate_items")
    if SELF_EVAL.search(response):
        out.append("self_eval")
    if len(re.findall(r"\w+", response)) < 4 and not any(_short_ok(c) for c in constraints):
        out.append("too_short")
    return out
