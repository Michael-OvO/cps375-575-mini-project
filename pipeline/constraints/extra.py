"""Constraint types written for this project, each with its own verifier.

The first 36 (`_V1`) were used for the v1 run: thirty in training and six dev-only. `_NEW`
adds 29 more types from under-represented families (positional, punctuation counts,
per-paragraph, ordering/repetition, character-level, simple formats); six of those are
dev-only too. Dev-only types appear only in the private dev sets, to measure
generalisation to constraint types never trained on.

`group` is the one slot a type occupies in a prompt; `blocks` lists further groups it
conflicts with, so the sampler never pairs it with a type of those groups either.
"""

import random
import re
from dataclasses import dataclass
from typing import Callable

_WORD = re.compile(r"[A-Za-z0-9]+(?:['’-][A-Za-z0-9]+)*")

KEYWORDS = [
    "river", "bridge", "garden", "signal", "lantern", "market", "harbor", "window", "puzzle", "engine",
    "forest", "candle", "mirror", "ticket", "village", "copper", "meadow", "anchor", "ribbon", "summit",
    "orchard", "compass", "thunder", "velvet", "pebble", "island", "ladder", "canvas", "feather", "tunnel",
]
PHRASES = [
    "To begin with,", "Here is the plan:", "In short,", "Let us be clear:", "First things first:",
    "For the record,", "As requested,", "Without delay,", "Picture this:", "The short version:",
]
TAGS = ["answer", "note", "reply", "output", "summary", "draft", "result", "message"]


def words(text: str) -> list[str]:
    return _WORD.findall(text)


_SENTENCE_END = re.compile(r"(?<=[.!?])[\"')\]]*\s+")
_LIST_MARKER = re.compile(r"^\s*(?:\d+[.)]|[-*•]|#+)\s+")
# Abbreviations that do not end a sentence when the next word is lower-case.
_ABBREV = re.compile(r"(?:^|[\s(\"'])(?:e\.g|i\.e|dr|mr|mrs|ms|vs|etc)\.$", re.IGNORECASE)


def sentences(text: str) -> list[str]:
    """Sentences split on . ! ? and on newlines; list markers and heading prefixes are removed."""
    out: list[str] = []
    for line in text.strip().split("\n"):
        line = _LIST_MARKER.sub("", line).strip()
        merged: list[str] = []
        for frag in _SENTENCE_END.split(line):
            if merged and _ABBREV.search(merged[-1]) and frag[:1].islower():
                merged[-1] = f"{merged[-1]} {frag}"
            else:
                merged.append(frag)
        out.extend(s for s in merged if re.search(r"\w", s))
    return out


def paragraphs(text: str) -> list[str]:
    return [p for p in re.split(r"\n\s*\n", text.strip()) if p.strip()]


def lines(text: str) -> list[str]:
    return [ln for ln in text.strip().split("\n") if ln.strip()]


def _count_word(text: str, word: str) -> int:
    return len(re.findall(rf"\b{re.escape(word)}\b", text, flags=re.IGNORECASE))


def _first_word(text: str) -> str:
    w = words(text)
    return w[0].lower() if w else ""


def _last_word(text: str) -> str:
    w = words(text)
    return w[-1].lower() if w else ""


def _numbered(text: str) -> list[int]:
    return [int(m.group(1)) for ln in lines(text) if (m := re.match(r"^\s*(\d+)\.", ln))]


def _table_ok(text: str, n: int) -> bool:
    rows = [ln for ln in lines(text) if ln.strip().startswith("|")]
    return len(rows) == n + 2 and bool(re.match(r"^\s*\|[\s:|-]+\|\s*$", rows[1]))


class _Bound:
    def __init__(self, spec: "Extra", kwargs: dict):
        self.spec, self.kwargs = spec, kwargs

    def check_following(self, response: str) -> bool:
        if not response.strip():
            return False
        try:
            return bool(self.spec.check(response, self.kwargs))
        except Exception:
            return False


@dataclass(frozen=True)
class Extra:
    id: str
    group: str
    sample: Callable[[random.Random], dict]
    template: str
    check: Callable[[str, dict], bool]
    dev_only: bool = False
    blocks: tuple[str, ...] = ()

    def describe(self, kwargs: dict) -> str:
        return self.template.format(**kwargs)

    def bind(self, kwargs: dict) -> _Bound:
        return _Bound(self, kwargs)


def _kw(r):
    return {"word": r.choice(KEYWORDS)}


def _two(r):
    first, second = r.sample(KEYWORDS, 2)
    return {"first": first, "second": second}


def _range(r):
    lo = r.randint(30, 300)
    return {"lo": lo, "hi": lo + r.randint(20, 80)}


def _n(lo, hi):
    return lambda r: {"n": r.randint(lo, hi)}


def _order(t, k):
    a = re.search(rf"\b{re.escape(k['first'])}\b", t, flags=re.IGNORECASE)
    b = re.search(rf"\b{re.escape(k['second'])}\b", t, flags=re.IGNORECASE)
    return bool(a and b and a.start() < b.start())


_V1 = [
    Extra("x:start_phrase", "start", lambda r: {"phrase": r.choice(PHRASES)},
          'Begin your response with the exact phrase "{phrase}".',
          lambda t, k: t.strip().startswith(k["phrase"])),
    Extra("x:first_word", "start", _kw, 'The first word of your response must be "{word}".',
          lambda t, k: _first_word(t) == k["word"]),
    Extra("x:last_word", "end", _kw, 'The last word of your response must be "{word}".',
          lambda t, k: _last_word(t) == k["word"]),
    Extra("x:exact_words", "length", _n(20, 150), "Your response must contain exactly {n} words.",
          lambda t, k: len(words(t)) == k["n"]),
    Extra("x:word_range", "length", _range, "Your response must be between {lo} and {hi} words long.",
          lambda t, k: k["lo"] <= len(words(t)) <= k["hi"]),
    Extra("x:exact_sentences", "length", _n(2, 12), "Use exactly {n} sentences.",
          lambda t, k: len(sentences(t)) == k["n"]),
    Extra("x:exact_paragraphs", "structure", _n(2, 6),
          "Write exactly {n} paragraphs, separated by blank lines.",
          lambda t, k: len(paragraphs(t)) == k["n"]),
    Extra("x:max_sentence_words", "sentence", _n(12, 25), "No sentence may be longer than {n} words.",
          lambda t, k: all(len(words(s)) <= k["n"] for s in sentences(t))),
    Extra("x:min_sentence_words", "sentence", _n(5, 10), "Every sentence must contain at least {n} words.",
          lambda t, k: all(len(words(s)) >= k["n"] for s in sentences(t))),
    Extra("x:numbered_list", "structure", _n(3, 10),
          "Present your answer as a numbered list with exactly {n} items, numbered 1. to {n}.",
          lambda t, k: _numbered(t) == list(range(1, k["n"] + 1))),
    Extra("x:no_digits", "digits", lambda r: {}, "Do not use any digits in your response.",
          lambda t, k: not re.search(r"\d", t)),
    Extra("x:include_number", "digits", _n(100, 9999), "Include the number {n} somewhere in your response.",
          lambda t, k: bool(re.search(rf"(?<!\d){k['n']}(?!\d)", t))),
    Extra("x:ban_char", "chars", lambda r: {"ch": r.choice("?!;:")},
          'Do not use the character "{ch}" anywhere in your response.',
          lambda t, k: k["ch"] not in t),
    Extra("x:question_count", "chars", _n(1, 4), "Your response must contain exactly {n} question marks.",
          lambda t, k: t.count("?") == k["n"]),
    Extra("x:end_exclaim", "end", lambda r: {}, "End your response with an exclamation mark.",
          lambda t, k: t.rstrip().endswith("!")),
    Extra("x:keyword_exact", "keyword", lambda r: {"word": r.choice(KEYWORDS), "n": r.randint(1, 5)},
          'Use the word "{word}" exactly {n} times.',
          lambda t, k: _count_word(t, k["word"]) == k["n"]),
    Extra("x:keyword_first_sentence", "keyword", _kw, 'The word "{word}" must appear in your first sentence.',
          lambda t, k: bool(sentences(t)) and _count_word(sentences(t)[0], k["word"]) > 0),
    Extra("x:keyword_last_sentence", "keyword", _kw, 'The word "{word}" must appear in your last sentence.',
          lambda t, k: bool(sentences(t)) and _count_word(sentences(t)[-1], k["word"]) > 0),
    Extra("x:keyword_order", "keyword", _two,
          'Mention "{first}" before you mention "{second}"; both words must appear.', _order),
    Extra("x:paragraph_start", "start", _kw, 'Start every paragraph with the word "{word}".',
          lambda t, k: bool(paragraphs(t)) and all(_first_word(p) == k["word"] for p in paragraphs(t))),
    Extra("x:line_count", "structure", _n(3, 12), "Your response must consist of exactly {n} non-empty lines.",
          lambda t, k: len(lines(t)) == k["n"]),
    Extra("x:caps_title", "start", lambda r: {},
          "Make the first line a title written entirely in capital letters.",
          lambda t, k: bool(lines(t)) and any(c.isalpha() for c in lines(t)[0])
          and lines(t)[0] == lines(t)[0].upper()),
    Extra("x:headings", "structure", _n(2, 5),
          'Organize your response under exactly {n} markdown headings that start with "## ".',
          lambda t, k: sum(1 for ln in lines(t) if ln.startswith("## ")) == k["n"]),
    Extra("x:no_markdown", "format", lambda r: {},
          "Do not use markdown: no asterisks, pound signs, or backticks.",
          lambda t, k: not any(c in t for c in "*#`")),
    Extra("x:code_fence", "wrap", lambda r: {},
          "Put your entire response inside one fenced code block delimited by triple backticks.",
          lambda t, k: t.strip().startswith("```") and t.strip().endswith("```") and t.count("```") == 2),
    Extra("x:xml_wrap", "wrap", lambda r: {"tag": r.choice(TAGS)},
          "Wrap your entire response in <{tag}> and </{tag}> tags.",
          lambda t, k: t.strip().startswith(f"<{k['tag']}>") and t.strip().endswith(f"</{k['tag']}>")),
    Extra("x:parentheticals", "brackets", _n(1, 3),
          "Include exactly {n} parenthetical remarks, each enclosed in round brackets.",
          lambda t, k: t.count("(") == k["n"] and t.count(")") == k["n"]),
    Extra("x:csv_line", "wrap", _n(3, 8),
          "Respond with a single line containing exactly {n} comma-separated items and nothing else.",
          lambda t, k: "\n" not in t.strip() and len(t.strip().split(",")) == k["n"]
          and all(p.strip() for p in t.strip().split(","))),
    Extra("x:table_rows", "structure", _n(2, 6),
          "Present the information as a markdown table with a header row and exactly {n} data rows.",
          lambda t, k: _table_ok(t, k["n"])),
    Extra("x:max_word_len", "wordlen", _n(8, 12), "Do not use any word longer than {n} letters.",
          lambda t, k: all(len(w) <= k["n"] for w in words(t))),
    # Dev-only types: never sampled for training data.
    Extra("x:min_chars", "length", lambda r: {"n": r.randrange(400, 1501, 100)},
          "Your response must be at least {n} characters long.",
          lambda t, k: len(t) >= k["n"], dev_only=True),
    Extra("x:semicolons", "chars", _n(1, 4), "Use exactly {n} semicolons.",
          lambda t, k: t.count(";") == k["n"], dev_only=True),
    Extra("x:open_question", "start", lambda r: {}, "Open your response with a question.",
          lambda t, k: bool(sentences(t)) and sentences(t)[0].rstrip().endswith("?"), dev_only=True),
    Extra("x:hashtags", "format", _n(1, 5), "Include exactly {n} hashtags (words prefixed with #).",
          lambda t, k: len(re.findall(r"(?<!\w)#\w+", t)) == k["n"], dev_only=True),
    Extra("x:line_max_words", "sentence", _n(6, 12), "No line may contain more than {n} words.",
          lambda t, k: all(len(words(ln)) <= k["n"] for ln in lines(t)), dev_only=True),
    Extra("x:quote_pairs", "brackets", _n(1, 3),
          "Include exactly {n} quoted phrases, each enclosed in double quotation marks.",
          lambda t, k: t.count('"') == 2 * k["n"], dev_only=True),
]

REPEAT_PHRASES = [
    "step by step", "in the end", "make it simple", "one more thing", "as it turns out",
    "on the other hand", "little by little", "at the same time",
]
ACROSTIC_WORDS = ["SUN", "OAK", "MAP", "SKY", "TEA", "ARC", "PATH", "WIND", "GOLD", "LAMP"]
CLOSING_LABELS = ["Takeaway", "Verdict", "Bottom line", "Next step", "In brief"]
SCORE_LABELS = ["Score", "Rating", "Confidence"]

_BULLET = re.compile(r"^\s*-\s+(\S.*)$")
_KV_LINE = re.compile(r"^\s*(?:[-*•]\s+|\d+[.)]\s+)?(?:\*\*)?[A-Za-z][A-Za-z-]*(?:\*\*)?:(?:\*\*)?\s+\S")
_TAG = re.compile(r"\[[A-Z]{2,}(?:[ _-][A-Z]+)*\]")
_ELLIPSIS = re.compile(r"(?<!\.)\.\.\.(?!\.)")


def _plain(line: str) -> str:
    """A line without list marker or inline markdown emphasis, stripped."""
    return _LIST_MARKER.sub("", re.sub(r"[*_`]", "", line)).strip()


def _first_pos(text: str, word: str) -> int:
    m = re.search(rf"\b{re.escape(word)}\b", text, flags=re.IGNORECASE)
    return m.start() if m else -1


def _nth_sentence_ok(t, k):
    s = sentences(t)
    return len(s) >= k["n"] and _first_word(s[k["n"] - 1]) == k["word"]


def _acrostic_ok(t, k):
    ps = paragraphs(t)
    if len(ps) != len(k["word"]):
        return False
    firsts = [words(_LIST_MARKER.sub("", p)) for p in ps]
    return all(f and f[0][0].lower() == ch.lower() for f, ch in zip(firsts, k["word"]))


def _distinct_starts(t, k):
    firsts = [_first_word(s) for s in sentences(t)]
    return bool(firsts) and len(firsts) == len(set(firsts))


def _final_one_sentence(t, k):
    ps = paragraphs(t)
    return len(ps) >= 2 and len(sentences(ps[-1])) == 1


def _bullets_ok(t, k):
    items = [m.group(1) for ln in lines(t) if (m := _BULLET.match(ln))]
    return len(items) == k["n"] and all(len(words(i)) < k["m"] for i in items)


def _steps_ok(t, k):
    nums = [int(x) for x in re.findall(r"\bStep (\d+)\b", t, flags=re.IGNORECASE)]
    return nums == list(range(1, k["n"] + 1))


def _alpha_ok(t, k):
    ws = [k["a"], k["b"], k["c"]]
    pos = {w: _first_pos(t, w) for w in ws}
    if min(pos.values()) < 0:
        return False
    return sorted(ws, key=lambda w: pos[w]) == sorted(ws)


def _not_first_paragraph(t, k):
    ps = paragraphs(t)
    return (len(ps) >= 2 and _count_word(ps[0], k["word"]) == 0
            and any(_count_word(p, k["word"]) > 0 for p in ps[1:]))


def _repeat_cap(t, k):
    counts: dict[str, int] = {}
    for w in words(t):
        if len(w) >= 4:
            counts[w.lower()] = counts.get(w.lower(), 0) + 1
    return max(counts.values(), default=0) <= k["n"]


def _all_caps_count(t, k):
    return sum(1 for w in words(t) if len(re.sub(r"[^A-Za-z]", "", w)) >= 2 and w.isupper()) == k["n"]


def _long_words(t, k):
    return len({w.lower() for w in words(t) if len(re.sub(r"[^A-Za-z]", "", w)) >= k["L"]}) >= k["n"]


def _double_letters(t, k):
    return len({w.lower() for w in words(t) if re.search(r"([a-z])\1", w.lower())}) >= k["n"]


def _closing_ok(t, k):
    ls = lines(t)
    if not ls:
        return False
    last = _plain(ls[-1])
    head = k["label"].lower() + ":"
    if not last.lower().startswith(head):
        return False
    return 1 <= len(words(last[len(head):])) <= k["n"]


def _score_ok(t, k):
    pat = re.compile(rf"{re.escape(k['label'])}:\s*(?:10|[1-9])\s*/\s*10\.?", flags=re.IGNORECASE)
    return any(pat.fullmatch(_plain(ln)) for ln in lines(t))


def _comma_range(r):
    lo = r.randint(3, 10)
    return {"lo": lo, "hi": lo + r.randint(4, 8)}


def _three(r):
    a, b, c = r.sample(KEYWORDS, 3)
    return {"a": a, "b": b, "c": c}


_NEW = [
    # Positional.
    Extra("x:nth_sentence_start", "position", lambda r: {"n": r.randint(2, 4), "word": r.choice(KEYWORDS)},
          'Sentence number {n} of your response must begin with the word "{word}".',
          _nth_sentence_ok, dev_only=True),
    Extra("x:line_end_char", "end", lambda r: {"ch": r.choice([".", "~"])},
          'End every non-empty line of your response with the character "{ch}".',
          lambda t, k: bool(lines(t)) and all(ln.rstrip().endswith(k["ch"]) for ln in lines(t))),
    Extra("x:paragraph_acrostic", "structure", lambda r: {"word": r.choice(ACROSTIC_WORDS)},
          'Write one paragraph per letter of "{word}", separated by blank lines, so that the first '
          'letters of the paragraphs spell "{word}" in order.',
          _acrostic_ok, blocks=("start", "wrap")),
    Extra("x:paragraph_end_word", "end", _kw, 'End every paragraph with the word "{word}".',
          lambda t, k: bool(paragraphs(t)) and all(_last_word(p) == k["word"] for p in paragraphs(t)),
          blocks=("keyword",)),
    Extra("x:distinct_sentence_starts", "start", lambda r: {},
          "No two sentences in your response may begin with the same word.", _distinct_starts),
    Extra("x:final_paragraph_one_sentence", "structure", lambda r: {},
          "Write at least two paragraphs separated by blank lines, and make the final paragraph a single sentence.",
          _final_one_sentence, blocks=("end", "wrap")),
    # Punctuation and symbol counts.
    Extra("x:colon_count", "chars", _n(1, 4), "Use exactly {n} colons in your response.",
          lambda t, k: t.count(":") == k["n"]),
    Extra("x:em_dash_count", "chars", _n(1, 3), 'Use the em dash character "—" exactly {n} times.',
          lambda t, k: t.count("—") == k["n"], dev_only=True),
    Extra("x:exclaim_count", "chars", _n(1, 3), "Use exactly {n} exclamation marks.",
          lambda t, k: t.count("!") == k["n"]),
    Extra("x:ellipsis_count", "chars", _n(1, 3),
          'Use exactly {n} ellipses, each written as three periods ("...").',
          lambda t, k: len(_ELLIPSIS.findall(t)) == k["n"]),
    Extra("x:comma_range", "chars", _comma_range, "Use between {lo} and {hi} commas (inclusive).",
          lambda t, k: k["lo"] <= t.count(",") <= k["hi"], blocks=("wrap",)),
    # Per-line and per-paragraph.
    Extra("x:paragraph_sentences", "structure", _n(2, 4),
          "Every paragraph (paragraphs are separated by blank lines) must contain exactly {n} sentences.",
          lambda t, k: bool(paragraphs(t)) and all(len(sentences(p)) == k["n"] for p in paragraphs(t)),
          dev_only=True, blocks=("length",)),
    Extra("x:bullet_item_max_words", "structure", lambda r: {"n": r.randint(3, 6), "m": r.randint(8, 15)},
          'Include a bulleted list of exactly {n} items, each starting with "- ", and keep every item '
          "under {m} words.", _bullets_ok, blocks=("wrap",)),
    Extra("x:paragraphs_min_words", "structure",
          lambda r: {"n": r.randint(2, 4), "m": r.randrange(30, 51, 5)},
          "Write exactly {n} paragraphs separated by blank lines, each at least {m} words long.",
          lambda t, k: len(paragraphs(t)) == k["n"] and all(len(words(p)) >= k["m"] for p in paragraphs(t)),
          blocks=("length", "wrap")),
    # Ordering and repetition.
    Extra("x:phrase_repeat", "repeat", lambda r: {"phrase": r.choice(REPEAT_PHRASES), "n": r.randint(2, 4)},
          'Use the exact phrase "{phrase}" exactly {n} times.',
          lambda t, k: _count_word(t, k["phrase"]) == k["n"]),
    Extra("x:keywords_alpha", "keyword", _three,
          'Use each of the words "{a}", "{b}" and "{c}", arranged so that their first appearances '
          "follow alphabetical order.", _alpha_ok, dev_only=True),
    Extra("x:keyword_not_first_paragraph", "keyword", _kw,
          'Do not use the word "{word}" in your first paragraph, but use it at least once in a later '
          "paragraph.", _not_first_paragraph, blocks=("wrap",)),
    Extra("x:keyword_every_paragraph", "keyword", _kw, 'Use the word "{word}" at least once in every paragraph.',
          lambda t, k: bool(paragraphs(t)) and all(_count_word(p, k["word"]) > 0 for p in paragraphs(t))),
    Extra("x:word_repeat_cap", "repeat", _n(3, 5),
          "No word of four or more letters may appear more than {n} times (ignoring case).",
          _repeat_cap, blocks=("keyword",)),
    # Character-level.
    Extra("x:all_caps_count", "case", _n(1, 4),
          "Include exactly {n} words written entirely in capital letters (two or more letters each).",
          _all_caps_count),
    Extra("x:initial_letter_count", "letters",
          lambda r: {"letter": r.choice("bcdfmpst"), "n": r.randint(5, 12)},
          'Use at least {n} words that begin with the letter "{letter}".',
          lambda t, k: sum(1 for w in words(t) if w.lower().startswith(k["letter"])) >= k["n"]),
    Extra("x:long_words", "wordlen", lambda r: {"n": r.randint(2, 5), "L": r.randint(10, 12)},
          "Use at least {n} different words that are {L} or more letters long.", _long_words),
    Extra("x:double_letter_words", "doubles", _n(3, 8),
          'Use at least {n} different words that contain the same letter twice in a row (as in "coffee" '
          'or "balloon").', _double_letters, dev_only=True),
    Extra("x:lipogram", "letters", lambda r: {"letter": r.choice("jqx")},
          'Do not use the letter "{letter}" anywhere in your response (upper or lower case).',
          lambda t, k: k["letter"] not in t.lower(), blocks=("keyword", "start", "end")),
    # Simple formats.
    Extra("x:key_value_lines", "kv", _n(3, 6),
          'Include exactly {n} lines formatted as "Key: value", where each key is a single word.',
          lambda t, k: sum(1 for ln in lines(t) if _KV_LINE.match(ln)) == k["n"], blocks=("chars", "wrap")),
    Extra("x:bracket_tags", "tags", _n(1, 4),
          "Include exactly {n} bracketed tags written in capital letters, such as [TIP] or [NOTE].",
          lambda t, k: len(_TAG.findall(t)) == k["n"], dev_only=True, blocks=("placeholders", "case")),
    Extra("x:closing_label_line", "end", lambda r: {"label": r.choice(CLOSING_LABELS), "n": r.randint(3, 8)},
          'Finish with a separate final line that starts with "{label}:" followed by no more than {n} words.',
          _closing_ok, blocks=("chars",)),
    Extra("x:score_line", "digits", lambda r: {"label": r.choice(SCORE_LABELS)},
          'Include a line that reads "{label}: X/10", where X is a whole number from 1 to 10.',
          _score_ok, blocks=("chars", "wrap")),
    Extra("x:step_labels", "structure", _n(3, 6),
          'Lay out your answer as exactly {n} steps labelled "Step 1", "Step 2", and so on up to "Step {n}".',
          _steps_ok, blocks=("digits",)),
]

_ALL = _V1 + _NEW
V1_IDS = frozenset(e.id for e in _V1)
REGISTRY: dict[str, Extra] = {e.id: e for e in _ALL}
