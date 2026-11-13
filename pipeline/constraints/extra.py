"""Constraint types written for this project, each with its own verifier.

Thirty are used in training. Six are dev-only: they appear only in the private
dev set, to measure generalisation to constraint types never trained on.
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


_ALL = [
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

REGISTRY: dict[str, Extra] = {e.id: e for e in _ALL}
