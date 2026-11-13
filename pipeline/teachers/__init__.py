"""Teacher registry and a tolerant JSON extractor."""

import json
import os
import re
import sys
from pathlib import Path

_FENCE = re.compile(r"```(?:json)?\n(.*?)```", re.DOTALL)
_HEX4 = re.compile(r"[0-9a-fA-F]{4}")
_ASCII_LETTER = re.compile(r"[A-Za-z]")
_LETTERS = re.compile(r"[A-Za-z]*")
# LaTeX commands starting with "n": only these make "\n<letters>" LaTeX; any other "\n" is a JSON newline.
_LATEX_N_COMMANDS = frozenset({
    "nu", "ne", "neq", "neg", "nabla", "not", "notin", "ni", "nmid", "nleq", "ngeq", "nless", "ngtr",
    "newline", "nonumber", "natural", "nexists", "nsubseteq", "nsupseteq", "nparallel", "nrightarrow",
    "nleftarrow", "ncong", "nsim", "nprec", "nsucc", "ntriangleleft", "ntriangleright", "nolimits",
    "normalsize", "noindent",
})
# Control characters that never belong in a problem or training text (newline and CR excluded).
_CONTROL = re.compile(r"[\x00-\x09\x0b\x0c\x0e-\x1f]")
_CONTROL_NO_TAB = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f]")


def has_control_chars(text: str, allow_tab: bool = False) -> bool:
    return bool((_CONTROL_NO_TAB if allow_tab else _CONTROL).search(text or ""))


def check_configured(names) -> None:
    """Fail fast, before any paid call, if a requested teacher has no credentials or model id."""
    missing = []
    for name in dict.fromkeys(names):
        if name == "gpt":
            if not os.environ.get("OPENAI_API_KEY"):
                missing.append("gpt: OPENAI_API_KEY is not set")
            if not os.environ.get("OPENAI_TEACHER_MODEL", "").strip():
                missing.append("gpt: OPENAI_TEACHER_MODEL is not set (the GPT model id to use)")
        elif name == "tinker":
            if not os.environ.get("TINKER_API_KEY"):
                missing.append("tinker: TINKER_API_KEY is not set")
        elif name == "claude":
            if not (os.environ.get("ANTHROPIC_API_KEY") or os.environ.get("ANTHROPIC_AUTH_TOKEN")
                    or (Path.home() / ".config" / "anthropic").is_dir()):
                missing.append("claude: set ANTHROPIC_API_KEY or ANTHROPIC_AUTH_TOKEN, "
                               "or log in with `ant auth login` (~/.config/anthropic)")
        else:
            missing.append(f"{name}: unknown teacher")
    if missing:
        raise RuntimeError("Teacher configuration incomplete:\n  " + "\n  ".join(missing))


def new_usage() -> dict:
    return {"calls": 0, "input_tokens": 0, "output_tokens": 0, "models": {}}


def record_usage(counter: dict, response) -> None:
    """Add one API response to a module-level counter: token usage and the model id actually served.

    Field names checked against the installed SDKs (anthropic 2.8 BetaMessage.usage / .model,
    openai 2.30 Response.usage / .model); read defensively since usage can be None.
    """
    counter["calls"] += 1
    usage = getattr(response, "usage", None)
    counter["input_tokens"] += int(getattr(usage, "input_tokens", 0) or 0)
    counter["output_tokens"] += int(getattr(usage, "output_tokens", 0) or 0)
    model = str(getattr(response, "model", None) or "unknown")
    counter["models"][model] = counter["models"].get(model, 0) + 1


def usage_snapshot() -> dict:
    """Usage so far in this process, per teacher module that has been loaded."""
    out = {}
    for name in ("claude", "gpt"):
        module = sys.modules.get(f"pipeline.teachers.{name}")
        if module is not None:
            out[name] = json.loads(json.dumps(module.USAGE))
    return out


def _merge(old, new):
    """Sum numbers key by key (nested), keeping keys present on either side."""
    if isinstance(old, dict) and isinstance(new, dict):
        out = dict(old)
        for key, value in new.items():
            out[key] = _merge(out[key], value) if key in out else value
        return out
    if isinstance(old, (int, float)) and isinstance(new, (int, float)):
        return old + new
    return new


def write_usage(out_path) -> dict:
    """Add this run's usage to <out_path>.usage.json (summing with earlier runs) and print it."""
    snap = usage_snapshot()
    path = Path(str(out_path) + ".usage.json")
    total = _merge(json.loads(path.read_text(encoding="utf-8")), snap) if path.exists() else snap
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(total, indent=2, sort_keys=True), encoding="utf-8")
    print(f"teacher usage this run: {json.dumps(snap, sort_keys=True)}")
    print(f"teacher usage total -> {path}")
    return total


def get(name: str):
    if name == "claude":
        from pipeline.teachers import claude
        return claude.complete
    if name == "gpt":
        from pipeline.teachers import gpt
        return gpt.complete
    if name == "tinker":
        from pipeline.teachers import tinker_teacher
        return tinker_teacher.complete
    raise ValueError(f"unknown teacher: {name}")


def _repair_backslashes(text: str, latex: bool) -> str:
    """Double every backslash that does not start a valid JSON escape.

    With latex=True, \\b \\f \\r \\t followed by a letter are read as LaTeX commands (\\frac,
    \\times, \\rho ...) and also doubled, instead of silently decoding to control characters.
    \\n is LaTeX only when it starts a known command (\\nu, \\neq, \\not ...) followed by a
    non-letter or the end; otherwise it stays a real newline ("Line one.\\nLine two").
    """
    out = []
    i = 0
    while i < len(text):
        ch = text[i]
        if ch != "\\":
            out.append(ch)
            i += 1
            continue
        nxt = text[i + 1:i + 2]
        if nxt in ('"', "\\", "/") and nxt:
            out.append(text[i:i + 2])
            i += 2
        elif nxt == "u" and _HEX4.fullmatch(text[i + 2:i + 6]):
            out.append(text[i:i + 6])
            i += 6
        elif nxt and nxt in "bfnrt" and not (latex and _is_latex_escape(text, i)):
            out.append(text[i:i + 2])
            i += 2
        else:
            out.append("\\\\")
            i += 1
    return "".join(out)


def _is_latex_escape(text: str, i: int) -> bool:
    """text[i] is a backslash before one of b f n r t: is this a LaTeX command rather than an escape?"""
    if text[i + 1] != "n":
        return bool(_ASCII_LETTER.fullmatch(text[i + 2:i + 3]))
    word = "n" + _LETTERS.match(text, i + 2).group(0)  # maximal letter run: followed by non-letter or end
    return word in _LATEX_N_COMMANDS


def _span_end(text: str, start: int) -> int | None:
    """Index just past the bracket that closes text[start], honouring JSON strings; None if unclosed."""
    depth = 0
    in_string = False
    i = start
    while i < len(text):
        ch = text[i]
        if in_string:
            if ch == "\\":
                i += 2
                continue
            if ch == '"':
                in_string = False
        elif ch == '"':
            in_string = True
        elif ch in "[{":
            depth += 1
        elif ch in "]}":
            depth -= 1
            if depth == 0:
                return i + 1
        i += 1
    return None


def parse_json(text: str, latex: bool = True):
    """The JSON array or object that starts at the first [ or { of a candidate (fenced blocks first,
    then the whole text). If that fails to decode even after repair, the candidate gives None: an
    inner bracket is never returned. Set latex=False for replies whose strings hold code, where
    "\\t" or "\\n" before a letter is real whitespace.
    """
    if not text:
        return None
    candidates = _FENCE.findall(text) + [text]
    for cand in candidates:
        cand = _repair_backslashes(cand, latex)
        starts = [p for p in (cand.find("["), cand.find("{")) if p != -1]
        if not starts:
            continue
        start = min(starts)
        end = _span_end(cand, start)
        if end is None:
            continue  # unclosed (e.g. truncated)
        try:
            return json.loads(cand[start:end])
        except json.JSONDecodeError:
            continue
    return None
