"""Teacher registry and a tolerant JSON extractor."""

import json
import re

_FENCE = re.compile(r"```(?:json)?\n(.*?)```", re.DOTALL)


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


def parse_json(text: str):
    if not text:
        return None
    candidates = _FENCE.findall(text) + [text]
    decoder = json.JSONDecoder()
    for cand in candidates:
        for i, ch in enumerate(cand):
            if ch in "[{":
                try:
                    value, _ = decoder.raw_decode(cand[i:])
                    return value
                except json.JSONDecodeError:
                    continue
    return None
