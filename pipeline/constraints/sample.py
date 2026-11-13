"""Compose one to four compatible constraints into a prompt."""

import random

from instruction_following_eval import instructions_registry

from pipeline.constraints import extra

# At most one constraint per group in a prompt, to avoid contradictions.
STANDARD_GROUPS = {
    "keywords:existence": "keyword",
    "keywords:frequency": "keyword",
    "keywords:forbidden_words": "keyword",
    "keywords:letter_frequency": "letters",
    "language:response_language": "language",
    "length_constraints:number_sentences": "length",
    "length_constraints:number_paragraphs": "structure",
    "length_constraints:number_words": "length",
    "length_constraints:nth_paragraph_first_word": "structure",
    "detectable_content:number_placeholders": "placeholders",
    "detectable_content:postscript": "end",
    "detectable_format:number_bullet_lists": "structure",
    "detectable_format:constrained_response": "wrap",
    "detectable_format:number_highlighted_sections": "format",
    "detectable_format:multiple_sections": "structure",
    "detectable_format:json_format": "wrap",
    "detectable_format:title": "start",
    "combination:two_responses": "wrap",
    "combination:repeat_prompt": "start",
    "startend:end_checker": "end",
    "change_case:capital_word_frequency": "case",
    "change_case:english_capital": "case",
    "change_case:english_lowercase": "case",
    "punctuation:no_comma": "chars",
    "startend:quotation": "wrap",
}
K_WEIGHTS = [(1, 0.20), (2, 0.35), (3, 0.30), (4, 0.15)]
STANDARD_IDS = sorted(STANDARD_GROUPS)
TRAIN_EXTRA_IDS = sorted(i for i, e in extra.REGISTRY.items() if not e.dev_only)
DEV_ONLY_IDS = sorted(i for i, e in extra.REGISTRY.items() if e.dev_only)


def group_of(cid: str) -> str:
    return extra.REGISTRY[cid].group if cid in extra.REGISTRY else STANDARD_GROUPS[cid]


def _build(cid: str, rng: random.Random, base_task: str) -> dict:
    if cid in extra.REGISTRY:
        spec = extra.REGISTRY[cid]
        kwargs = spec.sample(rng)
        return {"id": cid, "kwargs": kwargs, "text": spec.describe(kwargs)}
    inst = instructions_registry.INSTRUCTION_DICT[cid](cid)
    # The library samples its parameters with the global `random` module.
    state = random.getstate()
    random.seed(rng.getrandbits(64))
    try:
        if cid == "combination:repeat_prompt":
            text = inst.build_description(prompt_to_repeat=base_task)
        else:
            text = inst.build_description()
    finally:
        random.setstate(state)
    return {"id": cid, "kwargs": inst.get_instruction_args() or {}, "text": text}


def sample_constraints(rng: random.Random, base_task: str, mode: str = "train") -> list[dict]:
    k = rng.choices([k for k, _ in K_WEIGHTS], weights=[w for _, w in K_WEIGHTS])[0]
    chosen: list[str] = []
    groups: set[str] = set()
    if mode == "dev_unseen":
        first = rng.choice(DEV_ONLY_IDS)
        chosen.append(first)
        groups.add(group_of(first))
    for _ in range(50):
        if len(chosen) >= k:
            break
        pool = STANDARD_IDS if rng.random() < 0.5 else TRAIN_EXTRA_IDS
        cid = rng.choice(pool)
        group = group_of(cid)
        if group in groups:
            continue
        chosen.append(cid)
        groups.add(group)
    # A response that must be wrapped whole leaves little room for other rules.
    if "wrap" in groups:
        wrap = [c for c in chosen if group_of(c) == "wrap"]
        rest = [c for c in chosen if group_of(c) != "wrap"]
        chosen = wrap + rest[:1]
    return [_build(cid, rng, base_task) for cid in chosen]


def render_prompt(base_task: str, constraints: list[dict]) -> str:
    return " ".join([base_task.strip()] + [c["text"] for c in constraints])
