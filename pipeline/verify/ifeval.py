"""Strict and loose constraint checking over standard and project-specific types."""

from instruction_following_eval import instructions_registry
from instruction_following_eval.evaluation import ensure_nltk_resource

from pipeline.constraints import extra

ensure_nltk_resource()


def _instance(constraint: dict, prompt: str):
    cid, kwargs = constraint["id"], constraint.get("kwargs") or {}
    if cid in extra.REGISTRY:
        return extra.REGISTRY[cid].bind(kwargs)
    inst = instructions_registry.INSTRUCTION_DICT[cid](cid)
    inst.build_description(**kwargs)
    args = inst.get_instruction_args()
    if args and "prompt" in args:
        inst.build_description(prompt=prompt)
    return inst


def _follows(inst, response: str) -> bool:
    try:
        return bool(response.strip() and inst.check_following(response))
    except Exception:
        return False


def check(prompt: str, constraints: list[dict], response: str) -> list[bool]:
    return [_follows(_instance(c, prompt), response) for c in constraints]


def passes(prompt: str, constraints: list[dict], response: str) -> bool:
    return all(check(prompt, constraints, response))


def _loose_variants(response: str) -> list[str]:
    parts = response.split("\n")
    variants = [
        response,
        response.replace("*", ""),
        "\n".join(parts[1:]).strip(),
        "\n".join(parts[:-1]).strip(),
        "\n".join(parts[1:-1]).strip(),
    ]
    return variants + [v.replace("*", "") for v in variants[2:]]


def score(prompt: str, constraints: list[dict], response: str) -> dict:
    strict = check(prompt, constraints, response)
    variants = _loose_variants(response)
    loose = [
        any(_follows(_instance(c, prompt), v) for v in variants) for c in constraints
    ]
    return {
        "n": len(constraints),
        "inst_strict": sum(strict),
        "inst_loose": sum(loose),
        "prompt_strict": all(strict),
        "prompt_loose": all(loose),
    }
