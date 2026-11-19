"""Strict and loose constraint checking over standard, IFEvalG/IFBench-train and project-specific types."""

from instruction_following_eval import instructions_registry
from instruction_following_eval.evaluation import ensure_nltk_resource
from langdetect import DetectorFactory

from pipeline.constraints import extra, ifevalg

ensure_nltk_resource()
DetectorFactory.seed = 0  # language checks must be deterministic for rewards


def route(cid: str) -> str:
    if cid in extra.REGISTRY:
        return "extra"
    if cid in instructions_registry.INSTRUCTION_DICT:
        return "official"
    if cid in ifevalg.REGISTRY:
        return "ifevalg"
    raise KeyError(f"no checker for constraint id {cid!r}")


def _instance(constraint: dict, prompt: str):
    cid, kwargs = constraint["id"], constraint.get("kwargs") or {}
    where = route(cid)
    if where == "extra":
        return extra.REGISTRY[cid].bind(kwargs)
    if where == "official":
        # None would make build_description draw a random value: drop it so the check is deterministic.
        inst = instructions_registry.INSTRUCTION_DICT[cid](cid)
        inst.build_description(**{k: v for k, v in kwargs.items() if v is not None})
    else:
        inst = ifevalg.REGISTRY[cid](cid)  # drop only None: 0 and False are real values (e.g. n_start=0)
        inst.build_description(**{k: v for k, v in kwargs.items() if v is not None})
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
