"""IFEvalG constraint checkers, vendored from allenai/open-instruct (open_instruct/IFEvalG, Apache-2.0, see LICENSE).

Upstream: main as of 2026-11-15. Changed only the import paths and absl logging -> stdlib logging. These are the
checkers behind the IF-RLVR training data (allenai/IF_multi_constraints_upto5): the 25 IFEval types plus the
IFBench-train types. Use `passes()` to check a response.
"""

import ast

from pipeline.constraints.ifevalg import instructions_registry

REGISTRY = instructions_registry.INSTRUCTION_DICT


def parse_ground_truth(gt: str) -> tuple[list[str], list[dict | None]]:
    d = ast.literal_eval(gt)
    d = d[0] if isinstance(d, list) else d
    return list(d["instruction_id"]), list(d["kwargs"])


def check(response: str, instruction_ids: list[str], kwargs: list[dict | None], prompt: str = "") -> list[bool]:
    """Strict per-constraint results, built the way upstream IF-RLVR scoring builds them."""
    out = []
    for iid, kw in zip(instruction_ids, kwargs):
        checker = REGISTRY[iid](iid)
        checker.build_description(**{k: v for k, v in (kw or {}).items() if v})
        args = checker.get_instruction_args()
        if args and "prompt" in args:
            checker.build_description(prompt=prompt)
        out.append(bool(response.strip()) and bool(checker.check_following(response)))
    return out


def passes(response: str, instruction_ids: list[str], kwargs: list[dict | None], prompt: str = "") -> bool:
    return all(check(response, instruction_ids, kwargs, prompt))
