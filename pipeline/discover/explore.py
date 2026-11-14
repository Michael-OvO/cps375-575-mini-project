"""Explorer: Opus proposes facet values the taxonomy misses (the unknown unknowns).

Input per task: the taxonomy, the training coverage map, the staff's description of the hidden benchmark
and the v1 failure summary. Output: data/discover/proposals_<task>.json, every value with status
"proposed". We review and set status to "accepted" (or "rejected"); accepted values join the taxonomy
(taxonomy.facets) and get cells on the next `probes cells`.

Usage:
    python -m pipeline.discover.explore --task code
"""

import argparse
import asyncio
import json

from pipeline import teachers
from pipeline.discover.probes import STAFF
from pipeline.discover.taxonomy import DISCOVER, FACETS

FAILURES = {
    "math": "v1 scores 70% on MATH-500. Level 5: 39%. Weakest subjects: Geometry 44%, Intermediate Algebra 52%, "
            "Precalculus 60%. Most failures are wrong answers after short reasoning; some runs end without an answer line.",
    "code": "v1 scores 75% on MBPP and 67% on HumanEval (base model 71%). Most failures fail an assert that is shown "
            "in the prompt; weakest topics: dates and times, custom sort keys, recursion, dynamic programming.",
    "if": "v1 scores 72% on IFEval. Weakest: exact counts (words, sentences, letters), rules of types never seen in "
          "training (53% vs 66%), prompts with 3+ rules; 11% of answers on the public set hit the length cap, mostly "
          "in repetition loops, often when a non-English answer is requested.",
}

PROMPT = """We are fine-tuning a small open model and must do well on a hidden benchmark of {staff}.

Our training data is described by these facets (each value with its number of training examples; values with \
count 0 or missing are not covered):
{coverage}

What the current model gets wrong:
{failures}

Propose NEW facet values that a benchmark of this kind plausibly contains and that our data does not cover, \
including whole new problem types we have not thought of. For each give the facet (an existing facet name, or \
a new facet name if the value fits none), the value (short, usable as a description for writing test items), \
why a benchmark of this kind would contain it, and how likely it is to appear (high / medium / low).

Propose 8 to 15 values. Prefer values a small model is likely to fail on. Do not repeat existing values.
Return only a JSON list of objects with keys "facet", "value", "why", "likelihood"."""


def _coverage_text(task: str) -> str:
    p = DISCOVER / f"coverage_{task}.json"
    cov = json.loads(p.read_text()) if p.exists() else {}
    lines = []
    for f, vals in FACETS[task].items():
        c = cov.get(f, {})
        shown = [f"{v} ({int(c.get(v, 0))})" for v in vals]
        lines.append(f"- {f}: " + "; ".join(shown))
    return "\n".join(lines)


async def explore(task: str) -> list[dict]:
    complete = teachers.get("claude")
    text = await complete(PROMPT.format(staff=STAFF[task], coverage=_coverage_text(task), failures=FAILURES[task]),
                          max_tokens=16000)
    got = teachers.parse_json(text)
    return [g for g in got if isinstance(g, dict) and g.get("facet") and g.get("value")]


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--task", choices=list(FACETS), required=True)
    a = p.parse_args()
    teachers.check_configured(["claude"])
    got = asyncio.run(explore(a.task))
    out = DISCOVER / f"proposals_{a.task}.json"
    data = json.loads(out.read_text()) if out.exists() else {"facets": {}}
    for g in got:
        vals = data["facets"].setdefault(g["facet"], [])
        if all(v["value"] != g["value"] for v in vals):
            vals.append({"value": g["value"], "why": g.get("why", ""), "likelihood": g.get("likelihood", ""),
                         "status": "proposed"})
    out.write_text(json.dumps(data, indent=1))
    for g in got:
        print(f"[{g.get('likelihood')}] {g['facet']}: {g['value']} — {g.get('why', '')[:120]}")
    teachers.write_usage(out)


if __name__ == "__main__":
    main()
