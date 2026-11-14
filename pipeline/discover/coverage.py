"""Coverage map: how many training examples fall into each facet value (and value pair).

Rules where the facet is explicit in the data (IF constraint ids and genres, code categories, math subject
and level, answer type from the answer string, code output contract from the asserts). Math technique and
form are not in the data: a sample is labelled by Opus (`label_math`), cached in data/discover/math_labels.jsonl.

Usage:
    python -m pipeline.discover.coverage build            # rules only, plus cached math labels
    python -m pipeline.discover.coverage label-math --limit 1000
"""

import argparse
import ast
import asyncio
import json
import random
import re
from collections import Counter

from pipeline import jsonl, teachers
from pipeline.config import DATA, SEED
from pipeline.constraints.sample import group_of
from pipeline.discover.taxonomy import DISCOVER, FACETS
from pipeline.generate import ifeval as gif

MATH_LABELS = DISCOVER / "math_labels.jsonl"


# ---------------------------------------------------------------- math

def answer_type(ans: str) -> str:
    a = str(ans).strip().strip("$")
    if re.search(r"\^\\circ|°|degrees", a):
        return "angle in degrees"
    if re.search(r"\\\$|dollars|^\d+\.\d\d$", a):
        return "decimal or money amount"
    if re.search(r"\\cup|\\infty|[<>]|\\le|\\ge", a):
        return "interval or inequality"
    if re.fullmatch(r"\(.*,.*\)", a) or re.search(r"\\begin\{pmatrix\}", a):
        return "ordered pair or tuple"
    if "," in a:
        return "several values separated by commas"
    if "\\pi" in a:
        return "expression in pi"
    if "sqrt" in a:
        return "radical expression"
    if re.search(r"[a-zA-Z]", re.sub(r"\\(frac|dfrac|tfrac|left|right|cdot)", "", a)):
        return "polynomial or algebraic expression"
    if re.search(r"\\d?frac|/", a):
        return "fraction"
    if re.fullmatch(r"-?\d+", a.replace(",", "")):
        return "integer"
    if re.fullmatch(r"-?\d*\.\d+", a):
        return "decimal or money amount"
    return "polynomial or algebraic expression"


def math_rows() -> list[dict]:
    rows = {}
    for f in ("math_train_solutions.jsonl", "math_new_solutions.jsonl"):
        for r in jsonl.read(DATA / f):
            pid = r.get("pid", r["id"])
            rows.setdefault(pid, {"id": pid, "problem": r["problem"], "answer": r.get("answer") or r.get("target", ""),
                                  "subject": r.get("subject"), "level": str(r.get("level"))})
    return list(rows.values())


LABEL_PROMPT = """Label each competition math problem below. For each, give:
- "technique": the ONE main technique, chosen from this list: {techniques}; use "other: <short name>" if none fits.
- "form": one of: {forms}.

Return only a JSON list of objects with keys "id", "technique", "form", in the same order.

{problems}"""


async def _label_batch(batch: list[dict]) -> list[dict] | None:
    complete = teachers.get("claude")
    probs = "\n\n".join(f"[{r['id']}] {r['problem']}" for r in batch)
    text = await complete(LABEL_PROMPT.format(techniques="; ".join(FACETS["math"]["technique"]),
                                              forms="; ".join(FACETS["math"]["form"]), problems=probs),
                          max_tokens=8000)
    try:
        got = teachers.parse_json(text)
    except Exception:
        return None
    by_id = {g.get("id"): g for g in got if isinstance(g, dict)}
    out = [{"id": r["id"], "technique": by_id[r["id"]].get("technique"), "form": by_id[r["id"]].get("form")}
           for r in batch if r["id"] in by_id]
    return out or None


def label_math(limit: int, batch: int = 25, concurrency: int = 8) -> None:
    from pipeline.teachers.runner import run_many
    rows = math_rows()
    random.Random(SEED).shuffle(rows)
    rows = rows[:limit]
    batches = [{"id": f"lb-{i}", "rows": rows[i:i + batch]} for i in range(0, len(rows), batch)]

    async def fn(item):
        return await _label_batch(item["rows"])

    teachers.check_configured(["claude"])
    DISCOVER.mkdir(parents=True, exist_ok=True)
    try:
        asyncio.run(run_many(batches, fn, MATH_LABELS, concurrency))
    finally:
        teachers.write_usage(MATH_LABELS)


# ---------------------------------------------------------------- code

def _literal(node):
    try:
        return ast.literal_eval(node)
    except Exception:
        return None


_CONTRACT = [(type(None), "None or a value depending on input"), (bool, "boolean"),
             (float, "float rounded to given decimals"), (int, "single number"), (str, "exactly formatted string"),
             (list, "ordered list"), (tuple, "tuple"), (dict, "dictionary")]  # bool before int: bool is an int


def code_contract(tests: list[str]) -> str:
    """The return type the asserts compare against; any None result marks a 'None or a value' contract."""
    kinds = Counter()
    for t in tests:
        try:
            node = ast.parse(t.strip()).body[0]
        except (SyntaxError, IndexError):
            continue
        if not isinstance(node, ast.Assert):
            continue
        test = node.test
        if not (isinstance(test, ast.Compare) and len(test.comparators) == 1):
            kinds["boolean"] += 1  # assert f(x) / assert not f(x)
            continue
        v = _literal(test.comparators[0])
        label = next((lab for typ, lab in _CONTRACT if isinstance(v, typ)), None)
        if label and (v is not None or isinstance(test.comparators[0], ast.Constant)):
            kinds[label] += 1
    if kinds["None or a value depending on input"]:
        return "None or a value depending on input"
    return kinds.most_common(1)[0][0] if kinds else "single number"


def code_regimes(tests: list[str]) -> set[str]:
    s = " ".join(tests)
    out = {"typical inputs"}
    if re.search(r"\(\s*(\[\]|''|\"\"|\{\}|\(\))\s*[,)]", s):
        out.add("empty or degenerate inputs")
    if re.search(r"[(\[, ]-\d", s):
        out.add("negative numbers and zero")
    if re.search(r"\d\.\d", s):
        out.add("floating point values")
    if re.search(r"'[^']*[A-Z][^']*[ ,.!?][^']*'", s):
        out.add("mixed case, spaces and punctuation in strings")
    if re.search(r"\[\[\[|\[\(\[|\{'[^']*': \{", s):
        out.add("deeply nested structures")
    if re.search(r"\d{7,}", s) or len(s) > 1500:
        out.add("large values or long inputs")
    return out


# ---------------------------------------------------------------- build

def _form_of(row_id: str) -> str | None:
    m = re.search(r"bt-(\d+)-(\d+)", row_id)
    return gif.FORMS[int(m.group(2))] if m and int(m.group(2)) < len(gif.FORMS) else None


from instruction_following_eval.instructions_util import LANGUAGE_CODES as LANG


def build() -> dict:
    cov: dict[str, dict[str, Counter]] = {t: {} for t in FACETS}

    def add(task, facet, value, k=1):
        cov[task].setdefault(facet, Counter())[value] += k

    # IF: every training response file we may use
    for f in ("if_responses.jsonl", "v2_if_count.jsonl", "v2_if_new.jsonl"):
        p = DATA / f
        if not p.exists():
            continue
        for r in jsonl.read(p):
            ids = [c["id"] for c in r["constraints"]]
            for g in {group_of(i) for i in ids}:
                add("if", "family", g)
            add("if", "n_constraints", str(len(ids)))
            form = _form_of(r.get("id", ""))
            if form:
                add("if", "genre", form)
            lang = next((c["kwargs"].get("language") for c in r["constraints"] if c["id"] == "language:response_language"), None)
            add("if", "language", LANG.get(lang, lang) if lang else "English")
            add("if", "wording", "standard template wording" if r["prompt"].startswith(r["task"].strip()[:40])
                else "paraphrased in casual language")
            add("if", "position", "constraints after the task")

    # code
    for r in jsonl.read(DATA / "code_tasks.jsonl"):
        add("code", "topic", r.get("category", "unknown"))
        add("code", "output_contract", code_contract(r["tests"] + r.get("hidden_tests", [])))
        for reg in code_regimes(r["tests"] + r.get("hidden_tests", [])):
            add("code", "input_regime", reg)
        add("code", "prompt_style", "MBPP one-line description")

    # math
    rows = math_rows()
    for r in rows:
        add("math", "subject", r["subject"])
        add("math", "level", r["level"])
        add("math", "answer_type", answer_type(r["answer"]))
    if MATH_LABELS.exists():
        labels = list(jsonl.read(MATH_LABELS))
        scale = len(rows) / max(1, len(labels))  # labels are a uniform sample: scale to the pool
        for lab in labels:
            add("math", "technique", str(lab.get("technique")), scale)
            add("math", "form", str(lab.get("form")), scale)
    out = {t: {f: dict(c.most_common()) for f, c in fs.items()} for t, fs in cov.items()}
    DISCOVER.mkdir(parents=True, exist_ok=True)
    for t, fs in out.items():
        (DISCOVER / f"coverage_{t}.json").write_text(json.dumps(fs, indent=1))
    return out


def count(task: str, facet: str, value: str) -> float:
    p = DISCOVER / f"coverage_{task}.json"
    if not p.exists():
        return 0.0
    return float(json.loads(p.read_text()).get(facet, {}).get(value, 0))


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("command", choices=["build", "label-math"])
    p.add_argument("--limit", type=int, default=1000)
    a = p.parse_args()
    if a.command == "label-math":
        label_math(a.limit)
    out = build()
    for t, fs in out.items():
        for f, c in fs.items():
            known = FACETS[t].get(f, [])
            zero = [v for v in known if v not in c]
            print(f"{t}.{f}: {len(c)} values seen; never in training: {zero}")


if __name__ == "__main__":
    main()
