"""Facet taxonomy per task, and a covering selection of cells to probe.

A cell is one value per facet. We never probe the full product; `select_cells` picks a small set in
which every facet value appears and as many value pairs as possible co-occur (greedy pairwise cover).
Explorer values we accepted (data/discover/proposals_<task>.json, status "accepted") join the facets at runtime.
"""

import itertools
import json
import random

from pipeline.config import DATA, SEED

DISCOVER = DATA / "discover"

FACETS: dict[str, dict[str, list[str]]] = {
    "math": {
        "subject": ["Algebra", "Counting & Probability", "Geometry", "Intermediate Algebra",
                    "Number Theory", "Prealgebra", "Precalculus"],
        "technique": [
            "casework", "complementary counting", "modular arithmetic", "Vieta's formulas", "telescoping sums",
            "coordinate geometry", "trigonometric identities", "complex numbers", "AM-GM or other inequalities",
            "functional equations", "recursive sequences", "expected value", "similar triangles",
            "3D solid geometry", "floor and ceiling functions", "polynomial remainder and factor theorem",
            "logarithms and exponents", "vectors and matrices", "systems of equations", "geometric probability",
            "digit and base problems", "divisibility and gcd/lcm", "circle theorems", "arithmetic and geometric series",
        ],
        "answer_type": ["integer", "fraction", "radical expression", "polynomial or algebraic expression",
                        "interval or inequality", "ordered pair or tuple", "several values separated by commas",
                        "angle in degrees", "decimal or money amount", "expression in pi"],
        "form": ["word problem", "symbolic computation", "geometry described in words (no diagram)",
                 "find all solutions / count solutions"],
        "level": ["3", "4", "5"],
    },
    "code": {
        "topic": [
            # the 40 categories already in the pool
            "string manipulation", "string searching and counting", "string formatting and parsing",
            "list filtering", "list transformation", "list aggregation", "sorting with custom keys",
            "searching in sorted lists", "nested lists and matrices", "tuples", "lists of tuples", "dictionaries",
            "dictionaries of lists", "sets", "number theory", "primes and divisors", "digits of integers",
            "arithmetic and rounding", "sequences and series", "combinatorics", "geometry formulas",
            "bit manipulation", "base conversion", "recursion", "simple dynamic programming",
            "two-pointer techniques", "sliding windows", "stacks and queues", "regular expressions",
            "character classification", "palindromes and anagrams", "dates and times arithmetic",
            "intervals and ranges", "frequency counting", "removing duplicates", "merging and zipping collections",
            "boolean checks and validation", "math with fractions and gcd", "text tokenizing and word statistics",
            "heap and priority selection",
            # not covered by the pool
            "graphs and BFS/DFS on adjacency lists", "grid and 2D board simulation", "run-length and simple encodings",
            "parsing structured text (CSV, key=value, logs)", "matrix arithmetic", "trees given as nested tuples or dicts",
            "greedy scheduling and selection", "binary search on the answer", "generating permutations, subsets or combinations",
            "roman numerals and number words", "bank-account or inventory style state updates", "polynomial arithmetic on coefficient lists",
        ],
        "input_regime": ["typical inputs", "empty or degenerate inputs", "negative numbers and zero", "floating point values",
                         "mixed case, spaces and punctuation in strings", "deeply nested structures", "large values or long inputs"],
        "output_contract": ["single number", "boolean", "exactly formatted string", "ordered list", "tuple",
                            "dictionary", "float rounded to given decimals", "None or a value depending on input"],
        "prompt_style": ["MBPP one-line description", "HumanEval-style signature with docstring and examples",
                         "longer specification with explicit rules"],
    },
    "if": {
        # constraint groups of pipeline.constraints.sample (doubles, position, tags exist only as dev-only types);
        # the count regime (exact / bound / none) is recorded per probe and sliced afterwards
        "family": ["length", "sentence", "structure", "keyword", "chars", "start", "end", "wrap", "format", "digits",
                   "case", "letters", "repeat", "kv", "language", "brackets", "placeholders", "wordlen",
                   "doubles", "position", "tags"],
        "n_constraints": ["1", "2", "3", "4"],
        # the 15 FORMS of the training pool, then genres the pool never asks for
        "genre": ["a short essay", "an email to a colleague", "a short story", "a product description",
                  "an explanation for a beginner", "a speech", "a poem", "a review", "a letter to a friend",
                  "a blog post", "a set of instructions", "a list of tips", "a dialogue between two people",
                  "an advertisement", "a summary of the key ideas",
                  "a table comparing options", "an answer to a factual question", "an explanation of a piece of code",
                  "a technical explanation for experts", "a riddle or joke", "a social media post"],
        # languages the IFEval language checker supports (no Chinese there)
        "language": ["English", "Spanish", "German", "Japanese", "Hindi"],
        "wording": ["standard template wording", "paraphrased in casual language", "paraphrased formally with extra context"],
        "position": ["constraints after the task", "constraints before the task", "constraints mixed into the task"],
    },
}


def _proposals(task: str) -> dict[str, list[str]]:
    path = DISCOVER / f"proposals_{task}.json"
    if not path.exists():
        return {}
    data = json.loads(path.read_text())
    return {f: [p["value"] for p in vals if p.get("status") == "accepted"] for f, vals in data.get("facets", {}).items()}


def facets(task: str, with_proposals: bool = True) -> dict[str, list[str]]:
    out = {f: list(v) for f, v in FACETS[task].items()}
    if with_proposals:
        for f, vals in _proposals(task).items():
            out.setdefault(f, [])
            out[f] += [v for v in vals if v not in out[f]]
    return out


def select_cells(task: str, n: int, seed: int = SEED, with_proposals: bool = True,
                 must_include: dict[str, list[str]] | None = None, valid=None) -> list[dict]:
    """Greedy pairwise cover: every value at least once, then maximise new value pairs per cell.

    must_include: facet -> values that each get one extra cell, anchored first (e.g. values with no training data).
    valid: optional predicate rejecting impossible cells (e.g. English-letter rules in a Hindi answer).
    """
    fx = facets(task, with_proposals)
    names = list(fx)
    n = max(n, max(len(v) for v in fx.values()))  # enough cells for every value to appear once
    rng = random.Random(f"{seed}-{task}")
    uncovered_pairs = {((a, va), (b, vb)) for a, b in itertools.combinations(names, 2) for va in fx[a] for vb in fx[b]}
    unseen_values = {(f, v) for f in names for v in fx[f]}
    cells: list[dict] = []

    def score(cell):
        pairs = {((a, cell[a]), (b, cell[b])) for a, b in itertools.combinations(names, 2)}
        vals = {(f, cell[f]) for f in names}
        return 10 * len(vals & unseen_values) + len(pairs & uncovered_pairs)

    forced = [(f, v) for f, vs in (must_include or {}).items() for v in vs]
    n += len(forced)  # forced values get cells of their own on top of the covering set
    while len(cells) < n:
        best, best_s = None, -1
        anchor = forced.pop(0) if forced else None
        for _ in range(300):
            cand = {f: rng.choice(fx[f]) for f in names}
            if anchor:
                cand[anchor[0]] = anchor[1]
            if valid and not valid(cand):
                continue
            s = score(cand)
            if s > best_s:
                best, best_s = cand, s
        cells.append(best)
        unseen_values -= {(f, best[f]) for f in names}
        uncovered_pairs -= {((a, best[a]), (b, best[b])) for a, b in itertools.combinations(names, 2)}
    for i, c in enumerate(cells):
        c["cell_id"] = f"{task}-c{i:03d}"
    return cells


def describe(task: str, cell: dict) -> str:
    return "; ".join(f"{f}: {cell[f]}" for f in facets(task) if f in cell)
