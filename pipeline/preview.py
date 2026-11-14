"""Write a browsable HTML preview of the generated data pools.

Shows examples of each kind of training data next to the public source it was built from,
plus the distributions we chose. Reads only training-side data: the MATH *train* split and
our own pools under data/. It never loads an evaluation set.

Usage: python -m pipeline.preview            (writes reports/data_preview.html)
"""

import html
import random
import statistics
from collections import Counter

from pipeline import formats, jsonl
from pipeline.config import DATA, ROOT, SEED

OUT = ROOT / "reports" / "data_preview.html"

CSS = """
:root { --bg:#fbfaf7; --fg:#1d1c1a; --muted:#6b675f; --card:#ffffff; --line:#e4e0d8; --accent:#8a4b2a;
        --tag:#f1ece2; --good:#2f6b3c; }
@media (prefers-color-scheme: dark) {
  :root { --bg:#17171a; --fg:#e9e6df; --muted:#a09b90; --card:#202024; --line:#34343a; --accent:#e0a37d;
          --tag:#2a2a30; --good:#8fcf9c; } }
* { box-sizing: border-box; }
body { background:var(--bg); color:var(--fg); font:16px/1.55 -apple-system, "Segoe UI", sans-serif;
       margin:0; padding:32px 16px 80px; }
main { max-width: 1040px; margin: 0 auto; }
h1 { font-size: 30px; margin: 0 0 6px; }
h2 { font-size: 22px; margin: 48px 0 6px; padding-top: 18px; border-top: 1px solid var(--line); }
h3 { font-size: 16px; margin: 26px 0 8px; color: var(--accent); }
p, li { max-width: 72ch; }
.muted { color: var(--muted); }
.grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(300px, 1fr)); gap: 14px; margin: 12px 0; }
.card { background: var(--card); border: 1px solid var(--line); border-radius: 10px; padding: 14px 16px;
        min-width: 0; }
.label { font-size: 12px; letter-spacing: .06em; text-transform: uppercase; color: var(--muted);
         margin-bottom: 6px; }
.text { white-space: pre-wrap; overflow-wrap: anywhere; font-size: 14.5px; }
.code { white-space: pre-wrap; overflow-wrap: anywhere; font: 13px/1.5 ui-monospace, Menlo, monospace; }
.tag { display: inline-block; background: var(--tag); border-radius: 999px; padding: 1px 9px; font-size: 12px;
       margin: 0 4px 4px 0; }
table { border-collapse: collapse; margin: 10px 0; font-size: 14px; }
th, td { border-bottom: 1px solid var(--line); padding: 5px 12px; text-align: right; }
th:first-child, td:first-child { text-align: left; }
.scroll { overflow-x: auto; }
.note { border-left: 3px solid var(--accent); padding: 4px 14px; margin: 14px 0; }
.ok { color: var(--good); font-weight: 600; }
nav a { margin-right: 14px; color: var(--accent); }
"""

MATHJAX = """
<script>window.MathJax = { tex: { inlineMath: [['$','$'],['\\\\(','\\\\)']], displayMath: [['$$','$$'],['\\\\[','\\\\]']] },
  options: { skipHtmlTags: ['script','noscript','style','textarea'] } };</script>
<script async src="https://cdn.jsdelivr.net/npm/mathjax@3.2.2/es5/tex-chtml.js"></script>
"""


def esc(text) -> str:
    return html.escape(str(text))


def card(label: str, body: str, mono: bool = False) -> str:
    cls = "code" if mono else "text"
    return f'<div class="card"><div class="label">{esc(label)}</div><div class="{cls}">{esc(body)}</div></div>'


def table(header: list[str], rows: list[list]) -> str:
    head = "".join(f"<th>{esc(h)}</th>" for h in header)
    body = "".join("<tr>" + "".join(f"<td>{esc(c)}</td>" for c in r) + "</tr>" for r in rows)
    return f'<div class="scroll"><table><tr>{head}</tr>{body}</table></div>'


def pct(n: int, total: int) -> str:
    return f"{100 * n / total:.0f}%" if total else "-"


def median_len(texts: list[str]) -> int:
    return int(statistics.median(len(t) for t in texts)) if texts else 0


def math_section() -> str:
    from datasets import load_dataset

    from pipeline.generate import math as gm

    raw = load_dataset(gm.MATH_DATASET, split="train")  # train split only
    reference = {f"mt-{i}": row["solution"] for i, row in enumerate(raw)}
    train = gm.load_train()
    by_id = {r["id"]: r for r in train}
    solved = jsonl.read(gm.TRAIN_SOLUTIONS)
    new_problems = jsonl.read(gm.NEW_PROBLEMS)
    new_solutions = jsonl.read(gm.NEW_SOLUTIONS)
    rng = random.Random(SEED)
    out = ['<h2 id="math">1. Math</h2>']
    out.append(
        "<p>The public benchmark (MATH-500) is 500 problems sampled from the MATH <em>test</em> split. "
        "We never load it. Everything below is built from the MATH <em>train</em> split, which is the "
        "same source and style: 7 subjects, 5 difficulty levels, one final answer per problem.</p>"
    )

    subjects = sorted({r["subject"] for r in train})
    dist = Counter((r["subject"], r["level"]) for r in train if not r["dev"])
    new_dist = Counter((r["subject"], r["level"]) for r in new_problems)
    solved_pids = {r["pid"] for r in solved}
    rows = []
    for s in subjects:
        n_train = sum(dist[(s, lv)] for lv in range(1, 6))
        n_solved = sum(1 for p in solved_pids if by_id.get(p, {}).get("subject") == s)
        n_new = sum(new_dist[(s, lv)] for lv in range(1, 6))
        rows.append([s] + [dist[(s, lv)] for lv in range(1, 6)] + [n_train, n_solved, n_new])
    out.append("<h3>Where the problems come from</h3>")
    out.append(table(["Subject", "L1", "L2", "L3", "L4", "L5", "Train problems", "Solved so far",
                      "New problems written"], rows))
    lv_train = Counter(r["level"] for r in train if not r["dev"])
    lv_new = Counter(r["level"] for r in new_problems)
    out.append("<h3>Difficulty: original train split versus the new problems we ask for</h3>")
    out.append(table(["Level", "MATH train", "share", "New problems", "share"],
                     [[f"Level {lv}", lv_train[lv], pct(lv_train[lv], sum(lv_train.values())),
                       lv_new[lv], pct(lv_new[lv], sum(lv_new.values()))] for lv in range(1, 6)]))
    out.append('<p class="note">New problems are requested in proportion to difficulty level (level 5 five '
               "times as often as level 1), because hard problems are where a small model has the most to "
               "learn and where plain sampling leaves the fewest examples.</p>")

    out.append("<h3>The same train problem: dataset reference solution versus the teacher's solution</h3>")
    ref_lens = [reference[r["pid"]] for r in solved if r["pid"] in reference]
    out.append(f'<p class="muted">Median length so far: reference solutions {median_len(ref_lens)} characters, '
               f"teacher solutions {median_len([r['solution'] for r in solved])} characters. The teacher is asked "
               "for a concise, numbered, step-by-step solution in LaTeX; its final answer must match the "
               "dataset's answer under the benchmark's own scorer or the row is discarded.</p>")
    picks = []
    for lv in (2, 4, 5):
        cands = [r for r in solved if r["level"] == lv and r["pid"] in reference]
        if cands:
            picks.append(rng.choice(cands))
    for r in picks:
        out.append(f'<p><span class="tag">{esc(r["subject"])}</span><span class="tag">level {r["level"]}</span>'
                   f'<span class="tag">{esc(r["pid"])}</span></p>')
        out.append('<div class="grid">' + card("Problem (MATH train)", r["problem"])
                   + card("Dataset reference solution", reference[r["pid"]])
                   + card("Teacher solution (training target body)", r["solution"])
                   + card("Final line the student is trained to write",
                          f"ANSWER: {r.get('answer') or r['target']}\n\n(dataset answer string: {r['target']})")
                   + "</div>")

    if picks:
        r = picks[0]
        out.append("<h3>What one training example looks like to the student</h3>")
        prompt = formats.math_prompt(r["problem"])
        target = formats.math_target(r["solution"], r.get("answer") or r["target"])
        out.append('<div class="grid">' + card("User turn (exact evaluation-harness wording)", prompt)
                   + card("Assistant turn (what the loss is computed on)", target) + "</div>")
        out.append('<p class="muted">About 70% of math examples use this exact harness wording and 30% use one '
                   "of four paraphrases, so the model does not depend on a single phrasing.</p>")

    if new_problems:
        out.append("<h3>New problems written by the teacher</h3>")
        out.append("<p>Each call shows the teacher three train problems from one subject and level and asks for "
                   "five new ones in the same style. A new problem is kept only when two independent solves "
                   "agree on the answer (a third, higher-effort solve breaks ties).</p>")
        calls = {c["id"]: c for c in gm._write_calls([t for t in train if not t["dev"]], ["claude"], 8000)}
        sol_by_pid = {}
        for s in new_solutions:
            sol_by_pid.setdefault(s["pid"], []).append(s)
        shown = 0
        for call_id in sorted({p["id"].rsplit("-", 1)[0] for p in new_problems}):
            call = calls.get(call_id)
            probs = [p for p in new_problems if p["id"].rsplit("-", 1)[0] == call_id]
            if not call or not probs:
                continue
            out.append(f'<p><span class="tag">{esc(call["subject"])}</span>'
                       f'<span class="tag">level {call["level"]}</span></p>')
            cells = [card("Seed (MATH train)", call["seeds"][0]), card("Seed (MATH train)", call["seeds"][1])]
            for p in probs[:2]:
                sols = sol_by_pid.get(p["id"], [])
                verdict = (f"kept: {len(sols)} agreeing solves, answer {sols[0]['target']}" if sols
                           else "dropped: solves did not agree")
                cells.append(card(f"New problem ({verdict})", p["problem"]))
            out.append('<div class="grid">' + "".join(cells) + "</div>")
            shown += 1
            if shown == 3:
                break
        kept = len({s["pid"] for s in new_solutions})
        out.append(f'<p class="muted">Pilot so far: {len(new_problems)} new problems written, {kept} kept after '
                   "agreement.</p>")
    return "\n".join(out)


def code_section() -> str:
    raw = jsonl.read(DATA / "code_tasks_raw.jsonl")
    kept = jsonl.read(DATA / "code_tasks.jsonl")
    out = ['<h2 id="code">2. Code</h2>']
    out.append(
        "<p>The public benchmark (MBPP) gives a one-sentence task plus three assert statements that fix the "
        "function name and signature; the model must reply with just the function. We load no MBPP data in any "
        "split. The teacher writes fresh tasks in that shape across 40 categories; each is kept only if its "
        "reference solution runs and passes its own three tests, and then passes extra hidden tests written in "
        "a separate call that never sees the solution.</p>"
    )
    if not raw:
        return "\n".join(out + ["<p class='muted'>No code tasks generated yet.</p>"])
    out.append(f"<p>Pilot so far: <strong>{len(raw)}</strong> tasks passed their own tests; "
               f"<strong>{len(kept)}</strong> ({pct(len(kept), len(raw))}) also passed the hidden tests.</p>")
    words = [len(t["text"].split()) for t in raw]
    out.append(f'<p class="muted">Task descriptions: median {int(statistics.median(words))} words '
               f"(shortest {min(words)}, longest {max(words)}). MBPP's own descriptions are typically around "
               "10 to 20 words and often leave details to the tests. Ours run longer and spell out edge cases, "
               "which is a deliberate difference to watch: see the note below.</p>")
    cats = Counter(t["category"] for t in raw)
    kept_cats = Counter(t["category"] for t in kept)
    out.append(table(["Category (pilot)", "Passed own tests", "Passed hidden tests"],
                     [[c, n, kept_cats[c]] for c, n in cats.most_common()]))
    rng = random.Random(SEED)
    out.append("<h3>Examples</h3>")
    for t in rng.sample(kept, min(4, len(kept))):
        out.append(f'<p><span class="tag">{esc(t["category"])}</span><span class="tag">{esc(t["id"])}</span></p>')
        out.append('<div class="grid">' + card("Task description", t["text"])
                   + card("Visible tests (shown to the student in the prompt)", "\n".join(t["tests"]), mono=True)
                   + card("Reference solution (training target)", t["code"], mono=True)
                   + card("Hidden tests (used only to verify, and to score the private dev set)",
                          "\n".join(t["hidden_tests"]), mono=True) + "</div>")
    t = kept[0]
    out.append("<h3>What one training example looks like to the student</h3>")
    out.append('<div class="grid">' + card("User turn (harness layout, our own few-shot examples)",
                                           formats.code_prompt(t["text"], t["tests"]), mono=True)
               + card("Assistant turn", formats.code_target(t["code"]), mono=True) + "</div>")
    out.append('<p class="note">Open question this preview is meant to surface: our descriptions are longer and '
               "more specific than MBPP's. That makes tasks unambiguous (good for verification) but shifts the "
               "style away from the benchmark's terse one-liners. If you want them closer to MBPP, the fix is "
               "one prompt change before the full run.</p>")
    return "\n".join(out)


BASE_TASKS = [
    "Write a short blog post about why libraries still matter in small towns.",
    "Explain how a bicycle stays upright to a curious ten-year-old.",
    "Write an email to a colleague proposing a weekly team lunch.",
    "Describe the water cycle for a museum exhibit label.",
    "Write a product description for a reusable coffee cup.",
    "Give a list of tips for someone preparing for their first job interview.",
]


def if_section() -> str:
    from pipeline.constraints import extra
    from pipeline.constraints.sample import STANDARD_IDS, render_prompt, sample_constraints

    out = ['<h2 id="if">3. Instruction following</h2>']
    out.append(
        "<p>The public benchmark (IFEval) attaches one to three machine-checkable constraints to a writing "
        "request and scores only whether the constraints are met. We load none of its prompts. We reuse the "
        "open-source <em>checker code</em> for its 25 constraint types, add 36 types of our own with their own "
        "checkers, and attach them to fresh writing requests. A teacher response is kept only if every "
        "constraint passes in strict mode.</p>"
    )
    rng = random.Random(SEED)
    k_counts, type_counts = Counter(), Counter()
    n = 3000
    for _ in range(n):
        cons = sample_constraints(rng, BASE_TASKS[0], mode="train")
        k_counts[len(cons)] += 1
        for c in cons:
            type_counts[c["id"]] += 1
    total_c = sum(type_counts.values())
    std = sum(v for k, v in type_counts.items() if k in STANDARD_IDS)
    out.append("<h3>How many constraints per prompt (3,000 sampled training prompts)</h3>")
    out.append(table(["Constraints", "Prompts", "share"], [[k, v, pct(v, n)] for k, v in sorted(k_counts.items())]))
    train_extra = [e for e in extra.REGISTRY.values() if not e.dev_only]
    dev_only = [e for e in extra.REGISTRY.values() if e.dev_only]
    out.append(f"<p>Constraint instances: {pct(std, total_c)} from the 25 standard types, "
               f"{pct(total_c - std, total_c)} from our {len(train_extra)} new training types. "
               f"{len(dev_only)} further types are never trained on and appear only in the private dev set, "
               "to measure whether the skill transfers to constraints the model has not seen.</p>")
    top = type_counts.most_common()
    out.append(table(["Constraint type", "Times sampled", "Kind"],
                     [[k, v, "standard" if k in STANDARD_IDS else "new"] for k, v in top[:12]]
                     + [["...", "", ""]]
                     + [[k, v, "standard" if k in STANDARD_IDS else "new"] for k, v in top[-6:]]))
    out.append("<h3>Our new constraint types</h3><p>"
               + "".join(f'<span class="tag">{esc(e.describe(e.sample(random.Random(1))))}</span>' for e in train_extra)
               + "</p><p class='muted'>Held out for the dev set only:</p><p>"
               + "".join(f'<span class="tag">{esc(e.describe(e.sample(random.Random(1))))}</span>' for e in dev_only)
               + "</p>")
    out.append("<h3>Example prompts as the sampler composes them</h3>")
    cells = []
    for i, task in enumerate(BASE_TASKS):
        cons = sample_constraints(rng, task, mode="dev_unseen" if i == len(BASE_TASKS) - 1 else "train")
        tags = ", ".join(c["id"] for c in cons)
        label = "Dev-only example (contains a held-out type)" if i == len(BASE_TASKS) - 1 else "Training prompt"
        cells.append(card(f"{label}: {tags}", render_prompt(task, cons)))
    out.append('<div class="grid">' + "".join(cells) + "</div>")
    out.append('<p class="muted">The six base requests above are placeholders written for this preview; in the '
               "real run the teacher writes about 18,000 base requests across 60 topics and 15 forms.</p>")
    responses = jsonl.read(DATA / "if_responses.jsonl")
    if responses:
        out.append(f"<h3>Verified teacher responses ({len(responses)} so far)</h3>")
        for r in random.Random(SEED).sample(responses, min(3, len(responses))):
            out.append('<div class="grid">' + card("Prompt: " + ", ".join(c["id"] for c in r["constraints"]),
                                                   r["prompt"])
                       + card("Teacher response (passes every constraint, strict)", r["response"]) + "</div>")
    else:
        out.append("<p class='muted'>No teacher responses generated yet; this section fills in once the "
                   "instruction-following pilot has run.</p>")
    return "\n".join(out)


def general_section() -> str:
    rows = jsonl.read(DATA / "general.jsonl")
    out = ['<h2 id="general">4. General chat</h2>']
    out.append("<p>A small slice of ordinary single-turn conversations from the staff-suggested Tulu-3 mixture "
               "(train split), so fine-tuning on three narrow skills does not erode everyday behaviour. "
               "Short-answer task sets and refusal-focused sets are excluded and no source exceeds 40%.</p>")
    if rows:
        src = Counter(r["source"].split("/")[-1] for r in rows)
        out.append(table(["Source", "Rows", "share"], [[k, v, pct(v, len(rows))] for k, v in src.most_common()]))
    return "\n".join(out)


def overview() -> str:
    names = [("math_train_solutions", "Verified teacher solutions to MATH train problems"),
             ("math_new_problems", "New math problems written by the teacher"),
             ("math_new_solutions", "Agreed solutions to new problems"),
             ("code_tasks_raw", "Code tasks whose solution passes its own tests"),
             ("code_tasks", "Code tasks that also pass hidden tests"),
             ("if_base_tasks", "Fresh writing requests"),
             ("if_responses", "Constrained responses passing every checker"),
             ("general", "General chat conversations"),
             ("math_dev", "Private dev set: held-out MATH train problems (never trained on)")]
    rows = [[label, len(jsonl.read(DATA / f"{name}.jsonl"))] for name, label in names]
    return ("<h2>What exists right now</h2>" + table(["Pool", "Rows"], rows)
            + '<p class="muted">Generation is in progress, so counts grow; rerun '
              "<code>python -m pipeline.preview</code> to refresh this page.</p>")


def main() -> None:
    body = "\n".join([
        "<h1>Training data preview</h1>",
        '<p class="muted">How each kind of training example is built, with real rows from the pools, next to '
        "the public material it was built from. Benchmark test items are deliberately not shown: the project "
        "rules forbid using them in any way, so this page loads only train-split data and our own files. "
        "To see the benchmarks themselves, browse them on Hugging Face: "
        '<a href="https://huggingface.co/datasets/HuggingFaceH4/MATH-500">MATH-500</a>, '
        '<a href="https://huggingface.co/datasets/google-research-datasets/mbpp">MBPP</a>, '
        '<a href="https://huggingface.co/datasets/google/IFEval">IFEval</a>.</p>',
        '<nav><a href="#math">Math</a><a href="#code">Code</a><a href="#if">Instruction following</a>'
        '<a href="#general">General chat</a></nav>',
        overview(), math_section(), code_section(), if_section(), general_section(),
    ])
    page = (f'<!doctype html><html lang="en"><head><meta charset="utf-8">'
            f'<meta name="viewport" content="width=device-width, initial-scale=1">'
            f"<title>Training data preview</title><style>{CSS}</style>{MATHJAX}</head>"
            f"<body><main>{body}</main></body></html>")
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(page, encoding="utf-8")
    print(f"wrote {OUT} ({len(page) // 1024} KB)")


if __name__ == "__main__":
    main()
