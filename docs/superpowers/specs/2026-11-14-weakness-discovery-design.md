# Weakness discovery: a structured search for what the model cannot do yet

Date: 2026-11-14. Status: design for the team.

## Why

Failure mining (v2) only finds weaknesses inside the pools we already built. The hidden benchmarks
will contain problem types we never generated. We need a repeatable way to find areas where the
model fails that our data does not cover, measure them, and decide which ones to fix.

Examples of gaps visible already, none of which current mining can see:

- Code: our 40 task categories have no graphs/BFS, no matrix or grid simulation beyond basics, and
  every prompt is MBPP-style; HumanEval-style prompts (signature plus docstring) dropped 70.7 → 67.1.
- Math: MATH problems with `[asy]` diagrams, multi-answer formats ("enter all solutions separated by
  commas"), and answers with units or degrees are not tracked separately.
- Instruction following: all our constraints use one fixed wording per type; the hidden benchmark
  will phrase rules differently. Genres such as code, tables or non-English answers are thin.

## The system

### 1. Facet taxonomy (per task)

Each task is described by independent facets, not one topic list. A cell is a combination of facet
values. The taxonomy lives in `pipeline/discover/taxonomy.py` and is versioned.

| Task | Facets |
|---|---|
| Math | subject (7); technique (about 25: casework, complementary counting, modular arithmetic, Vieta, telescoping, coordinate geometry, trig identities, complex numbers, AM-GM, functional equations, recursions, expected value, similar triangles, 3D geometry, floor functions, polynomial remainders, logarithms, vectors and matrices, ...); answer type (integer, fraction, radical, expression, interval, tuple, several values, degrees or units); form (word problem, symbolic, diagram-based) |
| Code | topic (the 40 categories plus missing ones such as graphs, grids, simulation, parsing formats, classes-free OOP-like state); input edge regime (empty, single, nested, negative, float, unicode, large); output contract (tuple vs list, ordering, float rounding, None vs raise, bool, string formatting); prompt style (MBPP one-liner, HumanEval signature plus docstring, longer specification) |
| Instruction following | constraint family; parameter regime (small or large counts, exact vs bound); number of constraints; response genre (story, email, list, table, JSON, code, poem, dialogue, technical explanation); language; constraint wording (library template vs paraphrase); constraint position (before or after the task) |

### 2. Coverage map

A cheap classifier (Opus on a 2,000-example sample per task, plus rules where facets are explicit,
such as constraint ids) tags each training example with its facet values. Output: training count
per cell, written to `data/discover/coverage_<task>.json`.

### 3. Explorer (finding cells outside the taxonomy)

Once per round, Opus receives the taxonomy, the coverage map, the staff's description of the hidden
benchmarks and the current failure summary, and proposes new facet values or cells that a benchmark
of that kind could contain and we do not cover. Proposals are added to the taxonomy as
`proposed` and reviewed by us before probing. This is the step that finds unknown unknowns.

### 4. Probe sets

For every cell selected for measurement, Opus writes about 12 fresh, verified probe items:

- Math: two independent Opus solves that agree (or a third tie-break); the answer form must match the
  cell's answer type.
- Code: description, tests and hidden tests, with a reference solution that passes them, in the
  cell's prompt style.
- Instruction following: base task plus constraints rendered in the cell's wording and position; the
  verifier decides.

Probes are generated from scratch, never seeded from any test set, and pass the same test-overlap
filter as training data. Probes are measurement data: they are never trained on.

### 5. Measurement

- v1 (and later each new checkpoint) answers every probe once, greedy, with the evaluation prompt
  and limits. Cost: cents per hundred probes on Tinker.
- Teacher pass rate on the same probes estimates learnability (a cell the teacher cannot do is not
  worth targeting).
- Per cell: failure rate with a Wilson interval, coverage count, teacher pass rate.

### 6. Ranking

priority = lower bound of failure rate × relevance × learnability × coverage gap

- relevance: 1.0 for facets matching what staff described (function generation with unit tests,
  single-answer competition math, verifiable constraints), lower for speculative ones.
- learnability: teacher pass rate on the cell's probes.
- coverage gap: higher when the cell has few training examples.

### 7. Action and registry

The top cells get training data from the existing generators, given the cell spec (topic, style,
answer type, wording). `reports/weakness_registry.md` lists every measured cell with its numbers and
a status (open, targeted in vN, fixed, not worth fixing). After each new checkpoint the same probes
are re-scored, so the registry shows whether the data fixed the weakness.

## Round 1 scope

- About 150 cells: about 50 per task, chosen to span every facet value at least once (a covering
  design, not the full product), plus the explorer's proposals.
- 12 probes per cell: about 1,800 probes. Opus usage roughly $100 to $150 at list price; v1 scoring
  under $2 on Tinker.
- Output: ranked registry and the top 10 to 15 cells, fed into v2 if ready in time, otherwise v3.

## Guards

- Never seeded from or tuned on public test items; same overlap filter as all generated data.
- Probes are measurement only; training data for a weak cell is generated separately.
- Twelve probes per cell is enough to separate a 30% from a 70% cell, not to rank close cells; close
  calls are not acted on.
