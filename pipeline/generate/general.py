"""General chat slice: 2,000 single-turn conversations from the Tulu-3 SFT mixture (train split).

Usage: python -m pipeline.generate.general
"""

import random

from datasets import load_dataset

from pipeline import jsonl
from pipeline.config import DATA, SEED

OUT = DATA / "general.jsonl"
N = 2000
SCAN = 150_000
# Skip Tulu subsets that target the three graded skills; those come from our own verified data.
SKIP = ("math", "code", "persona", "ifdata", "numina", "gsm")


def main() -> None:
    rng = random.Random(SEED)
    stream = load_dataset("allenai/tulu-3-sft-mixture", split="train", streaming=True)
    kept: list[dict] = []
    seen = 0
    for row in stream:
        seen += 1
        if seen > SCAN:
            break
        source = str(row.get("source", "")).lower()
        msgs = row.get("messages") or []
        if any(s in source for s in SKIP) or len(msgs) != 2:
            continue
        user, assistant = msgs[0], msgs[1]
        if user.get("role") != "user" or assistant.get("role") != "assistant":
            continue
        if not (20 <= len(assistant["content"].split()) <= 500) or len(user["content"].split()) > 300:
            continue
        item = {"messages": [{"role": "user", "content": user["content"]},
                             {"role": "assistant", "content": assistant["content"]}],
                "source": f"tulu-3:{row.get('source', 'unknown')}"}
        # Reservoir sampling keeps a uniform sample of everything that qualifies.
        if len(kept) < N:
            kept.append(item)
        else:
            j = rng.randrange(seen)
            if j < N:
                kept[j] = item
    for i, item in enumerate(kept):
        item["id"] = f"gen-{i}"
    jsonl.write(OUT, kept)
    print(f"wrote {len(kept)} conversations to {OUT} (scanned {seen})")


if __name__ == "__main__":
    main()
