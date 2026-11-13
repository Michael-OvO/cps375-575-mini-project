"""Minimal JSONL helpers with done-id tracking for resumable jobs."""

import json
from pathlib import Path


def read(path) -> list[dict]:
    path = Path(path)
    if not path.exists():
        return []
    with path.open() as f:
        return [json.loads(line) for line in f if line.strip()]


def append(path, row: dict) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")


def write(path, rows) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def _done_path(path) -> Path:
    return Path(str(path) + ".done")


def done_ids(path) -> set[str]:
    p = _done_path(path)
    return set(p.read_text().split()) if p.exists() else set()


def mark_done(path, item_id: str) -> None:
    p = _done_path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("a") as f:
        f.write(item_id + "\n")
