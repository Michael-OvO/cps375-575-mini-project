"""Minimal JSONL helpers with done-id tracking for resumable jobs."""

import json
import logging
from pathlib import Path

logger = logging.getLogger(__name__)


def read(path) -> list[dict]:
    """All rows. A truncated or undecodable final line (an interrupted write) is skipped with a
    warning; a bad line anywhere else still raises."""
    path = Path(path)
    if not path.exists():
        return []
    with path.open(encoding="utf-8") as f:
        lines = [line for line in f if line.strip()]
    rows = []
    for i, line in enumerate(lines):
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            if i != len(lines) - 1:
                raise
            logger.warning("skipping undecodable final line of %s", path)
    return rows


def _repair_tail(path: Path) -> None:
    """Before appending: a final line without a newline is a torn write. Drop it if it does not
    decode (otherwise the next row would be glued onto it), or just end it if it does."""
    if not path.exists() or path.stat().st_size == 0:
        return
    with path.open("rb") as f:
        f.seek(-1, 2)
        if f.read(1) == b"\n":
            return  # the common case: one byte read
    data = path.read_bytes()
    start = data.rfind(b"\n") + 1
    try:
        json.loads(data[start:].decode("utf-8"))
    except (json.JSONDecodeError, UnicodeDecodeError):
        logger.warning("dropping torn final line of %s", path)
        with path.open("r+b") as f:
            f.truncate(start)
        return
    with path.open("ab") as f:
        f.write(b"\n")


def append(path, row: dict) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    _repair_tail(path)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")


def append_many(path, rows) -> None:
    """Append all rows in a single write call so an item's rows land together."""
    rows = list(rows)
    if not rows:
        return
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    data = "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows)
    _repair_tail(path)
    with path.open("a", encoding="utf-8") as f:
        f.write(data)


def write(path, rows) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def _done_path(path) -> Path:
    return Path(str(path) + ".done")


def done_ids(path) -> set[str]:
    p = _done_path(path)
    return set(p.read_text(encoding="utf-8").split()) if p.exists() else set()


def mark_done(path, item_id: str) -> None:
    p = _done_path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("a", encoding="utf-8") as f:
        f.write(item_id + "\n")
