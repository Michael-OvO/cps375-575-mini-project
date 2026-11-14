"""Run an async function over many items with bounded concurrency and resume."""

import asyncio
import logging

from pipeline import jsonl
from pipeline.teachers import TeacherUnavailable

logger = logging.getLogger(__name__)

MAX_CONSECUTIVE_FAILURES = 25


async def run_many(items: list[dict], fn, out_path, concurrency: int = 16) -> None:
    done = jsonl.done_ids(out_path)
    # Ids already on disk (e.g. a crash between writing rows and the done marker) are never re-appended.
    seen = {row["id"] for row in jsonl.read(out_path) if "id" in row}
    todo = [it for it in items if it["id"] not in done]
    sem = asyncio.Semaphore(concurrency)
    finished = 0
    unprocessed = 0
    consecutive_failures = 0
    stop_reason: str | None = None  # set by the circuit breaker; new items are then skipped
    started = 0

    def trip(reason: str) -> None:
        nonlocal stop_reason
        if stop_reason is None:
            stop_reason = reason

    async def one(item: dict) -> None:
        nonlocal finished, unprocessed, consecutive_failures, started
        async with sem:
            if stop_reason is not None:
                return  # circuit breaker open: never started, stays un-done
            started += 1
            try:
                rows = await fn(item)
            except TeacherUnavailable as exc:
                logger.warning("item %s failed: %r", item["id"], exc)
                trip(f"teacher unavailable: {exc}")
                return
            except Exception as exc:  # leave the item un-done so a rerun retries it
                logger.warning("item %s failed: %r", item["id"], exc)
                consecutive_failures += 1
                if consecutive_failures >= MAX_CONSECUTIVE_FAILURES:
                    trip(f"{consecutive_failures} consecutive item failures (last: {exc!r})")
                return
        consecutive_failures = 0
        if rows is None:  # fn could not process the item: leave it un-done so a rerun retries it
            unprocessed += 1
            logger.debug("item %s returned None; not marked done", item["id"])
            return
        fresh = []
        for row in rows:
            if row["id"] not in seen:
                seen.add(row["id"])
                fresh.append(row)
        jsonl.append_many(out_path, fresh)
        jsonl.mark_done(out_path, item["id"])
        finished += 1
        if finished % 200 == 0:
            logger.info("%d/%d done -> %s", finished, len(todo), out_path)

    await asyncio.gather(*(one(it) for it in todo))
    if stop_reason is not None:
        logger.error("stopping run early, %d of %d item(s) not started and left un-done -> %s: %s",
                     len(todo) - started, len(todo), out_path, stop_reason)
    if unprocessed:
        logger.warning("%d item(s) returned None and were not marked done -> %s", unprocessed, out_path)
