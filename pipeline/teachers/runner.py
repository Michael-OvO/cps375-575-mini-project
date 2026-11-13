"""Run an async function over many items with bounded concurrency and resume."""

import asyncio
import logging

from pipeline import jsonl

logger = logging.getLogger(__name__)


async def run_many(items: list[dict], fn, out_path, concurrency: int = 16) -> None:
    done = jsonl.done_ids(out_path)
    # Ids already on disk (e.g. a crash between writing rows and the done marker) are never re-appended.
    seen = {row["id"] for row in jsonl.read(out_path) if "id" in row}
    todo = [it for it in items if it["id"] not in done]
    sem = asyncio.Semaphore(concurrency)
    finished = 0
    unprocessed = 0

    async def one(item: dict) -> None:
        nonlocal finished, unprocessed
        async with sem:
            try:
                rows = await fn(item)
            except Exception as exc:  # leave the item un-done so a rerun retries it
                logger.warning("item %s failed: %r", item["id"], exc)
                return
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
    if unprocessed:
        logger.warning("%d item(s) returned None and were not marked done -> %s", unprocessed, out_path)
