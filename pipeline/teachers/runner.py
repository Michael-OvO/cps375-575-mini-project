"""Run an async function over many items with bounded concurrency and resume."""

import asyncio
import logging

from pipeline import jsonl

logger = logging.getLogger(__name__)


async def run_many(items: list[dict], fn, out_path, concurrency: int = 16) -> None:
    done = jsonl.done_ids(out_path)
    todo = [it for it in items if it["id"] not in done]
    sem = asyncio.Semaphore(concurrency)
    finished = 0

    async def one(item: dict) -> None:
        nonlocal finished
        async with sem:
            try:
                rows = await fn(item)
            except Exception as exc:  # leave the item un-done so a rerun retries it
                logger.warning("item %s failed: %r", item["id"], exc)
                return
        for row in rows or []:
            jsonl.append(out_path, row)
        jsonl.mark_done(out_path, item["id"])
        finished += 1
        if finished % 200 == 0:
            logger.info("%d/%d done -> %s", finished, len(todo), out_path)

    await asyncio.gather(*(one(it) for it in todo))
