from __future__ import annotations

import asyncio
from typing import Any


async def matan_gather_chunked(*coros_or_futures, chunk_size: int = 300) -> list[Any]:
    """Like asyncio.gather but processes in chunks to avoid overwhelming the event loop."""
    all_items = list(coros_or_futures)
    results: list[Any] = []
    for i in range(0, len(all_items), chunk_size):
        chunk = all_items[i : i + chunk_size]
        chunk_results = await asyncio.gather(*chunk)
        results.extend(chunk_results)
    return results


asyncio.matan_gather_chunked = matan_gather_chunked  # type: ignore[attr-defined]
