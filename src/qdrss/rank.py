"""Categorical relevance filter over the candidate pool.

Every candidate gets a verdict (strong / relevant / exclude); there is no
per-batch quota. A batch whose model call fails raises: the caller turns that
into a 503 rather than an empty feed that looks like "nothing new".
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import AsyncIterator
from typing import Any

from .models import Candidate, Match
from .providers import Ranker

INSTRUCTION = (
    "Act as a relevance filter, not a ranker. For every record in rs, classify m as "
    "strong, relevant, or exclude. Strong means the record directly satisfies the "
    "standing query q; relevant means it is substantially useful but partial. Exclude "
    "weak, tangential, or merely keyword-overlapping records. Return only JSON as "
    '{"results":[{"i":0,"m":"strong","why":"..."}]}, one entry per record. Each why is '
    "one sentence of at most 25 words naming the specific topic, product, event, or "
    "finding that matches. Never refer to 'the record' or 'the query'. Use only "
    "supplied facts."
)

CATEGORIES = ("strong", "relevant")
logger = logging.getLogger(__name__)


class RankingError(RuntimeError):
    pass


async def classify(
    query: str, candidates: list[Candidate], ranker: Ranker, batch_size: int = 10
) -> list[Match]:
    if not candidates:
        return []
    decide = getattr(ranker, "decide", None)
    if decide is not None:
        # A decision model (decisions.py) scores the whole page in one call and writes no reasons.
        try:
            return await decide(query, candidates)
        except Exception as exc:  # any failure is a failed ranking: 503, not an empty list
            raise RankingError(f"{type(exc).__name__}: {exc}") from exc
    batches = [candidates[i : i + batch_size] for i in range(0, len(candidates), batch_size)]
    results = await asyncio.gather(
        *(_classify_batch(query, batch, ranker) for batch in batches), return_exceptions=True
    )
    matches: list[Match] = []
    for result in results:
        if isinstance(result, BaseException):
            raise RankingError(f"{type(result).__name__}: {result}") from result
        matches.extend(result)
    return matches


async def classify_stream(
    query: str, candidates: list[Candidate], ranker: Ranker, batch_size: int = 10
) -> AsyncIterator[list[Match]]:
    """The matches of each batch as soon as that batch is judged (batches run at once, so the slowest one
    no longer holds back the rest). A decision model judges the whole page in one call: one yield."""
    if not candidates:
        return
    if getattr(ranker, "decide", None) is not None:
        yield await classify(query, candidates, ranker, batch_size)
        return
    tasks = [asyncio.ensure_future(_classify_batch(query, candidates[i : i + batch_size], ranker))
             for i in range(0, len(candidates), batch_size)]
    try:
        for done in asyncio.as_completed(tasks):
            try:
                yield await done
            except Exception as exc:  # any failed batch is a failed ranking, as in classify
                raise RankingError(f"{type(exc).__name__}: {exc}") from exc
    finally:
        for task in tasks:
            task.cancel()


async def _classify_batch(query: str, batch: list[Candidate], ranker: Ranker) -> list[Match]:
    payload = {
        "q": query,
        "rs": [
            {"i": i, "title": c.item.title, "text": c.item.content[:1500]}
            for i, c in enumerate(batch)
        ],
    }
    # One retry: a model occasionally returns truncated or malformed JSON, and a
    # second call almost always succeeds. A second failure surfaces as a 503.
    try:
        output = await ranker.structured(INSTRUCTION, payload)
    except Exception as first:  # noqa: BLE001 - any provider failure gets one retry
        logger.warning("ranking batch failed once (%s: %s); retrying", type(first).__name__, str(first)[:120])
        output = await ranker.structured(INSTRUCTION, payload)
    items: Any = output.get("results", [])
    if not isinstance(items, list):
        raise RankingError("ranker returned no results list")
    matches: list[Match] = []
    seen: set[int] = set()
    for entry in items:
        if not isinstance(entry, dict):
            continue
        try:
            i = int(entry.get("i"))
        except (TypeError, ValueError):
            continue
        if i in seen or not 0 <= i < len(batch):
            continue
        seen.add(i)
        category = str(entry.get("m", "")).lower()
        if category not in CATEGORIES:
            continue
        matches.append(Match(batch[i].item, category, str(entry.get("why", "")).strip()))
    return matches
