"""In-memory vector index over every stored item, rebuilt on refresh.

The date and collection mask is applied before top-K: a feed's candidates are
the best *new* items, not the new items among the best overall.
"""

from __future__ import annotations

import asyncio
from datetime import datetime

import numpy as np

from .models import Candidate, Item
from .providers import Embeddings
from .store import Store


class Index:
    """Items and their vectors (float32 rows), with the date, source and collection of each for masking.

    `matrix` may be a view on a larger `_buffer`, so new items can be appended (`extended`) without copying
    every vector: the in-memory collections (memory.py) hold several GB and take new items every few minutes.
    """

    def __init__(
        self, items: list[Item], matrix: np.ndarray, collections: dict[str, str] | None = None,
        *, _meta: tuple[np.ndarray, np.ndarray, np.ndarray] | None = None, _buffer: np.ndarray | None = None,
        _codes: dict[str, int] | None = None,
    ):
        self.items = items
        self.matrix = matrix
        self._buffer = matrix if _buffer is None else _buffer
        self._collections = collections or {}
        # Source and collection names as small integers: matching a tab's 140 sources against 500k string
        # objects took 2.5 s, against integers milliseconds. Shared with extended indexes (append-only).
        self._codes = {} if _codes is None else _codes
        # source name -> collection, from the manifest. Items whose source has
        # dropped out of the manifest belong to no collection.
        # An item's own collection wins (archive backfills file items under sources the manifest lacks).
        self._collection, self._source, self.ingested = _meta if _meta is not None else self._describe(items)

    def _describe(self, items: list[Item]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        return (
            np.asarray([self._code(item.collection or self._collections.get(item.source, "")) for item in items],
                       dtype=np.int32),
            np.asarray([self._code(item.source) for item in items], dtype=np.int32),
            np.asarray([item.ingested_at.timestamp() for item in items], dtype=np.float64),
        )

    def _code(self, name: str) -> int:
        return self._codes.setdefault(name, len(self._codes))

    def _codes_of(self, names) -> np.ndarray:
        return np.asarray([self._codes[n] for n in names if n in self._codes], dtype=np.int32)

    @classmethod
    def with_room(cls, items: list[Item], matrix: np.ndarray, collections: dict[str, str] | None = None,
                  spare: float = 0.05) -> Index:
        """An index whose float32 buffer has room for `spare` more items, filled from `matrix` (any float dtype)."""
        buffer = np.empty((len(items) + max(1024, int(len(items) * spare)), matrix.shape[1]), dtype=np.float32)
        buffer[: len(items)] = matrix
        return cls(items, buffer[: len(items)], collections, _buffer=buffer)

    def extended(self, items: list[Item], vectors: np.ndarray) -> Index:
        """This index plus `items`. The vectors go into the spare room of the buffer, which grows only when full;
        this index's own view is unchanged, so searches running on it are unaffected."""
        n, k = len(self.items), len(items)
        buffer = self._buffer
        if n + k > len(buffer):
            buffer = np.empty((n + k + max(1024, (n + k) // 10), self.matrix.shape[1]), dtype=np.float32)
            buffer[:n] = self.matrix
        buffer[n : n + k] = vectors
        meta = tuple(np.concatenate([old, new]) for old, new in
                     zip((self._collection, self._source, self.ingested), self._describe(items), strict=True))
        return Index(self.items + items, buffer[: n + k], self._collections, _meta=meta, _buffer=buffer,  # type: ignore[arg-type]
                     _codes=self._codes)

    def __len__(self) -> int:
        return len(self.items)

    def eligible(
        self, since: datetime | None, collection: str | None = None, sources: frozenset[str] | None = None,
        collections: frozenset[str] | None = None,
    ) -> np.ndarray:
        mask = np.ones(len(self.items), dtype=bool)
        if since is not None:
            mask &= self.ingested > since.timestamp()
        if collection:
            mask &= self._collection == self._codes.get(collection, -1)
        if sources is not None:
            mask &= np.isin(self._source, self._codes_of(sources))
        if collections is not None:
            mask &= np.isin(self._collection, self._codes_of(collections))
        return mask

    def _top(self, vector: np.ndarray, since, collection, limit, sources, collections) -> list[tuple[int, float]]:
        rows = np.flatnonzero(self.eligible(since, collection, sources, collections))
        if not len(rows):
            return []
        vector = np.asarray(vector, dtype=np.float32)
        # Scoring every row and picking the eligible ones beats copying most of a multi-GB matrix to score a subset.
        scores = (self.matrix @ vector)[rows] if len(rows) * 4 >= len(self.items) else self.matrix[rows] @ vector
        if len(scores) > 4 * limit:
            top = np.argpartition(-scores, limit - 1)[:limit]
            order = top[np.argsort(-scores[top], kind="stable")]
        else:
            order = np.argsort(-scores, kind="stable")[:limit]
        return [(int(rows[i]), float(scores[i])) for i in order]

    async def search(
        self, vector: np.ndarray, since: datetime | None, collection: str | None, limit: int,
        sources: frozenset[str] | None = None, collections: frozenset[str] | None = None,
    ) -> list[Candidate]:
        # numpy releases the GIL, so a large search runs off the event loop and other requests keep moving.
        work = (lambda: self._top(vector, since, collection, limit, sources, collections))
        top = await asyncio.to_thread(work) if len(self.items) > 50_000 else work()
        return [Candidate(self.items[row], "vector", score) for row, score in top]


async def build_index(
    store: Store, embedder: Embeddings, collections: dict[str, str] | None = None
) -> Index:
    """Embed only items with no cached vector for this embedding model."""
    items = store.all_items()
    cached = store.embeddings(embedder.model, [item.id for item in items])
    missing = [item for item in items if item.id not in cached]
    for start in range(0, len(missing), 100):
        chunk = missing[start : start + 100]
        vectors = await embedder.embed([item.text[:8000] for item in chunk])
        fresh = {item.id: vectors[i] for i, item in enumerate(chunk)}
        store.save_embeddings(embedder.model, fresh)
        cached.update(fresh)
    if items:
        matrix = np.vstack([cached[item.id] for item in items]).astype(np.float32)
    else:
        matrix = np.zeros((0, 1), dtype=np.float32)
    return Index(items, matrix, collections)
