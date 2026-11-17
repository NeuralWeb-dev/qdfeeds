"""Serve chosen collections from process memory instead of a Cosmos vector query.

A collection that is small, grows slowly and is read a lot (This American Life: 28k passages, a new episode a
week) gains nothing from a database round trip on every query and is exposed to its throttling and timeouts. Its
passages and vectors are loaded once at startup from a snapshot (scripts/export_snapshot.py) and searched with the
same in-memory Index the SQLite mode uses. Whatever was stored or embedded after the snapshot is read from the
corpus store (Postgres) at startup and again every few minutes, so the snapshot only has to be rebuilt to keep that
catch-up small. A collection that is not in memory goes to the store's own search, which Postgres does not offer:
the VM loads every collection.
"""

from __future__ import annotations

import asyncio
import gzip
import io
import json
import logging
import time
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from urllib.parse import quote

import httpx
import numpy as np

from .index import Index
from .models import Candidate, Item

logger = logging.getLogger("qdrss")
METADATA_TOKEN = "http://metadata.google.internal/computeMetadata/v1/instance/service-accounts/default/token"


def write_snapshot(directory: Path, collection: str, items: list[Item], matrix: np.ndarray) -> None:
    """items.jsonl.gz + float16 vectors (half the size; cosine ranking does not notice)."""
    directory.mkdir(parents=True, exist_ok=True)
    rows = (json.dumps({**asdict(i), "published_at": i.published_at and i.published_at.isoformat(),
                        "ingested_at": i.ingested_at.isoformat()}) for i in items)
    with gzip.open(directory / f"{collection}.jsonl.gz", "wt", encoding="utf-8") as fh:
        fh.write("\n".join(rows))
    np.save(directory / f"{collection}.npy", matrix.astype(np.float16))


def parse_snapshot(items_gz: bytes, vectors_npy: bytes) -> tuple[list[Item], np.ndarray]:
    lines = gzip.decompress(items_gz).decode("utf-8").splitlines()
    items = [Item(**{**row, "published_at": datetime.fromisoformat(row["published_at"]) if row["published_at"] else None,
                     "ingested_at": datetime.fromisoformat(row["ingested_at"])})
             for row in map(json.loads, lines)]
    matrix = np.load(io.BytesIO(vectors_npy))  # float16 as stored; Index.with_room widens it into its buffer
    if len(items) != len(matrix):
        raise ValueError(f"snapshot has {len(items)} items but {len(matrix)} vectors")
    return items, matrix


async def _read(location: str, name: str) -> bytes:
    """A file of the snapshot: a local directory, or gs://bucket/prefix read with the service's own credentials."""
    if not location.startswith("gs://"):
        return (Path(location) / name).read_bytes()
    bucket, _, prefix = location[5:].partition("/")
    async with httpx.AsyncClient(timeout=120) as client:
        token = (await client.get(METADATA_TOKEN, headers={"Metadata-Flavor": "Google"})).json()["access_token"]
        object_name = f"{prefix.strip('/')}/{name}".lstrip("/")
        response = await client.get(
            f"https://storage.googleapis.com/storage/v1/b/{bucket}/o/{quote(object_name, safe='')}",
            params={"alt": "media"}, headers={"Authorization": f"Bearer {token}"})
        response.raise_for_status()
        return response.content


class MemoryRetriever:
    def __init__(self, store, source_collections: dict[str, str], collections: list[str], snapshot: str,
                 catch_up_seconds: float = 300):
        # The corpus store new items are read from: Postgres (pgstore.py, items_changed) or Cosmos (items_since).
        self.store = store
        self.source_collections = source_collections
        self.names = collections
        self.snapshot = snapshot
        self.catch_up_seconds = catch_up_seconds
        self.indexes: dict[str, Index] = {}
        self.status: dict[str, str] = {name: "not loaded" for name in collections}
        self._checked: dict[str, float] = {}
        self._busy: dict[str, asyncio.Lock] = {name: asyncio.Lock() for name in collections}
        self._floor: dict[str, float] = {}
        self._ids: dict[str, set[str]] = {}  # every item id in memory, so a catch-up never adds one twice

    async def load(self) -> None:
        for name in self.names:
            try:
                started = time.monotonic()
                items, matrix = parse_snapshot(await _read(self.snapshot, f"{name}.jsonl.gz"),
                                               await _read(self.snapshot, f"{name}.npy"))
                # Catch-up starts an hour before the newest item in the snapshot: an item stored or embedded
                # while the snapshot was being taken is read again rather than missed (the id set drops repeats).
                self._floor[name] = max((i.ingested_at.timestamp() for i in items), default=0.0) - 3600
                self.indexes[name] = Index.with_room(items, matrix, self.source_collections)
                self._ids[name] = {i.id for i in items}
                del items, matrix
                await self.catch_up(name)
                self.status[name] = f"{len(self.indexes[name])} items in memory"
                logger.info("collection %s: %s (%.1fs)", name, self.status[name], time.monotonic() - started)
            except Exception as exc:  # noqa: BLE001 - serve from Cosmos instead, visibly
                self.status[name] = f"snapshot failed ({type(exc).__name__}: {str(exc)[:100]}); not in memory"
                logger.exception("collection %s: could not load the snapshot", name)

    async def catch_up(self, name: str) -> int:
        """Add what Cosmos holds from after the snapshot and is not in memory yet."""
        async with self._busy[name]:
            self._checked[name] = time.monotonic()
            index = self.indexes[name]
            known = self._ids[name]
            if hasattr(self.store, "items_changed"):
                # Postgres: inserted or embedded after the floor, by the database's clock; the floor becomes the
                # time of this read, so an item embedded long after it was ingested is still picked up.
                changed, floor = await self.store.items_changed(name, self._floor[name])
            else:
                changed, floor = await self.store.items_since(name, self._floor[name]), None
            fresh = [(i, v) for i, v in changed if i.id not in known]
            fresh = list({i.id: (i, v) for i, v in fresh}.values())
            if fresh:
                self.indexes[name] = index.extended([i for i, _ in fresh], np.vstack([v for _, v in fresh]))
                known.update(i.id for i, _ in fresh)
                if floor is None:
                    self._floor[name] = max(self._floor[name], *(i.ingested_at.timestamp() for i, _ in fresh))
            if floor is not None:
                self._floor[name] = floor
                logger.info("collection %s: %d items added since the snapshot", name, len(fresh))
            return len(fresh)

    async def search(self, vector: np.ndarray, since: datetime | None, collection: str | None, limit: int,
                     sources: frozenset[str] | None = None, collections: frozenset[str] | None = None) -> list[Candidate]:
        if not collection and collections is None and self.indexes and len(self.indexes) == len(self.names):
            collections = frozenset(self.indexes)  # every collection, and every one of them is in memory
        if not collection and collections is not None:
            # A tab over several collections (All: nytnews and nytimes): each from memory if it is loaded, else
            # from Cosmos, merged by score. Scores are comparable: the same embedding model and cosine scale.
            runs = await asyncio.gather(*(self.search(vector, since, name, limit, sources)
                                          for name in sorted(collections)))
            return sorted((c for run in runs for c in run), key=lambda c: -c.score)[:limit]
        index = self.indexes.get(collection or "")
        if index is None:
            return await self.store.search(vector, since, collection, limit, sources, collections)
        if time.monotonic() - self._checked[collection] > self.catch_up_seconds and not self._busy[collection].locked():
            asyncio.create_task(self._safe_catch_up(collection))
        return await index.search(vector, since, collection, limit, sources, collections)

    async def _safe_catch_up(self, name: str) -> None:
        try:
            await self.catch_up(name)
        except Exception:  # noqa: BLE001 - the next query tries again
            logger.exception("collection %s: catch-up failed; serving what is in memory", name)
