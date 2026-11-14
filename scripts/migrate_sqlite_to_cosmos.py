"""One-time load of a local SQLite corpus (items + cached vectors) into Cosmos DB.

    COSMOS_ENDPOINT=... COSMOS_KEY=... python scripts/migrate_sqlite_to_cosmos.py data/qdrss.sqlite \
        [--collections nytnews,nytimes] [--skip-existing]

Idempotent: documents are upserted by id. Items without a vector for the
configured embedding model are written without one; `qdrss-ingest` embeds them.
--collections loads only those collections; --skip-existing leaves items Cosmos
already holds with a vector, so an interrupted load resumes where it stopped.
An item's collection is its manifest section, or its own when the manifest lacks
its source (archive-only sources from scripts/backfill_nyt_news.py).
"""

from __future__ import annotations

import argparse
import asyncio
import os
import sys
import time

from azure.cosmos.exceptions import CosmosHttpResponseError

from qdrss.config import load_config
from qdrss.cosmos import CosmosStore, _document
from qdrss.ingest import load_sources
from qdrss.store import Store


async def main(path: str, only: set[str], skip_existing: bool) -> None:
    config = load_config()
    if not config.cosmos_endpoint:
        sys.exit("COSMOS_ENDPOINT / COSMOS_KEY must be set")
    collections = {s.name: s.collection for s in load_sources(config.sources_path)}
    model = config.embedding_model and f"openrouter:{config.embedding_model}"
    sqlite = Store(path)
    from dataclasses import replace

    cosmos = CosmosStore(config.cosmos_endpoint, config.cosmos_key, config.cosmos_database)
    await cosmos.setup()
    held: set[str] = set()
    if skip_existing:
        names = only or {collections.get(s["name"], "default") for s in await sqlite.all_sources()}
        for name in names:
            rows = cosmos._items.query_items("SELECT VALUE c.item_id FROM c WHERE IS_DEFINED(c.embedding)",
                                             partition_key=name)
            held |= {row async for row in rows}
        print(f"{len(held)} already in Cosmos with a vector; skipping them", flush=True)

    sem = asyncio.Semaphore(int(os.environ.get("MIGRATE_CONCURRENCY", "32")))
    done = 0
    started = time.time()

    async def put(item, vector):
        nonlocal done
        async with sem:
            doc = _document(item, vector, model if vector is not None else None)
            for attempt in range(30):
                try:
                    await cosmos._items.upsert_item(doc)
                    break
                except CosmosHttpResponseError as exc:
                    if exc.status_code != 429:
                        raise
                    wait = float(exc.headers.get("x-ms-retry-after-ms", 1000)) / 1000
                    await asyncio.sleep(max(wait, 0.5) * (1 + attempt / 5))
            else:
                raise RuntimeError(f"gave up on {item.id} after repeated 429s")
        done += 1
        if done % 1000 == 0:
            print(f"{done} written in {time.time() - started:.0f}s", flush=True)

    # Sources too, so the first ingest run sends conditional GETs.
    for src in await sqlite.all_sources():
        if only and collections.get(src["name"]) not in only:
            continue
        state = await sqlite.source_state(src["name"])
        await cosmos.remember_source(
            src["name"], src["url"], etag=state.get("etag"),
            last_modified=state.get("last_modified"), sha256=state.get("sha256"),
        )
    # A batch at a time: a document carries its vector as 1536 Python floats, so holding a large
    # corpus's documents (or its items and vectors) at once runs a machine out of memory.
    for batch in sqlite.iter_items(2000):
        batch = [replace(i, collection=collections.get(i.source, i.collection or "default")) for i in batch]
        batch = [i for i in batch if (not only or i.collection in only) and i.id not in held]
        if not batch:
            continue
        vectors = sqlite.embeddings(model, [i.id for i in batch])
        await asyncio.gather(*(put(i, vectors.get(i.id)) for i in batch))
    print(f"done: {done} documents, {await cosmos.count()} in Cosmos, {time.time() - started:.0f}s")
    await cosmos.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("path", nargs="?", default="data/qdrss.sqlite")
    parser.add_argument("--collections", default="", help="comma-separated; default all")
    parser.add_argument("--skip-existing", action="store_true")
    args = parser.parse_args()
    asyncio.run(main(args.path, {c.strip() for c in args.collections.split(",") if c.strip()}, args.skip_existing))
