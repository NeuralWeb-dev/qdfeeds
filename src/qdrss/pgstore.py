"""The corpus in Postgres on the neuralweb VM (replaces Cosmos DB).

Items, their vectors and the feeds' fetch state live in Postgres (schema: pg_schema.sql, loaded from the Cosmos
export by scripts/load_postgres.py). Ingest writes here; the VM's vector service (vectors.py) reads what changed
for its in-memory collections. Postgres does no vector search: every search is exact, in memory (memory.py), so
the table has no vector index and `search` is not offered.

`modified_at` is set by Postgres on every insert and vector update, so "what changed since" is read from the
database's own clock and an item embedded after it was ingested is still picked up.
"""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import numpy as np
from pgvector.psycopg import register_vector_async
from psycopg.rows import dict_row
from psycopg_pool import AsyncConnectionPool

from .models import Item

SCHEMA = Path(__file__).with_name("pg_schema.sql")
SNIPPET_CHARS = 1500  # what the ranker sees (as in Cosmos)
CONTENT_CHARS = 8000  # what gets embedded
ITEM_COLUMNS = ("id, source, source_title, collection, url, title, snippet, content, published_at, published_raw, "
                "ingested_at, extra")


class PostgresStore:
    def __init__(self, host: str, password: str, database: str = "qdrss", user: str = "qdrss", max_size: int = 8):
        self.pool = AsyncConnectionPool(
            f"host={host} dbname={database} user={user} password={password} connect_timeout=10",
            min_size=1, max_size=max_size, open=False, configure=self._configure,
            kwargs={"autocommit": True, "row_factory": dict_row},
            # Each connection is tested as it is handed out: a restart of the VM's Postgres leaves the pool's
            # idle connections closed, and without the check the first request to take one failed (a 500).
            check=AsyncConnectionPool.check_connection,
        )

    @staticmethod
    async def _configure(conn) -> None:
        await register_vector_async(conn)

    async def setup(self) -> None:
        await self.pool.open(wait=True, timeout=30)
        async with self.pool.connection() as conn:
            await conn.execute(SCHEMA.read_text())

    async def close(self) -> None:
        await self.pool.close()

    # --- items ---------------------------------------------------------------

    async def insert_missing(self, items: list[Item]) -> int:
        """Store items not yet stored; existing ones are untouched. Returns how many were new."""
        if not items:
            return 0
        rows = [(i.id, i.source, i.source_title, i.collection or "default", i.url, i.title,
                 i.content[:SNIPPET_CHARS], i.content[:CONTENT_CHARS], i.published_at, i.published_raw,
                 i.ingested_at, i.extra) for i in {i.id: i for i in items}.values()]
        async with self.pool.connection() as conn, conn.transaction(), conn.cursor() as cur:
            await cur.execute("CREATE TEMP TABLE incoming (LIKE items INCLUDING DEFAULTS) ON COMMIT DROP")
            async with cur.copy(f"COPY incoming ({ITEM_COLUMNS}) FROM STDIN") as copy:
                for row in rows:
                    await copy.write_row(row)
            await cur.execute(f"INSERT INTO items ({ITEM_COLUMNS}) SELECT {ITEM_COLUMNS} FROM incoming "
                              "ON CONFLICT (id) DO NOTHING")
            return cur.rowcount

    async def items_without_embedding(self, limit: int = 500) -> list[Item]:
        async with self.pool.connection() as conn:
            rows = await (await conn.execute(
                f"SELECT {ITEM_COLUMNS} FROM items WHERE embedding IS NULL ORDER BY ingested_at LIMIT %s",
                (limit,))).fetchall()
        return [_item(r) for r in rows]

    async def save_embeddings(self, model: str, vectors: dict[str, np.ndarray],
                              collections: dict[str, str] | None = None) -> None:
        if not vectors:
            return
        async with self.pool.connection() as conn, conn.cursor() as cur:
            await cur.executemany(
                "UPDATE items SET embedding = %s, embedding_model = %s, modified_at = now() WHERE id = %s",
                [(np.asarray(v, dtype=np.float32), model, item_id) for item_id, v in vectors.items()])

    async def items_changed(self, collection: str, since: float) -> tuple[list[tuple[Item, np.ndarray]], float]:
        """Embedded items of a collection inserted or given a vector after `since` (epoch seconds, by the
        database's clock), and the next call's `since`: a minute before this read, because a row's modified_at is
        its transaction's start and a write still committing during the read would otherwise be skipped for good.
        The caller drops ids it already holds."""
        async with self.pool.connection() as conn:
            now = (await (await conn.execute("SELECT extract(epoch FROM now()) AS t")).fetchone())["t"]
            rows = await (await conn.execute(
                f"SELECT {ITEM_COLUMNS}, embedding FROM items WHERE collection = %s AND embedding IS NOT NULL "
                "AND modified_at > %s", (collection, datetime.fromtimestamp(since, UTC)))).fetchall()
        return [(_item(r), r["embedding"].to_numpy().astype(np.float32)) for r in rows], float(now) - 60

    async def search(self, *args, **kwargs):
        raise NotImplementedError("Postgres holds no vector index: searches run in memory on the VM (vectors.py)")

    async def count(self) -> int:
        async with self.pool.connection() as conn:
            return (await (await conn.execute("SELECT count(*) AS n FROM items")).fetchone())["n"]

    async def counts_by_collection(self, collections: list[str]) -> dict[str, int]:
        async with self.pool.connection() as conn:
            rows = await (await conn.execute(
                "SELECT collection, count(*) AS n FROM items WHERE collection = ANY(%s) GROUP BY collection",
                (collections,))).fetchall()
        found = {r["collection"]: r["n"] for r in rows}
        return {name: found.get(name, 0) for name in collections}

    # --- sources -------------------------------------------------------------

    async def source_state(self, name: str) -> dict[str, str | None]:
        async with self.pool.connection() as conn:
            row = await (await conn.execute(
                "SELECT url, etag, last_modified, sha256, last_fetch, last_error FROM sources WHERE name = %s",
                (name,))).fetchone()
        if row is None:
            return {}
        return {**row, "last_fetch": row["last_fetch"].isoformat() if row["last_fetch"] else None}

    async def remember_source(self, name: str, url: str, *, etag: str | None = None, last_modified: str | None = None,
                              sha256: str | None = None, error: str | None = None) -> None:
        """After a fetch: its validators, or on an error the previous validators kept and the error recorded."""
        async with self.pool.connection() as conn:
            await conn.execute(
                """INSERT INTO sources (name, url, etag, last_modified, sha256, last_fetch, last_error)
                   VALUES (%(name)s, %(url)s, %(etag)s, %(last_modified)s, %(sha256)s, now(), %(error)s)
                   ON CONFLICT (name) DO UPDATE SET url = EXCLUDED.url, last_fetch = now(), last_error = %(error)s,
                     etag = CASE WHEN %(error)s::text IS NULL THEN EXCLUDED.etag ELSE sources.etag END,
                     last_modified = CASE WHEN %(error)s::text IS NULL THEN EXCLUDED.last_modified
                                          ELSE sources.last_modified END,
                     sha256 = CASE WHEN %(error)s::text IS NULL THEN EXCLUDED.sha256 ELSE sources.sha256 END""",
                {"name": name, "url": url, "etag": etag, "last_modified": last_modified, "sha256": sha256,
                 "error": error})

    async def all_sources(self) -> list[dict[str, str | None]]:
        async with self.pool.connection() as conn:
            rows = await (await conn.execute(
                "SELECT name, url, last_fetch, last_error FROM sources ORDER BY name")).fetchall()
        return [{**r, "last_fetch": r["last_fetch"].isoformat() if r["last_fetch"] else None} for r in rows]


def _item(row: dict) -> Item:
    return Item(
        id=row["id"], source=row["source"], source_title=row["source_title"], url=row["url"], title=row["title"],
        content=row["content"] or row["snippet"], published_at=row["published_at"],
        published_raw=row["published_raw"], ingested_at=row["ingested_at"], collection=row["collection"],
        extra=row["extra"],
    )
