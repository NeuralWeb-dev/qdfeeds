"""Load an export from scripts/export_cosmos.py into Postgres.

Each partition file is COPYed into a staging table and upserted into `items` (or
`sources`), so loading a later `--since` export on top of the first applies what changed
in Cosmos meanwhile: new items, and vectors attached to items that had none. A row's
`modified_at` is the document's Cosmos `_ts`.

    .venv/bin/python scripts/load_postgres.py data/azure-export [partition ...]

Needs QDRSS_PG_HOST and QDRSS_PG_PASSWORD (in ~/.config/nlweb/env); user and database are qdrss.
"""

from __future__ import annotations

import argparse
import base64
import gzip
import json
import os
import time
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import psycopg
from pgvector.psycopg import register_vector

SCHEMA = Path(__file__).resolve().parent.parent / "src" / "qdrss" / "pg_schema.sql"
COLUMNS = ("id", "source", "source_title", "collection", "url", "title", "snippet", "content", "published_at",
           "published_raw", "ingested_at", "extra", "embedding", "embedding_model", "modified_at")
# Binary COPY: the float32 vectors go over as bytes, with no text formatting or parsing.
TYPES = ("text",) * 8 + ("timestamptz", "text", "timestamptz", "text", "vector", "text", "timestamptz")


def vector(b64: str | None) -> np.ndarray | None:
    return np.frombuffer(base64.b64decode(b64), dtype=np.float32) if b64 else None


def when(iso: str | None) -> datetime | None:
    return datetime.fromisoformat(iso) if iso else None


def ts(seconds: float | None) -> datetime:
    return datetime.fromtimestamp(seconds or 0, UTC)


def item_row(doc: dict) -> tuple:
    return (doc["item_id"], doc["source"], doc.get("source_title") or "", doc.get("collection") or "default",
            doc.get("url") or "", doc.get("title") or "", doc.get("snippet") or "", doc.get("content") or "",
            when(doc.get("published_at")), doc.get("published_raw") or "", when(doc["ingested_at"]),
            doc.get("extra") or "", vector(doc.get("embedding_b64")), doc.get("embedding_model"), ts(doc.get("_ts")))


def load_items(conn: psycopg.Connection, path: Path) -> tuple[int, int]:
    rows = skipped = 0
    with conn.cursor() as cur:
        cur.execute("CREATE TEMP TABLE IF NOT EXISTS stage (LIKE items INCLUDING DEFAULTS) ON COMMIT DELETE ROWS")
        with gzip.open(path, "rt") as f, cur.copy(
                f"COPY stage ({', '.join(COLUMNS)}) FROM STDIN WITH (FORMAT BINARY)") as copy:
            copy.set_types(TYPES)
            for line in f:
                doc = json.loads(line)
                if doc.get("kind") != "item":
                    skipped += 1
                    continue
                copy.write_row(item_row(doc))
                rows += 1
        # Newer wins; a vector is never replaced by NULL (an item exported before it was embedded).
        keep = {"embedding", "embedding_model"}
        updates = ", ".join(
            f"{c} = COALESCE(EXCLUDED.{c}, items.{c})" if c in keep else f"{c} = EXCLUDED.{c}"
            for c in COLUMNS if c != "id")
        cur.execute(f"""
            INSERT INTO items ({', '.join(COLUMNS)}) SELECT {', '.join(COLUMNS)} FROM stage
            ON CONFLICT (id) DO UPDATE SET {updates}
            WHERE EXCLUDED.modified_at >= items.modified_at""")
    conn.commit()
    return rows, skipped


def load_sources(conn: psycopg.Connection, path: Path) -> int:
    n = 0
    with conn.cursor() as cur, gzip.open(path, "rt") as f:
        for line in f:
            doc = json.loads(line)
            if doc.get("kind") != "source":
                continue
            cur.execute("""
                INSERT INTO sources (name, url, etag, last_modified, sha256, last_fetch, last_error)
                VALUES (%s, %s, %s, %s, %s, %s, %s)
                ON CONFLICT (name) DO UPDATE SET url = EXCLUDED.url, etag = EXCLUDED.etag,
                    last_modified = EXCLUDED.last_modified, sha256 = EXCLUDED.sha256,
                    last_fetch = EXCLUDED.last_fetch, last_error = EXCLUDED.last_error
                WHERE sources.last_fetch IS NULL OR EXCLUDED.last_fetch >= sources.last_fetch""",
                (doc["name"], doc.get("url") or "", doc.get("etag"), doc.get("last_modified"), doc.get("sha256"),
                 doc.get("last_fetch"), doc.get("last_error")))
            n += 1
    conn.commit()
    return n


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("export", type=Path)
    parser.add_argument("partitions", nargs="*", help="only these (default: every file in the export)")
    args = parser.parse_args()
    dsn = f"host={os.environ['QDRSS_PG_HOST']} dbname=qdrss user=qdrss password={os.environ['QDRSS_PG_PASSWORD']}"
    with psycopg.connect(dsn) as conn:
        conn.execute(SCHEMA.read_text())
        conn.commit()
        register_vector(conn)
        started = time.time()
        for path in sorted(args.export.glob("*.jsonl.gz")):
            if args.partitions and path.name.removesuffix(".jsonl.gz") not in args.partitions:
                continue
            if path.name == "_sources.jsonl.gz":
                print(f"{'sources':20} {load_sources(conn, path):8}", flush=True)
                continue
            rows, skipped = load_items(conn, path)
            print(f"{path.name.removesuffix('.jsonl.gz'):20} {rows:8} items"
                  + (f", {skipped} other documents skipped" if skipped else "") + f"  ({time.time() - started:.0f}s)",
                  flush=True)
        conn.execute("ANALYZE items")
        total, embedded = conn.execute("SELECT COUNT(*), COUNT(embedding) FROM items").fetchone()
        print(f"items in Postgres: {total} ({embedded} with a vector)")


if __name__ == "__main__":
    main()
