"""Write in-memory snapshots (see qdrss/memory.py) of collections from Postgres, optionally uploading them.

    .venv/bin/python scripts/snapshot_from_pg.py OUTDIR [collection ...] [--upload gs://neuralweb-qdrss-snapshots]

With no collection named, every collection in Postgres. Items without a vector are skipped (they are not
searchable until ingest embeds them, and the service's catch-up adds them then). Runs nightly on the VM
(deploy/vm/qdrss-nightly.sh) so a restart of the vector service loads a day-old snapshot and catches up little.
Needs QDRSS_PG_HOST and QDRSS_PG_PASSWORD.
"""

from __future__ import annotations

import argparse
import gzip
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import psycopg
from pgvector.psycopg import register_vector

DIMENSIONS = 1536


def snapshot(conn: psycopg.Connection, collection: str, out: Path) -> int:
    # The vectors go straight to a memory-mapped .npy on disk: the VM holds every collection in memory for the
    # vector service, and a 1.5 GB array for nytnews here could push it out of memory.
    tmp_items, tmp_npy = out / f".{collection}.jsonl.gz", out / f".{collection}.npy"
    total = conn.execute("SELECT count(*) FROM items WHERE collection = %s AND embedding IS NOT NULL",
                         (collection,)).fetchone()[0]
    vectors = np.lib.format.open_memmap(tmp_npy, mode="w+", dtype=np.float16, shape=(total, DIMENSIONS))
    n = 0
    with conn.cursor(name=f"snap_{collection}") as cur, gzip.open(tmp_items, "wt", encoding="utf-8") as items:
        cur.itersize = 5000
        cur.execute(
            "SELECT id, source, source_title, url, title, coalesce(nullif(content, ''), snippet) AS content, "
            "published_at, published_raw, ingested_at, collection, extra, embedding FROM items "
            "WHERE collection = %s AND embedding IS NOT NULL ORDER BY ingested_at, id", (collection,))
        for row in cur:
            (id_, source, source_title, url, title, content, published_at, published_raw, ingested_at,
             coll, extra, embedding) = row
            if n == total:  # stored while we read (same transaction, so only if the count missed it): next night
                break
            vectors[n] = embedding.to_numpy()
            items.write(("\n" if n else "") + json.dumps({
                "id": id_, "source": source, "source_title": source_title, "url": url, "title": title,
                "content": content, "published_at": published_at.isoformat() if published_at else None,
                "published_raw": published_raw, "ingested_at": ingested_at.isoformat(), "collection": coll,
                "extra": extra}))
            n += 1
    vectors.flush()
    del vectors
    if n < total:  # fewer than counted: rewrite at the true length
        np.save(tmp_npy, np.load(tmp_npy, mmap_mode="r")[:n])
    # Renamed into place only when complete, so an upload never takes half a snapshot.
    tmp_items.replace(out / f"{collection}.jsonl.gz")
    tmp_npy.replace(out / f"{collection}.npy")
    return n


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("out", type=Path)
    parser.add_argument("collections", nargs="*")
    parser.add_argument("--upload", help="gs://bucket[/prefix] to copy each finished snapshot to")
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    dsn = f"host={os.environ['QDRSS_PG_HOST']} dbname=qdrss user=qdrss password={os.environ['QDRSS_PG_PASSWORD']}"
    with psycopg.connect(dsn) as conn:
        register_vector(conn)
        names = args.collections or [r[0] for r in conn.execute(
            "SELECT DISTINCT collection FROM items WHERE embedding IS NOT NULL ORDER BY 1")]
        for name in names:
            started = time.monotonic()
            n = snapshot(conn, name, args.out)
            print(f"{name}: {n} items ({time.monotonic() - started:.0f}s)", flush=True)
            if args.upload:
                subprocess.run(["gcloud", "storage", "cp", "--quiet", str(args.out / f"{name}.jsonl.gz"),
                                str(args.out / f"{name}.npy"), args.upload.rstrip("/") + "/"], check=True)


if __name__ == "__main__":
    sys.exit(main())
