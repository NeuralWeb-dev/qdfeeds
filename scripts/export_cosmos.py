"""Export every document of the qdrss Cosmos container to gzipped JSON lines, one file per
partition (collection), as a lossless backup and the source for loading Postgres.

Each line is the document as Cosmos returns it, except that `embedding` (1536 JSON
numbers, ~30 KB) is replaced by `embedding_b64`, the same float32 values base64-encoded
(6 KB→8 KB of text). Cosmos stores vectors as float32 (the container's vector policy), so
nothing is lost. `_ts` (last modified, seconds) is kept: `--since <ts>` exports only
documents written or patched after an earlier export, which is how the final copy before
switching off Cosmos is taken: use MANIFEST.json's `next_since`.

    .venv/bin/python scripts/export_cosmos.py data/azure-export [--since 1763251200]

Needs COSMOS_ENDPOINT and COSMOS_KEY (the prod key is COSMOS_KEY_2 in ~/.config/nlweb/env).
"""

from __future__ import annotations

import argparse
import base64
import gzip
import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
from azure.cosmos import CosmosClient

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from qdrss.ingest import load_sources

MANIFEST = Path(__file__).resolve().parent.parent / "sources.yaml"
SOURCES_PK = "_sources"


def encode(doc: dict) -> dict:
    vector = doc.pop("embedding", None)
    if vector is not None:
        doc["embedding_b64"] = base64.b64encode(np.asarray(vector, dtype=np.float32).tobytes()).decode()
    return doc


def container():
    return (CosmosClient(os.environ["COSMOS_ENDPOINT"], os.environ["COSMOS_KEY"])
            .get_database_client(os.environ.get("COSMOS_DATABASE", "qdrss")).get_container_client("items"))


def export(partition: str, out: Path, since: int | None) -> tuple[str, int, int]:
    """One partition, in its own process: decoding 38 KB of JSON numbers per document is CPU-bound."""
    path = out / f"{partition}.jsonl.gz"
    if path.exists() and since is None:  # finished by an earlier, interrupted run
        with gzip.open(path, "rt") as f:
            return partition, sum(1 for _ in f), 0
    query, params = "SELECT * FROM c", []
    if since is not None:
        query, params = "SELECT * FROM c WHERE c._ts > @since", [{"name": "@since", "value": since}]
    tmp = path.with_suffix(".part")
    n = newest = 0
    with gzip.open(tmp, "wt", compresslevel=3) as f:
        for doc in container().query_items(query, parameters=params, partition_key=partition, max_item_count=500):
            newest = max(newest, doc.get("_ts", 0))
            f.write(json.dumps(encode(doc), separators=(",", ":")) + "\n")
            n += 1
    tmp.rename(path)
    return partition, n, newest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("out", type=Path)
    parser.add_argument("--since", type=int, help="only documents with _ts after this")
    parser.add_argument("--workers", type=int, default=os.cpu_count() or 4)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    # Every partition Cosmos has, not only the manifest's: items of sources that left the manifest
    # stay in theirs (`default` holds the earliest ones).
    present = set(container().query_items("SELECT DISTINCT VALUE c.collection FROM c",
                                           enable_cross_partition_query=True))
    partitions = sorted(present | {s.collection for s in load_sources(MANIFEST)} | {SOURCES_PK})
    started = time.time()
    first_started = float(os.environ.get("EXPORT_STARTED", started))  # the first run's start, when resuming
    total = 0
    newest = 0
    with ProcessPoolExecutor(args.workers) as pool:
        jobs = [pool.submit(export, p, args.out, args.since) for p in partitions]
        for job in jobs:
            partition, n, ts = job.result()
            total += n
            newest = max(newest, ts)
            print(f"{partition:20} {n:8}  ({time.time() - started:.0f}s)", flush=True)
    # A document in a partition the manifest no longer names would be missed: compare with the total.
    query = "SELECT VALUE COUNT(1) FROM c" + (" WHERE c._ts > @since" if args.since is not None else "")
    params = [{"name": "@since", "value": args.since}] if args.since is not None else []
    in_cosmos = sum(container().query_items(query, parameters=params, enable_cross_partition_query=True))
    (args.out / "MANIFEST.json").write_text(json.dumps(
        {"exported": total, "in_cosmos": in_cosmos, "newest_ts": newest, "since": args.since,
         # A later `--since` export starts here: anything written after the first partition began.
         "next_since": int(first_started) - 60,
         "partitions": partitions}, indent=1))
    print(f"exported {total}, Cosmos has {in_cosmos}; newest _ts {newest}")
    if total != in_cosmos:
        # Documents written while the export ran: the `--since next_since` export picks them up.
        print(f"{in_cosmos - total} more documents in Cosmos than exported (written during the export)")


if __name__ == "__main__":
    main()
