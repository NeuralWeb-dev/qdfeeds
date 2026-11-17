"""Write a collection's in-memory snapshot (see qdrss/memory.py) from a Cosmos export (scripts/export_cosmos.py).

    .venv/bin/python scripts/snapshot_from_export.py data/azure-export nytnews OUTDIR
    gcloud storage cp OUTDIR/nytnews.* gs://neuralweb-qdrss-snapshots/

The export holds every item with its float32 vector (`embedding_b64`); the snapshot is the items as JSON lines
plus the vectors as float16, the format the service loads at startup. Items without a vector are skipped: they
are not in the vector index either. The service then fetches from Cosmos whatever was ingested after the newest
item here, so an older snapshot only makes that catch-up longer.
"""

from __future__ import annotations

import base64
import gzip
import json
import sys
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from qdrss.cosmos import _item  # the same row -> Item mapping the service uses

DIMENSIONS = 1536
MODEL = "openrouter:openai/text-embedding-3-small"


def main(export: Path, collection: str, out: Path) -> None:
    started = time.monotonic()
    out.mkdir(parents=True, exist_ok=True)
    vectors = np.zeros((1 << 16, DIMENSIONS), dtype=np.float16)
    count = skipped = 0
    newest = ""
    with gzip.open(export / f"{collection}.jsonl.gz", "rt", encoding="utf-8") as rows, \
            gzip.open(out / f"{collection}.jsonl.gz", "wt", encoding="utf-8") as items:
        for line in rows:
            row = json.loads(line)
            if row.get("kind") != "item" or not row.get("embedding_b64") or row.get("embedding_model", MODEL) != MODEL:
                skipped += 1
                continue
            vector = np.frombuffer(base64.b64decode(row["embedding_b64"]), dtype=np.float32)
            if vector.shape != (DIMENSIONS,):
                skipped += 1
                continue
            item = _item(row)
            if count == len(vectors):
                vectors = np.concatenate([vectors, np.zeros_like(vectors)])
            vectors[count] = vector
            items.write(("\n" if count else "") + json.dumps(
                {**asdict(item), "published_at": item.published_at and item.published_at.isoformat(),
                 "ingested_at": item.ingested_at.isoformat()}))
            newest = max(newest, row["ingested_at"])
            count += 1
    np.save(out / f"{collection}.npy", vectors[:count])
    print(f"{collection}: {count} items, {skipped} skipped, newest ingested {newest}, {time.monotonic() - started:.0f}s")


if __name__ == "__main__":
    main(Path(sys.argv[1]), sys.argv[2], Path(sys.argv[3]))
