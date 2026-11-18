"""Pull one collection's corpus and embeddings from a GCS bucket at startup.

On Cloud Run each collection is its own service with its own bucket, so the
image carries code only and a collection is updated by copying files into its
bucket and restarting the service. The bucket holds the collection's `.jsonl`
and the `.npz` embedding blocks tools/embed.py wrote; they land exactly where
the catalog loader already looks -- the data directory and the cache directory
-- so nothing downstream knows the files were downloaded.

Downloads run in parallel threads: in-region they take seconds, and the
largest collection is a few hundred megabytes.
"""

from __future__ import annotations

import logging
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

logger = logging.getLogger(__name__)


SNAPSHOT_PREFIX = "snapshot/"


def download_collection(
    bucket_url: str, data_dir: Path, cache_dir: Path, snapshot_dir: Path | None = None,
    snapshot_prefix: str = SNAPSHOT_PREFIX,
) -> list[Path]:
    from google.cloud import storage  # only needed where a bucket is configured

    name = bucket_url.removeprefix("gs://").strip("/")
    bucket_name, _, prefix = name.partition("/")
    client = storage.Client()
    every = list(client.list_blobs(bucket_name, prefix=prefix or None))
    # A prebuilt snapshot, when the bucket has one, is all the service needs;
    # the raw corpus and embeddings stay in the bucket for rebuilding it.
    snapshot = [b for b in every if Path(b.name).parent.name + "/" == snapshot_prefix]
    if snapshot and snapshot_dir is not None:
        blobs = snapshot
    else:
        blobs = [b for b in every if b.name.endswith((".jsonl", ".npz"))
                 and "/" not in b.name]
    if not blobs:
        raise RuntimeError(f"{bucket_url} holds no snapshot, .jsonl or .npz files")

    for directory in (data_dir, cache_dir, snapshot_dir):
        if directory is not None:
            directory.mkdir(parents=True, exist_ok=True)

    def fetch(blob) -> Path:
        if blobs is snapshot:
            directory = snapshot_dir
        else:
            directory = cache_dir if blob.name.endswith(".npz") else data_dir
        target = directory / Path(blob.name).name
        # A restarted instance may still have the file; size is enough to tell,
        # since a changed corpus changes the embedding file names too.
        if target.is_file() and target.stat().st_size == blob.size:
            return target
        temporary = target.with_suffix(target.suffix + ".part")
        blob.download_to_filename(temporary)
        temporary.replace(target)
        return target

    started = time.monotonic()
    with ThreadPoolExecutor(max_workers=8) as pool:
        paths = list(pool.map(fetch, blobs))
    total = sum(path.stat().st_size for path in paths)
    logger.info("downloaded %d files (%.0f MB) from %s in %.1fs",
                len(paths), total / 1e6, bucket_url, time.monotonic() - started)
    return paths
