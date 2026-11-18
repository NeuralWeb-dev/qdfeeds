"""Keep the corpus current without a redeploy.

The manifest names where each collection lives; this fetches them at startup and
rechecks on a timer. A check that finds nothing changed costs one conditional
request per source and no embedding, so a frequent schedule is cheap.

When something has changed, a replacement catalog is built off to the side and
swapped in with a single attribute assignment. Queries already in flight keep
the catalog they started with, and no request ever observes a half-built index.
"""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path

from .catalog import SNAPSHOT_STATE, MemoryCatalog
from .config import Config
from .pipeline import Services
from .sources import (
    RefreshReport,
    Source,
    fetch_all,
    kinds_by_collection,
    load_manifest,
)

logger = logging.getLogger(__name__)
BUNDLED_CORPUS = Path(__file__).resolve().parent / "corpus"


@dataclass(slots=True)
class RefreshState:
    """What the last refresh did, for /health and for operators."""

    last_check: str | None = None
    last_change: str | None = None
    checks: int = 0
    rebuilds: int = 0
    sources: int = 0
    refreshed: int = 0
    failures: list[str] = field(default_factory=list)

    def snapshot(self) -> dict[str, object]:
        return {
            "last_check": self.last_check,
            "last_change": self.last_change,
            "checks": self.checks,
            "rebuilds": self.rebuilds,
            "sources": self.sources,
            "refreshed": self.refreshed,
            "failures": self.failures,
        }


def _now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


def read_sources(config: Config) -> list[Source]:
    """Load the manifest, or return nothing if the deployment has no manifest.

    A missing manifest is not an error: the corpus can still be a directory of
    files, which is how tests and local checkouts run.
    """
    path = Path(config.sources_manifest)
    if not path.is_file():
        logger.info("no manifest at %s; using %s as-is", path, config.data_dir)
        return []
    # A malformed manifest raises: better to refuse to start than to silently
    # serve whatever happens to be on disk, which could be arbitrarily stale.
    sources = load_manifest(path)
    logger.info("manifest lists %d sources", len(sources))
    return sources


async def refresh_once(
    config: Config, sources: list[Source], state: RefreshState
) -> RefreshReport | None:
    """Fetch every source. Returns the report, or None when there is nothing to do."""
    if not sources:
        return None
    report = await fetch_all(sources, config.data_dir)
    state.checks += 1
    state.last_check = _now()
    # Collections in the manifest, and the subset that is fetched: an entry
    # shipping with the repository has no url and is never in a report.
    state.sources = len(sources)
    state.refreshed = len(report.outcomes)
    state.failures = [f"{o.source.name}: {o.error}" for o in report.failures]
    for failure in report.failures:
        # One unreachable feed must not stop the rest from updating.
        logger.warning("could not refresh %s: %s", failure.source.name, failure.error)
    logger.info("source check: %s", report.summary())
    return report


async def build_catalog(
    config: Config, sources: list[Source], embedder
) -> MemoryCatalog:
    kinds = kinds_by_collection(sources) if sources else None
    snapshot = config.snapshot_dir
    if config.vector_url and snapshot is not None and (snapshot / SNAPSHOT_STATE).is_file():
        from .vm_catalog import VMCatalog

        if not config.vector_token:
            raise RuntimeError("ASKHUB_VECTOR_URL is set but ASKHUB_VECTOR_TOKEN is not")
        catalog = await asyncio.to_thread(
            VMCatalog.connect, snapshot, config.vector_url, config.vector_token, kinds,
            config.bm25_field_weights, config.vector_field_weights,
        )
        logger.info("VM catalog %s: keyword index of %d records of %s", config.vector_url,
                    len(catalog.documents), catalog.sites)
        return catalog
    if (config.cosmos_endpoint and snapshot is not None
            and (snapshot / SNAPSHOT_STATE).is_file()):
        return await _cosmos_catalog(config, snapshot, kinds)
    if snapshot is not None and (snapshot / SNAPSHOT_STATE).is_file():
        started = time.monotonic()
        catalog = await asyncio.to_thread(
            MemoryCatalog.load_snapshot, snapshot, kinds,
            config.bm25_field_weights, config.vector_field_weights,
        )
        logger.info("loaded snapshot %s: %d documents in %.1fs",
                    snapshot, len(catalog.documents), time.monotonic() - started)
        return catalog
    return await MemoryCatalog.load(
        config.data_dir,
        embedder,
        config.cache_dir,
        kinds,
        bundled_directory=BUNDLED_CORPUS,
        field_weights=config.bm25_field_weights,
        field_b=config.bm25_field_b,
        vector_weights=config.vector_field_weights,
    )


async def _cosmos_catalog(config: Config, snapshot: Path, kinds) -> MemoryCatalog:
    from azure.cosmos.aio import CosmosClient

    from .cosmos_catalog import CosmosCatalog

    started = time.monotonic()
    # Held for the life of the process: every query reuses its connections.
    client = CosmosClient(config.cosmos_endpoint, config.cosmos_key)
    container = client.get_database_client(config.cosmos_database).get_container_client(
        config.cosmos_container
    )
    catalog = await asyncio.to_thread(
        CosmosCatalog.connect, snapshot, container, "", kinds,
        config.bm25_field_weights, config.vector_field_weights,
    )
    # One service, one collection: its partition is that collection's kind.
    sites = catalog.sites
    catalog.partition = config.cosmos_partition or (kinds or {}).get(sites[0], sites[0])
    catalog.client = client
    logger.info("cosmos catalog %s/%s partition %s: keyword index of %d records in %.1fs",
                config.cosmos_database, config.cosmos_container, catalog.partition,
                len(catalog.documents), time.monotonic() - started)
    return catalog


async def run_refresh_loop(
    services: Services,
    sources: list[Source],
    state: RefreshState,
    interval_seconds: float,
) -> None:
    """Recheck sources forever, rebuilding only when something changed."""
    if not sources or interval_seconds <= 0:
        return
    while True:
        try:
            await asyncio.sleep(interval_seconds)
            report = await refresh_once(services.config, sources, state)
            if report is None or not report.changed:
                continue

            catalog = await build_catalog(services.config, sources, services.embedder)
            if not catalog.documents:
                # Never swap in an empty catalog: a transient fetch problem
                # would otherwise take the whole corpus offline.
                logger.error("refresh produced an empty catalog; keeping the current one")
                continue

            previous = len(services.catalog.documents)
            services.catalog = catalog  # atomic rebind; in-flight queries unaffected
            state.rebuilds += 1
            state.last_change = _now()
            logger.info(
                "catalog rebuilt: %d records (was %d)", len(catalog.documents), previous
            )
        except asyncio.CancelledError:
            raise
        except Exception:
            # A refresh failure must never end the loop; the next tick retries.
            logger.exception("source refresh failed; will retry at the next interval")
