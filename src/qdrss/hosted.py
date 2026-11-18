"""Cooking and Wirecutter answered in this process by askhub (the NLWeb pipeline, src/askhub).

They used to be separate Cloud Run services (nlweb-recipes, nlweb-reviews) that the search
page called over HTTP. Here each is an askhub `Services` built at startup from a fixed
configuration: the service downloads only the collection's keyword index from its bucket
(`keyword/`, about 54 MB for recipes); the records and vectors stay on the neuralweb VM
(askhub-vectors, askhub/vector_service.py), as every other NYT vertical's do.

`events(vertical, payload)` runs one search and yields its events in the NLWeb wire format
(`message_type`, `content`) that nlweb.parse reads, the same events the HTTP services sent.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
import os
from collections.abc import AsyncIterator, Mapping
from pathlib import Path
from typing import Any

from .nlweb import NLWebError

logger = logging.getLogger(__name__)

# What nlweb-recipes and nlweb-reviews ran with (their Dockerfile defaults and Cloud Run
# variables), less what only an HTTP service needs (host, port, CORS, MCP hosts, rate limits).
COMMON = {
    "ASKHUB_GCS_SNAPSHOT_PREFIX": "keyword/",
    "ASKHUB_REFRESH_INTERVAL_HOURS": "0",
    "ASKHUB_BM25_RANK_COUNT": "20",
    "ASKHUB_VECTOR_RANK_COUNT": "20",
    "ASKHUB_DESCRIBE_MODEL": "google/gemini-3.8-flash",
    "ASKHUB_DESCRIBE_COUNT": "10",
    "ASKHUB_DESCRIBE_PROJECT": "1",
    "OPENROUTER_RANKING_PROVIDER_ORDER": "coreweave;novita;groq",
    "OPENROUTER_APP_TITLE": "nlweb-samples",
}
SITES: dict[str, dict[str, str]] = {
    "recipes": {
        "ASKHUB_GCS_BUCKET": "gs://neuralweb-nlweb-recipes",
        "ASKHUB_RESTRICTION_SITES": "recipes",
        "ASKHUB_SUBSTITUTION_SITES": "recipes",
        "ASKHUB_SIMPLE_SITES": "recipes",
        "ASKHUB_SIMPLE_SUMMARY_MODEL": "google/gemini-2.5-flash-lite",
    },
    "reviews": {
        "ASKHUB_GCS_BUCKET": "gs://neuralweb-nlweb-reviews",
    },
}
MANIFEST = Path(__file__).resolve().parents[2] / "askhub-sources.yaml"


class HostedError(NLWebError):
    pass


def settings(collection: str, data_root: Path, environ: Mapping[str, str]) -> dict[str, str]:
    """The askhub configuration for one collection: COMMON, its own settings, where its files go, and
    the keys and VM address from this process's environment."""
    data = data_root / collection
    return {
        **COMMON,
        **SITES[collection],
        "ASKHUB_DATA_DIR": str(data),
        "ASKHUB_CACHE_DIR": str(data / ".cache"),
        "ASKHUB_SNAPSHOT_DIR": str(data / "snapshot"),
        "ASKHUB_SOURCES_MANIFEST": str(MANIFEST),
        "ASKHUB_VECTOR_URL": environ.get("QDRSS_ASKHUB_VECTOR_URL", ""),
        "ASKHUB_VECTOR_TOKEN": environ.get("QDRSS_VECTOR_TOKEN", ""),
        "OPENROUTER_API_KEY": environ.get("OPENROUTER_API_KEY", ""),
        "OPENROUTER_BASE_URL": environ.get("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1"),
    }


class Hosted:
    """The askhub services of the NLWeb verticals, keyed by collection ("recipes", "reviews")."""

    def __init__(self, collections: list[str], data_root: Path, environ: Mapping[str, str] | None = None):
        self.collections = collections
        self.data_root = data_root
        self.environ = os.environ if environ is None else environ
        self.services: dict[str, Any] = {}
        self.configs: dict[str, Any] = {}
        self._decision_rankers: list[Any] = []
        self._ready: asyncio.Event = asyncio.Event()
        self._task: asyncio.Task | None = None
        self.error: str | None = None

    def start(self) -> None:
        """Build every collection's services in the background: the web app answers the other tabs while
        the keyword indexes download."""
        self._task = asyncio.create_task(self._build_all())

    async def _build_all(self) -> None:
        try:
            await asyncio.gather(*(self._build(c) for c in self.collections))
        except Exception as exc:  # reported by events() and /health, not lost in a task
            self.error = f"{type(exc).__name__}: {exc}"
            logger.exception("askhub services failed to start")
        finally:
            self._ready.set()

    async def _build(self, collection: str) -> None:
        from askhub.app import build_services
        from askhub.config import Config

        config = Config.from_env(settings(collection, self.data_root, self.environ))
        if not config.vector_url:
            raise HostedError("QDRSS_ASKHUB_VECTOR_URL is not set")
        self.configs[collection] = config
        self.services[collection] = await build_services(config, owned_decision_rankers=self._decision_rankers)
        logger.info("askhub %s ready", collection)

    async def wait(self, timeout: float = 120.0) -> None:
        await asyncio.wait_for(self._ready.wait(), timeout)

    async def events(self, collection: str, payload: dict) -> AsyncIterator[dict]:
        """One search's events, as the HTTP service streamed them."""
        from askhub.app import _request
        from askhub.pipeline import search

        await self.wait()
        if collection not in self.services:
            raise HostedError(self.error or f"{collection} is not hosted")
        request = _request(payload, self.configs[collection])
        try:
            async for event in search(request, self.services[collection]):
                yield event.wire()
        except Exception as exc:  # any pipeline failure is this tab being unavailable (a 503 or an error line)
            raise HostedError(f"{collection}: {type(exc).__name__}: {exc}") from exc

    def status(self) -> dict:
        return {"ready": sorted(self.services), "starting": not self._ready.is_set(), "error": self.error}

    async def aclose(self) -> None:
        if self._task:
            self._task.cancel()
            with contextlib.suppress(asyncio.CancelledError):  # _build_all keeps every other failure
                await self._task
        for ranker in self._decision_rankers:
            await ranker.aclose()
