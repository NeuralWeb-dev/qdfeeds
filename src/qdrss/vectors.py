"""Vector search as its own service on the neuralweb VM, and the web app's client for it.

The search page's collections are searched exactly in memory (memory.py): about 6 GB of vectors and items, loaded
from the GCS snapshots and caught up from Cosmos every five minutes. Held in the Cloud Run service, that needed an
always-on 4 vCPU / 12 GiB instance; the VM already runs and has the room. So the VM runs `qdrss-vectors`, listening
only on its internal address, and the web app (in the VPC through Direct VPC egress) asks it over HTTP.

Wire: POST /search with the query vector as base64 float32 and the same filters as Retriever.search; the answer is
the candidates with their items. A shared token (QDRSS_VECTOR_TOKEN) is required on every search.

If the VM does not answer (down, restarting, loading), the client searches Cosmos instead, which is slow but right,
and says so in the log and in /health: a failed vector service never becomes an empty result.
"""

from __future__ import annotations

import base64
import hmac
import logging
import os
import time
from dataclasses import asdict
from datetime import datetime
from pathlib import Path

import httpx
import numpy as np

from .models import Candidate, Item

logger = logging.getLogger("qdrss")


def _item_wire(item: Item) -> dict:
    return {**asdict(item), "published_at": item.published_at and item.published_at.isoformat(),
            "ingested_at": item.ingested_at.isoformat()}


def _item(row: dict) -> Item:
    return Item(**{**row, "published_at": datetime.fromisoformat(row["published_at"]) if row["published_at"] else None,
                   "ingested_at": datetime.fromisoformat(row["ingested_at"])})


def request_body(vector, since, collection, limit, sources, collections) -> dict:
    return {
        "vector": base64.b64encode(np.asarray(vector, dtype=np.float32).tobytes()).decode(),
        "since": since.isoformat() if since else None, "collection": collection, "limit": limit,
        "sources": sorted(sources) if sources is not None else None,
        "collections": sorted(collections) if collections is not None else None,
    }


class RemoteRetriever:
    """The web app's retriever when QDRSS_VECTOR_URL is set: the VM first, Cosmos when the VM fails."""

    def __init__(self, url: str, token: str, fallback, timeout: float = 10.0):
        self.url = url.rstrip("/")
        self.fallback = fallback
        self.client = httpx.AsyncClient(timeout=timeout, headers={"authorization": f"Bearer {token}"})
        self.status = {"url": self.url, "last_ok": None, "last_failure": None, "fallbacks": 0}

    async def search(self, vector, since, collection, limit, sources=None, collections=None) -> list[Candidate]:
        try:
            response = await self.client.post(f"{self.url}/search",
                                              json=request_body(vector, since, collection, limit, sources, collections))
            response.raise_for_status()
            rows = response.json()["candidates"]
            self.status["last_ok"] = time.time()
            return [Candidate(_item(r["item"]), "vector", float(r["score"])) for r in rows]
        except (httpx.HTTPError, KeyError, TypeError, ValueError) as exc:
            self.status["fallbacks"] += 1
            self.status["last_failure"] = f"{time.time():.0f} {type(exc).__name__}: {str(exc)[:160]}"
            logger.warning("vector service failed (%s: %s); searching Cosmos", type(exc).__name__, str(exc)[:160])
            return await self.fallback.search(vector, since, collection, limit, sources, collections)

    async def aclose(self) -> None:
        await self.client.aclose()


def create_vector_app(retriever, token: str, status):
    """The VM's service: /search for the web app, /health for people."""
    from starlette.applications import Starlette
    from starlette.requests import Request
    from starlette.responses import JSONResponse
    from starlette.routing import Route

    async def search(request: Request) -> JSONResponse:
        given = request.headers.get("authorization", "").removeprefix("Bearer ")
        if not hmac.compare_digest(given.encode(), token.encode()):
            return JSONResponse({"error": "unauthorized"}, status_code=401)
        body = await request.json()
        vector = np.frombuffer(base64.b64decode(body["vector"]), dtype=np.float32)
        since = datetime.fromisoformat(body["since"]) if body.get("since") else None
        sources = frozenset(body["sources"]) if body.get("sources") is not None else None
        collections = frozenset(body["collections"]) if body.get("collections") is not None else None
        found = await retriever.search(vector, since, body.get("collection"), int(body["limit"]), sources, collections)
        return JSONResponse({"candidates": [{"item": _item_wire(c.item), "score": c.score} for c in found]})

    async def health(_: Request) -> JSONResponse:
        return JSONResponse({"service": "qdrss-vectors", "collections": status()})

    return Starlette(routes=[Route("/search", search, methods=["POST"]), Route("/health", health)])


def main() -> None:
    """qdrss-vectors: load the snapshots, catch up from Cosmos, serve on QDRSS_VECTOR_HOST:QDRSS_VECTOR_PORT."""
    import asyncio

    import uvicorn
    from dotenv import load_dotenv

    from .app import configure_logging
    from .cosmos import CosmosStore
    from .ingest import load_sources
    from .memory import MemoryRetriever

    load_dotenv()
    configure_logging()
    env = os.environ
    token = env["QDRSS_VECTOR_TOKEN"]
    sources = load_sources(Path(env.get("QDRSS_SOURCES", "sources.yaml")))
    store = CosmosStore(env["COSMOS_ENDPOINT"], env["COSMOS_KEY"], env.get("COSMOS_DATABASE", "qdrss"))
    memory = MemoryRetriever(store, {s.name: s.collection for s in sources},
                             [c for c in env["QDRSS_MEMORY_COLLECTIONS"].split(",") if c.strip()],
                             env["QDRSS_MEMORY_SNAPSHOT"])

    async def run() -> None:
        await store.setup()
        await memory.load()
        config = uvicorn.Config(create_vector_app(memory, token, lambda: memory.status),
                                host=env.get("QDRSS_VECTOR_HOST", "127.0.0.1"),
                                port=int(env.get("QDRSS_VECTOR_PORT", "8090")), log_level="warning")
        try:
            await uvicorn.Server(config).serve()
        finally:
            await store.close()

    asyncio.run(run())
