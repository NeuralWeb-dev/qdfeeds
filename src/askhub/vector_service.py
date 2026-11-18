"""The NYT collections' records and vectors, served from the neuralweb VM.

    ASKHUB_VECTOR_TOKEN=... ASKHUB_VECTOR_SNAPSHOTS=/srv/recipes,/srv/reviews \
        python -m askhub.vector_service --host 10.128.0.2 --port 8091

Each directory is a full snapshot (tools/build_snapshot.py: catalog.pkl and the
per-field vectors, as in gs://neuralweb-nlweb-<collection>/snapshot/), loaded
with MemoryCatalog.load_snapshot -- the vectors memory-mapped -- and searched
exactly as an in-memory service searches it. The qdrss web app's VMCatalog asks
for vector search in named collections and for records by id; every request needs the shared token.
"""

from __future__ import annotations

import argparse
import base64
import hmac
import logging
import os
import time
from pathlib import Path

import numpy as np
from starlette.applications import Starlette
from starlette.requests import Request
from starlette.responses import JSONResponse
from starlette.routing import Route

from .catalog import MemoryCatalog
from .config import Config
from .vm_catalog import document_wire

logger = logging.getLogger("askhub.vector_service")


def create_vector_app(catalogs: list[MemoryCatalog], token: str) -> Starlette:
    by_site = {site: catalog for catalog in catalogs for site in catalog.sites}

    def authorized(request: Request) -> bool:
        given = request.headers.get("authorization", "").removeprefix("Bearer ")
        return hmac.compare_digest(given.encode(), token.encode())

    async def vector(request: Request) -> JSONResponse:
        if not authorized(request):
            return JSONResponse({"error": "unauthorized"}, status_code=401)
        body = await request.json()
        query = np.frombuffer(base64.b64decode(body["vector"]), dtype=np.float32)
        limit, exclude = int(body["limit"]), set(body.get("exclude_ids") or ())
        unknown = [site for site in body["sites"] if site not in by_site]
        if unknown:  # never an empty answer for a collection this VM does not hold
            return JSONResponse({"error": f"no collection {unknown}"}, status_code=404)
        found = [
            c
            for site in body["sites"]
            for c in await by_site[site].search_vector(query, site, limit, exclude)
        ]
        found.sort(key=lambda c: (-c.retrieval_score, c.document.id))
        return JSONResponse(
            {
                "candidates": [
                    {"document": document_wire(c.document), "score": c.retrieval_score}
                    for c in found[:limit]
                ]
            }
        )

    async def documents(request: Request) -> JSONResponse:
        if not authorized(request):
            return JSONResponse({"error": "unauthorized"}, status_code=401)
        ids = (await request.json())["ids"]
        out = []
        for catalog in catalogs:
            positions = catalog._positions()
            out += [document_wire(catalog.documents[positions[i]]) for i in ids if i in positions]
        return JSONResponse({"documents": out})

    async def health(_: Request) -> JSONResponse:
        return JSONResponse(
            {
                "service": "askhub-vectors",
                "sites": {site: len(catalog.documents) for site, catalog in by_site.items()},
            }
        )

    return Starlette(
        routes=[
            Route("/vector", vector, methods=["POST"]),
            Route("/documents", documents, methods=["POST"]),
            Route("/health", health),
        ]
    )


def main() -> None:
    import uvicorn
    from dotenv import load_dotenv

    load_dotenv()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default=os.getenv("ASKHUB_VECTOR_HOST", "127.0.0.1"))
    parser.add_argument("--port", type=int, default=int(os.getenv("ASKHUB_VECTOR_PORT", "8091")))
    args = parser.parse_args()
    config = Config.from_env()
    catalogs = []
    for directory in os.environ["ASKHUB_VECTOR_SNAPSHOTS"].split(","):
        started = time.monotonic()
        catalog = MemoryCatalog.load_snapshot(
            Path(directory.strip()), None, config.bm25_field_weights, config.vector_field_weights
        )
        catalog._positions()  # built now, not on the first request
        catalogs.append(catalog)
        logger.info(
            "loaded %s: %d records of %s in %.1fs",
            directory,
            len(catalog.documents),
            catalog.sites,
            time.monotonic() - started,
        )
    uvicorn.run(
        create_vector_app(catalogs, os.environ["ASKHUB_VECTOR_TOKEN"]),
        host=args.host,
        port=args.port,
        log_level="warning",
    )


if __name__ == "__main__":
    main()
