"""A catalog whose records and vectors are on the neuralweb VM (vector_service.py).

The NYT collections (recipes, Wirecutter reviews) are served by the qdrss Cloud
Run service with the other NYT verticals, and every NYT vector lives on the VM.
So the service holds only the keyword index, as with Cosmos (cosmos_catalog.py):
the in-memory BM25F arrays with one stub per record. Keyword hits are fetched
from the VM by id; vector search runs there, exactly as MemoryCatalog runs it
(the weighted sum of the query's similarity to each record's name, description
and body embeddings, over every record of the site).

If the VM does not answer, the search fails with VectorServiceError: there is no
fallback, and a failed vector service never becomes an empty result.
"""

from __future__ import annotations

import base64
from pathlib import Path
from typing import Any

import httpx
import numpy as np

from .catalog import MemoryCatalog
from .models import Candidate, Document


class VectorServiceError(RuntimeError):
    pass


def document_wire(document: Document) -> dict[str, Any]:
    return {
        "id": document.id,
        "url": document.url,
        "name": document.name,
        "site": document.site,
        "text": document.text,
        "schema_object": document.schema_object,
        "fields": [list(pair) for pair in document.fields],
    }


def document_from_wire(row: dict[str, Any]) -> Document:
    return Document(
        row["id"],
        row["url"],
        row["name"],
        row["site"],
        row["text"],
        row["schema_object"],
        tuple((name, text) for name, text in row["fields"]),
    )


class VMCatalog(MemoryCatalog):
    client: httpx.AsyncClient
    url: str

    @classmethod
    def connect(
        cls,
        keyword_snapshot: Path,
        url: str,
        token: str,
        kinds: dict[str, str] | None = None,
        field_weights: dict[str, float] | None = None,
        vector_weights: dict[str, float] | None = None,
        timeout: float = 10.0,
    ) -> VMCatalog:
        base = MemoryCatalog.load_snapshot(keyword_snapshot, kinds, field_weights, vector_weights)
        catalog = cls.__new__(cls)
        catalog.__dict__.update(base.__dict__)
        catalog.url = url.rstrip("/")
        catalog.client = httpx.AsyncClient(
            timeout=timeout, headers={"authorization": f"Bearer {token}"}
        )
        return catalog

    async def search_bm25(
        self, query: str, site: str | None, limit: int, exclude_ids: set[str] | None = None
    ) -> list[Candidate]:
        stubs = await super().search_bm25(query, site, limit, exclude_ids)
        records = await self.fetch([c.document.id for c in stubs])
        return [
            Candidate(
                records[c.document.id],
                c.vector_score,
                c.bm25_score,
                c.retrieval_score,
                c.retrieval_source,
            )
            for c in stubs
            if c.document.id in records
        ]

    async def search_vector(
        self,
        vector: np.ndarray,
        site: str | None,
        limit: int,
        exclude_ids: set[str] | None = None,
    ) -> list[Candidate]:
        sites = self._physical_sites(site)
        if not sites or limit <= 0:
            return []
        body = await self._post(
            "/vector",
            {
                "vector": base64.b64encode(np.asarray(vector, dtype=np.float32).tobytes()).decode(),
                "sites": sites,
                "limit": limit,
                "exclude_ids": sorted(exclude_ids or ()),
            },
        )
        return [
            Candidate(
                document_from_wire(row["document"]), row["score"], None, row["score"], "vector"
            )
            for row in body["candidates"]
        ]

    def _physical_sites(self, site: str | None) -> list[str]:
        """The collections a scope covers (a scope may be a kind, like "recipes", which only the
        manifest here maps to collections), so the VM need not know the manifest."""
        mask = self._scope_mask(site)
        if mask is None:
            return self.sites
        return sorted({self.documents[i].site for i in np.flatnonzero(mask)})

    async def fetch(self, doc_ids: list[str]) -> dict[str, Document]:
        if not doc_ids:
            return {}
        body = await self._post("/documents", {"ids": doc_ids})
        return {row["id"]: document_from_wire(row) for row in body["documents"]}

    async def _post(self, path: str, payload: dict[str, Any]) -> dict[str, Any]:
        try:
            response = await self.client.post(self.url + path, json=payload)
            response.raise_for_status()
            return response.json()
        except (httpx.HTTPError, ValueError) as exc:
            raise VectorServiceError(
                f"{self.url}{path}: {type(exc).__name__}: {str(exc)[:160]}"
            ) from exc

    async def aclose(self) -> None:
        await self.client.aclose()
