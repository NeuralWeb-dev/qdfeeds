"""A catalog whose documents and vectors live in Azure Cosmos DB.

Startup was dominated by downloading and mapping the snapshot -- 0.7 GB for
movies, 9s of a 15s cold start -- and nearly all of that is the embeddings and
the records. Here those stay in Cosmos (tools/load_cosmos.py), and the service
holds only the keyword index: the same BM25F arrays as the in-memory catalog,
with one small stub per record (id and collection) in place of the record. A
restart downloads tens of megabytes, not hundreds.

Keyword search is therefore exactly the in-memory BM25F -- Cosmos's own
full-text ranking cannot weight a name match 32x, which is what wins director
and place-name queries -- and its hits are fetched from Cosmos by id.

Vector search keeps the in-memory scoring too: the sum of the query's cosine
similarity to each record's name, description and body embeddings. Cosmos can
order by one VectorDistance at a time, so each field's top candidates are
queried in parallel, the union is re-scored on all three fields in one query,
and the best by summed similarity are fetched. A record outside every field's
top few could in principle be missed; quantizedFlat is itself approximate.
"""

from __future__ import annotations

import asyncio
import logging
from pathlib import Path
from typing import Any

import numpy as np

from .catalog import MemoryCatalog
from .models import Candidate, Document

logger = logging.getLogger(__name__)

FIELDS = ("name", "description", "body")
# Per-field candidates gathered before re-scoring, as a multiple of the limit.
# Against the in-memory sum over the 31 sample questions, top-20 overlap was
# 94.7% at 2x (worst query 50%), 99.2% at 4x (worst 90%), 100% at 8x -- at
# 396, 475 and 629 RU per vector search. 4x is the knee.
WIDEN = 4


class CosmosCatalog(MemoryCatalog):
    container: Any
    partition: str

    @classmethod
    def connect(
        cls,
        keyword_snapshot: Path,
        container,
        partition: str,
        kinds: dict[str, str] | None = None,
        field_weights: dict[str, float] | None = None,
        vector_weights: dict[str, float] | None = None,
    ) -> CosmosCatalog:
        base = MemoryCatalog.load_snapshot(keyword_snapshot, kinds, field_weights, vector_weights)
        catalog = cls.__new__(cls)
        catalog.__dict__.update(base.__dict__)
        catalog.container = container
        catalog.partition = partition
        return catalog

    # ---- keyword: in-memory BM25F, records fetched from Cosmos -------------

    async def search_bm25(
        self, query: str, site: str | None, limit: int, exclude_ids: set[str] | None = None
    ) -> list[Candidate]:
        stubs = await super().search_bm25(query, site, limit, exclude_ids)
        records = await self._fetch([c.document.id for c in stubs])
        return [
            Candidate(records[c.document.id], c.vector_score, c.bm25_score,
                      c.retrieval_score, c.retrieval_source)
            for c in stubs
            if c.document.id in records
        ]

    # ---- vector: per-field top-k in Cosmos, re-scored on all three ----------

    async def search_vector(
        self, vector: np.ndarray, site: str | None, limit: int,
        exclude_ids: set[str] | None = None,
    ) -> list[Candidate]:
        if not self.documents or limit <= 0:
            return []
        if site is not None and not self._scope_mask(site).any():
            return []
        excluded = exclude_ids or set()
        values = [float(x) for x in np.asarray(vector, dtype=np.float32)]
        width = limit * WIDEN + len(excluded)
        weights = {f: self.vector_weights.get(f, 0.0) for f in FIELDS}
        heads = await asyncio.gather(*(
            self._query(
                f"SELECT TOP {width} c.doc_id FROM c "
                f"ORDER BY VectorDistance(c.vec_{field}, @v)",
                [{"name": "@v", "value": values}],
            )
            for field in FIELDS if weights[field]
        ))
        union = list(dict.fromkeys(
            row["doc_id"] for head in heads for row in head if row["doc_id"] not in excluded
        ))
        if not union:
            return []
        rows = await self._query(
            "SELECT c.doc_id, " + ", ".join(
                f"VectorDistance(c.vec_{field}, @v) AS {field}" for field in FIELDS
            ) + " FROM c WHERE ARRAY_CONTAINS(@ids, c.doc_id)",
            [{"name": "@v", "value": values}, {"name": "@ids", "value": union}],
        )
        # A field with no embedding (empty text) scores 0, as its zero vector
        # does in memory.
        scored = sorted(
            ((sum(weights[f] * float(row.get(f) or 0.0) for f in FIELDS), row["doc_id"])
             for row in rows),
            key=lambda pair: (-pair[0], pair[1]),
        )[:limit]
        records = await self._fetch([doc_id for _, doc_id in scored])
        return [
            Candidate(records[doc_id], score, None, score, "vector")
            for score, doc_id in scored
            if doc_id in records
        ]

    # ---- Cosmos access -----------------------------------------------------

    async def _query(self, query: str, parameters: list[dict]) -> list[dict]:
        items = self.container.query_items(
            query, parameters=parameters, partition_key=self.partition
        )
        return [item async for item in items]

    async def _fetch(self, doc_ids: list[str]) -> dict[str, Document]:
        if not doc_ids:
            return {}
        rows = await self._query(
            "SELECT c.doc_id, c.url, c.name, c.site, c.schema, "
            "c.text_name, c.text_description, c.text_body "
            "FROM c WHERE ARRAY_CONTAINS(@ids, c.doc_id)",
            [{"name": "@ids", "value": doc_ids}],
        )
        return {row["doc_id"]: _document(row) for row in rows}


def _document(row: dict) -> Document:
    fields = tuple(
        (field, row.get(f"text_{field}") or "")
        for field in FIELDS if row.get(f"text_{field}")
    )
    return Document(
        row["doc_id"], row.get("url") or row["doc_id"], row.get("name") or "",
        row.get("site") or "", " ".join(text for _, text in fields),
        row.get("schema") or {}, fields,
    )
