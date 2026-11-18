from __future__ import annotations

import asyncio
import contextlib
import hashlib
import json
import logging
import math
import pickle
import re
from collections import Counter
from pathlib import Path

import numpy as np

from .adapters import SUPPORTED_SUFFIXES, load_file
from .config import (
    DEFAULT_BM25_FIELD_B,
    DEFAULT_BM25_FIELD_WEIGHTS,
    DEFAULT_VECTOR_FIELD_WEIGHTS,
)
from .models import RETRIEVAL_FIELDS, Candidate, Document
from .providers import Embeddings
from .sources import KINDS as KIND_SCOPES

SNAPSHOT_STATE = "catalog.pkl"
SNAPSHOT_VERSION = 1

# English function words, removed from queries before scoring. Without this a
# question phrased as a sentence is scored on its grammar: "A small Japanese
# language model I can run locally" put 68% of its top hit's score on the token
# `i`, which matched the model named IF-I-M-v1.0.
#
# Queries only, never the index. Nineteen of these words also appear as
# language codes on records in this corpus -- `it`, `is`, `be`, `as`, `am`,
# `my`, `or`, `he`, `in`, `to` among them -- and languages are recorded as
# codes, so stripping these from documents would delete the only marker an
# Italian, Icelandic or Burmese record carries. Measured on the 100-query
# benchmark: query-side removal lifts
# descriptive nDCG@10 from 0.627 to 0.650 and leaves known-item at 0.978, while
# stripping both sides reaches only 0.647 and drops known-item to 0.967.
STOPWORDS = frozenset(
    [
    "a", "an", "the", "of", "for", "on", "in", "to", "with", "that", "this", "these", "those",
    "and", "or", "but", "if", "then", "than", "is", "are", "was", "were", "be", "been", "am",
    "i", "me", "my", "we", "our", "you", "your", "it", "its", "they", "them", "their", "he",
    "she", "his", "her", "can", "could", "would", "should", "will", "what", "which", "who",
    "whom", "how", "when", "where", "why", "do", "does", "did", "need", "needs", "want",
    "find", "show", "give", "any", "some", "there", "here", "about", "from", "by", "at", "as",
    "into", "over", "under", "out", "up", "down"
    ]
)
logger = logging.getLogger(__name__)


class MemoryCatalog:
    def __init__(
        self,
        documents: tuple[Document, ...],
        matrix: np.ndarray,
        cache_hit: bool = False,
        kinds: dict[str, str] | None = None,
        field_weights: dict[str, float] | None = None,
        field_b: dict[str, float] | None = None,
        field_matrices: dict[str, np.ndarray] | None = None,
        vector_weights: dict[str, float] | None = None,
    ):
        self.documents = documents
        self.matrix = matrix
        # One embedding per field, when the corpus was indexed that way. A
        # single vector over the whole record is dominated by whatever field is
        # longest: for openai/whisper-large-v3 the name is 47 characters against
        # 2688 of body, so the name contributes almost nothing to the direction
        # of the vector and a query naming the model cannot match on it.
        self.field_matrices = field_matrices or {}
        self.vector_weights = dict(vector_weights or DEFAULT_VECTOR_FIELD_WEIGHTS)
        self.cache_hit = cache_hit
        # Collection name -> manifest section. The UI's aggregate scopes filter
        # on this, so adding a collection is a manifest edit, not a code change.
        self.kinds = kinds or {}
        self._sites = sorted({document.site for document in documents})
        self._scope_masks: dict[str, np.ndarray] = {}
        self._id_positions: dict[str, int] | None = None
        self._field_weights = dict(field_weights or DEFAULT_BM25_FIELD_WEIGHTS)
        self._field_b = dict(field_b or DEFAULT_BM25_FIELD_B)
        self._build_index()

    def _build_index(self) -> None:
        """Per-field postings and length normalisation for BM25F.

        Document frequency is counted once per document across all fields, not
        once per field. Per-field IDF is the trap in fielded scoring: a term
        that is everywhere in bodies but rare in names would earn a huge
        name-IDF, and a single incidental name token would then outrank a real
        match.
        """
        per_field = [document.field_text() for document in self.documents]
        names = {name for fields in per_field for name in fields}
        # Any field an adapter emits but config never weighted still scores, at
        # the neutral weight, rather than silently vanishing from retrieval.
        self._fields = tuple(sorted(names, key=lambda n: -self._field_weights.get(n, 1.0)))

        # Postings per field as flat arrays rather than lists of tuples: one
        # field's postings for a token are doc_ids[start:end] and freqs[start:end],
        # with (start, end) in that field's vocabulary. Millions of Python tuples
        # took seconds to build and gigabytes to hold; arrays score with numpy and
        # save and load in a snapshot without being rebuilt.
        self._vocab: dict[str, dict[str, tuple[int, int]]] = {}
        self._doc_ids: dict[str, np.ndarray] = {}
        self._freqs: dict[str, np.ndarray] = {}
        self._norms: dict[str, np.ndarray] = {}
        count = len(self.documents)
        seen: Counter[str] = Counter()

        for name in self._fields:
            tokens = [_tokens(fields.get(name, "")) for fields in per_field]
            lengths = np.asarray([len(t) for t in tokens], dtype=np.float32)
            # Empty fields are excluded from the average: a corpus where half
            # the records have no description should not halve the yardstick
            # the other half is measured against.
            present = lengths[lengths > 0]
            average = float(present.mean()) if present.size else 0.0
            b = self._field_b.get(name, 0.75)
            self._norms[name] = (
                1 - b + b * lengths / average if average else np.ones(count, dtype=np.float32)
            ).astype(np.float32)
            postings: dict[str, list[tuple[int, int]]] = {}
            for index, document_tokens in enumerate(tokens):
                for token, frequency in Counter(document_tokens).items():
                    postings.setdefault(token, []).append((index, frequency))
            vocab: dict[str, tuple[int, int]] = {}
            doc_ids: list[int] = []
            freqs: list[int] = []
            for token, entries in postings.items():
                vocab[token] = (len(doc_ids), len(doc_ids) + len(entries))
                for index, frequency in entries:
                    doc_ids.append(index)
                    freqs.append(frequency)
            self._vocab[name] = vocab
            self._doc_ids[name] = np.asarray(doc_ids, dtype=np.int32)
            self._freqs[name] = np.asarray(freqs, dtype=np.float32)

        for fields in per_field:
            seen.update({token for text in fields.values() for token in _tokens(text)})
        self._document_frequency = seen

    @property
    def sites(self) -> list[str]:
        return list(self._sites)

    async def search_vector(
        self,
        vector: np.ndarray,
        site: str | None,
        limit: int,
        exclude_ids: set[str] | None = None,
    ) -> list[Candidate]:
        if not self.documents:
            return []
        vector_scores = self._vector_scores(vector)
        return [
            Candidate(self.documents[index], float(vector_scores[index]), None,
                      float(vector_scores[index]), "vector")
            for index in self._top(vector_scores, site, limit, exclude_ids)
        ]

    def _vector_scores(self, vector: np.ndarray) -> np.ndarray:
        """Similarity to the query, combined across per-field embeddings.

        Empty fields were embedded as zero vectors, so they contribute nothing
        rather than dragging a record toward the origin of an averaged space.
        """
        if not self.field_matrices:
            return self.matrix @ vector
        total = np.zeros(len(self.documents), dtype=np.float32)
        for name, field_matrix in self.field_matrices.items():
            weight = self.vector_weights.get(name, 0.0)
            if weight:
                total += weight * (field_matrix @ vector)
        return total

    async def search_bm25(
        self,
        query: str,
        site: str | None,
        limit: int,
        exclude_ids: set[str] | None = None,
    ) -> list[Candidate]:
        if not self.documents:
            return []
        scores = self._bm25(query)
        return [
            Candidate(self.documents[index], None, float(scores[index]),
                      float(scores[index]), "bm25")
            for index in self._top(scores, site, limit, exclude_ids)
        ]

    def _top(
        self, scores: np.ndarray, site: str | None, limit: int, exclude_ids: set[str] | None
    ) -> list[int]:
        """Indices of the `limit` best in-scope records, by score then id.

        Was a Python pass over every record plus a full sort -- 95% of a search's
        time (~20ms of 21 locally, ~60ms on one Cloud Run vCPU), and it blocked
        the event loop, so query expansion's dozen extra searches delayed every
        other request in flight. Now numpy: a cached scope mask, excluded records
        pushed to -inf, a partial sort. Ties at the cut are all kept and ordered
        by id, so the result is exactly what the full sort gave.
        """
        working = np.asarray(scores, dtype=np.float64).copy()
        mask = self._scope_mask(site)
        if mask is not None:
            working[~mask] = -np.inf
        if exclude_ids:
            positions = self._positions()
            for document_id in exclude_ids:
                position = positions.get(document_id)
                if position is not None:
                    working[position] = -np.inf
        available = int(np.isfinite(working).sum())
        count = min(limit, available)
        if count <= 0:
            return []
        threshold = -np.partition(-working, count - 1)[count - 1]
        tied_or_better = np.nonzero(working >= threshold)[0]
        ordered = sorted(
            tied_or_better.tolist(),
            key=lambda i: (-working[i], self.documents[i].id),
        )
        return ordered[:count]

    def _scope_mask(self, site: str | None) -> np.ndarray | None:
        if site is None:
            return None
        if site not in self._scope_masks:
            self._scope_masks[site] = np.fromiter(
                (_matches_scope(document, site, self.kinds) for document in self.documents),
                dtype=bool, count=len(self.documents),
            )
        return self._scope_masks[site]

    def _positions(self) -> dict[str, int]:
        if self._id_positions is None:
            self._id_positions = {
                document.id: index for index, document in enumerate(self.documents)
            }
        return self._id_positions

    async def search(
        self,
        vector: np.ndarray,
        site: str | None,
        limit: int,
        query: str = "",
        method: str = "vector",
    ) -> list[Candidate]:
        """Compatibility helper for direct retrieval diagnostics."""
        if method == "bm25":
            return await self.search_bm25(query, site, limit)
        if method == "compare":
            bm25 = await self.search_bm25(query, site, limit)
            vector_results = await self.search_vector(
                vector, site, limit, {candidate.document.id for candidate in bm25}
            )
            return (bm25 + vector_results)[:limit]
        return await self.search_vector(vector, site, limit)

    def _bm25(self, query: str) -> np.ndarray:
        """BM25F: weighted term frequency across fields, then one saturation.

        Summing a separate BM25 score per field would be the obvious
        implementation and the wrong one -- saturation would apply per field, so
        a term appearing once in each of three fields would score three times a
        term appearing three times in one, and the weights would stop meaning
        what they say.
        """
        count = len(self.documents)
        scores = np.zeros(count, dtype=np.float32)
        if not query or not count:
            return scores
        k1 = 1.5
        for token in _query_tokens(query):
            document_frequency = self._document_frequency.get(token, 0)
            if not document_frequency:
                continue
            inverse_frequency = math.log(
                1 + (count - document_frequency + 0.5) / (document_frequency + 0.5)
            )
            weighted = np.zeros(count, dtype=np.float32)
            for name in self._fields:
                weight = self._field_weights.get(name, 1.0)
                span = self._vocab[name].get(token)
                if not weight or span is None:
                    continue
                rows = self._doc_ids[name][span[0]:span[1]]
                # A token appears at most once per document in a field's
                # postings, so plain fancy-index addition is exact here.
                frequencies = self._freqs[name][span[0]:span[1]]
                weighted[rows] += weight * frequencies / self._norms[name][rows]
            hit = weighted > 0
            scores[hit] += inverse_frequency * weighted[hit] / (k1 + weighted[hit])
        return scores

    # ---- snapshots ---------------------------------------------------------
    #
    # Parsing the corpus and building the keyword index took ~41s of a Cloud Run
    # cold start for the recipes collection. A snapshot is the finished catalog:
    # documents and the BM25F arrays in one pickle, and each per-field embedding
    # matrix as an uncompressed .npy that loads memory-mapped. Built once by
    # tools/build_snapshot.py, from the same files and code a cold build uses.
    #
    # Field *weights* apply at query time, so config still governs them; field
    # length normalisation (`b`) is baked into the norms when the snapshot is built.
    # The pickle is trusted input: it comes only from this service's own bucket.

    def save_snapshot(self, directory: Path) -> None:
        directory.mkdir(parents=True, exist_ok=True)
        for name, field_matrix in self.field_matrices.items():
            np.save(directory / f"vectors-{name}.npy",
                    np.ascontiguousarray(field_matrix, dtype=np.float32))
        state = {
            "version": SNAPSHOT_VERSION,
            "documents": self.documents,
            "dimensions": int(self.matrix.shape[1]),
            "vector_fields": sorted(self.field_matrices),
            "fields": self._fields,
            "field_b": self._field_b,
            "vocab": self._vocab,
            "doc_ids": self._doc_ids,
            "freqs": self._freqs,
            "norms": self._norms,
            "document_frequency": dict(self._document_frequency),
        }
        temporary = directory / (SNAPSHOT_STATE + ".tmp")
        with temporary.open("wb") as handle:
            pickle.dump(state, handle, protocol=pickle.HIGHEST_PROTOCOL)
        temporary.replace(directory / SNAPSHOT_STATE)

    @classmethod
    def load_snapshot(
        cls,
        directory: Path,
        kinds: dict[str, str] | None = None,
        field_weights: dict[str, float] | None = None,
        vector_weights: dict[str, float] | None = None,
    ) -> MemoryCatalog:
        with (directory / SNAPSHOT_STATE).open("rb") as handle:
            state = pickle.load(handle)
        if state.get("version") != SNAPSHOT_VERSION:
            raise ValueError(f"snapshot version {state.get('version')} != {SNAPSHOT_VERSION}")
        catalog = cls.__new__(cls)
        catalog.documents = state["documents"]
        catalog.matrix = np.empty((len(catalog.documents), state["dimensions"]), dtype=np.float32)
        catalog.field_matrices = {
            name: np.load(directory / f"vectors-{name}.npy", mmap_mode="r")
            for name in state["vector_fields"]
        }
        catalog.vector_weights = dict(vector_weights or DEFAULT_VECTOR_FIELD_WEIGHTS)
        catalog.cache_hit = True
        catalog.kinds = kinds or {}
        catalog._sites = sorted({document.site for document in catalog.documents})
        catalog._scope_masks = {}
        catalog._id_positions = None
        catalog._field_weights = dict(field_weights or DEFAULT_BM25_FIELD_WEIGHTS)
        catalog._field_b = state["field_b"]
        catalog._fields = state["fields"]
        catalog._vocab = state["vocab"]
        catalog._doc_ids = state["doc_ids"]
        catalog._freqs = state["freqs"]
        catalog._norms = state["norms"]
        catalog._document_frequency = Counter(state["document_frequency"])
        return catalog

    @classmethod
    async def load(
        cls,
        directory: Path,
        embedder: Embeddings,
        cache_directory: Path | None = None,
        kinds: dict[str, str] | None = None,
        bundled_directory: Path | None = None,
        field_weights: dict[str, float] | None = None,
        field_b: dict[str, float] | None = None,
        vector_weights: dict[str, float] | None = None,
    ) -> MemoryCatalog:
        """Build a catalog from runtime and bundled data, caching per source file.

        Caching per file rather than per corpus is what makes a periodic refresh
        affordable. A single fingerprint over every document means one changed
        podcast feed invalidates all of them, and re-embedding thousands of
        records is both slow and billed. Podcasts publish weekly, so exactly one
        file changing is the ordinary case.
        """
        cache_root = cache_directory or directory / ".cache"
        paths = await asyncio.to_thread(_catalog_paths, directory, bundled_directory)
        if not paths:
            return cls((), np.empty((0, embedder.dimensions), dtype=np.float32), kinds=kinds,
                       field_weights=field_weights, field_b=field_b)

        documents: list[Document] = []
        blocks: dict[str, list[np.ndarray]] = {}
        every_block_cached = True

        for path in paths:
            file_documents = await asyncio.to_thread(load_file, path)
            if not file_documents:
                continue
            fingerprint = _fingerprint(file_documents, embedder.cache_key)
            for name in RETRIEVAL_FIELDS:
                texts = [document.field_text().get(name, "") for document in file_documents]
                cache_file = cache_root / f"embeddings-{path.stem}-{name}-{fingerprint}.npz"
                matrix = await asyncio.to_thread(_read_cache, cache_file, file_documents)
                if matrix is None:
                    every_block_cached = False
                    populated = sum(1 for text in texts if text)
                    logger.info(
                        "embedding %s [%s] (%d of %d records populated)",
                        path.name, name, populated, len(file_documents),
                    )
                    matrix = await _embed_field(embedder, texts)
                    await asyncio.to_thread(_write_cache, cache_file, file_documents, matrix)
                blocks.setdefault(name, []).append(matrix)
            await asyncio.to_thread(_prune_stale_cache, cache_root, path.stem, fingerprint)
            documents.extend(file_documents)

        if not documents:
            return cls((), np.empty((0, embedder.dimensions), dtype=np.float32), kinds=kinds,
                       field_weights=field_weights, field_b=field_b)

        # Source adapters assign stable identities. Distinct offers can
        # legitimately share a vendor URL, so URL-based deduplication would lose
        # records; deduplicating by id keeps the vectors aligned with documents.
        field_matrices = {name: np.concatenate(parts, axis=0) for name, parts in blocks.items()}
        unique: dict[str, int] = {}
        for index, document in enumerate(documents):
            unique.setdefault(document.id, index)
        if len(unique) != len(documents):
            keep = sorted(unique.values())
            documents = [documents[i] for i in keep]
            field_matrices = {name: m[keep] for name, m in field_matrices.items()}

        return cls(
            tuple(documents),
            np.empty((len(documents), embedder.dimensions), dtype=np.float32),
            cache_hit=every_block_cached,
            kinds=kinds,
            field_weights=field_weights,
            field_b=field_b,
            field_matrices=field_matrices,
            vector_weights=vector_weights,
        )


def _matches_scope(
    document: Document, site: str | None, kinds: dict[str, str] | None = None
) -> bool:
    """Match a physical collection, or one of the manifest's aggregate scopes."""
    if site is None:
        return True
    kinds = kinds or {}
    if site in KIND_SCOPES:
        declared = kinds.get(document.site)
        if declared is not None:
            return declared == site
        # No manifest: infer from the item type, so a directory of loose files
        # still works.
        schema_type = document.schema_object.get("@type")
        types = schema_type if isinstance(schema_type, list) else [schema_type]
        return site in {str(t).lower() for t in types if t}
    return document.site == site


async def _embed_batches(embedder: Embeddings, texts: list[str], size: int = 64) -> np.ndarray:
    batches = [
        await embedder.embed(texts[start : start + size]) for start in range(0, len(texts), size)
    ]
    return np.concatenate(batches, axis=0)


async def _embed_field(embedder: Embeddings, texts: list[str]) -> np.ndarray:
    """Embed one field across a file, leaving empty entries as zero vectors.

    Empty strings are not sent: providers reject them, and a record with no
    description should score zero on the description channel rather than
    whatever an empty input happens to embed to.
    """
    populated = [index for index, text in enumerate(texts) if text.strip()]
    if not populated:
        return np.zeros((len(texts), embedder.dimensions), dtype=np.float32)
    embedded = await _embed_batches(embedder, [texts[index] for index in populated])
    matrix = np.zeros((len(texts), embedded.shape[1]), dtype=np.float32)
    matrix[populated] = embedded
    return matrix


def _fingerprint(documents: list[Document], embedding_key: str) -> str:
    digest = hashlib.sha256(embedding_key.encode())
    for document in documents:
        digest.update(document.id.encode())
        digest.update(b"\0")
        digest.update(document.text.encode())
        digest.update(b"\0")
    return digest.hexdigest()[:24]


def _read_cache(path: Path, documents: list[Document]) -> np.ndarray | None:
    if not path.is_file():
        return None
    try:
        with np.load(path, allow_pickle=False) as cache:
            ids = cache["ids"].tolist()
            matrix = cache["embeddings"].astype(np.float32, copy=False)
        if ids != [document.id for document in documents] or len(matrix) != len(documents):
            return None
        return matrix
    except (OSError, ValueError, KeyError):
        return None


def _write_cache(path: Path, documents: list[Document], matrix: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp.npz")
    np.savez_compressed(
        temporary,
        ids=np.asarray([document.id for document in documents]),
        embeddings=matrix,
        metadata=json.dumps({"documents": len(documents)}),
    )
    temporary.replace(path)


def _tokens(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", text.lower())


def _query_tokens(query: str) -> list[str]:
    """Query tokens with function words removed.

    A query that is nothing but stopwords keeps them: some models really are
    named `it` or `A`, and returning nothing at all is worse than returning a
    weak match.
    """
    tokens = _tokens(query)
    return [token for token in tokens if token not in STOPWORDS] or tokens


def _source_paths(directory: Path) -> list[Path]:
    if not directory.is_dir():
        return []
    # Skip dotfiles: the fetcher keeps its per-source state beside the data as
    # .sources-state.json, and a .json suffix would otherwise make it look like
    # a collection.
    return [
        path
        for path in sorted(directory.iterdir())
        if path.is_file()
        and not path.name.startswith(".")
        and path.suffix.lower() in SUPPORTED_SUFFIXES
    ]


def _catalog_paths(directory: Path, bundled_directory: Path | None) -> list[Path]:
    """Return one path per source name, preferring the bundled canonical copy.

    Preferring bundled content also makes upgrades safe when an older deployment
    left a formerly runtime-owned corpus file in its data directory.
    """
    paths = {path.name: path for path in _source_paths(directory)}
    if bundled_directory is not None:
        paths.update({path.name: path for path in _source_paths(bundled_directory)})
    return [paths[name] for name in sorted(paths)]


def _prune_stale_cache(cache_root: Path, stem: str, keep: str) -> None:
    """Drop embeddings for older revisions of one source."""
    if not cache_root.is_dir():
        return
    current = {f"embeddings-{stem}-{name}-{keep}.npz" for name in RETRIEVAL_FIELDS}
    for path in cache_root.glob(f"embeddings-{stem}-*.npz"):
        if path.name not in current:
            with contextlib.suppress(OSError):
                path.unlink()
