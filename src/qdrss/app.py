from __future__ import annotations

import asyncio
import contextlib
import hashlib
import json
import logging
import os
import re
import sys
import time
from collections.abc import AsyncIterator
from datetime import UTC, datetime
from pathlib import Path
from urllib.parse import urlencode

import numpy as np
import uvicorn
from starlette.applications import Starlette
from starlette.requests import Request
from starlette.responses import (
    FileResponse,
    JSONResponse,
    PlainTextResponse,
    Response,
    StreamingResponse,
)
from starlette.routing import Route

# Pages change with each deploy; without this browsers reuse a stale copy.
NO_CACHE = {"cache-control": "no-cache"}

from . import nlweb, querylog, usage
from .activity import Activity, iso
from .config import Config, load_config
from .converse import rewrite, summarize
from .decisions import DecisionRanker
from .feed import FeedRequest, Retriever, evaluate, evaluate_stream, render_rss
from .feeds import parse_date
from .hosted import Hosted
from .index import Index, build_index
from .ingest import load_sources, refresh, select_sources
from .providers import Embeddings, HashEmbeddings, KeywordRanker, OpenRouter, Ranker
from .rank import RankingError
from .usage import UsageLedger
from .vectors import VectorServiceError
from .verticals import BY_ID as VERTICALS
from .verticals import resolve as resolve_vertical

logger = logging.getLogger("qdrss")
# The search page: everything since 2000 (archive items are stamped with their publication date), newer
# items preferred with a half-life of 180 days, at most 20 results.
SEARCH_SINCE = datetime(2000, 1, 1, tzinfo=UTC)
SEARCH_FRESH = 180.0
SEARCH_LIMIT = 20
STATIC = Path(__file__).parent / "static"
DECISION_MODELS = nlweb.DECISION_MODELS
# Ranking models the search page offers, as on the NLWeb samples' page (askhub/static/app.js). "" is the
# service's default. Ids in DECISION_MODELS are decision models, called through the Decisions API.
RANKING_CHOICES = {
    "": "Auto",
    "typesafe/jev-1.14": "Jev 1.14",  # a decision model: scores only, so results show the item's description
    "google/gemini-decision-1": "Gemini Decision 1",  # likewise
    "openai/gpt-oss-20b": "GPT-OSS 20B",
    "openai/gpt-oss-120b": "GPT-OSS 120B",
    "google/gemma-4-26b-a4b-it": "Gemma 4 26B",
    "google/gemma-4-31b-it": "Gemma 4 31B",
    "google/gemini-2.5-flash-lite": "Gemini 2.5 Flash Lite",
    "google/gemini-3.1-flash-lite": "Gemini 3.1 Flash Lite",
    "google/gemini-3.5-flash-lite": "Gemini 3.5 Flash Lite",
    "google/gemini-3.8-flash": "Gemini 3.8 Flash",
}


class Services:
    """Two storage modes behind one interface.

    Cosmos (COSMOS_ENDPOINT set): items and vectors live in Cosmos DB, retrieval
    is a Cosmos vector query, and the process holds nothing. SQLite (default):
    items live in a local file and an in-memory matrix is rebuilt after every
    refresh. Tests and offline runs use SQLite.
    """

    def __init__(self, config: Config, embedder: Embeddings, ranker: Ranker, writer: Ranker | None = None):
        self.config = config
        self.embedder = embedder
        self.ranker = ranker
        # Rewrites follow-ups and writes the search page's summaries; None offline (follow-ups are
        # searched as typed and there is no summary).
        self.writer = writer
        self._rankers: dict[str, Ranker] = {}
        self.activity = Activity()
        self.sources = load_sources(config.sources_path)
        self.collections = {s.name: s.collection for s in self.sources}
        # Each search-page tab's scope (collections or sources), resolved once.
        self.verticals = {v.id: resolve_vertical(v, self.sources) for v in VERTICALS.values() if not v.askhub}
        # A shared store (Postgres on the VM, or Cosmos before it) is written by the ingest job and read by every
        # process; SQLite is local to this one (tests, offline runs).
        self.store_kind = "postgres" if config.pg_host else "cosmos" if config.cosmos_endpoint else "sqlite"
        self.shared = self.store_kind != "sqlite"
        if self.store_kind == "postgres":
            from .pgstore import PostgresStore

            self.store = PostgresStore(config.pg_host, config.pg_password)
        elif self.store_kind == "cosmos":
            from .cosmos import CosmosStore

            self.store = CosmosStore(config.cosmos_endpoint, config.cosmos_key, config.cosmos_database)
        else:
            from .store import Store

            self.store = Store(config.db_path)
        self.memory = None
        self.remote = None
        if self.shared and config.vector_url:
            from .vectors import RemoteRetriever

            self.remote = RemoteRetriever(config.vector_url, config.vector_token)
        elif self.shared and config.memory_collections and config.memory_snapshot:
            from .memory import MemoryRetriever

            self.memory = MemoryRetriever(self.store, self.collections, list(config.memory_collections),
                                          config.memory_snapshot)
        # Cooking and Wirecutter: askhub in this process (hosted.py), when its vectors' VM address is set.
        self.hosted: Hosted | None = None
        if config.askhub_vector_url:
            self.hosted = Hosted([v.askhub for v in VERTICALS.values() if v.askhub], config.data_dir / "askhub")
        self.index: Index = Index([], np.zeros((0, 1), dtype=np.float32))
        self.ready_at: datetime | None = None
        self.last_refresh: dict = {}
        self._task: asyncio.Task | None = None
        self._counts_cache: tuple[float, dict] | None = None
        # Rendered feeds, keyed by the request. A feed only changes when the
        # corpus does, so entries live until the next refresh completes (or the
        # TTL, whichever is first). Readers poll the same URL for months; without
        # this every poll pays a vector search and eight LLM calls.
        self._feeds: dict[tuple, tuple[float, bytes, str]] = {}
        self.feed_ttl = config.refresh_minutes * 60

    def ranker_for(self, model: str) -> Ranker | None:
        """The ranker for a model the search page chose: the default for "" or the default's own id, one
        OpenRouter client per other model, made on first use. None when no other model can be called
        (offline, no OpenRouter key)."""
        if not model or model == self.config.ranking_model:
            return self.ranker
        if not self.config.openrouter_key:
            return None
        if model not in self._rankers and model in DECISION_MODELS:
            self._rankers[model] = DecisionRanker(self.config.openrouter_key, model, self.config.openrouter_base_url)
        if model not in self._rankers:
            self._rankers[model] = OpenRouter(
                self.config.openrouter_key, self.config.openrouter_base_url, model, self.config.embedding_model,
                provider_sort=self.config.provider_sort or None,
            )
        return self._rankers[model]

    @property
    def retriever(self) -> Retriever:
        if self.remote:
            return self.remote
        if self.memory:
            return self.memory
        return self.store if self.shared else self.index  # type: ignore[return-value]

    async def setup(self) -> None:
        if self.hosted:
            self.hosted.start()
        await self.store.setup()
        if self.memory:
            await self.memory.load()
        if self.shared:
            self.ready_at = datetime.now(UTC)

    async def refresh_once(self) -> None:
        outcomes = await refresh(
            self.sources, self.store, concurrency=self.config.fetch_concurrency
        )
        self.last_refresh = {
            "at": datetime.now(UTC).isoformat(),
            "new_items": sum(o.new_items for o in outcomes),
            "errors": {o.source.name: o.error for o in outcomes if o.error},
        }
        try:
            if self.shared:
                await self.embed_missing()
            else:
                # Rebuild even when nothing changed: a failed previous build gets
                # retried for free, and the cost is bounded by the embedding cache.
                self.index = await build_index(self.store, self.embedder, self.collections)
            self.ready_at = datetime.now(UTC)
        except Exception:
            logger.exception("indexing failed; serving the previous state")
        self._counts_cache = None
        self._feeds.clear()

    async def embed_missing(self) -> int:
        """Shared-store mode: embed items that arrived without a vector, in batches."""
        total = 0
        while True:
            items = await self.store.items_without_embedding(100)
            if not items:
                return total
            vectors = await self.embedder.embed([item.text[:8000] for item in items])
            await self.store.save_embeddings(
                self.embedder.model,
                {item.id: vectors[i] for i, item in enumerate(items)},
                {item.id: item.collection for item in items},
            )
            total += len(items)
            logger.info("embedded %d items (%d so far)", len(items), total)

    async def collection_counts(self) -> dict[str, dict]:
        if self._counts_cache and time.monotonic() - self._counts_cache[0] < 60:
            return self._counts_cache[1]
        out: dict[str, dict] = {}
        for source in self.sources:
            out.setdefault(source.collection, {"sources": 0, "items": 0})["sources"] += 1
        if self.shared:
            for name, n in (await self.store.counts_by_collection(list(out))).items():
                out.setdefault(name, {"sources": 0, "items": 0})["items"] = n
        else:
            counts = self.store.counts_by_source()
            for source in self.sources:
                out[source.collection]["items"] += counts.get(source.name, 0)
        self._counts_cache = (time.monotonic(), out)
        return out

    async def refresh_in_subprocess(self) -> None:
        """Shared-store mode: run `qdrss-ingest` as a child process.

        Fetching and parsing 300 feeds (some 50 MB) and embedding what is new is
        CPU-bound work that would stall the event loop on a small instance; in
        its own interpreter it only competes for the core.
        """
        proc = await asyncio.create_subprocess_exec(
            sys.executable, "-c", "from qdrss.app import ingest_main; ingest_main()",
            stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.STDOUT,
        )
        summary: dict = {"at": None, "new_items": None, "errors": {}}
        assert proc.stdout is not None
        async for raw in proc.stdout:
            line = raw.decode(errors="replace").rstrip()
            if m := re.match(r"new items: (\d+); errors: (\d+)", line):
                summary["new_items"] = int(m.group(1))
            elif line.startswith("  ") and ": " in line and summary["new_items"] is not None:
                name, _, err = line.strip().partition(": ")
                summary["errors"][name] = err
            elif "HTTP Request" not in line:
                logger.info("ingest: %s", line[:200])
        code = await proc.wait()
        if code != 0:
            summary["errors"]["_process"] = f"exit {code}"
        summary["at"] = datetime.now(UTC).isoformat()  # completion, not start
        self.last_refresh = summary
        self._counts_cache = None
        self._feeds.clear()

    async def _loop(self) -> None:
        while True:
            try:
                if self.shared:
                    await self.refresh_in_subprocess()
                else:
                    await self.refresh_once()
            except Exception:
                logger.exception("refresh failed")
            await asyncio.sleep(self.config.refresh_minutes * 60)

    def start(self) -> None:
        if self.config.ingest_in_app or not self.shared:
            self._task = asyncio.create_task(self._loop())

    async def stop(self) -> None:
        if self._task:
            self._task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._task
        if self.remote:
            await self.remote.aclose()
        if self.hosted:
            await self.hosted.aclose()
        for ranker in self._rankers.values():
            if hasattr(ranker, "aclose"):
                await ranker.aclose()
        await self.store.close()


def build_services(config: Config | None = None) -> Services:
    config = config or load_config()
    if config.openrouter_key:
        provider = OpenRouter(
            config.openrouter_key,
            config.openrouter_base_url,
            config.ranking_model,
            config.embedding_model,
            provider_sort=config.provider_sort or None,
        )
        # A reasoning model spends its output budget thinking before it writes: askhub saw gpt-oss-120b
        # return empty summaries at 400 tokens, so the writer gets room.
        writer = OpenRouter(
            config.openrouter_key, config.openrouter_base_url, config.writing_model, config.embedding_model,
            provider_sort=config.provider_sort or None, max_tokens=4000, phase="writing",
        )
        return Services(config, provider, provider, writer)
    logger.warning("OPENROUTER_API_KEY not set: using hash embeddings and keyword ranking")
    return Services(config, HashEmbeddings(), KeywordRanker())


def _line(event: dict) -> bytes:
    return (json.dumps(event, ensure_ascii=False) + "\n").encode()


def _feed_response(request: Request, body: bytes, etag: str) -> Response:
    """The ETag is the list of item ids, so a reader whose copy is current gets a 304."""
    headers = {"ETag": etag, "Cache-Control": "public, max-age=900"}
    if request.headers.get("if-none-match") == etag:
        return Response(status_code=304, headers=headers)
    return Response(body, media_type="application/rss+xml; charset=utf-8", headers=headers)


def create_app(services: Services) -> Starlette:
    async def feed(request: Request) -> Response:
        started = time.monotonic()
        try:
            response = await _feed(request)
        except Exception as exc:
            services.activity.record(500, time.monotonic() - started, f"{type(exc).__name__}: {exc}")
            log_feed(request, 500, started, None, type(exc).__name__)
            raise
        body = bytes(response.body)
        error = body.decode("utf-8", "replace") if response.status_code >= 400 else None
        services.activity.record(response.status_code, time.monotonic() - started,
                                 error if response.status_code >= 500 else None)
        log_feed(request, response.status_code, started,
                 body.count(b"<item>") if response.status_code == 200 else None, error)
        return response

    def log_feed(request: Request, status: int, started: float, results: int | None, error: str | None) -> None:
        params = request.query_params
        querylog.emit({
            "endpoint": "/feed.xml", "query": params.get("q", "").strip(),
            "vertical": params.get("vertical") or None, "collection": params.get("collection") or None,
            "since": bool(params.get("since")), "status": status, "error": error, "results": results,
            "latency_ms": round((time.monotonic() - started) * 1000),
            "user_agent": request.headers.get("user-agent", ""),
        })

    async def _feed(request: Request) -> Response:
        params = request.query_params
        q = params.get("q", "").strip()
        if not q:
            return PlainTextResponse("q is required", status_code=400)
        since = None
        if raw := params.get("since", "").strip():
            since = parse_date(raw)
            if since is None:
                return PlainTextResponse("since must be RFC 2822 or ISO 8601", status_code=400)
        try:
            limit = int(params.get("limit", services.config.default_limit))
        except ValueError:
            return PlainTextResponse("limit must be an integer", status_code=400)
        limit = max(1, min(limit, services.config.max_limit))
        threshold = params.get("threshold", "relevant")
        if threshold not in ("relevant", "strong"):
            return PlainTextResponse("threshold must be relevant or strong", status_code=400)
        fresh = 0.0
        if raw := params.get("fresh", "").strip():
            try:
                fresh = 365.0 if raw.lower() in ("on", "true", "yes") else float(raw)
            except ValueError:
                return PlainTextResponse("fresh must be a number of days (the half-life of the boost), or on", status_code=400)
            if fresh != 0 and not 1 <= fresh <= 3650:
                return PlainTextResponse("fresh must be 0 (off) or between 1 and 3650 days", status_code=400)
        collection = params.get("collection", "").strip() or None
        if collection and collection not in services.collections.values():
            return PlainTextResponse(f"unknown collection {collection!r}", status_code=400)
        vertical = params.get("vertical", "").strip()
        if vertical and vertical not in services.verticals:
            return PlainTextResponse(f"unknown vertical {vertical!r} (or one with no feed)", status_code=400)
        if services.ready_at is None:
            return PlainTextResponse("index not built yet", status_code=503)

        scope = services.verticals.get(vertical)
        feed_request = FeedRequest(q, since, limit, threshold, collection, fresh,
                                   scope.sources if scope else None, scope.collections if scope else None, vertical)
        base = str(request.base_url).rstrip("/")
        # since= makes the request time-specific, so only default-window fetches
        # are cached (those are what feed readers send).
        key = (q, limit, threshold, collection, fresh, vertical, base) if since is None else None
        cached = services._feeds.get(key) if key else None
        if cached and time.monotonic() - cached[0] < services.feed_ttl:
            return _feed_response(request, cached[1], cached[2])
        try:
            matches = await evaluate(
                feed_request,
                services.retriever,
                services.embedder,
                services.ranker,
                candidate_count=services.config.candidate_count, max_candidates=services.config.max_candidates,
                batch_size=services.config.batch_size,
                window_days=services.config.window_days,
            )
        except RankingError as exc:
            logger.error("ranking failed: %s", exc)
            return PlainTextResponse(f"ranking unavailable: {exc}", status_code=503)
        except VectorServiceError as exc:
            return PlainTextResponse(f"vector search unavailable: {exc}", status_code=503)
        body = render_rss(feed_request, matches, base)
        etag = '"' + hashlib.sha256(",".join(m.item.id for m in matches).encode()).hexdigest()[:24] + '"'
        if key:
            if len(services._feeds) > 5000:
                services._feeds.pop(next(iter(services._feeds)))
            services._feeds[key] = (time.monotonic(), body, etag)
        return _feed_response(request, body, etag)

    async def search_api(request: Request) -> Response:
        """One turn of the NYT search page, streamed as JSON lines.

        A turn is a question in one tab, with the conversation's earlier queries (each already standing
        alone) and the last turn's result titles, so a follow-up is rewritten to stand alone before it is
        searched. Results come first, each as soon as it is found (time to the first result matters most):
        the archive and live feeds, newest first among the matches with a preference for recent items, or
        an NLWeb backend for Cooking and Wirecutter. The summary follows, then what this turn's model calls
        cost, by model and step. Lines: {"type": "query" | "results" | "summary" | "summary_error" |
        "error" | "usage", ...}; a "results" line with "partial": true is the list so far (it replaces the
        last), and a row with "pending": true is kept but not yet described. A search that fails before
        its first result is an HTTP error, never an empty list; one that fails after it ends in "error"."""
        started = time.monotonic()
        user_agent = request.headers.get("user-agent", "")
        turn: dict = {"endpoint": "/api/search", "query": "", "user_agent": user_agent}

        def log_turn(status, results: int | None = None, error: str | None = None, **more) -> None:
            querylog.emit({**turn, "status": status, "error": error, "results": results,
                           "latency_ms": round((time.monotonic() - started) * 1000), **more})

        def refuse(content: dict, status_code: int) -> JSONResponse:
            log_turn(status_code, error=content.get("error"))
            return JSONResponse(content, status_code=status_code)

        if request.method == "POST":
            try:
                body = await request.json()
            except ValueError:
                return refuse({"error": "the body must be JSON"}, 400)
            if not isinstance(body, dict):
                return refuse({"error": "the body must be a JSON object"}, 400)
        else:
            body = dict(request.query_params)
        q = str(body.get("q") or "").strip()[:300]
        vertical = VERTICALS.get(str(body.get("vertical") or "all"))
        turn.update(query=q, vertical=str(body.get("vertical") or "all"))
        if not q or vertical is None:
            return refuse({"error": "q and a known vertical are required"}, 400)
        previous = [str(x).strip()[:300] for x in body.get("previous_queries") or [] if str(x).strip()][-5:]
        shown = [str(x).strip()[:200] for x in body.get("previous_results") or [] if str(x).strip()][:10]
        want_summary = body.get("summarize", True) not in (False, "0", "false")
        model = str(body.get("ranking_model") or "")
        turn.update(turn=len(previous) + 1, ranking_model=model or None)
        if model not in RANKING_CHOICES:
            return refuse({"error": f"ranking_model must be one of: {', '.join(m for m in RANKING_CHOICES if m)}"}, 400)
        ranker = None if vertical.askhub else services.ranker_for(model)  # an NLWeb backend ranks for itself
        if ranker is None and not vertical.askhub:
            return refuse({"error": "only the default ranking model is available here"}, 400)
        ledger = UsageLedger()
        usage.CURRENT.set(ledger)
        notices: list[str] = []
        standalone = q
        if previous and services.writer is not None:
            try:
                standalone = await rewrite(services.writer, q, previous, shown)
            except Exception as exc:  # noqa: BLE001 - said on the page; the question is searched as typed
                logger.warning("follow-up rewrite failed: %s", type(exc).__name__)
                notices.append("This could not be read as a follow-up, so it was searched as typed.")
        key = ("api", standalone, vertical.id, want_summary, model)
        cached = services._feeds.get(key)
        hit = bool(cached and time.monotonic() - cached[0] < services.feed_ttl)
        turn.update(effective_query=standalone if standalone != q else None, cached=hit)
        if not hit and not vertical.askhub and services.ready_at is None:
            return refuse({"error": "index not built yet"}, 503)
        base = str(request.base_url).rstrip("/")
        feed = None if vertical.askhub else f"{base}/feed.xml?" + urlencode({"q": standalone, "vertical": vertical.id})

        def results(rows: list[dict], partial: bool = False) -> dict:
            line = {"type": "results", "results": rows, "feed": feed, "notices": notices, "cached": hit}
            return {**line, "partial": True} if partial else line

        async def found() -> AsyncIterator[dict]:
            """The results as they come, each a "results" line marked partial (the list so far, which replaces
            the last); then the full list, unmarked, and the summary. Raises before the first line on a failed
            search, so the turn is an HTTP error and never an empty list."""
            if hit:
                entry = cached[1]
                summary_error = ""
            elif vertical.askhub:
                mode = "summarize" if want_summary else "list"
                answer = None
                for _ in range(2):  # these backends now and then answer "no results" to a query they answer the next time
                    if services.hosted is None:
                        raise nlweb.NLWebError(f"{vertical.label} is not configured (QDRSS_ASKHUB_VECTOR_URL)")
                    events = services.hosted.events(vertical.askhub, nlweb.payload(
                        vertical.nlweb_site, standalone, SEARCH_LIMIT, mode, model))
                    async for kind, value in nlweb.parse(events, SEARCH_LIMIT, model):
                        if kind == "rows":
                            yield results(value, partial=True)
                        else:
                            answer = value
                    ledger.merge(answer.usage, vertical.label)
                    if answer.rows:
                        break
                entry = {"results": answer.rows, "feed": None, "summary": answer.summary}
                summary_error = ""
                if want_summary and answer.rows and not answer.summary:
                    summary_error = next((n for n in answer.notices if "unavailable" in n.lower()),
                                         f"{vertical.label} sent no summary")
            else:
                scope = services.verticals[vertical.id]
                feed_request = FeedRequest(standalone, SEARCH_SINCE, SEARCH_LIMIT, "relevant", None, SEARCH_FRESH,
                                           scope.sources, scope.collections, vertical.id)
                rows: list[dict] = []
                async for matches in evaluate_stream(
                    feed_request, services.retriever, services.embedder, ranker,
                    candidate_count=services.config.candidate_count, max_candidates=services.config.max_candidates,
                    batch_size=services.config.batch_size, window_days=services.config.window_days,
                ):
                    rows = [{
                        "title": m.item.title or m.item.url, "url": m.item.url, "summary": m.item.content[:400],
                        "why": m.why, "image": None,
                        "date": (m.item.published_at or m.item.ingested_at).isoformat(),
                        "source": m.item.source_title or m.item.source,
                    } for m in matches]
                    if rows:
                        yield results(rows, partial=True)
                # The feed a reader (or an email digest) follows: new matches as they are published. Feeds always
                # rank with the default model.
                entry = {"results": rows, "feed": feed, "summary": None}
                summary_error = ""
            if not hit and not summary_error:
                if len(services._feeds) > 5000:
                    services._feeds.pop(next(iter(services._feeds)))
                services._feeds[key] = (time.monotonic(), entry, "")
            yield results(entry["results"])
            if want_summary and entry["results"]:
                error = summary_error
                if not entry["summary"] and not error and not vertical.askhub:
                    if services.writer is None:
                        error = "no summary model is configured"
                    else:
                        try:
                            # Stored on the cached entry, so the same turn asked again is not summarized twice.
                            entry["summary"] = await summarize(services.writer, standalone, entry["results"])
                        except Exception as exc:  # noqa: BLE001 - reported on the page, not cached
                            logger.warning("summary failed: %s: %s", type(exc).__name__, str(exc)[:120])
                            error = "the summary model failed"
                if entry["summary"]:
                    yield {"type": "summary", "value": entry["summary"]}
                elif error:
                    yield {"type": "summary_error", "message": error}

        def failure(exc: Exception) -> tuple[str, int]:
            if isinstance(exc, RankingError):
                logger.error("ranking failed: %s", exc)
                return "ranking unavailable", 503
            if isinstance(exc, VectorServiceError):
                return "vector search unavailable", 503
            logger.error("nlweb %s: %s", vertical.id, exc)
            return f"{vertical.label} search is unavailable", 503

        events = found()
        try:
            first = await anext(events)  # the first result (or the whole answer), before the 200 is sent
        except (RankingError, VectorServiceError, nlweb.NLWebError) as exc:
            message, status = failure(exc)
            return refuse({"error": message}, status)

        async def lines():
            usage.CURRENT.set(ledger)
            results: int | None = None
            status, error = "ok", None
            try:
                yield _line({"type": "query", "value": standalone, "rewritten": standalone != q})
                yield _line(first)
                results = len(first.get("results") or [])
                try:
                    async for event in events:
                        if event.get("type") == "results":
                            results = len(event.get("results") or [])
                        elif event.get("type") == "summary_error":
                            error = event.get("message")
                        yield _line(event)
                except (RankingError, VectorServiceError, nlweb.NLWebError) as exc:  # after results were shown
                    status, error = "error", failure(exc)[0]
                    yield _line({"type": "error", "message": error})
                    return
                yield _line({"type": "usage", "cached": hit, **ledger.wire()})
            except (asyncio.CancelledError, GeneratorExit):
                status = "cancelled"  # the reader left before the turn finished
                raise
            except Exception as exc:
                status, error = "error", type(exc).__name__
                raise
            finally:
                spent = ledger.wire()
                log_turn(status, results, error, cost_usd=spent["cost"], model_calls=spent["calls"])

        return StreamingResponse(lines(), media_type="application/x-ndjson", headers=NO_CACHE)

    async def nyt_search(_: Request) -> Response:
        return FileResponse(STATIC / "nyt-search.html", headers=NO_CACHE)

    async def ranking_models(_: Request) -> Response:
        return JSONResponse({"default": services.config.ranking_model or "",
                             "models": [{"id": k, "label": v} for k, v in RANKING_CHOICES.items()]})

    async def verticals(_: Request) -> Response:
        return JSONResponse([{"id": v.id, "label": v.label, "examples": list(v.examples),
                              "feed": not v.askhub} for v in VERTICALS.values()])

    async def home(_: Request) -> Response:
        return FileResponse(STATIC / "index.html", headers=NO_CACHE)

    async def nyt(_: Request) -> Response:
        # A landing page for the NYT podcasts collection alone: one question box,
        # archive search on, results previewed and the topic feed's URL to copy.
        return FileResponse(STATIC / "nyt.html", headers=NO_CACHE)

    async def npr(_: Request) -> Response:
        # The same landing page for the NPR and public-radio podcasts collection.
        return FileResponse(STATIC / "npr.html", headers=NO_CACHE)

    async def tal_page(_: Request) -> Response:
        # This American Life by section: results are passages, each with the second it starts at.
        return FileResponse(STATIC / "tal.html", headers=NO_CACHE)

    async def collections(_: Request) -> Response:
        return JSONResponse(await services.collection_counts())

    async def health(_: Request) -> Response:
        sources = await services.store.all_sources()
        failing = [s["name"] for s in sources if s.get("last_error")]
        fetched = [s["last_fetch"] for s in sources if s.get("last_fetch")]
        problems = (["index not built"] if not services.ready_at else []) + (
            [f"{len(failing)} of {len(sources)} sources failed their last fetch"] if sources and len(failing) / len(sources) > 0.2 else [])
        return JSONResponse(headers={"access-control-allow-origin": "*"}, content=
            {
                "status": "ok" if not problems else "degraded",
                **({"degraded_because": problems} if problems else {}),
                "service": "qdrss",
                "revision": os.getenv("K_REVISION") or None,
                "started_at": iso(services.activity.started),
                "uptime_seconds": int(time.time() - services.activity.started),
                "feed_requests": services.activity.snapshot(),
                "sources_summary": {"total": len(sources), "failing": len(failing), "failing_names": failing[:20],
                                    "last_fetch_at": max(fetched) if fetched else None},
                "note": "Request counters are for this instance since it started; a restart resets them and several instances each count their own.",
                "store": services.store_kind,
                "ranking_providers": getattr(services.ranker, "health", None) and services.ranker.health.report() or {},
                "in_memory": services.memory.status if services.memory else {},
                "vector_service": services.remote.status if services.remote else None,
                "askhub": services.hosted.status() if services.hosted else None,
                "items": await services.store.count(),
                "index_built_at": services.ready_at.isoformat() if services.ready_at else None,
                "last_refresh": services.last_refresh,
                "sources": sources,
            }
        )

    @contextlib.asynccontextmanager
    async def lifespan(_: Starlette):
        await services.setup()
        services.start()
        try:
            yield
        finally:
            await services.stop()

    return Starlette(
        routes=[
            Route("/", home),
            Route("/nyt", nyt),
            Route("/nyt/search", nyt_search),
            Route("/api/search", search_api, methods=["GET", "POST"]),
            Route("/api/verticals", verticals),
            Route("/api/models", ranking_models),
            Route("/npr", npr),
            Route("/tal", tal_page),
            Route("/feed.xml", feed),
            Route("/collections", collections),
            Route("/health", health),
            Route("/healthz", health),
        ],
        lifespan=lifespan,
    )


def configure_logging() -> None:
    """INFO for this app. The Azure SDK logs every HTTP request and response with all of its headers at INFO, once per Cosmos call: ingest
    wrote 30-60 GiB of it a day to Cloud Logging, so its loggers are held at WARNING."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")
    logging.getLogger("azure").setLevel(logging.WARNING)


def main() -> None:
    configure_logging()
    services = build_services()
    uvicorn.run(create_app(services), host=services.config.host, port=services.config.port)


def ingest_main() -> None:
    """Fetch every source once, store new items, embed them, exit.

    Runs anywhere with the same .env as the web app; with Cosmos this is how the
    corpus is updated independently of serving.
    """
    configure_logging()

    async def run() -> None:
        services = build_services()
        # QDRSS_INGEST_COLLECTIONS=nytnews fetches just those collections (an hourly job for news).
        services.sources = select_sources(services.sources, os.environ.get("QDRSS_INGEST_COLLECTIONS", ""))
        await services.setup()
        try:
            await services.refresh_once()
            r = services.last_refresh
            print(f"new items: {r['new_items']}; errors: {len(r['errors'])}")
            for name, err in r["errors"].items():
                print(f"  {name}: {err}")
        finally:
            await services.store.close()

    asyncio.run(run())
