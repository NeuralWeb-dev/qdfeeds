"""One stateless fetch: q + since -> the newest qualifying items, as RSS 2.0."""

from __future__ import annotations

import json
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from email.utils import format_datetime
from typing import Protocol
from urllib.parse import urlencode

import numpy as np

from .models import Candidate, Match
from .providers import Embeddings, Ranker
from .rank import classify


class Retriever(Protocol):
    async def search(
        self, vector: np.ndarray, since: datetime | None, collection: str | None, limit: int,
        sources: frozenset[str] | None = None, collections: frozenset[str] | None = None,
    ) -> list[Candidate]: ...

NS = "https://qdrss.dev/ns"

# Feed readers fetch the same URL for months. Cache the query vector per
# embedding model so a fetch costs one embedding call ever, not one per poll.
_QUERY_VECTORS: dict[tuple[str, str], np.ndarray] = {}
_QUERY_CACHE_MAX = 5000


async def _query_vector(embedder: Embeddings, q: str) -> np.ndarray:
    key = (embedder.model, q)
    vector = _QUERY_VECTORS.get(key)
    if vector is None:
        vector = (await embedder.embed([q]))[0]
        if len(_QUERY_VECTORS) >= _QUERY_CACHE_MAX:
            _QUERY_VECTORS.pop(next(iter(_QUERY_VECTORS)))
        _QUERY_VECTORS[key] = vector
    return vector


@dataclass(frozen=True)
class FeedRequest:
    q: str
    since: datetime | None
    limit: int
    threshold: str  # "relevant" (includes strong) | "strong"
    collection: str | None = None  # manifest section; None = every source
    fresh: float = 0.0  # freshness boost: half-life in days of the preference for recent items; 0 = none
    sources: frozenset[str] | None = None  # only these sources (a vertical, see verticals.py); None = no limit
    collections: frozenset[str] | None = None  # only these collections (a vertical); None = no limit
    vertical: str = ""  # the vertical the sources came from, for the feed's own link


async def evaluate(
    request: FeedRequest,
    retriever: Retriever,
    embedder: Embeddings,
    ranker: Ranker,
    *,
    candidate_count: int,
    batch_size: int,
    window_days: int,
    max_candidates: int | None = None,
    now: datetime | None = None,
) -> list[Match]:
    """The ranking model judges the closest `candidate_count` candidates; while fewer than `limit` of them
    qualify it judges the next `candidate_count`, up to `max_candidates` in all. A fixed cut would drop
    the item that answers the question whenever a bigger story fills the closest places: one vector per
    article captures what it is mostly about, so a question naming a second person ("Bessent's role in
    the Huang case") sits nearer to fifty articles about the first than to the one about both."""
    # `now` is injectable so the default window can be tested against fixed
    # timestamps; reading the clock here made those tests expire a week after
    # their fixture date.
    now = now or datetime.now(UTC)
    since = request.since or now - timedelta(days=window_days)
    vector = await _query_vector(embedder, request.q)
    ceiling = max(max_candidates or candidate_count, candidate_count)
    # One retrieval for every page: asking again for a wider pool each round costs more each time
    # (TOP 600 took a minute in Cosmos). A freshness boost orders a pool three times the first page.
    pool = max(ceiling, candidate_count * FRESH_POOL) if request.fresh else ceiling
    candidates = one_per_url(await retriever.search(vector, since, request.collection, pool, request.sources, request.collections))
    if request.fresh:
        candidates = by_blend(candidates, request.fresh, now)
    candidates = candidates[:ceiling]
    matches: list[Match] = []
    for start in range(0, len(candidates), candidate_count):
        judged = await classify(request.q, candidates[start : start + candidate_count], ranker, batch_size)
        matches += [m for m in judged if request.threshold != "strong" or m.category == "strong"]
        if len(matches) >= request.limit:
            break
    # Newest first by publication time, the date each item shows as its pubDate;
    # ingestion time stands in for an item without one, and the id breaks ties so
    # equal timestamps order the same way on every fetch. (Ordering by ingestion
    # time put items out of date order whenever a source was ingested in batches.)
    matches.sort(key=lambda m: (-published(m.item).timestamp(), m.item.id))
    return matches[: request.limit]


def one_per_url(candidates: list[Candidate]) -> list[Candidate]:
    """The closest candidate for each URL. A publisher lists one article in several of its feeds (a
    section, its home page, its most-viewed list), each a separate source here, so without this one story
    fills several places in a feed. Exact URLs: This American Life passages share a page and differ in
    their #t= anchor, and they are different items. An item without a URL is kept."""
    seen: set[str] = set()
    out = []
    for candidate in candidates:
        url = candidate.item.url
        if url and url in seen:
            continue
        seen.add(url)
        out.append(candidate)
    return out


FRESH_POOL = 3          # a freshness boost chooses its candidates from this many times the usual number
FRESH_WEIGHT = 0.5      # the most recent item gains half the pool's rank range over an old one equally close


def by_blend(candidates: list[Candidate], half_life_days: float, now: datetime) -> list[Candidate]:
    """Candidates by their blend of closeness and recency, best first. Candidates arrive closest first, so closeness is the rank
    (1 for the closest, 0 for the last): the retrievers score on different scales, and a rank does not depend on the scale. Recency is
    0.5 ** (age / half-life), 1 for an item published now. A boost only changes the order in which the ranking model sees items; it
    still judges each one, so a recent item that does not fit the question is not returned."""
    n = len(candidates)

    def blend(rank_and_candidate):
        rank, c = rank_and_candidate
        age_days = max(0.0, (now - published(c.item)).total_seconds() / 86400)
        return (1 - rank / n) + FRESH_WEIGHT * 0.5 ** (age_days / half_life_days)

    return [c for _, c in sorted(enumerate(candidates), key=blend, reverse=True)]


def freshen(candidates: list[Candidate], half_life_days: float, keep: int, now: datetime) -> list[Candidate]:
    """The `keep` candidates with the best blend of closeness and recency (see by_blend), closest first."""
    if len(candidates) <= keep:
        return candidates
    best = set(map(id, by_blend(candidates, half_life_days, now)[:keep]))
    return [c for c in candidates if id(c) in best]


def published(item) -> datetime:
    return item.published_at or item.ingested_at


def render_rss(request: FeedRequest, matches: list[Match], base_url: str) -> bytes:
    ET.register_namespace("qd", NS)
    rss = ET.Element("rss", version="2.0")
    channel = ET.SubElement(rss, "channel")
    ET.SubElement(channel, "title").text = f"rssnlweb: {request.q}"
    params = {"q": request.q}
    if request.collection:
        params["collection"] = request.collection
    if request.vertical:
        params["vertical"] = request.vertical
    if request.fresh:
        params["fresh"] = f"{request.fresh:g}"
    if request.since:
        params["since"] = request.since.isoformat()
    ET.SubElement(channel, "link").text = f"{base_url}/feed.xml?{urlencode(params)}"
    ET.SubElement(channel, "description").text = (
        f"Items matching \"{request.q}\" at or above {request.threshold}"
    )
    ET.SubElement(channel, "lastBuildDate").text = format_datetime(datetime.now(UTC))
    for match in matches:
        item = match.item
        node = ET.SubElement(channel, "item")
        ET.SubElement(node, "title").text = item.title or item.url
        if item.url:
            ET.SubElement(node, "link").text = item.url
        ET.SubElement(node, "guid", isPermaLink="false").text = item.id
        ET.SubElement(node, "description").text = _description(match)
        if item.published_at:
            ET.SubElement(node, "pubDate").text = format_datetime(item.published_at)
        ET.SubElement(node, "source", url=item.url).text = item.source_title or item.source
        ET.SubElement(node, f"{{{NS}}}category").text = match.category
        ET.SubElement(node, f"{{{NS}}}ingestedAt").text = item.ingested_at.isoformat()
        ET.SubElement(node, f"{{{NS}}}source").text = item.source
        _section(node, item.extra)
    return ET.tostring(rss, encoding="utf-8", xml_declaration=True)


def _section(node: ET.Element, extra: str) -> None:
    """An item that is one section of a longer recording says where: episode, act, and the offset in seconds."""
    if not extra:
        return
    try:
        info = json.loads(extra)
    except ValueError:
        return
    for key in ("episode", "act", "actId", "start", "end", "transcript", "audio"):
        if info.get(key) not in (None, ""):
            text = f"{info[key]:.1f}" if isinstance(info[key], float) else str(info[key])
            ET.SubElement(node, f"{{{NS}}}{'offsetSeconds' if key == 'start' else 'endSeconds' if key == 'end' else key}").text = text


def _description(match: Match) -> str:
    why = f"[{match.category}] {match.why}".strip()
    body = match.item.content[:1000]
    return f"{why}\n\n{body}" if body else why
