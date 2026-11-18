"""Ask an NLWeb backend (/ask, server-sent events) and return its results as plain dicts.

Used for the verticals whose content is not RSS (NYT Cooking recipes, Wirecutter
reviews). An NLWeb answer streams `candidate` events as its ranker judges items, then
`result` events as its describer writes about the kept ones; the described results are
what we want, the kept candidates show (as pending) until then, and stand in when a
backend sends no `result` before it ends. In
summarize mode an `nlws` event carries the summary, and a `usage` event what the
backend's own model calls cost.
"""

from __future__ import annotations

import json
import re
from collections.abc import AsyncIterator
from dataclasses import dataclass, field

import httpx

TIMEOUT = httpx.Timeout(90.0, connect=10.0)
# Ranking models that only score (decisions.py); results they rank carry nothing written about them.
DECISION_MODELS = frozenset({"typesafe/jev-1.14", "google/gemini-decision-1"})


class NLWebError(RuntimeError):
    pass


def _row(item: dict, own_description: bool = False) -> dict | None:
    """One result for the page. With own_description (a decision model ranked it, so the backend wrote
    nothing about it) the snippet is the record's own schema.org description. A recipe without one gets
    a line made from the fields the model judged it on (`recipe_line`), instead of the flattened field
    dump the backend sends ("... author X total time PT2H ...")."""
    schema = item.get("schema_object") or {}
    if isinstance(schema, list):
        schema = schema[0] if schema else {}
    url = item.get("url") or schema.get("url")
    title = item.get("name") or schema.get("name")
    if not url or not title:
        return None
    image = schema.get("image")
    if isinstance(image, list):
        image = image[0] if image else None
    if isinstance(image, dict):
        image = image.get("url")
    return {
        "title": title,
        "url": url,
        "summary": ((schema.get("description") or recipe_line(schema) if own_description
                     else item.get("description") or schema.get("description")) or "")[:400],
        "why": item.get("explanation") or "",
        "image": image if isinstance(image, str) and image.startswith("https://") else None,
        "date": schema.get("datePublished") or None,
        "source": schema.get("@type") or "",
    }


def _duration(value: object) -> str:
    """ISO 8601 "PT4H30M" -> "4 hr 30 min"; anything else -> ""."""
    match = re.fullmatch(r"P(?:(\d+)D)?T?(?:(\d+)H)?(?:(\d+)M)?", str(value or ""))
    if not match or not any(match.groups()):
        return ""
    days, hours, minutes = (int(g or 0) for g in match.groups())
    hours += days * 24
    return " ".join(p for p in (f"{hours} hr" if hours else "", f"{minutes} min" if minutes else "") if p)


def _text(value: object) -> str:
    if isinstance(value, list):
        value = value[0] if value else ""
    if isinstance(value, dict):
        value = value.get("name", "")
    return str(value or "").strip()


def recipe_line(schema: dict) -> str:
    """A description for a recipe that has none, from its own fields and nothing else:
    "Italian dinner by Molly O'Neill. 4 hr 30 min · 6 servings · rated 4/5 by 284 cooks. Keywords: ..."."""
    if "Recipe" not in str(schema.get("@type", "")):
        return ""
    cuisine = _text(schema.get("recipeCuisine")).capitalize()
    category = _text(schema.get("recipeCategory")).split(",")[0].strip()
    author = _text(schema.get("author"))
    kind = " ".join(p for p in (cuisine, category or "recipe") if p)
    first = f"{kind[:1].upper()}{kind[1:]}{f' by {author}' if author else ''}." if cuisine or category or author else ""
    rating = schema.get("aggregateRating") or {}
    value, count = rating.get("ratingValue"), rating.get("ratingCount")
    facts = [_duration(schema.get("totalTime")), _text(schema.get("recipeYield")),
             f"rated {value}/5 by {int(count):,} cooks" if value and str(count or "").isdigit() else ""]
    second = " · ".join(f for f in facts if f)
    keywords = _text(schema.get("keywords")) if not isinstance(schema.get("keywords"), list) \
        else ", ".join(map(str, schema["keywords"]))
    parts = [first, f"{second[:1].upper()}{second[1:]}." if second else "", f"Keywords: {keywords}." if keywords else ""]
    return " ".join(p for p in parts if p)


@dataclass
class Answer:
    rows: list[dict]
    summary: str | None = None
    usage: dict = field(default_factory=dict)
    notices: list[str] = field(default_factory=list)


async def stream(url: str, site: str, query: str, limit: int, client: httpx.AsyncClient | None = None,
                 mode: str = "list", ranking_model: str = "") -> AsyncIterator[tuple[str, object]]:
    """The backend's answer as it comes: ("rows", the list so far) each time it grows, then ("answer", the
    Answer). The backend first sends the candidates its ranker kept, one by one, then the described results
    in rank order, then the summary. Until a result is described, a kept candidate shows as pending with the
    record's own description (the ranker's unchecked words are not shown); a described result replaces it."""
    payload = {"query": query, "site": site, "mode": mode, "max_results": limit}
    if ranking_model:
        payload["ranking_model"] = ranking_model
    own = client is None
    client = client or httpx.AsyncClient(timeout=TIMEOUT)
    results: list[dict] = []
    pending: list[dict] = []  # kept candidates, not yet described
    candidates: list[dict] = []
    answer = Answer([])
    try:
        async with client.stream("POST", url, json=payload, headers={"accept": "text/event-stream"}) as response:
            if response.status_code != 200:
                raise NLWebError(f"{url}: HTTP {response.status_code}")
            async for line in response.aiter_lines():
                if not line.startswith("data:"):
                    continue
                try:
                    event = json.loads(line[5:])
                except ValueError:
                    continue
                kind = event.get("message_type")
                content = event.get("content")
                if kind in ("result", "candidate") and isinstance(content, list):
                    if kind == "result":
                        results.extend(_rows(content, ranking_model in DECISION_MODELS))
                    else:
                        candidates.extend(content)
                        pending.extend(_rows([c for c in content if c.get("relevance") in KEPT], True))
                    yield "rows", _merged(results, [{**r, "pending": True} for r in pending], limit)
                elif kind == "nlws" and isinstance(content, dict) and isinstance(content.get("answer"), str):
                    answer.summary = content["answer"].strip() or None
                elif kind == "usage" and isinstance(content, dict):
                    answer.usage = content
                elif kind == "intermediate_message" and isinstance(content, str) and content.strip():
                    answer.notices.append(content.strip())
                elif kind in ("end-nlweb-response", "complete"):
                    break
    except httpx.HTTPError as exc:
        raise NLWebError(f"{url}: {type(exc).__name__}") from exc
    finally:
        if own:
            await client.aclose()
    own_description = ranking_model in DECISION_MODELS
    answer.rows = _merged(results or _rows([c for c in candidates if c.get("relevance") in KEPT], own_description),
                          [], limit)
    yield "answer", answer


async def ask(url: str, site: str, query: str, limit: int, client: httpx.AsyncClient | None = None,
              mode: str = "list", ranking_model: str = "") -> Answer:
    async for kind, value in stream(url, site, query, limit, client, mode, ranking_model):
        if kind == "answer":
            return value
    raise AssertionError("unreachable")


KEPT = ("strong", "relevant")


def _rows(items: list[dict], own_description: bool) -> list[dict]:
    return [row for row in (_row(item, own_description=own_description) for item in items) if row]


def _merged(first: list[dict], then: list[dict], limit: int) -> list[dict]:
    """The rows of `first`, then those of `then`, one per URL, at most `limit`."""
    seen: set[str] = set()
    out = []
    for row in first + then:
        if row["url"] not in seen:
            seen.add(row["url"])
            out.append(row)
    return out[:limit]
