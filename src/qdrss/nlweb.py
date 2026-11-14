"""Ask an NLWeb backend (/ask, server-sent events) and return its results as plain dicts.

Used for the verticals whose content is not RSS (NYT Cooking recipes, Wirecutter
reviews). An NLWeb answer streams `candidate` events as retrieval finds items and
`result` events once its ranker has judged them; the ranked results are what we want,
and the candidates stand in when a backend sends no `result` before it ends.
"""

from __future__ import annotations

import json

import httpx

TIMEOUT = httpx.Timeout(90.0, connect=10.0)


class NLWebError(RuntimeError):
    pass


def _row(item: dict) -> dict | None:
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
        "summary": (item.get("description") or schema.get("description") or "")[:400],
        "why": item.get("explanation") or "",
        "strong": item.get("relevance") == "strong" or (item.get("score") or 0) >= 80,
        "image": image if isinstance(image, str) and image.startswith("https://") else None,
        "date": schema.get("datePublished") or None,
        "source": schema.get("@type") or "",
    }


async def ask(url: str, site: str, query: str, limit: int, client: httpx.AsyncClient | None = None) -> list[dict]:
    payload = {"query": query, "site": site, "mode": "list", "max_results": limit}
    own = client is None
    client = client or httpx.AsyncClient(timeout=TIMEOUT)
    results: list[dict] = []
    candidates: list[dict] = []
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
                    (results if kind == "result" else candidates).extend(content)
                elif kind in ("end-nlweb-response", "complete"):
                    break
    except httpx.HTTPError as exc:
        raise NLWebError(f"{url}: {type(exc).__name__}") from exc
    finally:
        if own:
            await client.aclose()
    seen: set[str] = set()
    out = []
    for item in results or [c for c in candidates if c.get("relevance") in ("strong", "relevant")]:
        row = _row(item)
        if row and row["url"] not in seen:
            seen.add(row["url"])
            out.append(row)
    return out[:limit]
