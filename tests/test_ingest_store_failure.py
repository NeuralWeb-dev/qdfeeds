"""A source whose writes the store keeps refusing fails alone; the rest of the crawl goes on."""

import asyncio

import httpx

from qdrss.ingest import Source, refresh
from qdrss.store import StoreUnavailable

RSS = b"""<?xml version="1.0"?><rss version="2.0"><channel><title>t</title>
<item><title>One</title><link>https://example.com/{n}</link><guid>https://example.com/{n}</guid>
<pubDate>Sat, 14 Nov 2026 10:00:00 GMT</pubDate></item></channel></rss>"""


class FlakyStore:
    """Refuses every write for the source named `bad`."""

    def __init__(self):
        self.stored: dict[str, int] = {}
        self.remembered: set[str] = set()

    async def source_state(self, name):
        return {}

    async def remember_source(self, name, url, **kwargs):
        self.remembered.add(name)

    async def insert_missing(self, items):
        if any(i.source == "bad" for i in items):
            raise StoreUnavailable("Cosmos kept failing for 40 attempts")
        for i in items:
            self.stored[i.source] = self.stored.get(i.source, 0) + 1
        return len(items)


def test_a_store_failure_fails_only_its_source():
    def handler(request):
        return httpx.Response(200, content=RSS.replace(b"{n}", request.url.path.encode()))

    sources = [Source(name=n, url=f"https://feeds.example.com/{n}") for n in ("good", "bad", "also_good")]
    store = FlakyStore()

    async def run():
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            return await refresh(sources, store, client=client)

    outcomes = {o.source.name: o for o in asyncio.run(run())}
    assert store.stored == {"good": 1, "also_good": 1}
    assert outcomes["bad"].error and outcomes["good"].error is None
    # Not remembered: the next run fetches it in full again instead of getting a 304.
    assert "bad" not in store.remembered
