"""The VM's vector service and the web app's client for it, with Cosmos as the fallback."""

from datetime import UTC, datetime

import httpx
import numpy as np
from starlette.testclient import TestClient

from qdrss.models import Candidate
from qdrss.vectors import RemoteRetriever, create_vector_app, request_body

from .conftest import make_item


class Memory:
    def __init__(self):
        self.asked = None

    async def search(self, vector, since, collection, limit, sources=None, collections=None):
        self.asked = (vector.tolist(), since, collection, limit, sources, collections)
        return [Candidate(make_item("1", "rust release", "notes"), "vector", 0.9)]


class Cosmos:
    def __init__(self):
        self.searched = 0

    async def search(self, *args):
        self.searched += 1
        return [Candidate(make_item("2", "from cosmos"), "vector", 0.5)]


def test_the_service_answers_a_search_with_its_items_and_refuses_without_the_token():
    memory = Memory()
    client = TestClient(create_vector_app(memory, "secret", lambda: {"tal": "ok"}))
    body = request_body(np.asarray([0.5, 0.25], dtype=np.float32), datetime(2000, 1, 1, tzinfo=UTC), None, 7,
                        frozenset({"b", "a"}), frozenset({"nytnews"}))
    assert client.post("/search", json=body).status_code == 401
    assert client.post("/search", json=body, headers={"authorization": "Bearer nope"}).status_code == 401
    r = client.post("/search", json=body, headers={"authorization": "Bearer secret"})
    assert r.status_code == 200
    [row] = r.json()["candidates"]
    assert row["item"]["title"] == "rust release" and row["score"] == 0.9
    assert memory.asked == ([0.5, 0.25], datetime(2000, 1, 1, tzinfo=UTC), None, 7, frozenset({"a", "b"}),
                            frozenset({"nytnews"}))


async def test_the_client_returns_the_services_candidates_and_falls_back_to_cosmos_when_it_fails():
    app = create_vector_app(Memory(), "secret", dict)
    cosmos = Cosmos()
    r = RemoteRetriever("http://vm.test", "secret", cosmos)
    r.client = httpx.AsyncClient(transport=httpx.ASGITransport(app=app), headers={"authorization": "Bearer secret"})
    [c] = await r.search(np.zeros(2), None, None, 5, None, frozenset({"nytnews"}))
    assert c.item.title == "rust release" and c.item.published_at.tzinfo and cosmos.searched == 0
    r.client = httpx.AsyncClient(transport=httpx.ASGITransport(app=app), headers={"authorization": "Bearer wrong"})
    [c] = await r.search(np.zeros(2), None, None, 5)
    assert c.item.title == "from cosmos" and cosmos.searched == 1 and r.status["fallbacks"] == 1
    assert "401" in r.status["last_failure"]
