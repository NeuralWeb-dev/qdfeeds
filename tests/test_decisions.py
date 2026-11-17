"""Jev (a decision model) as the ranker: scores only, levels mapped to strong / relevant / excluded."""

import httpx
import pytest

from qdrss import usage
from qdrss.decisions import DecisionRanker
from qdrss.models import Candidate
from qdrss.rank import RankingError, classify

from .conftest import make_item


def ranker(handler):
    r = DecisionRanker("k", "typesafe/jev-1.14", "https://openrouter.test/api/v1")
    r.client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    return r


def candidates(n):
    return [Candidate(make_item(str(i), f"title {i}", f"description {i}"), "vector", 1.0) for i in range(n)]


async def test_levels_become_strong_relevant_or_excluded_with_no_reason():
    sent = {}

    def handler(request):
        sent["url"] = str(request.url)
        sent["body"] = request.read()
        return httpx.Response(200, json={"model": "typesafe/jev-1.14", "usage": {"prompt_tokens": 40, "cost": 0.001},
                                         "answers": {"r0": {"type": "score", "score": 3.4},
                                                     "r1": {"type": "score", "score": 1.0},
                                                     "r2": {"type": "score", "score": 2.2}}})

    ledger = usage.UsageLedger()
    token = usage.CURRENT.set(ledger)
    try:
        matches = await classify("q", candidates(3), ranker(handler), batch_size=5)
    finally:
        usage.CURRENT.reset(token)
    assert sent["url"] == "https://openrouter.test/api/alpha/decisions"
    assert b'"description":"description 1"' in sent["body"]
    assert [(m.item.title, m.category, m.why) for m in matches] == [("title 0", "strong", ""), ("title 2", "relevant", "")]
    assert ledger.wire()["models"][0]["phases"] == {"ranking": 1} and ledger.wire()["cost"] == 0.001


async def test_a_page_too_large_for_the_model_is_scored_in_halves():
    sizes = []

    def handler(request):
        import json
        records = json.loads(request.read())["state"]["records"]
        sizes.append(len(records))
        if len(records) > 2:
            return httpx.Response(400, text='{"error": "max_tokens_exceeded"}')
        return httpx.Response(200, json={"answers": {r["id"]: {"type": "score", "score": 4} for r in records}})

    matches = await classify("q", candidates(4), ranker(handler))
    assert len(matches) == 4 and sizes[0] == 4 and sorted(sizes[1:]) == [2, 2]


async def test_no_complete_scores_is_a_failed_ranking(monkeypatch):
    monkeypatch.setattr("qdrss.decisions.asyncio.sleep", lambda s: _done())
    r = ranker(lambda request: httpx.Response(200, json={"answers": {"r0": {"type": "score", "score": 3}}}))
    with pytest.raises(RankingError, match="no complete scores"):
        await classify("q", candidates(2), r)


async def _done():
    return None
