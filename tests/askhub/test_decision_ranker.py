"""A decision model scores records and writes nothing: whatever it ranks is shown with the
record's own text. Chat rerankers keep writing their own descriptions."""

import json
from dataclasses import replace

import httpx
import pytest

from askhub.catalog import MemoryCatalog
from askhub.config import Config
from askhub.decision_models import (
    ATTEMPTS,
    SCORE_LEVELS,
    DecisionRanker,
    DecisionRankingError,
    DecisionVerdict,
    OpenRouterDecisionRanker,
    is_decision_ranker,
    item_description,
)
from askhub.models import Candidate, Document, SearchRequest
from askhub.pipeline import Services, _retrieve_and_rank
from askhub.providers import HashEmbeddings
from askhub.ranking_models import RANKING_MODELS
from askhub.usage import UsageLedger


def document(n: int, schema: dict, text: str = "item text") -> Document:
    return Document(f"d{n}", f"https://example.com/{n}", schema.get("name", f"Item {n}"),
                    "site", text, schema)


def candidate(doc: Document) -> Candidate:
    return Candidate(doc, 0.8, None, 0.8, "vector")


CATALOG = [
    document(0, {"name": "Food Bank Finder", "description": "Find a food bank near you.  Exact."}),
    document(1, {"name": "Bank Ledger", "description": "Accounting for a bank."}),
    document(2, {"name": "Paper", "@type": "Article", "abstract": "The paper's own abstract."}),
    document(3, {"name": "Bare"}, text="  Only the item text.  "),
]


def ranker_with(levels, calls=None, retries_before=0):
    """A real adapter whose HTTP layer answers with `levels`, after `retries_before` bad replies."""
    sent = []

    def respond(request: httpx.Request) -> httpx.Response:
        sent.append(json.loads(request.content))
        if len(sent) <= retries_before:
            return httpx.Response(200, json={"answers": {"r0": {"type": "score", "score": 4}}})
        return httpx.Response(200, json={
            "model": "typesafe/jev-1.14",
            "answers": {f"r{i}": {"type": "score", "score": s} for i, s in enumerate(levels)},
            "usage": {"input_tokens": 300, "output_tokens": 12, "cost": 0.0005},
        })

    ranker = OpenRouterDecisionRanker("key", "typesafe/jev-1.14")
    ranker.client = httpx.AsyncClient(transport=httpx.MockTransport(respond))
    ranker.sent = sent
    return ranker


class OtherDecisionModel:
    """Any object with `rank` is a decision ranker: no model id is consulted."""

    async def rank(self, query, candidates, *, usage, usage_phase="ranking", project=None):
        order = list(range(len(candidates)))[::-1]
        return [DecisionVerdict(i, 90, "strong", 4.0) for i in order[:2]]


class ChatModel:
    """A plain chat model: it returns a verdict and a description it wrote."""

    def __init__(self):
        self.calls = 0

    async def structured(self, instruction, payload, **kwargs):
        self.calls += 1
        item = {"i": payload["rs"][0]["i"], "m": "strong", "why": "Written by the chat model."}
        return {"results": [item]}


async def services_for(ranker):
    config = Config.from_env()
    embedder = HashEmbeddings()
    catalog = MemoryCatalog(CATALOG, await embedder.embed([d.text for d in CATALOG]))
    return Services(config, catalog, embedder, ranker, ranker, {"m": ranker}, {"m": 100}, "m")


async def run(ranker, **request):
    services = await services_for(ranker)
    services.config = Config.from_env()
    candidates = [candidate(d) for d in CATALOG]
    results, notices = await _retrieve_and_rank(
        "food bank", SearchRequest(query="food bank", **request), services, UsageLedger("q"),
        candidates_override=candidates,
    )
    return results, notices


def test_description_prefers_description_then_abstract_then_text():
    assert item_description(CATALOG[0]) == "Find a food bank near you.  Exact."
    assert item_description(CATALOG[2]) == "The paper's own abstract."
    assert item_description(CATALOG[3]) == "Only the item text."


async def test_returned_descriptions_exactly_match_the_source_items():
    ranker = ranker_with([4.2, 0.4, 3.1, 2.0])
    results, _ = await run(ranker)
    by_name = {r.document.name: r for r in results}
    assert set(by_name) == {"Food Bank Finder", "Paper", "Bare"}        # 0.4 is excluded
    for result in results:
        assert result.description == item_description(result.document)
    assert [r.document.name for r in results] == ["Food Bank Finder", "Paper", "Bare"]
    assert by_name["Food Bank Finder"].relevance == "strong"
    assert by_name["Bare"].relevance == "relevant"


async def test_another_decision_adapter_gets_the_same_behaviour():
    other = OtherDecisionModel()
    assert is_decision_ranker(other) and isinstance(other, DecisionRanker)
    results, _ = await run(other)
    assert [r.document.name for r in results] == ["Bare", "Paper"]
    for result in results:
        assert result.description == item_description(result.document)


async def test_chat_rerankers_still_write_their_own_descriptions():
    chat = ChatModel()
    assert not is_decision_ranker(chat)
    results, _ = await run(chat)
    assert chat.calls and results[0].description == "Written by the chat model."


async def test_missing_scores_trigger_retries_and_record_usage(monkeypatch):
    async def no_wait(_):
        return None
    monkeypatch.setattr("askhub.decision_models.asyncio.sleep", no_wait)
    ranker = ranker_with([4, 0, 3, 2], retries_before=ATTEMPTS - 1)
    usage = UsageLedger("q")
    verdicts = await ranker.rank("food bank", [candidate(d) for d in CATALOG], usage=usage)
    assert len(ranker.sent) == ATTEMPTS
    assert [v.index for v in verdicts] == [0, 2, 3, 1]
    wire = usage.wire()
    assert wire["calls"] == 1 and wire["prompt_tokens"] == 300 and wire["cost"] == 0.0005
    assert wire["latency_seconds"] > 0


async def test_gives_up_after_the_attempts_and_falls_back_to_retrieval(monkeypatch):
    async def no_wait(_):
        return None
    monkeypatch.setattr("askhub.decision_models.asyncio.sleep", no_wait)
    ranker = ranker_with([4, 0, 3, 2], retries_before=99)
    with pytest.raises(DecisionRankingError):
        await ranker.rank("q", [candidate(d) for d in CATALOG], usage=UsageLedger("q"))
    results, notices = await run(ranker)
    assert notices == ["Decision ranking unavailable; results use retrieval scores."]
    assert results and all(r.relevance == "retrieval" for r in results)


async def test_ambiguous_term_is_scored_against_the_complete_query():
    ranker = ranker_with([4, 0, 0, 0])
    query = "grants for a food bank that feeds families"
    await ranker.rank(query, [candidate(d) for d in CATALOG], usage=UsageLedger("q"))
    sent = ranker.sent[0]
    assert sent["state"]["query"] == query
    question = sent["questions"]["r1"]
    assert "complete request" in question["instructions"]
    assert "ambiguous word" in question["instructions"]
    assert question["criteria"] == list(SCORE_LEVELS) and len(SCORE_LEVELS) == 5
    assert [r["name"] for r in sent["state"]["records"]][:2] == ["Food Bank Finder", "Bank Ledger"]


def test_jev_is_a_decision_model_by_metadata():
    option = RANKING_MODELS["typesafe/jev-1.14"]
    assert option.kind == "decision" and option.decision_threshold == 1.9
    assert RANKING_MODELS["openai/gpt-oss-20b"].kind == "chat"
    assert option.wire()["kind"] == "decision"


@pytest.mark.asyncio
async def test_a_request_too_large_for_the_model_is_split_and_scored_in_full():
    """The API answers 400 max_tokens_exceeded for too many records; halves score the same."""
    sent = []

    def respond(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        count = len(body["state"]["records"])
        sent.append(count)
        if count > 3:
            detail = '{"detail":{"error_type":"max_tokens_exceeded"}}'
            return httpx.Response(400, json={"error": {"message": detail}})
        return httpx.Response(200, json={
            "model": "typesafe/jev-1.14",
            "answers": {r["id"]: {"type": "score", "score": 4.0 if r["name"].endswith("7") else 1.0}
                        for r in body["state"]["records"]},
            "usage": {"input_tokens": 10, "output_tokens": 1, "cost": 0.0001},
        })

    ranker = OpenRouterDecisionRanker("key", "typesafe/jev-1.14")
    ranker.client = httpx.AsyncClient(transport=httpx.MockTransport(respond))
    usage = UsageLedger("q")
    candidates = [
        candidate(document(i, {"name": f"Item {i}", "description": f"text {i}"})) for i in range(10)
    ]
    verdicts = await ranker.rank("query", candidates, usage=usage)
    assert sent[0] == 10 and sent[1:3] == [5, 5] and min(sent) < 4
    assert len(verdicts) == 10 and verdicts[0].index == 7 and verdicts[0].relevance == "strong"
    assert {v.index for v in verdicts} == set(range(10))
    assert sum(1 for v in verdicts if v.relevance == "strong") == 1


def test_record_does_not_repeat_the_item():
    schema = {"@type": "Article", "name": "Best Dishwasher", "headline": "Best Dishwasher",
              "description": "We tested forty.", "keywords": "dishwasher"}
    sent = OpenRouterDecisionRanker("key", "m").payload(
        "q", [candidate(document(0, schema))], lambda s: s)["state"]["records"][0]
    assert sent["description"] == "We tested forty."
    assert sent["record"] == {"@type": "Article", "keywords": "dishwasher"}


@pytest.mark.parametrize("fields_of", ["chat", "decision"])
def test_a_nested_author_is_named_but_not_described(fields_of):
    """Wirecutter authors carry a biography of hundreds of characters; no ranker needs it."""
    from askhub.pipeline import _ranking_projection
    bio = "I research and report on dishwashers. " * 12
    schema = {"@type": "Article", "name": "Best", "description": "d",
              "author": [{"@type": "Person", "name": "A. Writer", "description": bio}],
              "hasPart": [{"@type": "ListItem", "name": "Bosch 500"}]}
    fields = {"@type", "name", "description", "author", "hasPart"}
    projected = _ranking_projection(schema, fields)
    assert projected["author"] == [{"@type": "Person", "name": "A. Writer"}]
    assert projected["hasPart"] == [{"@type": "ListItem", "name": "Bosch 500"}]
    item = {"@type": "Organization", "name": "Food Bank", "description": "Feeds."}
    org = _ranking_projection(item, fields)
    assert org["description"] == "Feeds."   # the item itself is not nested


def test_an_item_that_is_itself_an_organization_keeps_its_fields():
    schema = {"@type": "Organization", "name": "Food Bank", "description": "Feeds people.",
              "areaServed": "Austin"}
    sent = OpenRouterDecisionRanker("key", "m").payload(
        "q", [candidate(document(0, schema))], lambda s: s)["state"]["records"][0]
    assert sent["record"] == {"@type": "Organization", "areaServed": "Austin"}


async def run_with_fallback(ranker, fallback, **config):
    services = await services_for(ranker)
    services.config = Config.from_env()
    services.config = replace(services.config, **config)
    services.rankers = {"m": ranker, "openai/gpt-oss-20b": fallback}
    results, notices = await _retrieve_and_rank(
        "food bank", SearchRequest(query="food bank"), services, UsageLedger("q"),
        candidates_override=[candidate(d) for d in CATALOG],
    )
    return results, notices


async def test_a_chat_model_is_asked_when_the_decision_model_keeps_nothing():
    chat = ChatModel()
    results, _ = await run_with_fallback(ranker_with([0.1, 0.2, 0.3, 0.1]), chat)
    assert chat.calls and results[0].description == "Written by the chat model."


async def test_the_chat_model_is_not_asked_when_the_decision_model_keeps_something():
    chat = ChatModel()
    results, _ = await run_with_fallback(ranker_with([4.2, 0.4, 0.3, 0.1]), chat)
    assert chat.calls == 0 and [r.document.name for r in results] == ["Food Bank Finder"]


async def test_a_chat_model_is_asked_when_the_decision_call_fails(monkeypatch):
    async def no_wait(_):
        return None
    monkeypatch.setattr("askhub.decision_models.asyncio.sleep", no_wait)
    chat = ChatModel()
    results, notices = await run_with_fallback(ranker_with([4, 0, 3, 2], retries_before=99), chat)
    assert chat.calls and results[0].description == "Written by the chat model."
    assert notices == []


async def test_the_fallback_can_be_turned_off():
    chat = ChatModel()
    results, _ = await run_with_fallback(
        ranker_with([0.1, 0.2, 0.3, 0.1]), chat, decision_fallback_model=None)
    assert chat.calls == 0 and results == []
