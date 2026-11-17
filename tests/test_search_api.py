"""The NYT search page's turn endpoint: follow-ups, summaries and what each turn cost."""

import json

from starlette.testclient import TestClient

from qdrss import nlweb, usage
from qdrss.app import Services, create_app
from qdrss.index import build_index
from qdrss.providers import HashEmbeddings, KeywordRanker
from qdrss.usage import UsageLedger

from .conftest import NOW, make_item
from .test_app import make_services


class Writer:
    """Rewrites with a fixed answer and summarizes by listing titles, recording a priced call each time."""

    def __init__(self, standalone=None, fail=()):
        self.standalone, self.fail, self.calls = standalone, set(fail), []

    async def structured(self, instruction, payload, phase=None):
        self.calls.append((phase, payload))
        usage.record("test/writer", phase, {"prompt_tokens": 100, "completion_tokens": 20, "cost": 0.001})
        if phase in self.fail:
            raise RuntimeError("provider down")
        if phase == "rewrite":
            return {"standalone_query": self.standalone or payload["current_query"]}
        return {"summary": "Read " + "; ".join(a["title"] for a in payload["articles"])}


async def _services(tmp_path, writer) -> Services:
    services = make_services(tmp_path)
    # The All tab searches the nytimes and nytnews collections.
    (tmp_path / "sources.yaml").write_text("nytimes:\n- name: s\n  url: https://a.test/f\n")
    services = Services(services.config, HashEmbeddings(), KeywordRanker(), writer)
    services.start = lambda: None
    await services.store.insert_missing([
        make_item("1", "rust release", "rust release notes", ingested=NOW),
        make_item("2", "rust conference", "rust community conference", ingested=NOW),
    ])
    services.index = await build_index(services.store, services.embedder, services.collections)
    services.ready_at = NOW
    return services


def turn(client, **body):
    r = client.post("/api/search", json={"vertical": "all", **body})
    assert r.status_code == 200, r.text
    assert r.headers["content-type"].startswith("application/x-ndjson")
    return {e["type"]: e for e in map(json.loads, r.text.splitlines())}


async def test_a_follow_up_is_rewritten_searched_summarized_and_costed(tmp_path):
    writer = Writer(standalone="rust release notes")
    services = await _services(tmp_path, writer)
    with TestClient(create_app(services)) as client:
        first = turn(client, q="rust")
        assert first["query"] == {"type": "query", "value": "rust", "rewritten": False}
        assert {r["title"] for r in first["results"]["results"]} == {"rust release", "rust conference"}
        assert first["summary"]["value"].startswith("Read ")
        assert [c[0] for c in writer.calls] == ["summary"]  # a first question is not rewritten

        follow = turn(client, q="just the release", previous_queries=["rust"],
                      previous_results=["rust release", "rust conference"])
        assert follow["query"] == {"type": "query", "value": "rust release notes", "rewritten": True}
        assert [r["title"] for r in follow["results"]["results"]] == ["rust release"]
        assert "q=rust+release+notes" in follow["results"]["feed"]  # the feed follows the refined request
        assert "strong" not in follow["results"]["results"][0]
        rewrite_payload = writer.calls[1][1]
        assert rewrite_payload["previous_queries"] == ["rust"] and rewrite_payload["current_query"] == "just the release"

        cost = follow["usage"]
        assert cost["cached"] is False
        [row] = cost["models"]
        assert row["model"] == "test/writer" and row["phases"] == {"rewrite": 1, "summary": 1}
        assert row["calls"] == 2 and row["total_tokens"] == 240 and abs(cost["cost"] - 0.002) < 1e-9


async def test_the_same_turn_again_is_served_from_the_cache_and_costs_nothing(tmp_path):
    writer = Writer()
    services = await _services(tmp_path, writer)
    with TestClient(create_app(services)) as client:
        turn(client, q="rust")
        again = turn(client, q="rust")
        assert again["results"]["cached"] is True and again["usage"]["cached"] is True
        assert again["usage"]["models"] == [] and again["summary"]["value"].startswith("Read ")
        assert [c[0] for c in writer.calls] == ["summary"]


async def test_a_failed_rewrite_is_said_and_the_question_searched_as_typed(tmp_path):
    services = await _services(tmp_path, Writer(fail={"rewrite"}))
    with TestClient(create_app(services)) as client:
        t = turn(client, q="rust", previous_queries=["conference"])
        assert t["query"]["rewritten"] is False and t["query"]["value"] == "rust"
        assert t["results"]["notices"] == ["This could not be read as a follow-up, so it was searched as typed."]
        assert t["usage"]["models"][0]["phases"] == {"rewrite": 1, "summary": 1}


async def test_a_failed_summary_is_reported_and_retried_next_time(tmp_path):
    writer = Writer(fail={"summary"})
    services = await _services(tmp_path, writer)
    with TestClient(create_app(services)) as client:
        t = turn(client, q="rust")
        assert "summary" not in t and t["summary_error"]["message"] == "the summary model failed"
        assert len(t["results"]["results"]) == 2
        writer.fail.clear()
        assert turn(client, q="rust")["summary"]["value"].startswith("Read ")


async def test_no_summary_when_not_asked(tmp_path):
    writer = Writer()
    services = await _services(tmp_path, writer)
    with TestClient(create_app(services)) as client:
        t = turn(client, q="rust", summarize=False)
        assert "summary" not in t and "summary_error" not in t and writer.calls == []


async def test_a_cooking_turn_carries_the_backends_summary_and_cost(tmp_path, monkeypatch):
    seen = {}

    async def ask(url, site, query, limit, client=None, mode="list", ranking_model=""):
        seen.update(query=query, mode=mode, model=ranking_model)
        return nlweb.Answer(
            [{"title": "Lasagna", "url": "https://cooking.test/1", "summary": "", "why": "", "image": None,
              "date": None, "source": "Recipe"}],
            summary="Make the lasagna.",
            usage={"models": [{"model": "openai/gpt-oss-20b", "calls": 3, "total_tokens": 900, "cost": 0.0004,
                               "phases": {"ranking": 3}}]},
        )

    monkeypatch.setattr(nlweb, "ask", ask)
    services = await _services(tmp_path, Writer(standalone="vegetarian lasagna"))
    with TestClient(create_app(services)) as client:
        t = turn(client, vertical="cooking", q="make it vegetarian", previous_queries=["lasagna"],
                 ranking_model="google/gemma-4-31b-it")
        assert seen == {"query": "vegetarian lasagna", "mode": "summarize", "model": "google/gemma-4-31b-it"}
        assert t["summary"]["value"] == "Make the lasagna." and t["results"]["feed"] is None
        phases = {row["model"]: row["phases"] for row in t["usage"]["models"]}
        assert phases == {"openai/gpt-oss-20b": {"Cooking ranking": 3}, "test/writer": {"rewrite": 1}}


async def test_bad_requests(tmp_path):
    services = await _services(tmp_path, Writer())
    with TestClient(create_app(services)) as client:
        assert client.post("/api/search", json={"vertical": "all"}).status_code == 400
        assert client.post("/api/search", json={"q": "x", "vertical": "nope"}).status_code == 400
        assert client.post("/api/search", json={"q": "x", "ranking_model": "evil/model"}).status_code == 400
        # offline there is no OpenRouter key, so only the default ranker can serve a chosen model
        assert client.post("/api/search", json={"q": "x", "ranking_model": "openai/gpt-oss-120b"}).status_code == 400
        assert client.get("/api/models").json()["models"][0] == {"id": "", "label": "Auto"}
        assert client.post("/api/search", content=b"[", headers={"content-type": "application/json"}).status_code == 400


def test_an_unpriced_call_is_counted_as_such():
    ledger = UsageLedger()
    ledger.record("m", "ranking", {"prompt_tokens": 10, "completion_tokens": 2})
    ledger.record("m", "ranking", {"prompt_tokens": 10, "completion_tokens": 2, "cost": 0.5})
    wire = ledger.wire()
    assert wire["unpriced_calls"] == 1 and wire["cost"] == 0.5 and wire["total_tokens"] == 24
    assert wire["models"][0]["phases"] == {"ranking": 2}


def test_a_jev_ranked_recipe_shows_its_own_description_or_none():
    described = {"name": "Lasagna", "url": "https://c.test/1", "description": "Written by the describer",
                 "schema_object": {"description": "The recipe's own words."}}
    bare = {"name": "Ragu", "url": "https://c.test/2", "description": "Ragu author X total time PT4H",
            "schema_object": {"name": "Ragu"}}
    assert nlweb._row(described, own_description=True)["summary"] == "The recipe's own words."
    assert nlweb._row(bare, own_description=True)["summary"] == ""  # not a Recipe: no fields to describe it by
    assert nlweb._row(described)["summary"] == "Written by the describer"


def test_a_recipe_without_a_description_gets_one_from_its_own_fields():
    schema = {"@type": "Recipe", "name": "Ragu Alla Bolognese", "author": {"@type": "Person", "name": "Molly O'Neill"},
              "totalTime": "PT4H30M", "recipeYield": "6 servings", "recipeCuisine": "italian",
              "recipeCategory": "dinner, weeknight, main course", "keywords": "beef broth, ground beef, pork, fall",
              "aggregateRating": {"ratingValue": 4, "ratingCount": 284}}
    assert nlweb.recipe_line(schema) == ("Italian dinner by Molly O'Neill. 4 hr 30 min · 6 servings · rated 4/5 by 284 cooks. "
                                         "Keywords: beef broth, ground beef, pork, fall.")
    row = nlweb._row({"name": "Ragu", "url": "https://c.test/2", "description": "Ragu author X", "schema_object": schema},
                     own_description=True)
    assert row["summary"].startswith("Italian dinner by Molly O'Neill.")
    assert nlweb.recipe_line({"@type": "Recipe", "totalTime": "PT45M"}) == "45 min."
    assert nlweb.recipe_line({"@type": "Product", "name": "x"}) == ""
