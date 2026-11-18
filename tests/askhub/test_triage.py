"""Triage sorts recipe requests into tiers; a simple one is answered cheaply.

A request that only names a dish skips expansion and the describer: the ranker's
choices are shown with the page's own description and a cheap model writes the
summary. Everything else keeps the full treatment.
"""

import asyncio

from askhub.config import Config
from askhub.models import Candidate, Document, Mode, SearchRequest
from askhub.pipeline import Services, _markup_description, collect
from askhub.providers import HashEmbeddings
from askhub.triage import HARD, parse, triage


def test_a_dish_name_with_no_constraints_is_simple():
    result = parse({"tier": "simple", "constraints": []})
    assert result.simple and result.restrictions == ()


def test_a_stated_constraint_makes_a_request_hard_whatever_the_tier_says():
    result = parse({"tier": "simple",
                    "constraints": [{"name": "no butter", "kind": "ingredient"}]})
    assert not result.simple


def test_only_dietary_constraints_are_restrictions():
    result = parse({"tier": "hard", "constraints": [
        {"name": "gluten-free", "kind": "gluten"},
        {"name": "quick", "kind": "time"},
        {"name": "vegan", "kind": "ethical"},
    ]})
    assert result.restrictions == (("gluten-free", "gluten"), ("vegan", "ethical"))
    assert len(result.constraints) == 3


def test_anything_unreadable_gets_the_full_treatment():
    assert parse(None) == HARD
    assert parse({"tier": "trivial"}).tier == "hard"


async def test_a_failed_or_slow_triage_is_hard():
    class Broken:
        async def structured(self, *args, **kwargs):
            raise RuntimeError("down")

    class Slow:
        async def structured(self, *args, **kwargs):
            await asyncio.sleep(1)
            return {"tier": "simple"}

    assert await triage("cookies", Broken(), 1) == HARD
    assert await triage("cookies", Slow(), 0.01) == HARD


def test_markup_description_is_plain_text_cut_at_a_sentence():
    assert _markup_description({"description": "Simon &amp; the <b>Whale</b>."}) == \
        "Simon & the Whale."
    first = "These cookies are " + "very " * 25 + "good."
    assert _markup_description({"description": f"{first} {'More. ' * 60}"}).startswith(first)
    assert len(_markup_description({"description": f"{first} {'More. ' * 60}"})) <= 280
    # A first sentence too short to stand alone is cut at a word instead.
    assert _markup_description({"description": "Short. " + "word " * 100}).endswith("word…")
    assert _markup_description({}) == ""
    assert _markup_description({"description": ["not", "text"]}) == ""


# ---- the pipeline, with fake models ----------------------------------------

def _document(index: int) -> Document:
    return Document(
        id=f"r{index}", url=f"https://example.com/{index}", name=f"Cookie {index}",
        site="recipes", text="cookie",
        schema_object={"@type": "Recipe", "name": f"Cookie {index}",
                       "description": f"Markup description {index}."},
    )


class Catalog:
    async def search_bm25(self, query, site, count, exclude=None):
        exclude = exclude or set()
        return [Candidate(_document(i), None, 1.0, 1.0, "bm25")
                for i in range(3) if f"r{i}" not in exclude][:count]

    async def search_vector(self, vector, site, count, exclude=None):
        return []


class Model:
    """Answers each prompt by what it asks for, and records who was asked."""

    def __init__(self, name: str, calls: list[str], tier: str, rejects: bool = False):
        self.name, self.calls, self.tier, self.rejects = name, calls, tier, rejects

    async def structured(self, instruction, payload, **kwargs):
        if instruction.startswith("Triage"):
            self.calls.append(f"{self.name}:triage")
            return {"tier": self.tier, "constraints": [] if self.tier == "simple"
                    else [{"name": "no butter", "kind": "ingredient"}]}
        if "keyword search" in instruction:
            self.calls.append(f"{self.name}:expansion")
            return {"terms": "cookie", "names": []}
        if instruction.startswith("Search results have been chosen"):
            self.calls.append(f"{self.name}:describe")
            return {"descriptions": [{"i": i, "description": "Described.",
                                      **({"keep": False} if self.rejects and i == 1 else {})}
                                     for i in range(len(payload["items"]))],
                    "summary": "Described summary."}
        if "relevance filter" in instruction:
            self.calls.append(f"{self.name}:rank")
            return {"results": [{"i": i, "m": "strong"} for i in range(len(payload["rs"]))]}
        self.calls.append(f"{self.name}:compose")
        return {"summary": f"{self.name} summary."}


def _services(calls: list[str], tier: str, rejects: bool = False) -> Services:
    config = Config(
        restriction_sites=("recipes",), substitution_sites=("recipes",),
        simple_sites=("recipes",), simple_summary_model="cheap",
        describe_model="describer", ranking_batch_count=1,
    )
    llm = Model("utility", calls, tier)
    rankers = {name: Model(name, calls, tier, rejects and name == "describer")
               for name in ("ranker", "cheap", "describer")}
    return Services(config, Catalog(), HashEmbeddings(), llm, rankers["ranker"], rankers,
                    {}, "ranker")


def _run(tier: str, rejects: bool = False) -> tuple[list[str], dict]:
    calls: list[str] = []
    events = asyncio.run(collect(
        SearchRequest(query="chocolate chip cookies", site="recipes", mode=Mode.SUMMARIZE,
                      retrieval="bm25"),
        _services(calls, tier, rejects),
    ))
    by_type: dict = {}
    for event in events:
        by_type.setdefault(event.type, []).append(event.data)
    return calls, by_type


def test_a_simple_request_skips_the_describer_and_uses_the_cheap_summary():
    calls, events = _run("simple")
    assert "describer:describe" not in calls
    assert "cheap:compose" in calls
    assert not any(call.endswith(":rank") and call != "ranker:rank" for call in calls)
    shown = [item for batch in events["result"] for item in batch]
    assert [item["description"] for item in shown] == \
        ["Markup description 0.", "Markup description 1.", "Markup description 2."]
    assert events["nlws"][0]["answer"] == "cheap summary."
    assert events["triage"] == [{"tier": "simple", "constraints": []}]


def test_a_hard_request_keeps_the_full_treatment():
    calls, events = _run("hard")
    assert "describer:describe" in calls and "utility:expansion" in calls
    assert "cheap:compose" not in calls
    shown = [item for batch in events["result"] for item in batch]
    assert {item["description"] for item in shown} == {"Described."}
    assert events["nlws"][0]["answer"] == "Described summary."


def test_the_describer_never_removes_a_result_the_ranker_chose():
    # It once dropped results the page had already shown as candidates.
    _calls, events = _run("hard", rejects=True)
    shown = [item for batch in events["result"] for item in batch]
    assert sorted(item["name"] for item in shown) == ["Cookie 0", "Cookie 1", "Cookie 2"]
    assert {item["description"] for item in shown} == {"Described."}


# ---- a careless summary is still shown --------------------------------------

def test_a_summary_with_unescaped_quotes_is_recovered():
    from askhub.providers import _loads_lenient
    assert _loads_lenient('{"summary": "Try the "Easy Banana Bread" for a classic."}') == \
        {"summary": 'Try the "Easy Banana Bread" for a classic.'}
    assert _loads_lenient('{"summary": "Cut off mid-sentence, the summary') == \
        {"summary": "Cut off mid-sentence, the summary"}
    assert _loads_lenient('{"summary": "One line.\\nTwo \\"quoted\\" lines.') == \
        {"summary": 'One line.\nTwo "quoted" lines.'}


def test_other_malformed_json_still_raises():
    import json

    import pytest

    from askhub.providers import _loads_lenient
    with pytest.raises(json.JSONDecodeError):
        _loads_lenient('{"results": [{"i": 0, "m": "strong"')
    with pytest.raises(json.JSONDecodeError):
        _loads_lenient('{"a": "x", "b": "y" oops}')


def test_unreadable_composition_falls_back_to_a_plain_overview():
    calls: list[str] = []

    class Garbled(Model):
        async def structured(self, instruction, payload, **kwargs):
            if instruction.startswith("Write the short overview"):
                raise ValueError("Expecting ',' delimiter")
            return await super().structured(instruction, payload, **kwargs)

    services = _services(calls, "simple")
    services.rankers["cheap"] = Garbled("cheap", calls, "simple")
    events = asyncio.run(collect(
        SearchRequest(query="cookies", site="recipes", mode=Mode.SUMMARIZE, retrieval="bm25"),
        services,
    ))
    answer = next(event.data["answer"] for event in events if event.type == "nlws")
    assert answer.startswith("The leading matches are Cookie 0")
