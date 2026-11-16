import xml.etree.ElementTree as ET
from datetime import timedelta

import pytest

from qdrss.feed import NS, FeedRequest, evaluate, freshen, render_rss
from qdrss.index import build_index
from qdrss.models import Candidate
from qdrss.providers import HashEmbeddings, KeywordRanker
from qdrss.rank import RankingError, classify
from qdrss.store import Store

from .conftest import NOW, make_item

OPTS = {"candidate_count": 3, "batch_size": 2, "window_days": 7, "now": NOW}


async def populated(store: Store):
    old = NOW - timedelta(days=30)
    await store.insert_missing(
        [
            make_item(f"old{i}", "rust compiler rust compiler", "rust", ingested=old)
            for i in range(5)
        ]
        + [
            make_item("new1", "rust release", "the rust team", ingested=NOW),
            make_item("new2", "rust notes", "compiler", ingested=NOW),
            make_item("new3", "gardening", "tomatoes", ingested=NOW - timedelta(days=1)),
        ]
    )
    return await build_index(store, HashEmbeddings())


async def test_date_mask_applies_before_top_k(store: Store):
    index = await populated(store)
    req = FeedRequest("rust compiler", since=NOW - timedelta(days=7), limit=10, threshold="relevant")
    matches = await evaluate(req, index, HashEmbeddings(), KeywordRanker(), **OPTS)
    ids = [m.item.id for m in matches]
    # Five older items are the closest vectors and would fill a top-3 pool; the
    # mask keeps them out so the two new rust items are what comes back.
    assert ids == ["s:new1", "s:new2"]
    assert {m.category for m in matches} == {"relevant", "strong"}


async def test_strong_threshold_and_limit(store: Store):
    index = await populated(store)
    req = FeedRequest("rust compiler", since=NOW - timedelta(days=7), limit=1, threshold="strong")
    matches = await evaluate(req, index, HashEmbeddings(), KeywordRanker(), **OPTS)
    assert [m.item.id for m in matches] == ["s:new2"]


async def test_default_window_when_no_since(store: Store):
    index = await populated(store)
    req = FeedRequest("rust", since=None, limit=10, threshold="relevant")
    matches = await evaluate(req, index, HashEmbeddings(), KeywordRanker(), **OPTS)
    # Non-empty, or the check below passes vacuously (it did, once the window
    # was read from the wall clock and drifted past the fixtures).
    assert matches
    assert all(m.item.ingested_at > NOW - timedelta(days=7) for m in matches)


async def test_newest_published_first_with_id_tiebreak(store: Store):
    index = await populated(store)
    req = FeedRequest("rust", since=None, limit=10, threshold="relevant")
    matches = await evaluate(req, index, HashEmbeddings(), KeywordRanker(), **OPTS)
    keys = [(-(m.item.published_at or m.item.ingested_at).timestamp(), m.item.id) for m in matches]
    assert keys == sorted(keys)


async def test_order_follows_the_published_date_not_the_ingestion_batch(store: Store):
    # A source ingested in one batch stamps every item the same moment; the feed shows each item's pubDate, so that is the order.
    from dataclasses import replace
    items = [replace(make_item(f"e{i}", "rust episode", "rust"), published_at=NOW - timedelta(days=d), ingested_at=NOW - timedelta(minutes=m))
             for i, (d, m) in enumerate([(300, 1), (5, 9), (900, 3)])]
    await store.insert_missing(items)
    index = await build_index(store, HashEmbeddings())
    req = FeedRequest("rust", since=None, limit=10, threshold="relevant")
    matches = await evaluate(req, index, HashEmbeddings(), KeywordRanker(), **OPTS)
    assert [m.item.id for m in matches] == ["s:e1", "s:e0", "s:e2"]


async def test_no_matches_is_empty_not_error(store: Store):
    index = await populated(store)
    req = FeedRequest("quantum chromodynamics", since=None, limit=10, threshold="relevant")
    assert await evaluate(req, index, HashEmbeddings(), KeywordRanker(), **OPTS) == []


async def test_ranker_failure_raises():
    class Broken:
        async def structured(self, instruction, payload):
            raise TimeoutError("slow")

    with pytest.raises(RankingError):
        await classify("q", [Candidate(make_item("1", "t"), "vector", 1.0)], Broken())


async def test_ranker_output_validation():
    class Odd:
        async def structured(self, instruction, payload):
            return {
                "results": [
                    {"i": 0, "m": "STRONG", "why": "ok"},
                    {"i": 0, "m": "relevant"},  # duplicate
                    {"i": 9, "m": "strong"},  # unknown
                    {"i": 1, "m": "maybe"},  # bad category
                    "junk",
                ]
            }

    cands = [Candidate(make_item(str(i), "t"), "vector", 1.0) for i in range(2)]
    matches = await classify("q", cands, Odd())
    assert [(m.item.id, m.category) for m in matches] == [("s:0", "strong")]


async def test_render_rss_escapes_and_is_stable():
    item = make_item("x", 'Tom & Jerry <3', "a & b", ingested=NOW)
    from qdrss.models import Match

    req = FeedRequest("cats & dogs", since=NOW, limit=5, threshold="relevant")
    one = render_rss(req, [Match(item, "strong", "why")], "http://h")
    two = render_rss(req, [Match(item, "strong", "why")], "http://h")
    root = ET.fromstring(one)
    node = root.find("channel/item")
    assert node.findtext("title") == "Tom & Jerry <3"
    assert node.find("guid").get("isPermaLink") == "false"
    assert node.findtext("guid") == "s:x"
    assert node.findtext("pubDate").endswith("+0000")
    assert node.findtext(f"{{{NS}}}category") == "strong"
    assert node.findtext(f"{{{NS}}}ingestedAt") == NOW.isoformat()
    assert "since=" in root.findtext("channel/link")
    assert one.split(b"<lastBuildDate>")[0] == two.split(b"<lastBuildDate>")[0]


async def test_collection_mask(store: Store):
    await store.insert_missing(
        [
            make_item("a1", "rust news", "rust", ingested=NOW, source="ai_show"),
            make_item("h1", "rust in history", "rust", ingested=NOW, source="hist_show"),
            make_item("x1", "rust orphan", "rust", ingested=NOW, source="gone"),
        ]
    )
    index = await build_index(
        store, HashEmbeddings(), {"ai_show": "ai", "hist_show": "history"}
    )
    async def ids(collection):
        req = FeedRequest("rust", since=None, limit=10, threshold="relevant", collection=collection)
        return sorted(m.item.id for m in await evaluate(req, index, HashEmbeddings(), KeywordRanker(), **OPTS))
    assert await ids("ai") == ["ai_show:a1"]
    assert await ids("history") == ["hist_show:h1"]
    assert await ids(None) == ["ai_show:a1", "gone:x1", "hist_show:h1"]
    req = FeedRequest("rust", since=None, limit=10, threshold="relevant", collection="ai")
    assert b"collection=ai" in render_rss(req, [], "http://h")


async def test_ranker_retries_once():
    calls = []

    class Flaky:
        async def structured(self, instruction, payload):
            calls.append(1)
            if len(calls) == 1:
                raise ValueError("malformed json")
            return {"results": [{"i": 0, "m": "strong", "why": "second try"}]}

    matches = await classify("q", [Candidate(make_item("1", "t"), "vector", 1.0)], Flaky())
    assert len(calls) == 2 and matches[0].why == "second try"


def _candidates(ages_days):
    """Candidates closest first, each published that many days before NOW."""
    return [Candidate(make_item(f"c{i}", "t", ingested=NOW - timedelta(days=a)), "vector", 1.0 - i / 100) for i, a in enumerate(ages_days)]


def test_a_freshness_boost_lets_a_recent_item_displace_an_old_one_that_is_only_slightly_closer():
    cands = _candidates([2000, 2000, 2000, 2000, 3])       # the recent item is last of five
    kept = [c.item.id for c in freshen(cands, 365, 4, NOW)]
    assert "s:c4" in kept and len(kept) == 4
    assert kept == sorted(kept)                              # closest-first order is kept


def test_a_freshness_boost_does_not_rescue_a_recent_item_that_is_far_down_the_pool():
    cands = _candidates([2000] * 60 + [3])
    kept = [c.item.id for c in freshen(cands, 365, 20, NOW)]
    assert "s:c60" not in kept                               # closeness still counts: the boost is half the rank range at most


def test_with_no_more_candidates_than_kept_nothing_changes():
    cands = _candidates([10, 3000])
    assert freshen(cands, 365, 5, NOW) == cands


async def test_the_boost_widens_the_pool_and_the_feed_url_carries_it(store: Store):
    asked = []
    class Spy:
        async def search(self, vector, since, collection, limit, sources=None, collections=None):
            asked.append(limit)
            return []
    for fresh in (0.0, 180.0):
        await evaluate(FeedRequest("rust", None, 10, "relevant", None, fresh), Spy(), HashEmbeddings(), KeywordRanker(), **OPTS)
    assert asked == [OPTS["candidate_count"], OPTS["candidate_count"] * 3]  # no max_candidates: one page
    assert "fresh=180" in ET.fromstring(render_rss(FeedRequest("rust", None, 10, "relevant", None, 180.0), [], "http://x")).find("channel/link").text


async def test_one_result_per_url_across_sources(store: Store):
    # The same article in a section feed and the home-page feed is two sources' items with one URL.
    from dataclasses import replace
    a = make_item("a", "rust release", "rust", source="world")
    await store.insert_missing([a, replace(make_item("a", "rust release", "rust", source="home"), url=a.url),
                                make_item("b", "rust notes", "rust", source="home")])
    index = await build_index(store, HashEmbeddings())
    req = FeedRequest("rust", since=None, limit=10, threshold="relevant")
    matches = await evaluate(req, index, HashEmbeddings(), KeywordRanker(), **OPTS)
    urls = [m.item.url for m in matches]
    assert len(urls) == len(set(urls)) == 2


async def test_sources_limit_the_candidates(store: Store):
    # A vertical searches only its sources: the closer item from another source is not a candidate.
    await store.insert_missing([make_item("a", "rust compiler", "rust", source="travel"),
                                make_item("b", "rust compiler rust compiler", "rust", source="world")])
    index = await build_index(store, HashEmbeddings())
    req = FeedRequest("rust compiler", since=None, limit=10, threshold="relevant", sources=frozenset({"travel"}))
    matches = await evaluate(req, index, HashEmbeddings(), KeywordRanker(), **OPTS)
    assert [m.item.source for m in matches] == ["travel"]


async def test_collections_limit_the_candidates_and_use_the_items_own_collection(store: Store):
    # An archive item's source is not in the manifest; its own collection still places it.
    from dataclasses import replace
    await store.insert_missing([replace(make_item("a", "rust compiler", "rust", source="nytnews_archive_x"), collection="nytnews"),
                                replace(make_item("b", "rust compiler rust", "rust", source="other"), collection="npr")])
    index = await build_index(store, HashEmbeddings())
    req = FeedRequest("rust compiler", since=None, limit=10, threshold="relevant", collections=frozenset({"nytnews"}))
    matches = await evaluate(req, index, HashEmbeddings(), KeywordRanker(), **OPTS)
    assert [m.item.source for m in matches] == ["nytnews_archive_x"]


class _Pool:
    """A retriever over fixed candidates, closest first, that records the pool sizes asked for."""
    def __init__(self, candidates):
        self.candidates, self.asked = candidates, []

    async def search(self, vector, since, collection, limit, sources=None, collections=None):
        self.asked.append(limit)
        return self.candidates[:limit]


class _Judge:
    """Ranks as relevant only items whose title names the answer; records every title it saw."""
    def __init__(self):
        self.seen = []

    async def structured(self, instruction, payload):
        self.seen += [r["title"] for r in payload["rs"]]
        return {"results": [{"i": r["i"], "m": "relevant" if "bessent" in r["title"] else "exclude", "why": "names it"}
                            for r in payload["rs"]]}


def _story(n_about_the_big_story, answer_at):
    out = [Candidate(make_item(f"huang{i}", f"huang affair {i}", "huang", ingested=NOW), "vector", 1 - i / 1000)
           for i in range(n_about_the_big_story)]
    out.insert(answer_at, Candidate(make_item("answer", "bessent megaspeed", "bessent", ingested=NOW), "vector", 0.5))
    return out


async def test_the_answer_just_past_the_first_page_is_still_judged():
    # Fifty items about the bigger story are closer than the one that answers the question.
    pool = _Pool(_story(50, answer_at=45))
    matches = await evaluate(FeedRequest("bessent role in huang case", None, 10, "relevant"), pool, HashEmbeddings(), _Judge(),
                             candidate_count=40, batch_size=5, window_days=7, max_candidates=200, now=NOW)
    assert [m.item.id for m in matches] == ["s:answer"]
    assert pool.asked == [200]                               # one retrieval serves every page


async def test_paging_stops_when_enough_qualify_and_at_the_ceiling():
    judge = _Judge()
    many = [Candidate(make_item(f"b{i}", f"bessent {i}", "x", ingested=NOW), "vector", 1 - i / 1000) for i in range(100)]
    await evaluate(FeedRequest("q", None, 10, "relevant"), _Pool(many), HashEmbeddings(), judge,
                   candidate_count=40, batch_size=5, window_days=7, max_candidates=200, now=NOW)
    assert len(judge.seen) == 40                             # the first page gave enough
    judge = _Judge()
    none = _story(500, answer_at=499)
    await evaluate(FeedRequest("q", None, 10, "relevant"), _Pool(none), HashEmbeddings(), judge,
                   candidate_count=40, batch_size=5, window_days=7, max_candidates=100, now=NOW)
    assert len(judge.seen) == 100 and len(set(judge.seen)) == 100   # the ceiling, nothing judged twice
