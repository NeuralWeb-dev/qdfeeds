"""The shipped sources.yaml: what a deploy would ingest."""

from collections import Counter
from pathlib import Path

from starlette.testclient import TestClient

from qdrss.app import create_app
from qdrss.ingest import load_sources

from .test_app import make_services

MANIFEST = Path(__file__).resolve().parent.parent / "sources.yaml"


def test_manifest_loads_with_unique_names_and_https_urls():
    sources = load_sources(MANIFEST)
    names = [s.name for s in sources]
    assert len(names) == len(set(names))
    assert all(s.url.startswith(("http://", "https://")) for s in sources)


def test_npr_collection():
    npr = [s for s in load_sources(MANIFEST) if s.collection == "npr"]
    assert 280 <= len(npr) <= 320
    assert all(s.name.startswith("npr_") for s in npr)
    # A feed listed twice would ingest the same episodes under two source names.
    assert not [url for url, n in Counter(s.url for s in npr).items() if n > 1]
    names = {s.name for s in npr}
    for show in ("npr_this_american_life", "npr_radiolab", "npr_fresh_air", "npr_planet_money"):
        assert show in names


def test_npr_landing_page(tmp_path):
    with TestClient(create_app(make_services(tmp_path))) as client:
        page = client.get("/npr")
        assert page.status_code == 200
        assert b"NPR podcasts by topic" in page.content
        assert b"collection: 'npr'" in page.content
        assert client.get("/nyt").status_code == 200


def test_nytnews_collection():
    news = [s for s in load_sources(MANIFEST) if s.collection == "nytnews"]
    assert len(news) >= 90
    assert all(s.name.startswith("nytnews_") for s in news)
    assert len({s.url for s in news}) == len(news)
    names = {s.name for s in news}
    # scripts/backfill_nyt_news.py files archive articles under these by section.
    for section in ("nytnews_world", "nytnews_politics", "nytnews_business", "nytnews_athletic_nfl"):
        assert section in names


def test_select_sources_by_collection():
    from qdrss.ingest import select_sources
    sources = load_sources(MANIFEST)
    assert select_sources(sources, "") == sources
    news = select_sources(sources, "nytnews")
    assert news and {s.collection for s in news} == {"nytnews"}
    assert {s.collection for s in select_sources(sources, "nytnews, nytimes")} == {"nytnews", "nytimes"}
    import pytest
    with pytest.raises(ValueError):
        select_sources(sources, "nope")


def test_every_vertical_resolves_against_the_manifest():
    from qdrss.verticals import VERTICALS, resolve
    sources = load_sources(MANIFEST)
    for vertical in VERTICALS:
        if vertical.nlweb_url:
            continue
        names = resolve(vertical, sources, strict=True)
        assert names, vertical.id
    by = {v.id: resolve(v, sources) for v in VERTICALS if not v.nlweb_url}
    assert by["travel"] == {"nytnews_travel"}
    assert {"nytnews_athletic_nfl", "nyt_athletic_hockey_show", "athletic_fc", "nytnews_baseball"} <= by["athletic"]
    assert by["audio"] == {s.name for s in sources if s.collection == "nytimes"}
    assert by["us"] <= by["all"] and by["opinion"] <= by["all"]
