"""Every query answered (a feed fetch or a search-page turn) writes one JSON line to stdout for the
private query log."""

import io
import json
from datetime import timedelta

from starlette.testclient import TestClient

from qdrss import querylog
from qdrss.app import create_app
from qdrss.index import build_index

from .conftest import NOW, make_item
from .test_app import make_services
from .test_search_api import Writer, _services

BROWSER = "Mozilla/5.0 (Macintosh) Chrome/157"


def logged(capsys) -> list[dict]:
    out = capsys.readouterr().out.splitlines()
    return [line for line in map(json.loads, (s for s in out if s.startswith("{"))) if line.get("log") == "query"]


async def test_a_feed_fetch_logs_its_query_results_and_reader(tmp_path, capsys):
    services = make_services(tmp_path)
    await services.store.insert_missing([make_item("1", "rust release", "rust", ingested=NOW - timedelta(days=1))])
    services.index = await build_index(services.store, services.embedder, services.collections)
    services.ready_at = NOW
    services.start = lambda: None
    with TestClient(create_app(services)) as client:
        since = (NOW - timedelta(days=7)).isoformat()
        assert client.get("/feed.xml", params={"q": "rust", "since": since},
                          headers={"user-agent": "node"}).status_code == 200
        assert client.get("/feed.xml?q=rust&threshold=huge", headers={"user-agent": BROWSER}).status_code == 400
    fetched, refused = logged(capsys)
    assert fetched["endpoint"] == "/feed.xml" and fetched["query"] == "rust" and fetched["status"] == 200
    assert fetched["results"] == 1 and fetched["client"] == "program" and fetched["user_agent"] == "node"
    assert fetched["since"] is True and fetched["latency_ms"] >= 0
    assert refused["status"] == 400 and refused["client"] == "browser" and "threshold" in refused["error"]


async def test_a_search_turn_logs_the_question_its_rewrite_and_spend(tmp_path, capsys):
    services = await _services(tmp_path, Writer(standalone="rust release notes"))
    with TestClient(create_app(services)) as client:
        r = client.post("/api/search", headers={"user-agent": BROWSER},
                        json={"vertical": "all", "q": "just the release", "previous_queries": ["rust"]})
        assert r.status_code == 200
        assert client.post("/api/search", json={"vertical": "nope", "q": "x"}).status_code == 400
    turn, refused = logged(capsys)
    assert turn["endpoint"] == "/api/search" and turn["vertical"] == "all"
    assert turn["query"] == "just the release" and turn["effective_query"] == "rust release notes"
    assert turn["turn"] == 2 and turn["status"] == "ok" and turn["results"] == 1
    assert turn["cost_usd"] > 0 and turn["model_calls"] == 2 and turn["client"] == "browser"
    assert refused["status"] == 400 and refused["query"] == "x" and refused["vertical"] == "nope"


def test_the_line_has_no_address_and_bounds_its_text():
    stream = io.StringIO()
    querylog.emit({"endpoint": "/feed.xml", "query": "q" * 2000, "status": 200, "user_agent": "u" * 900,
                   "error": None}, stream)
    line = json.loads(stream.getvalue())
    assert len(line["query"]) == querylog.QUERY_CHARS and len(line["user_agent"]) == querylog.TEXT_CHARS
    assert not {"ip", "remote_ip", "remoteIp", "headers"} & set(line)
    assert line["log"] == "query" and line["severity"] == "INFO"
