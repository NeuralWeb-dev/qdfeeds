"""Verdicts are surfaced as each one completes, never early and never twice."""

import json

from askhub.providers import ArrayItemStream

RESPONSE = json.dumps({"results": [
    {"i": 3, "m": "strong", "why": "Braised {short} ribs, \"Korean\" style [30 min]"},
    {"i": 0, "m": "relevant", "why": "Chicken thighs, 25 minutes"},
    {"i": 7, "m": "strong", "why": "Sheet-pan, one } brace in the text"},
]})


def test_every_split_point_yields_the_same_items_in_order():
    expected = json.loads(RESPONSE)["results"]
    for cut in range(len(RESPONSE) + 1):
        stream = ArrayItemStream("results")
        got = stream.feed(RESPONSE[:cut]) + stream.feed(RESPONSE[cut:])
        assert got == expected, cut


def test_an_item_is_reported_the_moment_it_closes():
    stream = ArrayItemStream("results")
    start = RESPONSE.index("{", RESPONSE.index("["))
    first_end = json.JSONDecoder().raw_decode(RESPONSE, start)[1]
    assert stream.feed(RESPONSE[: first_end - 1]) == []
    assert stream.feed(RESPONSE[first_end - 1 : first_end]) == [json.loads(RESPONSE)["results"][0]]


def test_character_by_character():
    stream = ArrayItemStream("results")
    got = [item for ch in RESPONSE for item in stream.feed(ch)]
    assert [item["i"] for item in got] == [3, 0, 7]
