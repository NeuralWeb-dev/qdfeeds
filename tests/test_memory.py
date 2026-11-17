"""In-memory collections: snapshot round trip, catch-up from the database, fallback."""

from datetime import UTC, datetime

import numpy as np

from qdrss.memory import MemoryRetriever, parse_snapshot, write_snapshot
from qdrss.models import Item


def item(n, day):
    return Item(id=f"s:{n}", source="s", source_title="S", url=f"http://x/{n}", title=f"t{n}", content=f"c{n}",
                published_at=None, published_raw="", ingested_at=datetime(2020, 1, day, tzinfo=UTC), collection="tal")


class FakeCosmos:
    def __init__(self, fresh=()):
        self.fresh, self.searched = list(fresh), 0

    async def items_since(self, collection, since_ts):
        return [(i, v) for i, v in self.fresh if i.ingested_at.timestamp() > since_ts]

    async def search(self, *args):
        self.searched += 1
        return []


def snapshot(tmp_path):
    items = [item(1, 1), item(2, 2)]
    write_snapshot(tmp_path, "tal", items, np.asarray([[1, 0], [0, 1]], dtype=np.float32))
    return items


def test_round_trip(tmp_path):
    snapshot(tmp_path)
    items, matrix = parse_snapshot((tmp_path / "tal.jsonl.gz").read_bytes(), (tmp_path / "tal.npy").read_bytes())
    assert [i.id for i in items] == ["s:1", "s:2"] and items[0].ingested_at.tzinfo and matrix.dtype == np.float16


async def test_searches_memory_and_adds_what_came_after_the_snapshot(tmp_path):
    snapshot(tmp_path)
    fresh = [(item(3, 9), np.asarray([0.8, 0.6], dtype=np.float32)), (item(2, 2), np.asarray([0, 1], dtype=np.float32))]
    cosmos = FakeCosmos(fresh)
    r = MemoryRetriever(cosmos, {"s": "tal"}, ["tal"], str(tmp_path))
    await r.load()
    assert len(r.indexes["tal"]) == 3                        # s:2 is not added twice
    got = await r.search(np.asarray([0.8, 0.6], dtype=np.float32), None, "tal", 2)
    assert [c.item.id for c in got][0] == "s:3" and cosmos.searched == 0


async def test_other_collections_and_failed_snapshots_use_cosmos(tmp_path):
    cosmos = FakeCosmos()
    r = MemoryRetriever(cosmos, {"s": "tal"}, ["tal"], str(tmp_path / "missing"))
    await r.load()
    assert "using Cosmos" in r.status["tal"]
    await r.search(np.zeros(2, dtype=np.float32), None, "tal", 5)
    await r.search(np.zeros(2, dtype=np.float32), None, None, 5)
    assert cosmos.searched == 2


async def test_a_tab_over_several_collections_searches_each_in_memory_and_merges(tmp_path):
    snapshot(tmp_path)
    other = [item(5, 3), item(6, 4)]
    other = [Item(**{**i.__dict__, "collection": "news"}) for i in other]
    write_snapshot(tmp_path, "news", other, np.asarray([[0.6, 0.8], [0, -1]], dtype=np.float32))
    cosmos = FakeCosmos()
    r = MemoryRetriever(cosmos, {"s": "tal"}, ["tal", "news"], str(tmp_path))
    await r.load()
    got = await r.search(np.asarray([0.6, 0.8], dtype=np.float32), None, None, 3, None, frozenset({"tal", "news"}))
    assert [c.item.id for c in got] == ["s:5", "s:2", "s:1"] and cosmos.searched == 0
    # a collection that is not in memory still comes from Cosmos
    await r.search(np.zeros(2, dtype=np.float32), None, None, 3, None, frozenset({"tal", "elsewhere"}))
    assert cosmos.searched == 1


async def test_catch_up_appends_without_rebuilding_and_moves_its_floor(tmp_path):
    snapshot(tmp_path)
    cosmos = FakeCosmos()
    r = MemoryRetriever(cosmos, {"s": "tal"}, ["tal"], str(tmp_path))
    await r.load()
    buffer = r.indexes["tal"]._buffer
    cosmos.fresh = [(item(7, 20), np.asarray([0.6, 0.8], dtype=np.float32))]
    assert await r.catch_up("tal") == 1 and await r.catch_up("tal") == 0
    index = r.indexes["tal"]
    assert len(index) == 3 and index._buffer is buffer and index.matrix.dtype == np.float32
    got = await r.search(np.asarray([0.6, 0.8], dtype=np.float32), None, "tal", 1)
    assert got[0].item.id == "s:7"


async def test_an_index_grows_past_its_spare_room_and_the_old_view_is_untouched():
    from qdrss.index import Index

    base = Index.with_room([item(1, 1)], np.asarray([[1, 0]], dtype=np.float16), {"s": "tal"}, spare=0)
    room = len(base._buffer)
    fresh = [item(n, 2) for n in range(2, room + 3)]
    grown = base.extended(fresh, np.tile(np.asarray([[0, 1]], dtype=np.float32), (len(fresh), 1)))
    assert len(grown) == room + 2 and len(grown._buffer) > room and len(base) == 1
    assert [c.item.id for c in await base.search(np.asarray([0, 1], dtype=np.float32), None, None, 5)] == ["s:1"]
    assert (await grown.search(np.asarray([1, 0], dtype=np.float32), None, "tal", 1))[0].item.id == "s:1"
