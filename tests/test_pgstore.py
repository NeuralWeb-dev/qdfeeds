"""PostgresStore against a real Postgres (the VM's qdrss_test database); skipped without QDRSS_TEST_PG_HOST."""

import os
from datetime import timedelta

import numpy as np
import pytest

from qdrss.pgstore import PostgresStore

from .conftest import NOW, make_item

HOST = os.environ.get("QDRSS_TEST_PG_HOST")
pytestmark = pytest.mark.skipif(not HOST, reason="no test Postgres (QDRSS_TEST_PG_HOST)")


@pytest.fixture
async def store():
    s = PostgresStore(HOST, os.environ["QDRSS_PG_PASSWORD"], database="qdrss_test")
    await s.setup()
    async with s.pool.connection() as conn:
        await conn.execute("TRUNCATE items, sources")
    yield s
    await s.close()


async def test_items_are_stored_once_embedded_and_read_back_as_changed(store):
    a, b = make_item("1", "rust release", "notes " * 2000), make_item("2", "go release", ingested=NOW - timedelta(days=1))
    assert await store.insert_missing([a, b, a]) == 2
    assert await store.insert_missing([a]) == 0
    assert {i.id for i in await store.items_without_embedding()} == {a.id, b.id}
    assert (await store.items_changed("default", 0))[0] == []  # nothing has a vector yet
    await store.save_embeddings("m", {a.id: np.ones(1536, dtype=np.float32)})
    changed, t1 = await store.items_changed("default", 0)
    assert [(i.id, round(float(v[0]), 3)) for i, v in changed] == [(a.id, 1.0)]
    assert [i.id for i in await store.items_without_embedding()] == [b.id]
    assert [i.id for i, _ in (await store.items_changed("default", t1))[0]] == [a.id]  # a minute's overlap
    item = changed[0][0]
    assert item.title == "rust release" and item.ingested_at == a.ingested_at and len(item.content) == 8000
    assert await store.count() == 2 and await store.counts_by_collection(["default", "x"]) == {"default": 2, "x": 0}


async def test_a_failed_fetch_keeps_the_previous_validators(store):
    await store.remember_source("s", "https://a.test/f", etag="e1", sha256="h1")
    assert (await store.source_state("s"))["etag"] == "e1"
    await store.remember_source("s", "https://a.test/f", error="HTTP 500")
    state = await store.source_state("s")
    assert state["etag"] == "e1" and state["sha256"] == "h1" and state["last_error"] == "HTTP 500" and state["last_fetch"]
    await store.remember_source("s", "https://a.test/f", etag="e2")
    assert (await store.source_state("s"))["last_error"] is None
    [row] = await store.all_sources()
    assert row["name"] == "s" and row["last_error"] is None


async def test_connections_closed_by_a_postgres_restart_are_replaced_not_failed(store):
    """A VM restart closes every idle connection in the pool; the next request still succeeds."""
    async with store.pool.connection() as conn:
        mine = (await (await conn.execute("SELECT pg_backend_pid() AS pid")).fetchone())["pid"]
    other = PostgresStore(HOST, os.environ["QDRSS_PG_PASSWORD"], database="qdrss_test", max_size=1)
    await other.setup()
    async with other.pool.connection() as conn:
        await conn.execute("SELECT pg_terminate_backend(%s)", (mine,))
    await other.close()
    assert await store.count() == 0
