"""The VM-backed catalog (keyword index here, records and vectors on the VM) answers exactly
as the in-memory catalog does, and a failed VM is an error, never an empty result."""

import httpx
import numpy as np
import pytest

from askhub.catalog import MemoryCatalog
from askhub.models import Document
from askhub.vector_service import create_vector_app
from askhub.vm_catalog import VectorServiceError, VMCatalog


def documents(site, n):
    return tuple(
        Document(
            f"{site}-{i}",
            f"https://{site}.test/{i}",
            f"{site} item {i}",
            site,
            f"lemon soup number {i}",
            {"@type": "Recipe", "name": f"item {i}"},
            (("name", f"{site} item {i}"), ("body", f"lemon soup number {i}")),
        )
        for i in range(n)
    )


def catalog(site, n, seed):
    rng = np.random.default_rng(seed)
    matrices = {
        f: rng.normal(size=(n, 8)).astype(np.float32) for f in ("name", "description", "body")
    }
    return MemoryCatalog(
        documents(site, n), np.empty((n, 8), dtype=np.float32), field_matrices=matrices
    )


def remote(full: MemoryCatalog, served: list[MemoryCatalog], token="t", header="t") -> VMCatalog:
    """A VMCatalog over `full`'s keyword index (records cut to stubs, as in a keyword snapshot)."""
    vm = VMCatalog.__new__(VMCatalog)
    vm.__dict__.update(full.__dict__)
    vm.documents = tuple(Document(d.id, "", "", d.site, "", {}) for d in full.documents)
    vm.field_matrices = {}
    vm.url = "http://vm.test"
    vm.client = httpx.AsyncClient(
        transport=httpx.ASGITransport(app=create_vector_app(served, token)),
        headers={"authorization": f"Bearer {header}"},
    )
    return vm


async def test_vector_and_keyword_search_match_the_in_memory_catalog():
    recipes, reviews = catalog("nytimes", 30, 1), catalog("wirecutter", 10, 2)
    vm = remote(recipes, [recipes, reviews])
    query = np.random.default_rng(3).normal(size=8).astype(np.float32)
    for site in ("nytimes", None):
        expected = await recipes.search_vector(query, site, 5, {"nytimes-0"})
        got = await vm.search_vector(query, site, 5, {"nytimes-0"})
        assert [c.document for c in got] == [c.document for c in expected]
        assert [c.retrieval_score for c in got] == pytest.approx(
            [c.retrieval_score for c in expected]
        )
    expected = await recipes.search_bm25("lemon soup 7", "nytimes", 5)
    got = await vm.search_bm25("lemon soup 7", "nytimes", 5)
    assert [(c.document, c.bm25_score) for c in got] == [
        (c.document, c.bm25_score) for c in expected
    ]


async def test_a_failed_or_refusing_vm_is_an_error():
    recipes = catalog("nytimes", 5, 1)
    with pytest.raises(VectorServiceError):  # wrong token: 401
        await remote(recipes, [recipes], header="wrong").search_vector(
            np.ones(8, np.float32), None, 3
        )
    with pytest.raises(VectorServiceError):  # the VM does not hold this collection: 404
        await remote(recipes, [catalog("wirecutter", 5, 2)]).search_vector(
            np.ones(8, np.float32), None, 3
        )
