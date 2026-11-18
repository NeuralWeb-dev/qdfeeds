"""The projection decides what the relevance filter is allowed to see.

Getting it wrong is silent: the ranker still returns confident verdicts, just
without the fields that distinguish one record from another.
"""

import json

from askhub.config import DEFAULT_RANKING_FIELDS, Config
from askhub.pipeline import _ranking_projection

RECORD = {
    "@type": ["SoftwareApplication", "hf:Model"],
    "@id": "https://huggingface.co/acme/demo",
    "name": "Demo 1B",
    "description": "A biomedical language model.",
    "keywords": ["transformers"],
    "hf:category": ["Token Classification"],
    "hf:capability": ["biomedical entity extraction"],
    "hf:subject": ["biomedicine"],
    "inLanguage": ["en"],
    "license": "https://opensource.org/license/mit",
    "creator": {"@type": "Organization", "name": "acme"},
    "hf:trainedOn": ["trained on PubMed abstracts"],
    "hf:evaluatedOn": ["BLURB benchmark"],
    "hf:repository": "acme/demo",
    "hf:revision": "abc123",
    "hf:modelCard": "https://huggingface.co/acme/demo/blob/abc123/README.md",
}


RECIPE = {
    "@type": "Recipe",
    "name": "Black Bean Chili",
    "recipeCuisine": "Tex-Mex",
    "recipeIngredient": ["2 cans black beans", "1 onion"],
    "aggregateRating": {"@type": "AggregateRating", "ratingValue": 4.6, "ratingCount": 812},
    "recipeInstructions": [{"@type": "HowToStep", "text": "Simmer for an hour."}],
}
HOME = {
    "@type": "SingleFamilyResidence",
    "name": "2826 Harrisburg Ave",
    "address": {"@type": "PostalAddress", "addressLocality": "Fremont", "addressRegion": "CA"},
    "numberOfBedrooms": 4,
    "price": {"@type": "PriceSpecification", "price": 2248000, "priceCurrency": "USD"},
}


def test_the_fields_a_query_constrains_on_survive_the_projection():
    """Nested values matter: a rating or a city that arrives as an empty object
    is a constraint the judge cannot check."""
    recipe = _ranking_projection(RECIPE, DEFAULT_RANKING_FIELDS)
    assert recipe["recipeCuisine"] == "Tex-Mex"
    assert recipe["recipeIngredient"] == ["2 cans black beans", "1 onion"]
    assert recipe["aggregateRating"]["ratingValue"] == 4.6
    home = _ranking_projection(HOME, DEFAULT_RANKING_FIELDS)
    assert home["address"]["addressLocality"] == "Fremont"
    assert home["price"]["price"] == 2248000
    assert home["numberOfBedrooms"] == 4


def test_long_text_the_judge_does_not_need_is_dropped():
    assert "recipeInstructions" not in _ranking_projection(RECIPE, DEFAULT_RANKING_FIELDS)


def test_plumbing_is_still_trimmed():
    """The projection exists to save ranking tokens, so it must still drop the
    keys that carry no signal."""
    kept = _ranking_projection(RECORD, DEFAULT_RANKING_FIELDS)
    for field in ("@id", "hf:revision", "hf:modelCard"):
        assert field not in kept
    assert len(json.dumps(kept)) < len(json.dumps(RECORD))


def test_a_list_from_another_corpus_blinds_the_ranker():
    """Regression: the inherited product-catalog allowlist stripped every field
    that distinguishes one model record from another."""
    product_catalog = {"@type", "name", "description", "category", "audience",
                       "eligibleRegion", "offers", "price", "brand", "url",
                       "keywords", "additionalProperty", "value", "valueReference"}
    blinded = _ranking_projection(RECORD, product_catalog)
    assert "about" not in blinded and "subjectOf" not in blinded and "featureList" not in blinded


def test_the_list_is_configurable_not_hardcoded():
    assert Config.from_env().ranking_fields == DEFAULT_RANKING_FIELDS
    narrow = Config(ranking_fields=("name",))
    assert _ranking_projection(RECORD, narrow.ranking_fields) == {"name": "Demo 1B"}
