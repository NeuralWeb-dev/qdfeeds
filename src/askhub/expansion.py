"""Query expansion: what else to search for, run alongside the ordinary search.

Retrieval only finds records whose text resembles the question. "AI" matches
Italian and Japanese titles that contain the word "ai", and nothing in Big
Hero 6's record says "AI" at all -- only knowledge about the world connects it
to "movies that portray AI positively". So a language model, in parallel with
the ordinary search, rewrites the request into search terms and names items it
knows fit. It never adds a result itself: what it suggests is searched for in
the collection, and whatever is found goes through the same relevance check as
every other candidate, so a wrong or invented suggestion costs a little time
and finds nothing.
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass

from .providers import LanguageModel

logger = logging.getLogger(__name__)

MAX_NAMES = 10


@dataclass(frozen=True, slots=True)
class Expansion:
    terms: str
    names: tuple[str, ...]


async def expand(
    query: str, collection: str | None, llm: LanguageModel, timeout_seconds: float
) -> Expansion | None:
    """Search terms and item names for `query`, or None if the model is slow or fails.

    Fails open, like decontextualization: the ordinary search has already run,
    so a missing expansion only means no extra candidates.
    """
    try:
        output = await asyncio.wait_for(
            llm.structured(
                "You help a keyword search over a catalog find records for a request. "
                "Return JSON with two fields. terms: a keyword search query that spells "
                "out abbreviations and adds the words a matching record's own text would "
                "use (synonyms, related concepts), without filler. names: up to "
                f"{MAX_NAMES} names of specific well-known items of the catalog's kind "
                "that you are confident satisfy every part of the request, as their "
                "usual titles without years or other annotations; an empty list if "
                'unsure. Return only {"terms":"...","names":["..."]}.',
                {"request": query, "catalog": collection or "general"},
                max_tokens=2000,
                temperature=0,
            ),
            timeout_seconds,
        )
        terms = str(output.get("terms") or "").strip()
        raw_names = output.get("names") or []
        names = tuple(
            str(name).strip() for name in raw_names[:MAX_NAMES]
            if isinstance(name, str | int) and str(name).strip()
        )
    except Exception as error:
        logger.warning("query expansion failed (%s); ordinary search only",
                       type(error).__name__)
        return None
    if not terms and not names:
        return None
    return Expansion(terms, names)
