"""Decontextualization: rewriting a conversational follow-up to stand alone.

This was a plugin registry with a Protocol, a decision record and a concurrent
executor, serving one live operation. Reading it meant reconstructing a general
handler architecture in order to understand a single branch, so it is now one
function. If a second pre-retrieval operation arrives, how it orders against
this one should decide the control flow then, from two concrete cases rather
than one speculative abstraction.
"""

from __future__ import annotations

import asyncio
import logging

from .models import SearchRequest
from .providers import LanguageModel

logger = logging.getLogger(__name__)

# The bare "rewrite as standalone, preserve constraints" lost the subject on any
# follow-up that did not restate it: after a dessert request, "give me more from
# Melissa Clark" became "Melissa Clark additional content". A conversation here
# is one request being refined, and the prompt has to say so.
# The rewrite also has to keep the user's wording. Paraphrased ("Find a recipe for an
# open-faced sandwich that layers white beans and vegetables...") a follow-up of
# "without cheese" retrieved a pool sharing 0-1 of 5 recipes with the same request
# typed in one go, and the follow-up showed one result where the single query showed
# several; one rewrite also added "vegetarian", which nobody had said.
DECONTEXTUALIZE = (
    "A user is refining one search request over several turns. Rewrite their "
    "current message as a single standalone request that a search engine could "
    "answer with no other context. previous_queries are the earlier turns, oldest "
    "first, each already standing alone; previous_results are what the last turn "
    "showed.\n"
    "- Unless the current message clearly starts an unrelated topic, carry forward "
    "the subject and every constraint from earlier turns (who it is for, diet, "
    "occasion, setting, timing, budget, size, location), then apply the message.\n"
    "- The message may add a constraint (\"something with coconut\"), narrow to an "
    "author, brand, director or place (\"more from X\" means items by X that still "
    "meet the whole request), reject results (\"the granita will melt\" means "
    "exclude that kind of item and keep the requirement behind the objection), or "
    "replace an earlier constraint (the new one wins).\n"
    "- Resolve references to shown results (\"the second one\", a result's name) "
    "using previous_results.\n"
    "- Make the smallest edit to the latest previous query, keeping the rest of its "
    "words exactly as written: add the new constraint in a few natural words, replace "
    "the constraint it changes, name any result it refers to, and state the requirement "
    "behind an objection. Do not paraphrase or summarize the rest, and do not prefix "
    "it with \"Find a recipe for\" or the like. If the message adds nothing new, "
    "return the previous query as it was.\n"
    "- Add nothing the user did not say: no inferred diet, cuisine or occasion.\n"
    "- If the message is unrelated to the earlier turns, return it unchanged.\n"
    "Return JSON with standalone_query."
)


async def decontextualize(
    request: SearchRequest, llm: LanguageModel, timeout_seconds: float
) -> str | None:
    """Rewrite a follow-up as a standalone query, or None if it already stands.

    Fails open: a timeout or a provider error returns None, so the original
    query is searched rather than the request failing. A degraded rewrite is
    much better than a dead search.
    """
    try:
        output = await asyncio.wait_for(
            llm.structured(
                DECONTEXTUALIZE,
                {
                    "current_query": request.query,
                    "previous_queries": request.previous_queries[-5:],
                    "previous_results": request.previous_results[:10],
                },
            ),
            timeout_seconds,
        )
        # Interpreting the response is inside the try on purpose: a provider that
        # returns a list or a string instead of a mapping is a provider failure,
        # and should degrade the same way a timeout does rather than fail the
        # request. CancelledError is a BaseException, so cancellation still
        # propagates and does not become "search the original query".
        replacement = str(output.get("standalone_query") or request.query).strip()
    except Exception as error:
        # No query content in the log; this line goes to shared operator output.
        logger.warning("decontextualization failed (%s); searching the original query",
                       type(error).__name__)
        return None
    # Whether the query changed is read from the rewrite itself, never from a flag the
    # model sets: gpt-oss-120b wrote a correct rewrite of "I am vegetarian" after a
    # vegetarian Indian request and still said changed=false 8 times in 20, and the
    # follow-up was searched on its own.
    return replacement if replacement != request.query.strip() else None
