"""The search page as a conversation: a follow-up rewritten to stand alone, and a summary over the results.

Both are single scoped calls to the writing model (gpt-oss-120b by default), made by the harness at a fixed
point: the rewrite before retrieval, the summary after ranking. Neither decides what runs next.

The rewrite prompt is the NLWeb samples' (askhub/pre_retrieval.py), which was tuned on follow-ups there:
a conversation is one request being refined, so the subject and constraints carry forward, and the rewrite
is the smallest edit to the previous query in the user's own words (a paraphrase retrieved a different pool).
"""

from __future__ import annotations

import logging
from typing import Any

from .providers import Ranker

logger = logging.getLogger(__name__)

REWRITE = (
    "A user is refining one search request over several turns. Rewrite their "
    "current message as a single standalone request that a search engine could "
    "answer with no other context. previous_queries are the earlier turns, oldest "
    "first, each already standing alone; previous_results are what the last turn "
    "showed.\n"
    "- Unless the current message clearly starts an unrelated topic, carry forward "
    "the subject and every constraint from earlier turns (who, where, when, which "
    "people, companies or places, which angle), then apply the message.\n"
    "- The message may narrow (\"just the opinion pieces\", \"only since the election\"), "
    "add a constraint, reject results (\"not the sports ones\" means exclude those and "
    "keep the rest of the request), or replace an earlier constraint (the new one wins). \"What about X?\" asks the same "
    "question about X instead.\n"
    "- Resolve references to shown results (\"the second one\", an article's title) "
    "using previous_results.\n"
    "- Make the smallest edit to the latest previous query, keeping the rest of its "
    "words exactly as written: add the new constraint in a few natural words, replace "
    "the constraint it changes, name any result it refers to. Do not paraphrase or "
    "summarize the rest, and do not prefix it with \"Find articles about\" or the like. "
    "If the message adds nothing new, return the previous query as it was.\n"
    "- The result must read as one natural request, never the old query with the message "
    "tacked on. Examples: \"A classic lasagna\" + \"make it vegetarian\" -> \"A classic "
    "vegetarian lasagna\"; \"Should Jensen Huang have been forced out of Nvidia?\" + \"only "
    "the opinion pieces\" -> \"Opinion pieces on whether Jensen Huang should have been forced "
    "out of Nvidia\"; \"Latest on the war in Ukraine\" + \"what about Gaza?\" -> \"Latest on "
    "the war in Gaza\".\n"
    "- Add nothing the user did not say.\n"
    "- A message with no subject of its own (\"something good for kids\", \"for my dad\", "
    "\"cheaper ones\") always refines the earlier request. Only a message that names a new "
    "subject unrelated to the earlier turns starts over; return that one unchanged.\n"
    "Return JSON with standalone_query."
)

SUMMARY = (
    "Write the short overview shown above a list of New York Times search results. "
    "Use only the supplied articles: each one's title, date, source and text are the only "
    "source of facts. In 2-4 concise sentences, answer the user's question as far as these "
    "articles do, and say which articles to read for what, naming them by title or by what "
    "they cover. Every fact you state must appear in an article; if the articles do not "
    "answer the question, say what they do cover. Prose for a reader, never a list, table, "
    "code or JSON inside the value. Return exactly one JSON object with one string field "
    "named summary, for example {\"summary\":\"...\"}."
)


class SummaryError(RuntimeError):
    pass


async def rewrite(writer: Ranker, query: str, previous_queries: list[str], previous_results: list[str]) -> str:
    """The follow-up as a standalone query (the query itself when it already stands alone).

    Raises on a provider failure or an unusable reply; the caller says so on the page and searches the
    query as typed, so a broken rewrite is visible rather than silently ignored."""
    output = await writer.structured(
        REWRITE,
        {"current_query": query, "previous_queries": previous_queries[-5:], "previous_results": previous_results[:10]},
        phase="rewrite",
    )
    value = output.get("standalone_query") if isinstance(output, dict) else None
    if not isinstance(value, str) or not value.strip():
        raise ValueError("rewrite returned no standalone_query")
    return value.strip()[:300]


async def summarize(writer: Ranker, query: str, rows: list[dict[str, Any]]) -> str:
    output = await writer.structured(
        SUMMARY,
        {"query": query, "articles": [
            {"title": r["title"], "date": (r.get("date") or "")[:10], "source": r.get("source") or "",
             "text": (r.get("summary") or "")[:600]}
            for r in rows[:12]
        ]},
        phase="summary",
    )
    value = output.get("summary") if isinstance(output, dict) else None
    if not isinstance(value, str) or not value.strip() or value.strip().startswith(("{", "[")):
        raise SummaryError("summary model returned no usable summary")
    return value.strip()
