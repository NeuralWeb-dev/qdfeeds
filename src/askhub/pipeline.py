from __future__ import annotations

import asyncio
import contextlib
import html
import json
import logging
import math
import re
import uuid
from collections.abc import AsyncIterator, Callable, Collection
from dataclasses import dataclass, field, replace
from typing import Any

from .catalog import MemoryCatalog
from .config import Config
from .decision_models import (
    DecisionRanker,
    DecisionRankingError,
    is_decision_ranker,
    item_description,
)
from .expansion import expand
from .models import Candidate, Event, Mode, Result, SearchRequest
from .pre_retrieval import decontextualize
from .providers import (
    Embeddings,
    LanguageModel,
    MeteredLanguageModel,
    NullLanguageModel,
    embed_with_usage,
    similarity_score,
)
from .triage import HARD, Triage, triage
from .usage import UsageLedger

logger = logging.getLogger(__name__)

# Records per batch when asking for a verdict on every one. Ten is the largest
# size the judges were observed to label exhaustively; at thirty-three they
# shortlist regardless of the instruction.
FULL_MATRIX_BATCH_SIZE = 10


@dataclass(slots=True)
class Services:
    config: Config
    catalog: MemoryCatalog
    embedder: Embeddings
    llm: LanguageModel
    ranker: LanguageModel | DecisionRanker | None = None
    rankers: dict[str, LanguageModel | DecisionRanker] = field(default_factory=dict)
    ranking_max_tokens: dict[str, int] = field(default_factory=dict)
    default_ranking_model: str | None = None


async def search(request: SearchRequest, services: Services) -> AsyncIterator[Event]:
    query_id = str(uuid.uuid4())
    usage = UsageLedger(query_id)
    sequence = 0

    def event(kind: str, data=None) -> Event:
        nonlocal sequence
        sequence += 1
        return Event(kind, data, query_id, sequence)

    yield event("begin-nlweb-response")
    if not request.query.strip():
        yield event("error", "No query provided.")
        yield event("end-nlweb-response")
        return

    # Match NLWeb's reconciliation gate: fast track can run concurrently with
    # prechecks, but nothing from it is published until the checks finish. A
    # conversation follow-up is not fast-track eligible because it may need a
    # decontextualized query.
    fast_task = None
    fast_updates = None
    # Triage (triage.py) sorts the request into a tier and finds its dietary
    # restrictions. Restrictions change the ranking prompt (substitutions) for
    # any model; they change the model only when the user left it to us and one
    # is configured. A simple request skips expansion and the describer. A first
    # turn triages the raw query beside the fast track, so a hard request loses
    # nothing; a follow-up triages its rewrite.
    config = services.config
    route_restrictions = request.site in config.restriction_sites
    can_simplify = (request.site in config.simple_sites
                    and config.simple_summary_model in services.rankers)
    triaged = route_restrictions or can_simplify
    triage_task: asyncio.Task[Triage] | None = None

    def start_triage(query: str) -> asyncio.Task[Triage]:
        return asyncio.create_task(triage(
            query, MeteredLanguageModel(services.llm, usage, "pre_retrieval"),
            config.triage_timeout,
        ))

    if triaged and not request.previous_queries:
        triage_task = start_triage(request.query)
    if not request.previous_queries:
        fast_task, fast_updates, fast_retrieved = _start_ranking(
            request.query, request, services, usage
        )
    replacement = None
    if request.previous_queries:
        replacement = await decontextualize(
            request,
            MeteredLanguageModel(services.llm, usage, "pre_retrieval"),
            config.decontextualize_timeout,
        )
    if replacement:
        request.canonical_query = replacement
        yield event("decontextualized_query", replacement)
    if triaged and (triage_task is None or replacement is not None):
        if triage_task is not None:
            triage_task.cancel()
        triage_task = start_triage(request.effective_query)

    rerouted = switched = False
    # Without a model to switch to, restrictions only inform the describer, so
    # triage runs beside retrieval and ranking (which read restrictions from the
    # request themselves) and is awaited just before description. Switching models
    # changes the ranking, so that mode must wait for triage first.
    switching = (route_restrictions
                 and config.restriction_ranking_model in services.rankers)
    if switching and triage_task is not None:
        restrictions = (await triage_task).restrictions
        if restrictions:
            request.restrictions = tuple(name for name, _kind in restrictions)
            if (request.ranking_model is None
                    and any(kind in config.restriction_route_kinds
                            for _name, kind in restrictions)):
                request.ranking_model = config.restriction_ranking_model
                switched = True
            rerouted = True
            yield event("restrictions", {
                "restrictions": list(request.restrictions),
                "ranking_model": request.ranking_model or services.default_ranking_model,
            })

    # Fast track speculated on the raw query. It is only valid if the query it
    # ran against is still the one being searched.
    if fast_task is None or replacement is not None or rerouted:
        if fast_task is not None:
            fast_task.cancel()
            await asyncio.gather(fast_task, return_exceptions=True)
        ranking_task, updates, retrieved = _start_ranking(
            request.effective_query, request, services, usage
        )
    else:
        ranking_task, updates, retrieved = fast_task, fast_updates, fast_retrieved

    # Expansion runs beside the ordinary search and feeds the same stream, so
    # the first results are never held for it; its extra verdicts follow.
    expansion_task = None
    if _expansion_enabled(request, services):
        # A simple request gets no expansion, but triage takes about as long as
        # the expansion call, so a hard one would lose that much if expansion
        # waited for it. It starts at once and only ranks what it found if the
        # request turned out to be hard.
        gate = triage_task if can_simplify and request.expansion is None else None
        expansion_task = asyncio.create_task(
            _expand_and_rank(request.effective_query, request, services, usage,
                             updates, retrieved, gate)
        )

    try:
        producers = 1 + (expansion_task is not None)
        while producers:
            update = await updates.get()
            if update is None:
                producers -= 1
                continue
            # Verdicts arrive as each completes, from either search.
            # The authoritative, sorted top-N follows as result events.
            yield event("candidate", [update.wire()])
        results, notices = await ranking_task
    except Exception:
        ranking_task.cancel()
        if expansion_task is not None:
            expansion_task.cancel()
        await asyncio.gather(ranking_task, *([expansion_task] if expansion_task else []),
                             return_exceptions=True)
        expansion_task = None
        ranking_task, updates, _ = _start_ranking(
            request.effective_query, request, services, usage
        )
        while (update := await updates.get()) is not None:
            yield event("candidate", [update.wire()])
        results, notices = await ranking_task

    if expansion_task is not None:
        extra = await expansion_task
        if extra:
            results = _deduplicate_ranked_results(
                _group_by_relevance_band(results + extra)
            )[: request.max_results]

    # The stronger model is stricter; it has returned nothing where the default
    # model found good answers (12 of 100 diet queries). Empty is never better.
    if switched and not results:
        request.ranking_model = None
        ranking_task, updates, _ = _start_ranking(
            request.effective_query, request, services, usage
        )
        while (update := await updates.get()) is not None:
            yield event("candidate", [update.wire()])
        results, notices = await ranking_task

    analysis = HARD
    if triage_task is not None:
        analysis = await triage_task
        yield event("triage", analysis.wire())
    restriction_kinds: set[str] = set()
    if route_restrictions and not switching:
        restriction_kinds = {kind for _name, kind in analysis.restrictions}
        if analysis.restrictions:
            request.restrictions = tuple(name for name, _kind in analysis.restrictions)
            yield event("restrictions", {
                "restrictions": list(request.restrictions),
                "ranking_model": request.ranking_model or services.default_ranking_model,
            })
    simple = can_simplify and analysis.simple

    described_summary = None
    describe_id = config.describe_model or ""
    if (config.describe_restricted_model
            and restriction_kinds & set(config.restriction_route_kinds)):
        describe_id = config.describe_restricted_model
    # A decision model writes no prose, and what it ranks is shown with the records' own
    # descriptions: nothing downstream rewrites them.
    ranking_id = request.ranking_model or services.default_ranking_model
    decision_ranked = is_decision_ranker(
        (services.rankers.get(ranking_id) if ranking_id else None) or services.ranker
    )
    describer = None if simple or decision_ranked else services.rankers.get(describe_id)
    if describer is not None and is_decision_ranker(describer):
        describer = None
    if simple and not decision_ranked:
        # The ranker's choices, shown with the page's own description: a request
        # that only names a dish has nothing for a describer to check.
        results = [
            replace(result, description=_markup_description(result.document.schema_object))
            for result in results[: config.describe_count]
        ]
    if results and describer is not None:
        # Stream the describer: each item goes out as a final result the moment its
        # description completes. Anything past the reviewed top would carry no
        # checked text, so it is not shown.
        top = results[: services.config.describe_count]
        described: asyncio.Queue[tuple[int, str]] = asyncio.Queue()
        task = asyncio.create_task(_describe(
            request, top, MeteredLanguageModel(describer, usage, "description"),
            services.config.ranking_fields if services.config.describe_project else None,
            lambda index, text: described.put_nowait((index, text)),
        ))
        shown: list[Result] = []
        handled: set[int] = set()

        def accept(index: int, text: str) -> Result:
            handled.add(index)
            result = replace(top[index], description=text or top[index].description)
            shown.append(result)
            return result

        while True:
            waiter = asyncio.create_task(described.get())
            done, _ = await asyncio.wait({task, waiter}, return_when=asyncio.FIRST_COMPLETED)
            if waiter in done:
                yield event("result", [accept(*waiter.result()).wire()])
                continue
            waiter.cancel()
            while not described.empty():
                yield event("result", [accept(*described.get_nowait()).wire()])
            break
        try:
            described_summary = task.result()
        except Exception as error:   # whatever was not described is shown as ranked
            logger.warning("description failed (%s)", type(error).__name__)
        # Items the describer never mentioned stand as ranked.
        rest = [index for index in range(len(top)) if index not in handled]
        for index in rest:
            shown.append(top[index])
            yield event("result", [top[index].wire()])
        results = shown
    else:
        for result in results:
            yield event("result", [result.wire()])
    for notice in notices:
        yield event("intermediate_message", notice)

    if not results:
        yield event("intermediate_message", "No matching results found.")
    elif request.mode is not Mode.LIST:
        try:
            composer = services.rankers[config.simple_summary_model] if simple else services.llm
            answer = described_summary if request.mode is Mode.SUMMARIZE and described_summary \
                else await _compose(
                    request, results, MeteredLanguageModel(composer, usage, "generation"),
                    config.ranking_fields if simple else None,
                )
            yield event("nlws", {"answer": answer, "items": [result.wire() for result in results]})
        except Exception as exc:
            yield event("intermediate_message", f"Answer generation unavailable: {exc}")

    yield event("usage", usage.wire())
    yield event("complete")
    yield event("end-nlweb-response")


async def _retrieve_and_rank(
    query: str,
    request: SearchRequest,
    services: Services,
    usage: UsageLedger,
    updates: asyncio.Queue[Result | None] | None = None,
    candidates_override: list[Candidate] | None = None,
    retrieved: Retrieved | None = None,
) -> tuple[list[Result], list[str]]:
    """Retrieve candidates for `query` and rank them, streaming verdicts to `updates`.

    With `candidates_override` the retrieval step is skipped and those records
    are ranked instead (query expansion supplies its own). `retrieved` is told
    which records the ordinary retrieval produced, so expansion can skip them.
    """
    notices: list[str] = []
    ranking_id = request.ranking_model or services.default_ranking_model
    ranking_model = services.rankers.get(ranking_id) if ranking_id else None
    ranking_model = ranking_model or services.ranker or services.llm
    ranking_max_tokens = services.ranking_max_tokens.get(
        ranking_id, services.config.ranking_max_tokens
    )
    embedding_task = (
        asyncio.create_task(embed_with_usage(services.embedder, [query], usage, "query_embedding"))
        if request.retrieval in {"vector", "compare"} and candidates_override is None
        else None
    )

    async def rank(candidate: Candidate) -> Result | None:
        try:
            output = await MeteredLanguageModel(
                ranking_model, usage, "ranking"
            ).structured(
                "Act as a relevance filter, not a ranker. Classify the supplied record as "
                "strong, relevant, or exclude. Strong means it directly satisfies the user's "
                "intent; relevant means it is substantially useful but partial. Exclude weak, "
                "tangential, or merely keyword-overlapping records. Return only JSON as "
                "{\"m\":\"strong\",\"why\":\"...\"}. The why is one sentence of at most "
                "25 words naming the specific matching product, episode, eligibility rule, "
                "benefit, or topic. Start with a concrete fact or item name. Never refer to "
                "'the record', 'the query', 'relevant information', or say that something "
                "'provides information'. Use only supplied facts.",
                {"q": query, "r": _ranking_projection(candidate.document.schema_object,
                                                      services.config.ranking_fields)},
                max_tokens=(ranking_max_tokens * 8 if request.include_excluded
                            else ranking_max_tokens),
                temperature=0,
            )
            parsed = _parse_relevance(
                output, request.min_score, services.config.strong_score_threshold
            )
            if parsed is None:
                if not request.include_excluded:
                    return None
                score, relevance = 0, "exclude"
            else:
                score, relevance = parsed
            description = _clean_description(
                str(output.get("why", output.get("description", "")))
            )
            result = Result(
                candidate.document,
                score,
                description,
                candidate.vector_score,
                candidate.bm25_score,
                candidate.retrieval_source,
                relevance,
            )
            if updates is not None:
                updates.put_nowait(result)
            return result
        except Exception:
            return None

    async def rank_batch(
        indexed_candidates: list[tuple[int, Candidate]],
    ) -> list[Result]:
        # Serving stops after a few good records; a full matrix needs every one.
        # This break -- not the prompt -- was the real ceiling: three per batch
        # times three batches is nine verdicts however wide the retrieval.
        return_count = (
            len(indexed_candidates) if request.include_excluded
            else min(max(1, services.config.ranking_batch_return_count),
                     len(indexed_candidates))
        )
        # Serving asks for a short list; evaluation asks for a verdict on every
        # record. The difference is not a threshold -- "select at most 3 of these
        # 33" caps output at 9 per query however wide the retrieval, so agreement
        # between two judges measures overlap of two shortlists rather than
        # whether they assess the same record the same way.
        instruction = (
            "Act as a relevance filter, not a ranker. "
            f"Return a verdict for EVERY one of the {len(indexed_candidates)} records, "
            "in input order, omitting none. For each record return m as "
            "strong, relevant, or exclude. Strong means it directly satisfies the "
            "user's intent; relevant means it is substantially useful but partial; "
            "exclude means weak, tangential, or merely keyword-overlapping. "
            "Return only JSON as "
            "{\"results\":[{\"i\":0,\"m\":\"strong\",\"why\":\"...\"}]}. "
            "Each why is one sentence of at most 20 words naming the specific "
            "matching topic or capability. Use only supplied facts."
        ) if request.include_excluded else (
                "Act as a relevance filter, not a ranker. "
                f"Select at most {return_count} records. For each selected record return m as "
                "either strong or relevant. Strong means the record directly satisfies the "
                "user's intent; relevant means it is substantially useful but partial. Omit "
                "weak, tangential, or merely keyword-overlapping records, and do not fill the "
                "quota. Return only JSON as "
                "{\"results\":[{\"i\":0,\"m\":\"strong\",\"why\":\"...\"}]}. Group "
                "strong before relevant and preserve input order within each group. Each why is "
                "one sentence of at most 25 words naming the specific matching product, "
                "episode, eligibility rule, benefit, or topic. Never refer to 'the record', "
                "'the query', 'relevant information', or say that something 'provides "
                "information'. Use only supplied facts: every claim in a why must be stated "
                "in that record. Never attribute a feature, category or specification to a "
                "named product from your own knowledge; when the record lists products "
                "without saying which one meets the requirement, say what the record does "
                "say (e.g. that it has picks in that category) and name no product. Such a "
                "record is still relevant when it covers what the user asked for: select it."
        ) if not services.config.describe_model else (
                # A describer writes what is shown, so the ranker only judges and
                # writes nothing. The strict rules on its whys above made 20B omit
                # records it could not describe from the markup alone (9 of 97 test
                # queries came back empty), and its notes, when passed on, were copied
                # by cheaper describers as unsupported claims.
                "Act as a relevance filter, not a ranker. "
                f"Select at most {return_count} records. For each selected record return m as "
                "either strong or relevant. Strong means the record directly satisfies the "
                "user's intent; relevant means it is substantially useful but partial. Omit "
                "weak, tangential, or merely keyword-overlapping records, and do not fill the "
                "quota. A guide or list that covers what the user asked for is relevant. "
                "Return only JSON as {\"results\":[{\"i\":0,\"m\":\"strong\"}]}, with no other "
                "fields. Group strong before relevant and preserve input order within each group."
        )
        if request.restrictions:
            instruction += (
                " The user's dietary restrictions are hard requirements: "
                + "; ".join(request.restrictions) + ". Check every ingredient against "
                "them, including hidden ones (flour, pastry, pasta, soy sauce, stock, "
                "butter, honey, sugar). A recipe that conflicts may still be selected "
                "if a simple, common substitution fixes it; its why must then name the "
                "substitution (e.g. 'use gluten-free puff pastry', 'swap butter for "
                "olive oil'), in up to 40 words. Omit recipes no simple substitution "
                "can fix."
            )
        elif request.site in services.config.substitution_sites:
            # The same guidance with no detection call: the ranker reads any
            # restriction from the request itself. Without it, dropping the
            # detection call cut recipe relevance from .83 to .66.
            instruction += (
                " If the request states dietary restrictions, treat them as hard "
                "requirements: check every ingredient, including hidden ones (flour, "
                "pastry, pasta, soy sauce, stock, butter, honey, sugar). A recipe that "
                "conflicts may still be selected if a simple, common substitution fixes "
                "it" + ("" if services.config.describe_model else
                        "; its why must then name the substitution, in up to 40 words")
                + ". Omit recipes no simple substitution can fix."
            )
        # The model echoes positions within the list it was shown, not the
        # global indices we supplied. Batches hold non-contiguous global
        # indices (round-robin), so a returned "3" matched only when global
        # index 3 happened to sit in that batch -- silently discarding most
        # verdicts. Number each batch locally and map back.
        candidates_by_index = {
            local: candidate for local, (_global, candidate) in enumerate(indexed_candidates)
        }
        results: list[Result] = []
        seen: set[int] = set()

        def take(item) -> None:
            """Handle one verdict the moment the stream completes it.

            Time to the first result shown is what a visitor feels, so a
            selected record goes out as soon as its own JSON object closes
            rather than when the batch's whole response has finished.
            """
            if not isinstance(item, dict) or len(seen) >= return_count:
                return
            try:
                index = int(item.get("i", item.get("index")))
            except (TypeError, ValueError):
                return
            candidate = candidates_by_index.get(index)
            if candidate is None or index in seen:
                return
            seen.add(index)
            parsed = _parse_relevance(
                item, request.min_score, services.config.strong_score_threshold
            )
            if parsed is None:
                if not request.include_excluded:
                    return
                parsed = (0, "exclude")
            score, relevance = parsed
            result = Result(
                candidate.document,
                score,
                _clean_description(str(item.get("why", item.get("description", "")))),
                candidate.vector_score,
                candidate.bm25_score,
                candidate.retrieval_source,
                relevance,
            )
            results.append(result)
            if updates is not None:
                # Each verdict independently feeds the SSE stream. The final
                # authoritative threshold groups follow after all batches settle.
                updates.put_nowait(result)

        # Verdicts that streamed before a failure still count; a broken tail
        # should not discard the records already shown.
        with contextlib.suppress(Exception):
            await MeteredLanguageModel(ranking_model, usage, "ranking").structured_items(
                instruction,
                {
                    "q": query,
                    "rs": [
                        {
                            "i": local,
                            "r": _ranking_projection(candidate.document.schema_object,
                                                     services.config.ranking_fields),
                        }
                        for local, (_index, candidate) in enumerate(indexed_candidates)
                    ],
                },
                take,
                max_tokens=(ranking_max_tokens * 8 if request.include_excluded
                            else ranking_max_tokens),
                temperature=0,
            )
        return results

    # One query must see one catalog. A refresh rebinds services.catalog between
    # the two searches below - they are separated by an await on the embedding -
    # which would otherwise mix BM25 hits from the old generation with vector
    # hits from the new one in a single compare request.
    catalog = services.catalog

    if candidates_override is not None:
        candidates = list(candidates_override)
    else:
        # Per-request retrieval width, capped. Defaults to the configured counts, so
        # an ordinary query is unchanged; an evaluation asks for far more.
        bm25_width = _bounded(request.retrieval_count, services.config.bm25_rank_count,
                              services.config.max_retrieval_count)
        vector_width = _bounded(request.retrieval_count, services.config.vector_rank_count,
                                services.config.max_retrieval_count)

        bm25_candidates: list[Candidate] = []
        bm25_ids: set[str] = set()
        if request.retrieval in {"bm25", "compare"}:
            bm25_candidates = await catalog.search_bm25(query, request.site, bm25_width)
            bm25_ids = {candidate.document.id for candidate in bm25_candidates}

        vector_candidates: list[Candidate] = []
        if embedding_task is not None:
            try:
                vector = (await embedding_task)[0]
                vector_candidates = await catalog.search_vector(
                    vector,
                    request.site,
                    vector_width,
                    bm25_ids if request.retrieval == "compare" else None,
                )
            except asyncio.CancelledError:
                raise
            except Exception:
                notices.append("Vector retrieval unavailable; returning BM25 results.")

        candidates = bm25_candidates + vector_candidates
        if request.ranking_count is not None:
            # Judge only the top of what was retrieved. Separate from retrieval
            # width on purpose: retrieving 200 and ranking 50 is a different
            # experiment from retrieving 50 and ranking all of them.
            candidates = candidates[: _bounded(request.ranking_count, len(candidates),
                                               services.config.max_ranking_count)]

    if retrieved is not None:
        retrieved.ids.update(candidate.document.id for candidate in candidates)
        retrieved.ready.set()

    def by_retrieval_score() -> list[Result]:
        ranked = []
        for candidate in candidates:
            score = (
                similarity_score(candidate.vector_score)
                if candidate.vector_score is not None
                else min(100, int((candidate.bm25_score or 0) * 5))
            )
            result = Result(
                candidate.document,
                score,
                candidate.document.text[:240],
                candidate.vector_score,
                candidate.bm25_score,
                candidate.retrieval_source,
                "retrieval",
            )
            ranked.append(result)
            if updates is not None:
                updates.put_nowait(result)
        return ranked

    ranked: list[Result] | None = None
    if isinstance(ranking_model, NullLanguageModel):
        notices.append("No language model is configured; results use retrieval scores.")
        ranked = by_retrieval_score()
    elif is_decision_ranker(ranking_model):
        # A decision model scores every candidate together against the whole query
        # and writes nothing; each result carries the record's own description.
        try:
            verdicts = await ranking_model.rank(
                query,
                candidates,
                usage=usage,
                project=lambda schema: _ranking_projection(schema, services.config.ranking_fields),
            )
        except DecisionRankingError as error:
            logger.warning("decision ranking failed (%s)", error)
            verdicts = None
        else:
            ranked = []
            for verdict in verdicts:
                if verdict.relevance == "exclude" and not request.include_excluded:
                    continue
                if not 0 <= verdict.index < len(candidates):
                    continue
                candidate = candidates[verdict.index]
                result = Result(
                    candidate.document,
                    verdict.score,
                    item_description(candidate.document),
                    candidate.vector_score,
                    candidate.bm25_score,
                    candidate.retrieval_source,
                    verdict.relevance,
                )
                ranked.append(result)
                if updates is not None:
                    updates.put_nowait(result)
        fallback_id = services.config.decision_fallback_model
        fallback = services.rankers.get(fallback_id) if fallback_id else None
        if not ranked and fallback is not None and not is_decision_ranker(fallback):
            # A decision model that finds nothing scores every candidate below its bar,
            # which on a long, many-part request says the records do not state every part,
            # not that none helps. A chat model, which reads the candidates for partial
            # answers, is asked instead; the same applies when the decision call fails.
            logger.info("decision ranker returned no results (%s); asking %s",
                        "failed" if verdicts is None else "empty", fallback_id)
            ranking_model = fallback
            ranking_max_tokens = services.ranking_max_tokens.get(
                fallback_id, services.config.ranking_max_tokens
            )
            ranked = None
        elif verdicts is None:
            notices.append("Decision ranking unavailable; results use retrieval scores.")
            ranked = by_retrieval_score()
    if ranked is None:
        if services.config.ranking_batch_count > 0:
            if request.include_excluded:
                # A verdict on every record only survives a batch the model will
                # actually work through. Given 33 records and told to label all of
                # them, it quietly returns its top few instead; given 10, it labels
                # 10. Batch to that size rather than to a fixed batch count.
                batch_count = max(1, math.ceil(len(candidates) / FULL_MATRIX_BATCH_SIZE))
            else:
                batch_count = min(services.config.ranking_batch_count, len(candidates))
            batches: list[list[tuple[int, Candidate]]] = [[] for _ in range(batch_count)]
            for index, candidate in enumerate(candidates):
                batches[index % batch_count].append((index, candidate))
            ranked = []
            for completed_batch in asyncio.as_completed(
                [rank_batch(batch) for batch in batches]
            ):
                ranked.extend(await completed_batch)
        else:
            ranked = [
                result for result in await asyncio.gather(*(rank(c) for c in candidates)) if result
            ]
    ranked = _group_by_relevance_band(ranked)
    if request.include_excluded:
        # Deduplication is a presentation concern: it collapses records that
        # share a display name so a result list does not show "Phi 4 mini
        # instruct" four times. 301 names in this corpus are shared by 649
        # records, and collapsing them silently dropped 45 of 1,000 verdicts the
        # judges had already produced. An evaluation needs every record.
        return ranked[: request.max_results], notices
    return _deduplicate_ranked_results(ranked)[: request.max_results], notices


@dataclass(slots=True)
class Retrieved:
    """Which records the ordinary retrieval produced, once it has."""

    ids: set[str] = field(default_factory=set)
    ready: asyncio.Event = field(default_factory=asyncio.Event)


def _start_ranking(
    query: str, request: SearchRequest, services: Services, usage: UsageLedger
):
    updates: asyncio.Queue[Result | None] = asyncio.Queue()
    retrieved = Retrieved()

    async def run():
        try:
            return await _retrieve_and_rank(
                query, request, services, usage, updates, retrieved=retrieved
            )
        finally:
            retrieved.ready.set()
            updates.put_nowait(None)

    return asyncio.create_task(run()), updates, retrieved


def _expansion_enabled(request: SearchRequest, services: Services) -> bool:
    """Serving queries only: evaluation paths measure the ordinary search."""
    wanted = (services.config.query_expansion if request.expansion is None
              else request.expansion)
    return (
        wanted
        and not request.include_excluded
        and request.ranking_count is None
        and not isinstance(services.llm, NullLanguageModel)
    )


async def _expand_and_rank(
    query: str,
    request: SearchRequest,
    services: Services,
    usage: UsageLedger,
    updates: asyncio.Queue[Result | None],
    retrieved: Retrieved,
    gate: asyncio.Future[Triage] | None = None,
) -> list[Result]:
    """Search for what expansion suggests and rank only records not already found.

    With `gate`, nothing is ranked unless triage finds the request hard, and the
    expansion call is abandoned if triage says simple before it returns.
    """
    expanding = None
    try:
        expanding = asyncio.ensure_future(expand(
            query,
            request.site,
            MeteredLanguageModel(services.llm, usage, "expansion"),
            services.config.expansion_timeout,
        ))
        if gate is not None:
            await asyncio.wait({expanding, gate}, return_when=asyncio.FIRST_COMPLETED)
            if gate.done() and not gate.cancelled() and gate.result().simple:
                expanding.cancel()
                return []
        expansion = await expanding
        if expansion is None:
            return []
        # Records the ordinary search retrieved are already being judged.
        await retrieved.ready.wait()
        catalog = services.catalog
        seen = set(retrieved.ids)
        extra: list[Candidate] = []
        for name in expansion.names:
            # A suggested name counts only if a record's own name matches it;
            # an invented or absent title then simply finds nothing.
            for candidate in await catalog.search_bm25(name, request.site, 5, seen):
                if _same_name(candidate.document.name, name):
                    extra.append(candidate)
                    seen.add(candidate.document.id)
                    break
        if expansion.terms:
            for candidate in await catalog.search_bm25(
                expansion.terms, request.site, services.config.expansion_term_count, seen
            ):
                extra.append(candidate)
                seen.add(candidate.document.id)
        if not extra or (gate is not None and (await gate).simple):
            return []
        results, _ = await _retrieve_and_rank(
            query, request, services, usage, updates, candidates_override=extra
        )
        return results
    except Exception as error:
        logger.warning("expanded search failed (%s)", type(error).__name__)
        return []
    finally:
        if expanding is not None:
            expanding.cancel()
        updates.put_nowait(None)


def _same_name(record_name: str, suggested: str) -> bool:
    def norm(value: str) -> str:
        # "Dial M for Murder (1954)" names the record "Dial M for Murder".
        value = re.sub(r"\s*\([^)]*\)", "", html.unescape(value))
        return " ".join(re.findall(r"[a-z0-9]+", value.lower()))
    return bool(norm(suggested)) and norm(record_name) == norm(suggested)


def _deduplicate_ranked_results(results: list[Result]) -> list[Result]:
    unique: list[Result] = []
    seen: set[tuple[str, str]] = set()
    for result in results:
        key = (result.document.site, " ".join(result.document.name.lower().split()))
        if key in seen:
            continue
        seen.add(key)
        unique.append(result)
    return unique


def _relevance_band(score: int, strong_threshold: int = 90) -> str:
    return "strong" if score >= strong_threshold else "relevant"


def _parse_relevance(
    output: dict, relevant_threshold: int = 70, strong_threshold: int = 90
) -> tuple[int, str] | None:
    """Accept categorical output, with numeric scores as a compatibility fallback."""
    match = str(output.get("m", output.get("match", ""))).strip().lower()
    if match == "strong":
        return strong_threshold, "strong"
    if match == "relevant":
        return relevant_threshold, "relevant"
    if match in {"exclude", "excluded", "none", "irrelevant"}:
        return None
    try:
        score = max(0, min(100, int(output.get("s", output.get("score", 0)))))
    except (TypeError, ValueError):
        return None
    if score < relevant_threshold:
        return None
    return score, _relevance_band(score, strong_threshold)


def _group_by_relevance_band(results: list[Result]) -> list[Result]:
    """Group strong matches first without ranking or tie-breaking inside a band."""
    return [result for result in results if result.relevance == "strong"] + [
        result for result in results if result.relevance != "strong"
    ]


def _bounded(requested: int | None, fallback: int, ceiling: int) -> int:
    """Honour a request's override within the server's ceiling."""
    if requested is None:
        return fallback
    return max(1, min(int(requested), ceiling))


# A person or organisation nested in an item (an article's author) is named, not described: a
# biography says nothing about the item and costs hundreds of tokens on every record.
_NAME_ONLY_TYPES = frozenset({"Person", "Organization"})


def _ranking_projection(value, fields: Collection[str], nested: bool = False):
    """Trim a schema.org object down to the keys worth spending ranking tokens on.

    `fields` is corpus-shaped and comes from config: a product catalog needs
    `offers` and `eligibleRegion`, a model catalog needs `hf:subject` and the
    evidence properties.
    Inheriting one corpus's list silently blinds the ranker on another -- with a
    product-catalog list in place, every evidence-bearing field on a model
    record was stripped before the relevance filter ever saw it.
    """
    if isinstance(value, dict):
        keep = {"@type", "name"} if nested and value.get("@type") in _NAME_ONLY_TYPES else fields
        return {
            key: _ranking_projection(item, fields, nested=True)
            for key, item in value.items()
            if key in keep
        }
    if isinstance(value, list):
        return [_ranking_projection(item, fields, nested) for item in value[:20]]
    if isinstance(value, str):
        return value[:1000]
    return value


_DESCRIPTION_PREAMBLES = (
    "the record provides relevant information about ",
    "this record provides relevant information about ",
    "the record provides information about ",
    "this record provides information about ",
)


def _clean_description(value: str) -> str:
    description = value.strip().lstrip(". ")
    lowered = description.lower()
    for preamble in _DESCRIPTION_PREAMBLES:
        if lowered.startswith(preamble):
            description = description[len(preamble) :].lstrip()
            break
    if description:
        description = description[0].upper() + description[1:]
    return description


# Markup the composer may state as fact. Without it the composer saw only the
# ranker's one-line descriptions and filled gaps from memory: on 30 Wirecutter
# questions its summaries made ~2 claims per question that no record supported.
_RECORD_SKIP = {"@context", "image", "url", "author", "review", "video", "thumbnailUrl", "logo"}
_RECORD_LIMIT = 3000


def _record_facts(schema: dict[str, Any] | None) -> str:
    facts = {key: value for key, value in (schema or {}).items() if key not in _RECORD_SKIP}
    return json.dumps(facts, ensure_ascii=False, default=str)[:_RECORD_LIMIT]


_MARKUP_DESCRIPTION_LIMIT = 280


def _markup_description(schema: dict[str, Any] | None) -> str:
    """The record's own schema.org description, as plain text cut at a sentence."""
    value = (schema or {}).get("description")
    if not isinstance(value, str):
        return ""
    text = re.sub(r"<[^>]+>", "", re.sub(r"<(?:br|/?p|/?li)\b[^>]*>", " ", value))
    text = " ".join(html.unescape(text).split())
    if len(text) <= _MARKUP_DESCRIPTION_LIMIT:
        return text
    head = text[:_MARKUP_DESCRIPTION_LIMIT]
    end = max(head.rfind(". "), head.rfind("! "), head.rfind("? "))
    if end >= _MARKUP_DESCRIPTION_LIMIT // 3:
        return head[: end + 1]
    return head.rsplit(" ", 1)[0] + "…"


# The describer writes; it does not choose. Allowed to drop items, it removed results the
# ranker had kept on literal readings ("lacks vegetables beyond shallots and herbs" for
# a white-bean bruschetta asked for as "white beans and some veggies"), and the page
# showed them and then took them away. Which results stand is the ranker's decision.
DESCRIBE = (
    "Search results have been chosen for the user's request; write what is shown for the "
    "top ones. Each item's record is the page's own markup and the only source of facts. "
    "For each item write a description: one sentence of at most 30 words saying how it "
    "helps with the request. Every fact in it must be stated in that item's record; never "
    "attribute a feature, category or specification to a named product from your own "
    "knowledge, even one you believe is true. When a record lists products without saying "
    "which one meets the request, say what the record does say and name no product. Then "
    "write a summary: 2-4 sentences naming the most useful items and the concrete "
    "differences between them, under the same rule. If the request states dietary or other "
    "restrictions, check each item's ingredients against them: if an item conflicts, its "
    "description must name the "
    "simple, common substitution that fixes it, or say plainly what does not fit. A "
    "substitution is advice, not a fact, and is allowed. Check ingredients carefully, "
    "since some contain gluten, dairy or nuts without saying so (anything made from "
    "wheat, barley or rye; soy sauce; butter and cheese; pesto). Describe every item; "
    "none is removed. Prose only, no lists. Return JSON "
    "{\"descriptions\": [{\"i\": <item index>, \"description\": \"...\"}], "
    "\"summary\": \"...\"}."
)


async def _describe(
    request: SearchRequest, top: list[Result], llm: LanguageModel,
    fields: Collection[str] | None, emit: Callable[[int, str], None],
) -> str | None:
    """Describe `top` in one streamed call; return the summary.

    `emit(index, description)` fires as each item's entry completes, so a caller
    can show it at once. Every item is shown: the describer writes, it does not
    choose.

    The default ranker judges every candidate cheaply but states product facts the
    markup does not hold in about a quarter of its descriptions (30 Wirecutter
    questions, 2026-10-05); a stronger model writing only what is shown keeps the
    ranking cheap and the shown text grounded."""
    reported: set[int] = set()

    def on_item(entry) -> None:
        if not isinstance(entry, dict):
            return
        try:
            index = int(entry.get("i"))
        except (TypeError, ValueError):
            return
        if not 0 <= index < len(top) or index in reported:
            return
        reported.add(index)
        emit(index, _clean_description(str(entry.get("description") or "")))

    output = await llm.structured_items(
        DESCRIBE,
        {
            "request": request.effective_query,
            "restrictions": list(request.restrictions),
            "items": [
                {"i": index, "name": result.document.name,
                 "record": json.dumps(_ranking_projection(result.document.schema_object, fields),
                                      ensure_ascii=False, default=str)
                 if fields else _record_facts(result.document.schema_object)}
                for index, result in enumerate(top)
            ],
        },
        on_item,
        key="descriptions",
        max_tokens=2000,
        temperature=0,
    )
    summary = output.get("summary") if isinstance(output, dict) else None
    return summary.strip() if isinstance(summary, str) and summary.strip() else None


async def _compose(
    request: SearchRequest, results: list[Result], llm: LanguageModel,
    fields: Collection[str] | None = None,
) -> str:
    key = "summary" if request.mode is Mode.SUMMARIZE else "answer"
    instruction = (
        "Write the short overview shown above ranked search results. Use only the supplied "
        "evidence: each item's record is the page's own markup and the only source of facts. "
        "Every fact you state about a product or article must appear in its record; you may "
        "say that something may suit the user's need, but add no features, categories or "
        "specifications the records do not state, even ones you believe are true. Write 2-4 concise natural-language sentences that name the most useful "
        "results and explain concrete differences between them. The summary value must be "
        "reader-facing prose, never a list, table, object, code block, tool call, or JSON/Python "
        "serialization. Do not put recipe or product names in quotation marks. Return exactly "
        "one JSON object with one string field named summary, "
        "for example {\"summary\":\"Several options fit...\"}."
        if request.mode is Mode.SUMMARIZE
        else "Answer the user's question using only the supplied search evidence. Write concise "
        "reader-facing prose and cite relevant supplied URLs. The answer value must never be a "
        "list, table, object, code block, tool call, or JSON/Python serialization. Return exactly "
        "one JSON object with one string field named answer."
    )
    try:
        output = await _compose_call(llm, instruction, request, results, fields)
    except ValueError:   # unreadable JSON: a plain overview beats none
        return _fallback_composition(results)
    value = output.get(key)
    if isinstance(value, str):
        prose = value.strip()
        if prose.startswith("```") and prose.endswith("```"):
            prose = prose.strip("`").removeprefix("json").strip()
        if prose.startswith(("{", "[")):
            try:
                nested = json.loads(prose)
            except (TypeError, json.JSONDecodeError):
                nested = None
            if isinstance(nested, dict) and isinstance(nested.get(key), str):
                prose = nested[key].strip()
            else:
                prose = ""
        if prose:
            return prose
    return _fallback_composition(results)


async def _compose_call(
    llm: LanguageModel, instruction: str, request: SearchRequest, results: list[Result],
    fields: Collection[str] | None,
) -> dict:
    return await llm.structured(
        instruction,
        {
            "query": request.effective_query,
            "evidence": [
                {
                    "name": result.document.name,
                    "url": result.document.url,
                    "description": result.description,
                    "record": json.dumps(_ranking_projection(result.document.schema_object,
                                                             fields),
                                         ensure_ascii=False, default=str)
                    if fields else _record_facts(result.document.schema_object),
                }
                for result in results
            ],
        },
        # A reasoning model spends its budget thinking before it writes. At 400
        # tokens gpt-oss-120b returned an empty completion for every summary and
        # answer -- the prose never got a turn. Composition is one call per
        # query, so the headroom is cheap.
        max_tokens=4000,
        temperature=0,
    )


def _fallback_composition(results: list[Result]) -> str:
    selected = results[:4]
    names = [result.document.name for result in selected]
    if not names:
        return "No matching results were available to summarize."
    name_list = (
        names[0]
        if len(names) == 1
        else f"{', '.join(names[:-1])}, and {names[-1]}"
    )
    details = []
    for result in selected[:3]:
        description = result.description.strip()
        if not description:
            continue
        details.append(description if description.endswith((".", "!", "?")) else f"{description}.")
    overview = f"The leading matches are {name_list}."
    return f"{overview} {' '.join(details)}".strip()


async def collect(request: SearchRequest, services: Services) -> list[Event]:
    return [event async for event in search(request, services)]
