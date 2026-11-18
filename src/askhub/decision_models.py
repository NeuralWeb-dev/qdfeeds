"""Rankers that return verdicts, not prose.

A decision model (OpenRouter's Decisions API) answers a closed question about each
record -- how well does it satisfy the request, on a fixed scale -- and nothing else.
It writes no description, so the pipeline shows the record's own text for every result
it ranks (`item_description`) rather than anything generated about it.

A ranker is a decision ranker by capability: it has an async `rank` method (`DecisionRanker`).
Which ranking models are built as one is stated in `RankingModelOption.kind`, never read
from a model id.
"""
from __future__ import annotations

import asyncio
import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Protocol, runtime_checkable

import httpx

from .models import Candidate, Document
from .usage import UsageLedger

# The five relevance levels, lowest first. A score is a position on this scale.
SCORE_LEVELS = (
    "Wrong meaning, incompatible, or no credible connection to the request",
    "Only generic usefulness, a shared keyword, or a tangential connection",
    "Meaningful but partial help with one aspect of the request",
    "Directly and substantially addresses the request",
    "Exact, purpose-built, or exceptionally strong match for the request",
)

# Initial cut: a record scoring below this is excluded. Recalibrate on a held-out
# corpus before changing it.
DEFAULT_THRESHOLD = 1.9

# One call may be repeated when the response is malformed or leaves a record unscored.
ATTEMPTS = 3

# The score from which a verdict is "strong" rather than "relevant".
STRONG_SCORE = 3

DESCRIPTION_LIMIT = 1200


@dataclass(frozen=True, slots=True)
class DecisionVerdict:
    index: int      # position in the candidates passed to `rank`
    score: int      # 0-100, the same scale the chat rankers' verdicts map to
    relevance: str  # "strong", "relevant" or "exclude"
    level: float    # the raw position on the five-level scale


@runtime_checkable
class DecisionRanker(Protocol):
    """Scores every candidate against the whole query, in one call."""

    async def rank(
        self,
        query: str,
        candidates: list[Candidate],
        *,
        usage: UsageLedger,
        usage_phase: str = "ranking",
        project: Callable[[dict[str, Any]], Any] | None = None,
    ) -> list[DecisionVerdict]:
        """A verdict for each candidate, best first (ties in retrieval order).

        Candidates below the ranker's threshold come back as "exclude"; the caller drops them.
        """


class DecisionRankingError(RuntimeError):
    """The decision model never produced a complete set of scores."""


def is_decision_ranker(ranker: object) -> bool:
    return isinstance(ranker, DecisionRanker)


def item_description(document: Document) -> str:
    """What the record says about itself: its schema.org description, else the article's
    abstract, else its text. Copied as published; nothing is written about the record."""
    for key in ("description", "abstract"):
        value = document.schema_object.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return document.text.strip()


def _decisions_url(base_url: str) -> str:
    return f"{base_url.rstrip('/').removesuffix('/v1')}/alpha/decisions"


def _record(
    index: int, candidate: Candidate, project: Callable[[dict[str, Any]], Any] | None
) -> dict[str, Any]:
    document = candidate.document
    schema = document.schema_object
    record: dict[str, Any] = {
        "id": f"r{index}",
        "name": document.name,
        "description": item_description(document)[:DESCRIPTION_LIMIT],
    }
    if project is not None:
        record["record"] = _lean(project(schema), record, root=True)
    return record


def _lean(projected: Any, sent: dict[str, Any], root: bool = False) -> Any:
    """The projected schema object without what the record already says.

    `name` and `description` are sent beside the record, and a title is often repeated as
    `headline`; repeating them costs tokens on every record.
    """
    if isinstance(projected, list):
        return [_lean(item, sent) for item in projected]
    if not isinstance(projected, dict):
        return projected
    repeated = (sent["name"], sent["description"]) if root else ()
    return {
        key: _lean(value, sent)
        for key, value in projected.items()
        if not (isinstance(value, str) and value in repeated)
    }


def _verdicts(levels: list[float], threshold: float) -> list[DecisionVerdict]:
    ordered = sorted(range(len(levels)), key=lambda i: (-levels[i], i))
    verdicts = []
    for index in ordered:
        level = levels[index]
        if level < threshold:
            score, relevance = 0, "exclude"
        elif level >= STRONG_SCORE:
            score, relevance = 90, "strong"
        else:
            score, relevance = 70, "relevant"
        verdicts.append(DecisionVerdict(index, score, relevance, level))
    return verdicts


def _too_large(response: httpx.Response) -> bool:
    """The Decisions API answers 400 `max_tokens_exceeded` when the records do not fit the model."""
    return response.status_code == 400 and "max_tokens_exceeded" in response.text


class OpenRouterDecisionRanker:
    """OpenRouter Decisions adapter: one `score` question per record on the five-level scale."""

    def __init__(
        self,
        key: str,
        model: str,
        base_url: str = "https://openrouter.ai/api/v1",
        app_url: str | None = None,
        app_title: str = "ask-samples",
        threshold: float = DEFAULT_THRESHOLD,
    ):
        headers = {
            "Authorization": f"Bearer {key}",
            "Content-Type": "application/json",
            "X-OpenRouter-Title": app_title,
        }
        if app_url:
            headers["HTTP-Referer"] = app_url
        self.model = model
        self.threshold = threshold
        self.endpoint = _decisions_url(base_url)
        self.client = httpx.AsyncClient(headers=headers, timeout=90)

    def payload(
        self,
        query: str,
        candidates: list[Candidate],
        project: Callable[[dict[str, Any]], Any] | None = None,
    ) -> dict[str, Any]:
        records = [_record(index, candidate, project) for index, candidate in enumerate(candidates)]
        return {
            "model": self.model,
            "state": {"query": query, "records": records},
            "questions": {
                record["id"]: {
                    "type": "score",
                    "instructions": (
                        f'How well does the record with id "{record["id"]}" satisfy `query`? '
                        "Interpret the complete request before scoring. Treat compound names "
                        "as units, reject a match to a different sense of an ambiguous word, "
                        "and use only facts from that record."
                    ),
                    "criteria": list(SCORE_LEVELS),
                }
                for record in records
            },
        }

    @staticmethod
    def _levels(body: Any, count: int) -> list[float]:
        if not isinstance(body, dict) or not isinstance(body.get("answers"), dict):
            raise ValueError("decision response has no answers object")
        levels = []
        for index in range(count):
            answer = body["answers"].get(f"r{index}")
            if not isinstance(answer, dict) or answer.get("type") != "score":
                raise ValueError(f"decision response has no score for r{index}")
            levels.append(float(answer["score"]))
        return levels

    async def _score(
        self,
        query: str,
        candidates: list[Candidate],
        project: Callable[[dict[str, Any]], Any] | None,
        usage: UsageLedger,
        usage_phase: str,
    ) -> list[float]:
        """Levels for `candidates`, in order. A request the model cannot hold is split in half."""
        started = time.perf_counter()
        for attempt in range(ATTEMPTS):
            try:
                response = await self.client.post(
                    self.endpoint, json=self.payload(query, candidates, project)
                )
                if _too_large(response) and len(candidates) > 1:
                    # The levels are absolute (each record is judged against the five criteria, not
                    # against the others), so halves scored separately are comparable.
                    middle = len(candidates) // 2
                    first, second = await asyncio.gather(
                        self._score(query, candidates[:middle], project, usage, usage_phase),
                        self._score(query, candidates[middle:], project, usage, usage_phase),
                    )
                    return first + second
                response.raise_for_status()
                body = response.json()
                levels = self._levels(body, len(candidates))
            except (httpx.HTTPError, KeyError, TypeError, ValueError) as error:
                if attempt == ATTEMPTS - 1:
                    raise DecisionRankingError(
                        f"{self.model} gave no complete scores in {ATTEMPTS} attempts: {error}"
                    ) from error
                await asyncio.sleep(0.5 * 2**attempt)
            else:
                usage.record(
                    str(body.get("model") or self.model),
                    usage_phase,
                    body.get("usage"),
                    latency=time.perf_counter() - started,
                )
                return levels
        raise AssertionError("unreachable")

    async def rank(
        self,
        query: str,
        candidates: list[Candidate],
        *,
        usage: UsageLedger,
        usage_phase: str = "ranking",
        project: Callable[[dict[str, Any]], Any] | None = None,
    ) -> list[DecisionVerdict]:
        if not candidates:
            return []
        levels = await self._score(query, candidates, project, usage, usage_phase)
        return _verdicts(levels, self.threshold)

    async def aclose(self) -> None:
        await self.client.aclose()
