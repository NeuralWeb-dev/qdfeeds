"""A decision model as the ranker (Jev or Gemini Decision 1, on OpenRouter's Decisions API).

A decision model answers one closed question per record, how well it satisfies the query on a fixed
five-level scale, and writes nothing else. So a match it keeps carries no reason, and the page shows the
item's own description as its snippet. The levels are absolute (each record is judged against the
criteria, not against the others), so a page of 40 goes in one call and halves of a page that is too large
for the model are comparable. Ported from the NLWeb samples (askhub/decision_models.py), with the same
scale, threshold and retries.
"""

from __future__ import annotations

import asyncio
import time
from typing import Any

import httpx

from . import usage
from .models import Candidate, Match

# The five relevance levels, lowest first. A score is a position on this scale.
SCORE_LEVELS = (
    "Wrong meaning, incompatible, or no credible connection to the request",
    "Only generic usefulness, a shared keyword, or a tangential connection",
    "Meaningful but partial help with one aspect of the request",
    "Directly and substantially addresses the request",
    "Exact, purpose-built, or exceptionally strong match for the request",
)
THRESHOLD = 1.9  # below this a record is excluded (askhub's initial cut)
STRONG_LEVEL = 3
ATTEMPTS = 3
DESCRIPTION_LIMIT = 1200


class DecisionError(RuntimeError):
    pass


class DecisionRanker:
    def __init__(self, key: str, model: str, base_url: str = "https://openrouter.ai/api/v1", threshold: float = THRESHOLD):
        self.model = model
        self.threshold = threshold
        self.endpoint = f"{base_url.rstrip('/').removesuffix('/v1')}/alpha/decisions"
        self.client = httpx.AsyncClient(
            headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json", "X-OpenRouter-Title": "qdrss"},
            timeout=90,
        )

    def payload(self, query: str, candidates: list[Candidate]) -> dict[str, Any]:
        records = [{"id": f"r{i}", "name": c.item.title, "description": c.item.content[:DESCRIPTION_LIMIT]}
                   for i, c in enumerate(candidates)]
        return {
            "model": self.model,
            "state": {"query": query, "records": records},
            "questions": {
                r["id"]: {
                    "type": "score",
                    "instructions": (
                        f'How well does the record with id "{r["id"]}" satisfy `query`? '
                        "Interpret the complete request before scoring. Treat compound names "
                        "as units, reject a match to a different sense of an ambiguous word, "
                        "and use only facts from that record."
                    ),
                    "criteria": list(SCORE_LEVELS),
                }
                for r in records
            },
        }

    @staticmethod
    def _levels(body: Any, count: int) -> list[float]:
        if not isinstance(body, dict) or not isinstance(body.get("answers"), dict):
            raise TypeError("decision response has no answers object")
        levels = []
        for i in range(count):
            answer = body["answers"].get(f"r{i}")
            if not isinstance(answer, dict) or answer.get("type") != "score":
                raise ValueError(f"decision response has no score for r{i}")
            levels.append(float(answer["score"]))
        return levels

    async def _score(self, query: str, candidates: list[Candidate]) -> list[float]:
        started = time.monotonic()
        for attempt in range(ATTEMPTS):
            try:
                response = await self.client.post(self.endpoint, json=self.payload(query, candidates))
                if response.status_code == 400 and "max_tokens_exceeded" in response.text and len(candidates) > 1:
                    middle = len(candidates) // 2
                    first, second = await asyncio.gather(self._score(query, candidates[:middle]),
                                                         self._score(query, candidates[middle:]))
                    return first + second
                response.raise_for_status()
                body = response.json()
                levels = self._levels(body, len(candidates))
            except (httpx.HTTPError, KeyError, TypeError, ValueError) as error:
                if attempt == ATTEMPTS - 1:
                    raise DecisionError(f"{self.model} gave no complete scores in {ATTEMPTS} attempts: {error}") from error
                await asyncio.sleep(0.5 * 2**attempt)
            else:
                usage.record(str(body.get("model") or self.model), "ranking", body.get("usage"), time.monotonic() - started)
                return levels
        raise AssertionError("unreachable")

    async def decide(self, query: str, candidates: list[Candidate]) -> list[Match]:
        """The candidates at or above the threshold, as matches with no reason, in candidate order."""
        if not candidates:
            return []
        levels = await self._score(query, candidates)
        return [Match(c.item, "strong" if level >= STRONG_LEVEL else "relevant", "")
                for c, level in zip(candidates, levels, strict=True) if level >= self.threshold]

    async def aclose(self) -> None:
        await self.client.aclose()
