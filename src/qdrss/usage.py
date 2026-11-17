"""What one search-page request spent on models: calls, tokens and dollars, by model and step.

The ledger for the request in progress sits in a context variable, so the OpenRouter client records
every call it makes (embedding, ranking, rewrite, summary) without each caller passing it down, and the
ranking batches gathered in parallel all land in the same ledger. Feed fetches set no ledger and record
nothing. OpenRouter reports each call's cost in its usage block; a call without one counts as unpriced.
Adapted from the NLWeb samples' askhub/usage.py, whose wire format the search page already knows.
"""

from __future__ import annotations

import contextvars
from dataclasses import dataclass, field
from typing import Any


def _mapping(value: Any) -> dict[str, Any]:
    if value is None:
        return {}
    if isinstance(value, dict):
        return value
    if hasattr(value, "model_dump"):
        return value.model_dump()
    return {}


def _integer(value: Any) -> int:
    try:
        return int(value or 0)
    except (TypeError, ValueError):
        return 0


def _number(value: Any) -> float | None:
    try:
        return None if value is None else float(value)
    except (TypeError, ValueError):
        return None


@dataclass(slots=True)
class ModelUsage:
    model: str
    calls: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    reasoning_tokens: int = 0
    cost: float = 0.0
    unpriced_calls: int = 0
    latency_seconds: float = 0.0
    phases: dict[str, int] = field(default_factory=dict)

    def wire(self) -> dict[str, Any]:
        return {
            "model": self.model, "calls": self.calls, "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens, "total_tokens": self.total_tokens,
            "reasoning_tokens": self.reasoning_tokens, "cost": round(self.cost, 12),
            "unpriced_calls": self.unpriced_calls, "latency_seconds": round(self.latency_seconds, 3),
            "phases": dict(sorted(self.phases.items())),
        }


@dataclass(slots=True)
class UsageLedger:
    by_model: dict[str, ModelUsage] = field(default_factory=dict)

    def record(self, model: str, phase: str, usage: Any, latency: float = 0.0) -> None:
        data = _mapping(usage)
        prompt = _integer(data.get("prompt_tokens"))
        completion = _integer(data.get("completion_tokens"))
        item = self.by_model.setdefault(model, ModelUsage(model))
        item.calls += 1
        item.prompt_tokens += prompt
        item.completion_tokens += completion
        item.total_tokens += _integer(data.get("total_tokens")) or prompt + completion
        item.reasoning_tokens += _integer(_mapping(data.get("completion_tokens_details")).get("reasoning_tokens"))
        item.phases[phase] = item.phases.get(phase, 0) + 1
        item.latency_seconds += latency
        cost = _number(data.get("cost"))
        if cost is None:
            item.unpriced_calls += 1
        else:
            item.cost += cost

    def merge(self, wire: dict[str, Any] | None, prefix: str) -> None:
        """Add another service's ledger (an NLWeb backend's usage event), its steps named under `prefix`."""
        for row in (wire or {}).get("models") or []:
            if not isinstance(row, dict) or not row.get("model"):
                continue
            item = self.by_model.setdefault(row["model"], ModelUsage(row["model"]))
            item.calls += _integer(row.get("calls"))
            item.prompt_tokens += _integer(row.get("prompt_tokens"))
            item.completion_tokens += _integer(row.get("completion_tokens"))
            item.total_tokens += _integer(row.get("total_tokens"))
            item.reasoning_tokens += _integer(row.get("reasoning_tokens"))
            item.cost += _number(row.get("cost")) or 0.0
            item.unpriced_calls += _integer(row.get("unpriced_calls"))
            item.latency_seconds += _number(row.get("latency_seconds")) or 0.0
            for phase, count in (row.get("phases") or {}).items():
                name = f"{prefix} {phase}"
                item.phases[name] = item.phases.get(name, 0) + _integer(count)

    def wire(self) -> dict[str, Any]:
        items = sorted(self.by_model.values(), key=lambda x: x.model)
        return {
            "models": [item.wire() for item in items],
            "calls": sum(i.calls for i in items),
            "prompt_tokens": sum(i.prompt_tokens for i in items),
            "completion_tokens": sum(i.completion_tokens for i in items),
            "total_tokens": sum(i.total_tokens for i in items),
            "reasoning_tokens": sum(i.reasoning_tokens for i in items),
            "cost": round(sum(i.cost for i in items), 12),
            "unpriced_calls": sum(i.unpriced_calls for i in items),
        }


CURRENT: contextvars.ContextVar[UsageLedger | None] = contextvars.ContextVar("qdrss_usage", default=None)


def record(model: str, phase: str, usage: Any, latency: float = 0.0) -> None:
    ledger = CURRENT.get()
    if ledger is not None:
        ledger.record(model, phase, usage, latency)
