from __future__ import annotations

import hashlib
import json
import math
import re
from collections.abc import Callable
from typing import Any, Protocol

import numpy as np
from openai import AsyncOpenAI

from .usage import UsageLedger


class Embeddings(Protocol):
    dimensions: int
    cache_key: str

    async def embed(self, texts: list[str]) -> np.ndarray: ...


class LanguageModel(Protocol):
    async def structured(
        self,
        instruction: str,
        payload: dict[str, Any],
        *,
        max_tokens: int | None = None,
        temperature: float | None = None,
    ) -> dict[str, Any]: ...


class HashEmbeddings:
    """Deterministic local embedding for demos and tests."""

    dimensions = 384
    cache_key = "hash-embeddings-v1-384"

    async def embed(self, texts: list[str]) -> np.ndarray:
        output = np.zeros((len(texts), self.dimensions), dtype=np.float32)
        for row, text in enumerate(texts):
            for token in re.findall(r"[a-z0-9]+", text.lower()):
                digest = hashlib.blake2b(token.encode(), digest_size=8).digest()
                number = int.from_bytes(digest, "little")
                output[row, number % self.dimensions] += 1 if number & 1 else -1
            norm = float(np.linalg.norm(output[row]))
            if norm:
                output[row] /= norm
        return output


class NullLanguageModel:
    async def structured(
        self, instruction: str, payload: dict[str, Any], **kwargs: Any
    ) -> dict[str, Any]:
        raise RuntimeError("No language model is configured")


class OpenRouterProvider:
    dimensions = 1536

    def __init__(
        self,
        key: str,
        llm_model: str,
        embedding_model: str,
        base_url: str = "https://openrouter.ai/api/v1",
        app_url: str | None = None,
        app_title: str = "ask-samples",
        reasoning_effort: str | None = None,
        provider_sort: str | None = None,
        provider_order: tuple[str, ...] = (),
    ):
        headers = {"X-OpenRouter-Title": app_title}
        if app_url:
            headers["HTTP-Referer"] = app_url
        self.client = AsyncOpenAI(api_key=key, base_url=base_url, default_headers=headers)
        self.llm_model = llm_model
        self.embedding_model = embedding_model
        self.reasoning_effort = reasoning_effort
        self.provider_sort = provider_sort
        # Preferred providers, tried in order with fallback. Sorting by
        # throughput picked Groq for gpt-oss-20b at 2.5x CoreWeave's price for
        # the same latency (0.7 s vs 0.5 s on a 4k-token ranking call).
        self.provider_order = tuple(provider_order)
        self.cache_key = f"openrouter:{embedding_model}"

    async def embed(
        self,
        texts: list[str],
        *,
        usage: UsageLedger | None = None,
        usage_phase: str = "embedding",
    ) -> np.ndarray:
        options = {"user": usage.query_id} if usage is not None else {}
        response = await self.client.embeddings.create(
            model=self.embedding_model, input=texts, **options
        )
        if usage is not None:
            usage.record(
                response.model or self.embedding_model,
                usage_phase,
                getattr(response, "usage", None),
            )
        matrix = np.asarray([item.embedding for item in response.data], dtype=np.float32)
        matrix /= np.maximum(np.linalg.norm(matrix, axis=1, keepdims=True), 1e-12)
        self.dimensions = matrix.shape[1]
        return matrix

    async def structured(
        self,
        instruction: str,
        payload: dict[str, Any],
        *,
        max_tokens: int | None = None,
        temperature: float | None = None,
        usage: UsageLedger | None = None,
        usage_phase: str = "language_model",
    ) -> dict[str, Any]:
        options: dict[str, Any] = {}
        if max_tokens is not None:
            options["max_tokens"] = max_tokens
        if temperature is not None:
            options["temperature"] = temperature
        if usage is not None:
            options["user"] = usage.query_id
        extra_body: dict[str, Any] = {}
        if self.reasoning_effort:
            extra_body["reasoning"] = {"effort": self.reasoning_effort}
        if self.provider_order:
            extra_body["provider"] = {"order": list(self.provider_order), "allow_fallbacks": True}
        elif self.provider_sort:
            extra_body["provider"] = {"sort": self.provider_sort}
        if extra_body:
            options["extra_body"] = extra_body
        response = await self.client.chat.completions.create(
            model=self.llm_model,
            messages=[
                {"role": "system", "content": instruction},
                {"role": "user", "content": json.dumps(payload)},
            ],
            response_format={"type": "json_object"},
            **options,
        )
        if usage is not None:
            usage.record(
                response.model or self.llm_model,
                usage_phase,
                getattr(response, "usage", None),
            )
        content = response.choices[0].message.content
        if not content:
            raise RuntimeError("OpenRouter returned an empty response")
        result = _loads_lenient(content)
        # Some otherwise-compatible models return a one-element JSON array even
        # when response_format requests an object. Accept that harmless wrapper.
        if isinstance(result, list) and len(result) == 1 and isinstance(result[0], dict):
            result = result[0]
        if not isinstance(result, dict):
            raise RuntimeError("OpenRouter returned JSON with an unsupported shape")
        return result

    async def structured_items(
        self,
        instruction: str,
        payload: dict[str, Any],
        on_item: Callable[[Any], None],
        *,
        key: str = "results",
        max_tokens: int | None = None,
        temperature: float | None = None,
        usage: UsageLedger | None = None,
        usage_phase: str = "language_model",
    ) -> dict[str, Any]:
        """structured(), streamed: call on_item for each `key` element as it completes.

        Returns the whole parsed object at the end, as structured() does, so a
        caller can still check what the model said in full.
        """
        options: dict[str, Any] = {}
        if max_tokens is not None:
            options["max_tokens"] = max_tokens
        if temperature is not None:
            options["temperature"] = temperature
        if usage is not None:
            options["user"] = usage.query_id
        extra_body: dict[str, Any] = {}
        if self.reasoning_effort:
            extra_body["reasoning"] = {"effort": self.reasoning_effort}
        if self.provider_order:
            extra_body["provider"] = {"order": list(self.provider_order), "allow_fallbacks": True}
        elif self.provider_sort:
            extra_body["provider"] = {"sort": self.provider_sort}
        if extra_body:
            options["extra_body"] = extra_body
        stream = await self.client.chat.completions.create(
            model=self.llm_model,
            messages=[
                {"role": "system", "content": instruction},
                {"role": "user", "content": json.dumps(payload)},
            ],
            response_format={"type": "json_object"},
            stream=True,
            stream_options={"include_usage": True},
            **options,
        )
        items = ArrayItemStream(key)
        content: list[str] = []
        model_name = self.llm_model
        final_usage = None
        async for chunk in stream:
            model_name = getattr(chunk, "model", None) or model_name
            if getattr(chunk, "usage", None) is not None:
                final_usage = chunk.usage
            if not chunk.choices:
                continue
            delta = chunk.choices[0].delta.content or ""
            if delta:
                content.append(delta)
                for item in items.feed(delta):
                    on_item(item)
        if usage is not None:
            usage.record(model_name, usage_phase, final_usage)
        text = "".join(content)
        if not text:
            raise RuntimeError("OpenRouter returned an empty response")
        result = json.loads(text)
        if isinstance(result, list) and len(result) == 1 and isinstance(result[0], dict):
            result = result[0]
        if not isinstance(result, dict):
            raise RuntimeError("OpenRouter returned JSON with an unsupported shape")
        return result


# One string field, as a summary or answer comes back: {"summary": "..."}.
_ONE_STRING_FIELD = re.compile(r'\s*\{\s*"(\w+)"\s*:\s*"(.*?)(?:"\s*\}?)?\s*', re.S)
_ANOTHER_FIELD = re.compile(r'"\s*,\s*"\w+"\s*:')


def _loads_lenient(content: str) -> Any:
    """json.loads, recovering a one-string-field object a model wrote carelessly.

    Gemini 2.5 Flash Lite broke 3 of 22 recipe summaries (2026-10-05) with
    malformed JSON -- an unescaped quotation mark inside the text, or a string
    left unterminated -- and the request then showed no summary at all. Only that
    shape is repaired; anything else still raises.
    """
    try:
        return json.loads(content)
    except json.JSONDecodeError:
        text = content.strip().removeprefix("```json").removeprefix("```")
        text = text.removesuffix("```").strip()
        match = _ONE_STRING_FIELD.fullmatch(text)
        if match is None or _ANOTHER_FIELD.search(match.group(2)) or not match.group(2).strip():
            raise
        value = match.group(2).replace('\\"', '"').replace("\\n", "\n")
        return {match.group(1): value.strip()}


class ArrayItemStream:
    """Pull complete objects out of a JSON response's array while it streams.

    Fed the response text as it arrives, returns each element of the
    `key` array the moment its closing brace lands, so a caller can act on the
    first verdict without waiting for the last. An element is only ever decoded
    once it is complete -- a half-written object fails to parse and is simply
    retried when more text arrives -- so nothing is reported early or twice.
    """

    def __init__(self, key: str):
        self._marker = f'"{key}"'
        self._text = ""
        self._position: int | None = None
        self._done = False
        self._decoder = json.JSONDecoder()

    def feed(self, chunk: str) -> list[Any]:
        self._text += chunk
        items: list[Any] = []
        if self._done:
            return items
        if self._position is None:
            at = self._text.find(self._marker)
            if at < 0:
                return items
            bracket = self._text.find("[", at + len(self._marker))
            if bracket < 0:
                return items
            self._position = bracket + 1
        text = self._text
        while True:
            position = self._position
            while position < len(text) and text[position] in " \t\r\n,":
                position += 1
            if position >= len(text):
                break
            if text[position] == "]":
                self._done = True
                break
            try:
                item, end = self._decoder.raw_decode(text, position)
            except json.JSONDecodeError:
                break
            items.append(item)
            self._position = end
        return items


class MeteredLanguageModel:
    def __init__(self, model: LanguageModel, usage: UsageLedger, phase: str):
        self.model = model
        self.usage = usage
        self.phase = phase

    async def structured(
        self,
        instruction: str,
        payload: dict[str, Any],
        *,
        max_tokens: int | None = None,
        temperature: float | None = None,
    ) -> dict[str, Any]:
        options = {"max_tokens": max_tokens, "temperature": temperature}
        options = {key: value for key, value in options.items() if value is not None}
        if isinstance(self.model, OpenRouterProvider):
            return await self.model.structured(
                instruction,
                payload,
                usage=self.usage,
                usage_phase=self.phase,
                **options,
            )
        return await self.model.structured(instruction, payload, **options)

    async def structured_items(
        self,
        instruction: str,
        payload: dict[str, Any],
        on_item: Callable[[Any], None],
        *,
        key: str = "results",
        max_tokens: int | None = None,
        temperature: float | None = None,
    ) -> dict[str, Any]:
        options = {"max_tokens": max_tokens, "temperature": temperature}
        options = {k: v for k, v in options.items() if v is not None}
        if isinstance(self.model, OpenRouterProvider):
            return await self.model.structured_items(
                instruction, payload, on_item, key=key,
                usage=self.usage, usage_phase=self.phase, **options,
            )
        # Models without streaming still work: items arrive all at once.
        output = await self.model.structured(instruction, payload, **options)
        for item in output.get(key, []) if isinstance(output.get(key), list) else []:
            on_item(item)
        return output


async def embed_with_usage(
    embedder: Embeddings,
    texts: list[str],
    usage: UsageLedger,
    phase: str,
) -> np.ndarray:
    if isinstance(embedder, OpenRouterProvider):
        return await embedder.embed(texts, usage=usage, usage_phase=phase)
    return await embedder.embed(texts)


def similarity_score(value: float) -> int:
    return max(0, min(100, math.floor((value + 1) * 50)))
