# Copyright (C) 2026 Microsoft Corporation.
# Licensed under the MIT License
# ruff: noqa: RUF029

"""Tests for aggregating forced SSE completion responses."""

from collections.abc import AsyncIterator, Iterator

import pytest
from graphrag_llm.completion.lite_llm_completion import _with_sse_aggregation
from graphrag_llm.types import LLMCompletionChunk
from graphrag_llm.utils import create_completion_response


def _chunk(content: str | None = None, *, empty_choices: bool = False):
    data = {
        "id": "chunk-id",
        "object": "chat.completion.chunk",
        "created": 0,
        "model": "test-model",
        "choices": [],
    }
    if not empty_choices:
        data["choices"] = [
            {
                "index": 0,
                "delta": {"content": content} if content is not None else {},
                "finish_reason": None,
            }
        ]
    return LLMCompletionChunk(**data)


def test_forced_sse_aggregates_sync_chunks_and_ignores_usage_chunk():
    seen = {}

    def sync_model(**kwargs) -> Iterator[LLMCompletionChunk]:
        seen.update(kwargs)
        return iter([_chunk("你"), _chunk("好"), _chunk(empty_choices=True)])

    async def async_model(**_kwargs):
        return create_completion_response("unused")

    sync, _ = _with_sse_aggregation(sync_model, async_model)
    response = sync(messages="hello")
    assert not isinstance(response, Iterator)
    assert response.content == "你好"
    assert seen["stream"] is True


@pytest.mark.asyncio
async def test_forced_sse_aggregates_async_chunks():
    seen = {}

    def sync_model(**_kwargs):
        return create_completion_response("unused")

    async def async_model(**kwargs) -> AsyncIterator[LLMCompletionChunk]:
        seen.update(kwargs)

        async def chunks():
            for chunk in (_chunk("O"), _chunk("K"), _chunk(empty_choices=True)):
                yield chunk

        return chunks()

    _, async_completion = _with_sse_aggregation(sync_model, async_model)
    response = await async_completion(messages="hello")
    assert not isinstance(response, AsyncIterator)
    assert response.content == "OK"
    assert seen["stream"] is True


def test_explicit_streaming_call_is_not_aggregated():
    def sync_model(**_kwargs):
        return iter([_chunk("stream")])

    async def async_model(**_kwargs):
        return create_completion_response("unused")

    sync, _ = _with_sse_aggregation(sync_model, async_model)
    assert isinstance(sync(messages="hello", stream=True), Iterator)
