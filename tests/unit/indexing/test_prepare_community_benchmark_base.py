# Copyright (c) 2026 Microsoft Corporation.
# Licensed under the MIT License
# ruff: noqa: SLF001, RUF029, EM101

import json
from types import SimpleNamespace

import pandas as pd
import pytest
from graphrag.index.operations.summarize_communities.community_reports_extractor import (
    CommunityReportResponse,
)
from pydantic import BaseModel

from scripts.base_sse_completion import BaseSSECompletion, TruncatedResponseError
from scripts.prepare_community_benchmark_base import fingerprint, validate_documents


@pytest.mark.parametrize(
    "bad",
    [
        "",
        "not json",
        '{"title":"only title"}',
        json.dumps({
            "title": " ",
            "summary": "summary",
            "findings": [],
            "rating": 1,
            "rating_explanation": "reason",
        }),
    ],
)
async def test_invalid_report_retries_then_caches_valid_report(model, monkeypatch, bad):
    valid = json.dumps({
        "title": "Title",
        "summary": "Summary",
        "findings": [{"summary": "Finding", "explanation": "Evidence"}],
        "rating": 1,
        "rating_explanation": "Reason",
    })
    responses = iter([bad, valid])
    calls = []

    async def read(args):
        calls.append(args)
        return next(responses), None

    async def no_delay(_):
        pass

    monkeypatch.setattr(model, "_read_stream", read)
    monkeypatch.setattr("scripts.base_sse_completion.asyncio.sleep", no_delay)
    args = {
        "messages": [{"role": "user", "content": "report"}],
        "response_format": CommunityReportResponse,
    }
    result = await model._bounded_completion(**args)
    assert result.content == valid
    assert len(calls) == 2
    assert (await model._bounded_completion(**args)).content == valid
    assert len(calls) == 2


def test_document_validation_rejects_duplicates_missing_and_changed_text():
    rows = [{"chunk_id": "a", "text": "first"}, {"chunk_id": "b", "text": "second"}]
    frame = pd.DataFrame([{"id": row["chunk_id"], "text": row["text"]} for row in rows])
    validate_documents(frame.iloc[::-1], rows)
    assert fingerprint(frame) == fingerprint(frame.iloc[::-1])
    for invalid in (
        frame.iloc[:1],
        pd.concat([frame, frame.iloc[:1]]),
        frame.assign(text="changed"),
    ):
        with pytest.raises(ValueError, match="Document IDs"):
            validate_documents(invalid, rows)


class FakeStream:
    def __init__(self, finish):
        self.finish = finish
        self.closed = False

    async def __aenter__(self):
        return self

    async def __aexit__(self, *_):
        self.closed = True

    def __aiter__(self):
        return self.chunks()

    async def chunks(self):
        for text, finish in (("complete answer", None), (None, self.finish)):
            yield SimpleNamespace(
                usage=None,
                choices=[
                    SimpleNamespace(
                        delta=SimpleNamespace(content=text),
                        finish_reason=finish,
                    )
                ],
            )


@pytest.fixture
def model(tmp_path):
    instance = object.__new__(BaseSSECompletion)
    instance.checkpoints = tmp_path
    instance._model_config = SimpleNamespace(
        call_args={},
        model="test-model",
        api_base="https://example.invalid/v1",
    )
    instance.test_streams = []

    async def create(**_):
        stream = FakeStream("stop")
        instance.test_streams.append(stream)
        return stream

    instance.client = SimpleNamespace(
        chat=SimpleNamespace(completions=SimpleNamespace(create=create))
    )
    return instance


async def test_completed_response_is_cached_without_repeating_api_call(model):
    first = await model._bounded_completion(
        messages=[{"role": "user", "content": "test"}]
    )
    second = await model._bounded_completion(
        messages=[{"role": "user", "content": "test"}]
    )
    assert first.content == second.content == "complete answer"
    assert len(model.test_streams) == 1
    assert model.test_streams[0].closed


async def test_length_retry_increases_budget_and_reuses_original_cache_key(
    model, monkeypatch
):
    budgets = []

    async def read(args):
        budgets.append(args["max_tokens"])
        if len(budgets) < 3:
            raise TruncatedResponseError("length")
        return "complete", None

    async def no_delay(_):
        pass

    monkeypatch.setattr(model, "_read_stream", read)
    monkeypatch.setattr("scripts.base_sse_completion.asyncio.sleep", no_delay)
    args = {"messages": [{"role": "user", "content": "test"}]}
    assert (await model._bounded_completion(**args)).content == "complete"
    assert budgets == [8192, 16384, 32768]
    assert (await model._bounded_completion(**args)).content == "complete"
    assert len(budgets) == 3


async def test_extraction_failure_stops_before_summarization(monkeypatch):
    from unittest.mock import AsyncMock

    from graphrag.callbacks.noop_workflow_callbacks import NoopWorkflowCallbacks
    from graphrag.config.enums import AsyncType
    from graphrag.index.workflows import extract_graph as workflow

    model = SimpleNamespace(
        completion_async=AsyncMock(side_effect=ValueError("length"))
    )
    summarize = AsyncMock()
    monkeypatch.setattr(workflow, "get_summarized_entities_relationships", summarize)
    with pytest.raises(Exception, match=r"parallel|Parallel|error"):
        await workflow.extract_graph(
            text_units=pd.DataFrame([{"id": "failed-unit", "text": "text"}]),
            callbacks=NoopWorkflowCallbacks(),
            extraction_model=model,
            extraction_prompt="{input_text}",
            entity_types=[],
            max_gleanings=0,
            extraction_num_threads=1,
            extraction_async_type=AsyncType.AsyncIO,
            summarization_model=model,
            max_summary_length=100,
            max_input_tokens=1000,
            summarization_prompt="summary",
            summarization_num_threads=1,
        )
    summarize.assert_not_called()


@pytest.mark.parametrize("finish", ["error", "length", None])
async def test_incomplete_stream_is_rejected_and_closed(model, finish):
    stream = FakeStream(finish)

    async def create(**_):
        return stream

    model.client.chat.completions.create = create
    with pytest.raises(ValueError, match=r"Incomplete SSE|without a stop"):
        await model._read_stream({})
    assert stream.closed


async def test_exhausted_retries_record_failure_and_do_not_cache(model, monkeypatch):
    async def read_stream(_):
        raise TimeoutError("test timeout")

    async def no_delay(_):
        pass

    monkeypatch.setattr(model, "_read_stream", read_stream)
    monkeypatch.setattr("scripts.base_sse_completion.asyncio.sleep", no_delay)
    monkeypatch.setattr(BaseSSECompletion, "failures", [])
    with pytest.raises(TimeoutError):
        await model._bounded_completion(messages=[{"role": "user", "content": "test"}])
    assert len(BaseSSECompletion.failures) == 1
    assert not list(model.checkpoints.glob("*.json"))


async def test_structured_response_retries_invalid_json_and_returns_typed_result(
    model, monkeypatch
):
    class Answer(BaseModel):
        answer: str

    responses = iter(["not json", '{"answer":"yes"}'])

    async def read_stream(_):
        return next(responses), None

    async def no_delay(_):
        pass

    monkeypatch.setattr(model, "_read_stream", read_stream)
    monkeypatch.setattr("scripts.base_sse_completion.asyncio.sleep", no_delay)
    model._track_metrics = False
    model._completion_async = model._bounded_completion
    response = await model.completion_async(messages="test", response_format=Answer)
    assert response.formatted_response.answer == "yes"
    assert len(list(model.checkpoints.glob("*.json"))) == 1


async def test_structured_retry_adds_json_reminder_and_accepts_fenced_json(model):
    class Answer(BaseModel):
        answer: str

    args = {"messages": [{"role": "user", "content": "original"}]}
    retry_args = model._args_for_attempt(args, Answer, attempt=2)
    assert retry_args["messages"][-1]["content"] == model._structured_retry_reminder
    assert args["messages"] == [{"role": "user", "content": "original"}]
    assert (
        model._validated_structured_content('```json\n{"answer": "yes"}\n```', Answer)
        == '{"answer": "yes"}'
    )
