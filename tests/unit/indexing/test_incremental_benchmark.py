# Copyright (C) 2026 Microsoft Corporation.
# Licensed under the MIT License

"""Tests for the incremental benchmark scoring helpers."""

import asyncio
from argparse import Namespace
from types import SimpleNamespace

import pandas as pd
import pytest

import scripts.run_community_incremental_benchmark as community_benchmark
from scripts.run_community_incremental_benchmark import normalized_mutual_information
from scripts.run_incremental_benchmark import (
    BASE_WORKFLOWS,
    QueryTimingCallbacks,
    aggregate_answers,
    context_recall,
    merge_pipeline_stats,
    next_base_workflow,
    read_stage_questions,
    recovered_base_index_metrics,
    resumed_workflows,
    score_objective,
    stage_queries_complete,
    successful_embedding_recovery,
)


def _question(question_type: str, gold_answer: str) -> dict:
    return {"question_type": question_type, "gold_answer": gold_answer}


def test_objective_scoring_formats():
    assert score_objective(_question("MC", "C"), "ANSWER: C")
    assert score_objective(_question("MS", "ACD"), "ANSWER: DCA")
    assert score_objective(_question("TF", "False"), "ANSWER: False")
    assert score_objective(_question("FB", "hash function"), "ANSWER: hash function")
    assert not score_objective(_question("MC", "C"), "ANSWER: A")


def test_context_recall_finds_gold_document_ids():
    hits, total, recall = context_recall(
        {"text_units": [{"document_id": "chunk-a"}, {"document_id": "chunk-b"}]},
        ["chunk-a", "chunk-c"],
    )
    assert (hits, total, recall) == (1, 2, 0.5)


def test_aggregate_answers_reports_tokens_accuracy_and_recall():
    rows = [
        {
            "question_type": "MC",
            "latency_seconds": 1.0,
            "objective_correct": True,
            "retrieval_hits": 1,
            "retrieval_gold_count": 1,
            "llm_metrics": {
                "prompt_tokens": 10,
                "completion_tokens": 2,
                "total_tokens": 12,
            },
        },
        {
            "question_type": "MC",
            "latency_seconds": 3.0,
            "objective_correct": False,
            "retrieval_hits": 0,
            "retrieval_gold_count": 1,
            "llm_metrics": {
                "prompt_tokens": 20,
                "completion_tokens": 3,
                "total_tokens": 23,
            },
        },
    ]
    summary = aggregate_answers(rows)["MC"]
    assert summary["objective_accuracy"] == pytest.approx(0.5)
    assert summary["retrieval_recall"] == pytest.approx(0.5)
    assert summary["mean_latency_seconds"] == pytest.approx(2.0)
    assert summary["total_tokens"] == 35


def test_resume_helpers_use_first_unfinished_workflow(tmp_path):
    stats_path = tmp_path / "stats.json"
    stats_path.write_text(
        '{"workflows": {"create_base_text_units": {}, "create_final_documents": {}}}',
        encoding="utf-8",
    )
    assert next_base_workflow(stats_path) == "extract_graph"
    assert (
        resumed_workflows(BASE_WORKFLOWS, "create_communities")[0]
        == "create_communities"
    )


def test_resume_helpers_retry_embedding_and_merge_metrics(tmp_path):
    stats_path = tmp_path / "stats.json"
    stats_path.write_text(
        '{"workflows": {'
        + ",".join(f'"{name}": {{}}' for name in BASE_WORKFLOWS)
        + "}}",
        encoding="utf-8",
    )
    assert next_base_workflow(stats_path) == "generate_text_embeddings"
    merged = merge_pipeline_stats(
        {"total_runtime": 10, "workflows": {"extract_graph": {"overall": 9}}},
        {"total_runtime": 2, "workflows": {"generate_text_embeddings": {"overall": 2}}},
    )
    assert merged["total_runtime"] == 12
    assert set(merged["workflows"]) == {"extract_graph", "generate_text_embeddings"}


def test_increment_stage_combines_old_and_new_questions(tmp_path):
    (tmp_path / "old.jsonl").write_text('{"qid": "old-1"}\n', encoding="utf-8")
    (tmp_path / "new.jsonl").write_text(
        '{"qid": "new-1"}\n{"qid": "new-2"}\n', encoding="utf-8"
    )
    questions = read_stage_questions(tmp_path, ("old.jsonl", "new.jsonl"))
    assert [item["qid"] for item in questions] == ["old-1", "new-1", "new-2"]


def test_query_resume_rejects_artifacts_with_old_question_count(tmp_path):
    answers_path = tmp_path / "answers.jsonl"
    summary_path = tmp_path / "summary.json"
    answers_path.write_text('{"qid": "old-1"}\n', encoding="utf-8")
    summary_path.write_text('{"question_count": 1}', encoding="utf-8")
    expected = [{"qid": "old-1"}, {"qid": "new-1"}]
    assert not stage_queries_complete(answers_path, summary_path, expected)

    answers_path.write_text('{"qid": "old-1"}\n{"qid": "new-1"}\n', encoding="utf-8")
    summary_path.write_text('{"question_count": 2}', encoding="utf-8")
    assert stage_queries_complete(answers_path, summary_path, expected)


def test_successful_embedding_recovery_is_a_base_index_marker(tmp_path):
    vector_store = tmp_path / "lancedb"
    marker = tmp_path / "embedding_recovery.json"
    marker.write_text(
        "{"
        '"stage":"01_base",'
        '"workflow":"generate_text_embeddings",'
        '"namespace":"benchmark",'
        f'"vector_store":"{vector_store}",'
        '"errors":[]'
        "}",
        encoding="utf-8",
    )
    recovery = successful_embedding_recovery(marker, "benchmark", vector_store)
    assert recovery is not None
    assert successful_embedding_recovery(marker, "other", vector_store) is None


def test_recovered_base_metrics_are_successful(tmp_path):
    corpus = tmp_path / "corpus.jsonl"
    corpus.write_text('{"id":"one"}\n{"id":"two"}\n', encoding="utf-8")
    stats = tmp_path / "stats.json"
    stats.write_text(
        '{"workflows":{"extract_graph":{"peak_memory_bytes":42}}}',
        encoding="utf-8",
    )
    metrics = recovered_base_index_metrics(
        recovery={"workflow": "generate_text_embeddings"},
        corpus=corpus,
        stats_path=stats,
    )
    assert metrics["document_count"] == 2
    assert metrics["recovered_from_existing_neo4j"] is True
    assert metrics["errors"] == []


def test_query_timing_callbacks_capture_nonzero_phases(monkeypatch):
    timestamps = iter([10.0, 12.0, 15.0, 16.0, 17.0])
    monkeypatch.setattr(
        "scripts.run_incremental_benchmark.time.perf_counter",
        lambda: next(timestamps),
    )
    callbacks = QueryTimingCallbacks()
    callbacks.on_context({})
    callbacks.on_llm_new_token("first")
    callbacks.on_llm_new_token("second")
    metrics = callbacks.finish()
    assert metrics["context_build_seconds"] == pytest.approx(2.0)
    assert metrics["llm_time_to_first_token_seconds"] == pytest.approx(3.0)
    assert metrics["llm_stream_seconds"] == pytest.approx(1.0)
    assert metrics["llm_total_seconds"] == pytest.approx(4.0)
    assert metrics["local_search_total_seconds"] == pytest.approx(7.0)
    assert normalized_mutual_information(
        {"a": "x", "b": "x", "c": "y"},
        {"a": "m", "b": "m", "c": "n"},
    ) == (1.0, 3)
    assert normalized_mutual_information({}, {}) == (None, 0)


def test_community_benchmark_runs_base_questions_before_updates(tmp_path, monkeypatch):
    calls = []

    async def questions(**kwargs):
        await asyncio.sleep(0)
        calls.append(("questions", kwargs))
        return [{"correct": True}]

    async def update(**kwargs):
        await asyncio.sleep(0)
        calls.append(("update", kwargs))
        return {"errors": []}

    monkeypatch.setattr(
        community_benchmark, "copy_base_tables", lambda *args, **kwargs: None
    )
    monkeypatch.setattr(community_benchmark, "run_questions", questions)
    monkeypatch.setattr(community_benchmark, "run_update", update)
    monkeypatch.setattr(
        community_benchmark, "collect_stage_nmi", lambda *args, **kwargs: (1.0, 1)
    )

    source = tmp_path / "source"
    output = tmp_path / "output"
    result = community_benchmark.orchestrate(
        Namespace(
            source_run_dir=source,
            source_namespace="source-base",
            output=output,
            namespace="comparison",
            overwrite=False,
            resume=False,
            community_level=0,
            index_concurrency=4,
            embedding_batch_size=4,
            community_embedding_batch_size=1,
            embedding_concurrency=1,
            embedding_batch_max_tokens=1024,
            vector_write_batch_size=128,
        )
    )

    assert result == 0
    assert calls[0][0] == "questions"
    assert calls[0][1]["method"] == "base"
    assert calls[0][1]["stage"] == 0
    assert calls[0][1]["namespace"] == "source-base"
    assert calls[0][1]["vector_source_dir"] == source.resolve()
    assert [call[1]["stage"] for call in calls if call[0] == "update"] == [
        1,
        1,
        2,
        2,
        3,
        3,
        4,
        4,
    ]


def test_base_copy_resume_restarts_with_server_side_clone(tmp_path, monkeypatch):
    frames = {
        "entities": pd.DataFrame([{"id": "entity-1", "title": "Entity"}]),
        "relationships": pd.DataFrame([{"id": "relationship-1"}]),
        "text_units": pd.DataFrame([{"id": "text-unit-1"}]),
        "documents": pd.DataFrame([{"id": "chunk-1", "text": "Body"}]),
        "communities": pd.DataFrame([{"id": "community-1"}]),
        "community_reports": pd.DataFrame([{"id": "report-1"}]),
    }

    class FakeNeo4jProvider:
        def __init__(self, tables, *, vector_version=None):
            self.tables = tables
            self.vector_version = vector_version
            self.writes = []
            self.projection_rebuilds = 0
            self.server_copies = 0
            self.projection_deferred = False

        async def get_active_provider(self):
            return self

        async def get_active_version(self):
            return None

        async def clear_namespace_tree(self):
            msg = "resume must not clear the namespace"
            raise AssertionError(msg)

        async def has(self, table_name):
            return table_name in self.tables

        async def row_count(self, table_name):
            return len(self.tables[table_name])

        async def read_dataframe(self, table_name):
            return self.tables[table_name].copy(deep=True)

        async def write_dataframe(self, table_name, frame, *, rebuild_projection=True):
            self.writes.append((table_name, rebuild_projection))
            self.tables[table_name] = frame.copy(deep=True)

        def defer_graph_projection(self):
            self.projection_deferred = True

        async def copy_namespace_from(self, source, *, rebuild_projection=False):
            self.server_copies += 1
            self.tables = {
                name: frame.copy(deep=True) for name, frame in source.tables.items()
            }
            return {name: len(frame) for name, frame in self.tables.items()}

        async def rebuild_graph_projection(self):
            self.projection_rebuilds += 1
            return {"RELATED": 1}

        async def close(self):
            return None

    source = FakeNeo4jProvider(
        {name: frame.copy(deep=True) for name, frame in frames.items()},
        vector_version="base-version",
    )
    destination_tables = {
        name: frame.copy(deep=True)
        for name, frame in frames.items()
        if name not in {"communities", "community_reports"}
    }
    destination_tables["communities"] = pd.DataFrame([{"id": "wrong"}])
    destination = FakeNeo4jProvider(destination_tables)

    monkeypatch.setattr(community_benchmark, "Neo4jTableProvider", FakeNeo4jProvider)
    monkeypatch.setattr(
        community_benchmark,
        "configure_run",
        lambda workspace, run_dir, namespace, metrics: SimpleNamespace(
            table_provider=namespace, output_storage=namespace
        ),
    )
    monkeypatch.setattr(community_benchmark, "create_storage", lambda value: value)
    monkeypatch.setattr(
        community_benchmark,
        "create_table_provider",
        lambda config, storage: source if config == "source-base" else destination,
    )
    monkeypatch.setattr(
        community_benchmark,
        "read_jsonl",
        lambda path: [{"chunk_id": "chunk-1", "text": "Body"}],
    )

    run_dir = tmp_path / "destination"
    community_benchmark.copy_base_tables(
        tmp_path / "source",
        run_dir,
        "source-base",
        "comparison-base",
        resume=True,
    )

    assert destination.writes == []
    assert destination.server_copies == 1
    assert destination.projection_deferred is True
    assert set(destination.tables) == set(frames)
    assert destination.projection_rebuilds == 1
    checkpoint = community_benchmark.load_json(run_dir / "base_copy_checkpoint.json")
    assert checkpoint["status"] == "complete"
    assert set(checkpoint["tables"]) == set(frames)
    assert (run_dir / "lancedb").is_dir()
