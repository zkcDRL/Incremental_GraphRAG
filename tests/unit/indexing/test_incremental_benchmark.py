# Copyright (C) 2026 Microsoft Corporation.
# Licensed under the MIT License

"""Tests for the incremental benchmark scoring helpers."""

import pytest

from scripts.run_incremental_benchmark import (
    aggregate_answers,
    context_recall,
    score_objective,
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
            "llm_metrics": {"prompt_tokens": 10, "completion_tokens": 2, "total_tokens": 12},
        },
        {
            "question_type": "MC",
            "latency_seconds": 3.0,
            "objective_correct": False,
            "retrieval_hits": 0,
            "retrieval_gold_count": 1,
            "llm_metrics": {"prompt_tokens": 20, "completion_tokens": 3, "total_tokens": 23},
        },
    ]
    summary = aggregate_answers(rows)["MC"]
    assert summary["objective_accuracy"] == pytest.approx(0.5)
    assert summary["retrieval_recall"] == pytest.approx(0.5)
    assert summary["mean_latency_seconds"] == pytest.approx(2.0)
    assert summary["total_tokens"] == 35
