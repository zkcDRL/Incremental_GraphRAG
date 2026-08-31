#!/usr/bin/env python3
# Copyright (c) 2026 Microsoft Corporation.
# Licensed under the MIT License
# ruff: noqa: S404, S603, T201, EM102, TRY003
"""Run the three-stage GraphRAG-Bench incremental Local Search benchmark."""

from __future__ import annotations

import argparse
import asyncio
import json
import re
import resource
import shutil
import subprocess
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import pandas as pd
from graphrag.api.index import build_index
from graphrag.api.query import local_search
from graphrag.config.enums import IndexingMethod
from graphrag.config.load_config import load_config
from graphrag.data_model.data_reader import DataReader
from graphrag_llm.config.metrics_config import MetricsConfig
from graphrag_storage import create_storage
from graphrag_storage.tables.neo4j_table_provider import Neo4jTableProvider
from graphrag_storage.tables.table_provider_factory import create_table_provider

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATASET = ROOT / "GraphRAG-Bench-incremental-101"
DEFAULT_WORKSPACE = ROOT / "rag_workspace"
STAGES = (
    (
        "01_base",
        "textbooks/01_base/compact/03_neighbor_records_pm2_851.jsonl",
        "questions/01_base_single_40.jsonl",
    ),
    (
        "02_increment_single",
        "textbooks/02_increment_single/all_chunks.jsonl",
        "questions/02_increment_single_41.jsonl",
    ),
    (
        "03_increment_multi",
        "textbooks/03_increment_multi/all_chunks.jsonl",
        "questions/03_increment_multi_20.jsonl",
    ),
)
OBJECTIVE_TYPES = {"FB", "MC", "MS", "TF"}


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    """Read a UTF-8 JSON Lines file."""
    with path.open(encoding="utf-8") as file:
        return [json.loads(line) for line in file if line.strip()]


def write_json(path: Path, value: Any) -> None:
    """Write formatted JSON, creating parent directories."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding="utf-8")


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    """Write records as UTF-8 JSON Lines."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        for row in rows:
            file.write(json.dumps(row, ensure_ascii=False) + "\n")


def configure_run(
    workspace: Path,
    run_dir: Path,
    namespace: str,
    metrics_dir: Path,
):
    """Load user models while isolating benchmark state and storage."""
    config = load_config(root_dir=workspace)
    config.output_storage.base_dir = str((run_dir / "output").resolve())
    config.update_output_storage.base_dir = str((run_dir / "update_output").resolve())
    config.reporting.base_dir = str((run_dir / "logs").resolve())
    if config.cache.storage is not None:
        config.cache.storage.base_dir = str((run_dir / "cache").resolve())
    config.vector_store.db_uri = str((run_dir / "lancedb").resolve())
    config.table_provider.namespace = namespace  # type: ignore[attr-defined]
    for model in (*config.completion_models.values(), *config.embedding_models.values()):
        model.metrics = MetricsConfig(writer="file", base_dir=str(metrics_dir.resolve()))
    return config


def documents_dataframe(corpus_path: Path) -> pd.DataFrame:
    """Convert benchmark chunks to GraphRAG input documents without reformatting text."""
    rows = []
    for index, item in enumerate(read_jsonl(corpus_path)):
        chunk_id = str(item.get("chunk_id") or item["id"])
        rows.append({
            "id": chunk_id,
            "human_readable_id": index,
            "title": chunk_id,
            "text": str(item["text"]),
            "creation_date": "",
            "raw_data": {
                "chunk_id": chunk_id,
                "source_record_id": item.get("source_record_id"),
                "textbook_id": item.get("textbook_id"),
                "experiment_stage": item.get("experiment_stage"),
            },
        })
    return pd.DataFrame(rows)


def document_count(corpus_path: Path) -> int:
    """Count non-empty JSONL records without retaining them."""
    with corpus_path.open(encoding="utf-8") as file:
        return sum(bool(line.strip()) for line in file)


def peak_rss_bytes() -> int:
    """Return this process's peak resident memory on Linux."""
    return int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024)


def collect_token_metrics(metrics_dir: Path) -> dict[str, int | float]:
    """Aggregate token counters emitted by GraphRAG LLM metrics writers."""
    totals: defaultdict[str, int | float] = defaultdict(int)
    for path in metrics_dir.glob("*.jsonl"):
        for row in read_jsonl(path):
            metrics = row.get("metrics", {})
            for key in (
                "prompt_tokens",
                "completion_tokens",
                "total_tokens",
                "successful_response_count",
                "failed_response_count",
                "total_cost",
            ):
                value = metrics.get(key)
                if isinstance(value, (int, float)):
                    totals[key] += value
    return dict(totals)


def format_question(item: dict[str, Any]) -> str:
    """Format a benchmark question with strict objective-answer instructions."""
    question = str(item["question"])
    choices = item.get("choices")
    if choices:
        question += "\n" + "\n".join(f"{key}. {value}" for key, value in choices.items())
    qtype = item["question_type"]
    instructions = {
        "MC": "Return only `ANSWER: X`, where X is one option letter.",
        "MS": "Return only `ANSWER: XYZ`, with all correct option letters sorted.",
        "TF": "Return only `ANSWER: True` or `ANSWER: False`.",
        "FB": "Return only `ANSWER: <filled text>` without explanation.",
        "OE": "Answer concisely using only supported evidence and include enough detail for manual judging.",
    }
    return f"{question}\n\n{instructions.get(qtype, instructions['OE'])}"


def json_safe(value: Any) -> Any:
    """Convert DataFrames and non-JSON values from query context."""
    if isinstance(value, pd.DataFrame):
        return value.to_dict("records")
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    if hasattr(value, "item"):
        return value.item()
    return value


def context_recall(context: Any, gold_ids: list[str]) -> tuple[int, int, float]:
    """Measure gold document-ID recall from the Local Search context payload."""
    serialized = json.dumps(json_safe(context), ensure_ascii=False)
    hits = sum(gold_id in serialized for gold_id in gold_ids)
    total = len(gold_ids)
    return hits, total, hits / total if total else 0.0


def normalize_text(value: str) -> str:
    """Normalize answer text for deterministic scoring."""
    value = re.sub(r"^\s*answer\s*:\s*", "", value, flags=re.IGNORECASE)
    return re.sub(r"[^a-z0-9]+", " ", value.lower()).strip()


def score_objective(item: dict[str, Any], response: str) -> bool:
    """Score FB/MC/MS/TF answers without an LLM judge."""
    qtype = item["question_type"]
    gold = str(item["gold_answer"])
    normalized = normalize_text(response)
    if qtype == "MC":
        letters = re.findall(r"\b[A-D]\b", response.upper())
        return bool(letters) and letters[0] == gold.upper()
    if qtype == "MS":
        letters = set(re.findall(r"[A-D]", response.upper().split("ANSWER:")[-1]))
        return letters == set(gold.upper())
    if qtype == "TF":
        match = re.search(r"\b(true|false)\b", normalized)
        return bool(match) and match.group(1) == normalize_text(gold)
    gold_normalized = normalize_text(gold)
    return normalized == gold_normalized or gold_normalized in normalized


async def active_provider(config):
    """Create the query-visible provider, resolving the active Neo4j version."""
    provider = create_table_provider(
        config.table_provider, create_storage(config.output_storage)
    )
    if isinstance(provider, Neo4jTableProvider):
        return await provider.get_active_provider()
    return provider


async def index_worker(args: argparse.Namespace) -> int:
    """Execute one isolated index stage and emit raw measurements."""
    workspace, run_dir = Path(args.workspace), Path(args.run_dir)
    metrics_dir = run_dir / "metrics" / args.stage / "index"
    config = configure_run(workspace, run_dir, args.namespace, metrics_dir)
    corpus = Path(args.corpus)
    started = time.perf_counter()
    results = await build_index(
        config,
        method=IndexingMethod.Standard,
        is_update_run=args.update,
        input_documents=documents_dataframe(corpus),
        verbose=False,
    )
    elapsed = time.perf_counter() - started
    errors = [str(result.error) for result in results if result.error is not None]
    stats_path = run_dir / "output" / "stats.json"
    stats = json.loads(stats_path.read_text(encoding="utf-8")) if stats_path.exists() else {}
    context_path = run_dir / "output" / "context.json"
    context = (
        json.loads(context_path.read_text(encoding="utf-8"))
        if context_path.exists()
        else {}
    )
    workflows = stats.get("workflows", {})
    write_json(Path(args.result), {
        "stage": args.stage,
        "kind": "incremental_update" if args.update else "full_build",
        "corpus": str(corpus),
        "document_count": document_count(corpus),
        "wall_time_seconds": elapsed,
        "process_peak_rss_bytes": peak_rss_bytes(),
        "max_workflow_peak_memory_bytes": max(
            (item.get("peak_memory_bytes", 0) for item in workflows.values()), default=0
        ),
        "workflow_metrics": workflows,
        "pipeline_stats": stats,
        "community_drift_state": context.get("community_drift_state"),
        "community_update": (context.get("community_update_history") or [None])[-1],
        "errors": errors,
    })
    return 1 if errors else 0


async def query_worker(args: argparse.Namespace) -> int:
    """Run one stage's Local Search questions and save answers/context evidence."""
    workspace, run_dir = Path(args.workspace), Path(args.run_dir)
    metrics_dir = run_dir / "metrics" / args.stage / "query"
    config = configure_run(workspace, run_dir, args.namespace, metrics_dir)
    provider = await active_provider(config)
    reader = DataReader(provider)
    graph_loader = (
        provider.read_local_search_context
        if isinstance(provider, Neo4jTableProvider)
        else None
    )
    vector_version = getattr(provider, "vector_version", None)
    if graph_loader:
        entities = await reader.entities()
        communities = community_reports = text_units = relationships = pd.DataFrame()
        covariates = None
    else:
        entities = await reader.entities()
        communities = await reader.communities()
        community_reports = await reader.community_reports()
        text_units = await reader.text_units()
        relationships = await reader.relationships()
        covariates = await reader.covariates() if await provider.has("covariates") else None

    rows: list[dict[str, Any]] = []
    try:
        for item in read_jsonl(Path(args.questions)):
            started = time.perf_counter()
            request_metrics: dict[str, Any] = {}
            response_value, context = await local_search(
                config=config,
                entities=entities,
                communities=communities,
                community_reports=community_reports,
                text_units=text_units,
                relationships=relationships,
                covariates=covariates,
                community_level=2,
                response_type="A concise answer",
                query=format_question(item),
                graph_context_loader=graph_loader,
                vector_store_version=vector_version,
                metrics=request_metrics,
            )
            latency = time.perf_counter() - started
            response = (
                response_value
                if isinstance(response_value, str)
                else json.dumps(response_value, ensure_ascii=False)
            )
            gold_ids = [str(value) for value in item.get("gold_chunk_ids", [])]
            hits, total, recall = context_recall(context, gold_ids)
            qtype = item["question_type"]
            rows.append({
                "stage": args.stage,
                "qid": item["qid"],
                "question_type": qtype,
                "question": item["question"],
                "choices": item.get("choices"),
                "gold_answer": item.get("gold_answer"),
                "gold_rationale": item.get("gold_rationale"),
                "gold_chunk_ids": gold_ids,
                "response": response,
                "latency_seconds": latency,
                "llm_metrics": request_metrics,
                "retrieval_hits": hits,
                "retrieval_gold_count": total,
                "retrieval_recall": recall,
                "objective_correct": (
                    score_objective(item, response) if qtype in OBJECTIVE_TYPES else None
                ),
                "context": json_safe(context),
            })
    finally:
        close = getattr(provider, "close", None)
        if close is not None:
            await close()
    write_jsonl(Path(args.result), rows)
    write_json(Path(args.summary), {
        "stage": args.stage,
        "question_count": len(rows),
        "process_peak_rss_bytes": peak_rss_bytes(),
    })
    return 0


def run_worker(command: list[str], result_path: Path) -> None:
    """Run a measured worker process and fail with its captured logs."""
    completed = subprocess.run(command, cwd=ROOT, text=True, capture_output=True, check=False)
    result_path.with_suffix(".stdout.log").write_text(completed.stdout, encoding="utf-8")
    result_path.with_suffix(".stderr.log").write_text(completed.stderr, encoding="utf-8")
    if completed.returncode:
        raise RuntimeError(
            f"Worker failed ({completed.returncode}); see {result_path.with_suffix('.stderr.log')}"
        )


def aggregate_answers(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Aggregate latency, objective accuracy, and retrieval recall by question type."""
    groups: defaultdict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[row["question_type"]].append(row)
    result: dict[str, Any] = {}
    for qtype, items in sorted(groups.items()):
        objective = [item["objective_correct"] for item in items if item["objective_correct"] is not None]
        result[qtype] = {
            "count": len(items),
            "mean_latency_seconds": sum(item["latency_seconds"] for item in items) / len(items),
            "p95_latency_seconds": sorted(item["latency_seconds"] for item in items)[
                max(0, int(0.95 * len(items)) - 1)
            ],
            "prompt_tokens": sum(
                item.get("llm_metrics", {}).get("prompt_tokens", 0) for item in items
            ),
            "completion_tokens": sum(
                item.get("llm_metrics", {}).get("completion_tokens", 0)
                for item in items
            ),
            "total_tokens": sum(
                item.get("llm_metrics", {}).get("total_tokens", 0) for item in items
            ),
            "objective_accuracy": (
                sum(objective) / len(objective) if objective else None
            ),
            "retrieval_recall": sum(item["retrieval_hits"] for item in items)
            / max(1, sum(item["retrieval_gold_count"] for item in items)),
        }
    return result


def orchestrate(args: argparse.Namespace) -> int:
    """Run all three index/query stages and prepare final scoring artifacts."""
    dataset, workspace = Path(args.dataset).resolve(), Path(args.workspace).resolve()
    run_dir = Path(args.output).resolve()
    if run_dir.exists():
        if not args.overwrite:
            raise FileExistsError(f"Output already exists: {run_dir}; pass --overwrite")
        shutil.rmtree(run_dir)
    run_dir.mkdir(parents=True)

    script = Path(__file__).resolve()
    all_answers: list[dict[str, Any]] = []
    stage_summaries = []
    for index, (stage, corpus_rel, questions_rel) in enumerate(STAGES):
        stage_dir = run_dir / stage
        stage_dir.mkdir()
        index_result = stage_dir / "index_metrics.json"
        index_command = [
            sys.executable,
            str(script),
            "_index-worker",
            "--workspace", str(workspace),
            "--run-dir", str(run_dir),
            "--namespace", args.namespace,
            "--stage", stage,
            "--corpus", str(dataset / corpus_rel),
            "--result", str(index_result),
        ]
        if index:
            index_command.append("--update")
        run_worker(index_command, index_result)

        answers_path = stage_dir / "answers.jsonl"
        query_summary_path = stage_dir / "query_process_metrics.json"
        run_worker([
            sys.executable,
            str(script),
            "_query-worker",
            "--workspace", str(workspace),
            "--run-dir", str(run_dir),
            "--namespace", args.namespace,
            "--stage", stage,
            "--questions", str(dataset / questions_rel),
            "--result", str(answers_path),
            "--summary", str(query_summary_path),
        ], answers_path)
        answers = read_jsonl(answers_path)
        all_answers.extend(answers)
        index_metrics = json.loads(index_result.read_text(encoding="utf-8"))
        query_process = json.loads(query_summary_path.read_text(encoding="utf-8"))
        stage_summaries.append({
            "stage": stage,
            "index": index_metrics,
            "index_llm": collect_token_metrics(run_dir / "metrics" / stage / "index"),
            "query_process": query_process,
            "query_llm": collect_token_metrics(run_dir / "metrics" / stage / "query"),
            "questions_by_type": aggregate_answers(answers),
        })

    open_rows = [row for row in all_answers if row["question_type"] == "OE"]
    write_jsonl(run_dir / "open_questions_for_manual_judging.jsonl", [
        {
            "qid": row["qid"],
            "stage": row["stage"],
            "question": row["question"],
            "gold_answer": row["gold_answer"],
            "gold_rationale": row["gold_rationale"],
            "model_response": row["response"],
            "manual_correct": None,
            "manual_score_0_to_1": None,
            "judge_reason": None,
        }
        for row in open_rows
    ])
    write_json(run_dir / "benchmark_summary.json", {
        "dataset": str(dataset),
        "workspace": str(workspace),
        "namespace": args.namespace,
        "search_mode": "local",
        "stages": stage_summaries,
        "objective_questions": sum(row["question_type"] in OBJECTIVE_TYPES for row in all_answers),
        "open_questions_pending_manual_judgment": len(open_rows),
        "overall_by_type": aggregate_answers(all_answers),
    })
    print(f"Benchmark complete: {run_dir / 'benchmark_summary.json'}")
    print(f"Manual OE judging file: {run_dir / 'open_questions_for_manual_judging.jsonl'}")
    return 0


def apply_judgments(args: argparse.Namespace) -> int:
    """Merge human/assistant OE judgments and compute final accuracy by type."""
    run_dir = Path(args.run_dir).resolve()
    judgments = {row["qid"]: row for row in read_jsonl(Path(args.judgments))}
    answers = [
        row
        for stage, _, _ in STAGES
        for row in read_jsonl(run_dir / stage / "answers.jsonl")
    ]
    missing = [
        row["qid"] for row in answers
        if row["question_type"] == "OE"
        and judgments.get(row["qid"], {}).get("manual_correct") is None
    ]
    if missing:
        raise ValueError(f"Missing manual_correct judgments for: {', '.join(missing)}")
    for row in answers:
        if row["question_type"] == "OE":
            judgment = judgments[row["qid"]]
            row["objective_correct"] = bool(judgment["manual_correct"])
            row["manual_score_0_to_1"] = judgment.get("manual_score_0_to_1")
            row["judge_reason"] = judgment.get("judge_reason")
    final = {
        "questions_by_type": aggregate_answers(answers),
        "overall_accuracy": sum(bool(row["objective_correct"]) for row in answers)
        / len(answers),
        "question_count": len(answers),
    }
    write_json(run_dir / "final_scored_summary.json", final)
    print(f"Final score written: {run_dir / 'final_scored_summary.json'}")
    return 0


def parser() -> argparse.ArgumentParser:
    """Build the public and worker command-line interface."""
    result = argparse.ArgumentParser(description=__doc__)
    subparsers = result.add_subparsers(dest="command", required=True)
    run = subparsers.add_parser("run", help="Run all index and Local Search stages")
    run.add_argument("--dataset", default=str(DEFAULT_DATASET))
    run.add_argument("--workspace", default=str(DEFAULT_WORKSPACE))
    run.add_argument("--output", default=str(DEFAULT_WORKSPACE / "benchmark_results" / "incremental-101"))
    run.add_argument("--namespace", default="graphrag-bench-incremental-101")
    run.add_argument("--overwrite", action="store_true")

    score = subparsers.add_parser("score-open", help="Apply completed OE judgments")
    score.add_argument("--run-dir", required=True)
    score.add_argument("--judgments", required=True)

    index = subparsers.add_parser("_index-worker")
    for name in ("workspace", "run-dir", "namespace", "stage", "corpus", "result"):
        index.add_argument(f"--{name}", required=True)
    index.add_argument("--update", action="store_true")

    query = subparsers.add_parser("_query-worker")
    for name in ("workspace", "run-dir", "namespace", "stage", "questions", "result", "summary"):
        query.add_argument(f"--{name}", required=True)
    return result


def main() -> int:
    """Dispatch public orchestration or internal worker commands."""
    args = parser().parse_args()
    if args.command == "run":
        return orchestrate(args)
    if args.command == "score-open":
        return apply_judgments(args)
    if args.command == "_index-worker":
        return asyncio.run(index_worker(args))
    return asyncio.run(query_worker(args))


if __name__ == "__main__":
    raise SystemExit(main())
