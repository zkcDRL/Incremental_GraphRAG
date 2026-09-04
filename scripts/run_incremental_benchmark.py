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
from graphrag.callbacks.noop_query_callbacks import NoopQueryCallbacks
from graphrag.config.enums import IndexingMethod
from graphrag.config.load_config import load_config
from graphrag.data_model.data_reader import DataReader
from graphrag.index.workflows.factory import PipelineFactory
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
        ("questions/01_base_single_40.jsonl",),
    ),
    (
        "02_increment_single",
        "textbooks/02_increment_single/all_chunks.jsonl",
        (
            "questions/01_base_single_40.jsonl",
            "questions/02_increment_single_41.jsonl",
        ),
    ),
    (
        "03_increment_multi",
        "textbooks/03_increment_multi/all_chunks.jsonl",
        ("questions/03_increment_multi_20.jsonl",),
    ),
)
OBJECTIVE_TYPES = {"FB", "MC", "MS", "TF"}
BASE_WORKFLOWS = (
    "create_base_text_units",
    "create_final_documents",
    "extract_graph",
    "finalize_graph",
    "extract_covariates",
    "create_communities",
    "create_final_text_units",
    "create_community_reports",
    "generate_text_embeddings",
)


def load_json(path: Path, default: Any = None) -> Any:
    """Read JSON when present, otherwise return the supplied default."""
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else default


def successful_result(path: Path) -> bool:
    """Return whether a worker artifact exists and records no errors."""
    value = load_json(path)
    return isinstance(value, dict) and not value.get("errors")


def successful_embedding_recovery(
    path: Path, namespace: str, vector_store: Path
) -> dict[str, Any] | None:
    """Validate an embedding-only recovery as a completed base index."""
    value = load_json(path)
    if not isinstance(value, dict):
        return None
    if (
        value.get("stage") != "01_base"
        or value.get("workflow") != "generate_text_embeddings"
        or value.get("namespace") != namespace
        or value.get("errors")
    ):
        return None
    recorded_store = Path(str(value.get("vector_store", ""))).resolve()
    if recorded_store != vector_store.resolve():
        return None
    return value


def recovered_base_index_metrics(
    *,
    recovery: dict[str, Any],
    corpus: Path,
    stats_path: Path,
) -> dict[str, Any]:
    """Create benchmark index metadata for a base index recovered outside the worker."""
    stats = load_json(stats_path, {}) or {}
    return {
        "stage": "01_base",
        "kind": "full_build_recovered",
        "corpus": str(corpus),
        "document_count": document_count(corpus),
        "wall_time_seconds": None,
        "process_peak_rss_bytes": None,
        "max_workflow_peak_memory_bytes": max(
            (
                item.get("peak_memory_bytes", 0)
                for item in stats.get("workflows", {}).values()
            ),
            default=0,
        ),
        "workflow_metrics": stats.get("workflows", {}),
        "pipeline_stats": stats,
        "community_drift_state": None,
        "community_update": None,
        "recovered_from_existing_neo4j": True,
        "embedding_recovery": recovery,
        "errors": [],
    }


def merge_pipeline_stats(previous: dict[str, Any], current: dict[str, Any]) -> dict[str, Any]:
    """Preserve workflow metrics collected before a resumed pipeline run."""
    if not previous:
        return current
    merged = {**previous, **current}
    merged["workflows"] = {
        **previous.get("workflows", {}),
        **current.get("workflows", {}),
    }
    merged["total_runtime"] = previous.get("total_runtime", 0) + current.get(
        "total_runtime", 0
    )
    return merged


def resumed_workflows(workflows: list[str] | tuple[str, ...], start: str) -> list[str]:
    """Return a workflow suffix beginning at the requested resume point."""
    workflow_list = list(workflows)
    try:
        start_index = workflow_list.index(start)
    except ValueError as error:
        msg = f"Unknown workflow {start!r}; choose from {workflow_list}"
        raise ValueError(msg) from error
    return workflow_list[start_index:]


def next_base_workflow(stats_path: Path) -> str:
    """Find the first unfinished base workflow, retrying the last if all were recorded."""
    completed = set((load_json(stats_path, {}) or {}).get("workflows", {}))
    return next(
        (workflow for workflow in BASE_WORKFLOWS if workflow not in completed),
        BASE_WORKFLOWS[-1],
    )


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    """Read a UTF-8 JSON Lines file."""
    with path.open(encoding="utf-8") as file:
        return [json.loads(line) for line in file if line.strip()]


def read_stage_questions(dataset: Path, question_paths: tuple[str, ...]) -> list[dict[str, Any]]:
    """Read and combine a stage's question sets, rejecting duplicate IDs."""
    rows = [row for path in question_paths for row in read_jsonl(dataset / path)]
    qids = [str(row["qid"]) for row in rows]
    if len(qids) != len(set(qids)):
        duplicates = sorted(qid for qid in set(qids) if qids.count(qid) > 1)
        msg = f"Duplicate question IDs in stage question sets: {', '.join(duplicates)}"
        raise ValueError(msg)
    return rows


def stage_queries_complete(
    answers_path: Path,
    summary_path: Path,
    expected_questions: list[dict[str, Any]],
) -> bool:
    """Return whether saved answers exactly cover the stage's current question sets."""
    if not answers_path.exists() or not summary_path.exists():
        return False
    answers = read_jsonl(answers_path)
    expected_qids = [str(row["qid"]) for row in expected_questions]
    answer_qids = [str(row.get("qid")) for row in answers]
    summary = load_json(summary_path, {}) or {}
    return (
        answer_qids == expected_qids
        and summary.get("question_count") == len(expected_questions)
    )


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


class QueryTimingCallbacks(NoopQueryCallbacks):
    """Capture query phase boundaries without crossing Pydantic-copied dictionaries."""

    def __init__(self) -> None:
        self.started_at = time.perf_counter()
        self.context_completed_at: float | None = None
        self.first_token_at: float | None = None
        self.completed_at: float | None = None
        self.output_chunks = 0

    def on_context(self, context: Any) -> None:
        """Record when retrieval and context construction have completed."""
        self.context_completed_at = time.perf_counter()

    def on_llm_new_token(self, token: Any) -> None:
        """Record first-token latency and streamed chunk count."""
        now = time.perf_counter()
        if self.first_token_at is None:
            self.first_token_at = now
        self.completed_at = now
        self.output_chunks += 1

    def finish(self) -> dict[str, int | float]:
        """Return timing metrics based on callback-observed phase boundaries."""
        finished_at = time.perf_counter()
        context_at = self.context_completed_at or finished_at
        first_token_at = self.first_token_at or finished_at
        stream_end_at = self.completed_at or finished_at
        return {
            "context_build_seconds": context_at - self.started_at,
            "llm_time_to_first_token_seconds": first_token_at - context_at,
            "llm_stream_seconds": max(0.0, stream_end_at - first_token_at),
            "llm_total_seconds": max(0.0, stream_end_at - context_at),
            "local_search_total_seconds": finished_at - self.started_at,
            "output_chunks": self.output_chunks,
        }


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
    stats_path = run_dir / "output" / "stats.json"
    previous_stats = load_json(stats_path, {}) or {}
    if args.start_workflow:
        workflows = list(
            BASE_WORKFLOWS
            if not args.update
            else PipelineFactory.pipelines[IndexingMethod.StandardUpdate]
        )
        config.workflows = resumed_workflows(workflows, args.start_workflow)

    started = time.perf_counter()
    results = await build_index(
        config,
        method=IndexingMethod.Standard,
        is_update_run=args.update,
        input_documents=None if args.start_workflow else documents_dataframe(corpus),
        verbose=False,
    )
    elapsed = time.perf_counter() - started
    errors = [str(result.error) for result in results if result.error is not None]
    current_stats = load_json(stats_path, {}) or {}
    stats = merge_pipeline_stats(previous_stats, current_stats)
    if args.start_workflow and not errors:
        write_json(stats_path, stats)
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
    result_path = Path(args.result)
    summary_path = Path(args.summary)
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

    questions = read_stage_questions(Path(args.dataset), tuple(args.question_files))
    rows: list[dict[str, Any]] = []
    write_jsonl(result_path, rows)
    write_json(summary_path, {
        "stage": args.stage,
        "status": "running",
        "question_count": 0,
        "expected_question_count": len(questions),
        "current_qid": None,
        "process_peak_rss_bytes": peak_rss_bytes(),
    })
    print(
        f"Starting Local Search evaluation for {args.stage}: "
        f"{len(questions)} questions",
        flush=True,
    )
    try:
        for question_number, item in enumerate(questions, start=1):
            qid = str(item["qid"])
            print(
                f"[{args.stage}] question {question_number}/{len(questions)} "
                f"started: {qid}",
                flush=True,
            )
            write_json(summary_path, {
                "stage": args.stage,
                "status": "running",
                "question_count": len(rows),
                "expected_question_count": len(questions),
                "current_question_number": question_number,
                "current_qid": qid,
                "process_peak_rss_bytes": peak_rss_bytes(),
            })
            started = time.perf_counter()
            request_metrics: dict[str, Any] = {}
            timing_callbacks = QueryTimingCallbacks()
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
                callbacks=[timing_callbacks],
            )
            latency = time.perf_counter() - started
            request_metrics.update(timing_callbacks.finish())
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
            context_seconds = float(request_metrics.get("context_build_seconds", 0))
            llm_seconds = float(request_metrics.get("llm_total_seconds", 0))
            overhead_seconds = max(0.0, latency - context_seconds - llm_seconds)
            request_metrics["benchmark_total_seconds"] = latency
            request_metrics["benchmark_overhead_seconds"] = overhead_seconds
            write_jsonl(result_path, rows)
            print(
                f"[{args.stage}] question {question_number}/{len(questions)} "
                f"completed: {qid} ({latency:.2f}s total; "
                f"context={context_seconds:.2f}s, llm={llm_seconds:.2f}s, "
                f"other={overhead_seconds:.2f}s)",
                flush=True,
            )
            write_json(summary_path, {
                "stage": args.stage,
                "status": "running",
                "question_count": len(rows),
                "expected_question_count": len(questions),
                "current_question_number": question_number,
                "current_qid": None,
                "last_completed_qid": qid,
                "last_latency_seconds": latency,
                "last_context_build_seconds": context_seconds,
                "last_llm_total_seconds": llm_seconds,
                "last_llm_time_to_first_token_seconds": request_metrics.get(
                    "llm_time_to_first_token_seconds"
                ),
                "last_llm_stream_seconds": request_metrics.get("llm_stream_seconds"),
                "last_overhead_seconds": overhead_seconds,
                "process_peak_rss_bytes": peak_rss_bytes(),
            })
    finally:
        close = getattr(provider, "close", None)
        if close is not None:
            await close()
    write_jsonl(result_path, rows)
    write_json(summary_path, {
        "stage": args.stage,
        "status": "completed",
        "question_count": len(rows),
        "expected_question_count": len(questions),
        "current_qid": None,
        "process_peak_rss_bytes": peak_rss_bytes(),
    })
    return 0


def run_worker(command: list[str], result_path: Path) -> None:
    """Run a worker while streaming output to the terminal and log files."""
    stdout_path = result_path.with_suffix(".stdout.log")
    stderr_path = result_path.with_suffix(".stderr.log")
    print(f"Starting worker: {' '.join(command[2:])}", flush=True)
    with stdout_path.open("w", encoding="utf-8") as stdout_log, stderr_path.open(
        "w", encoding="utf-8"
    ) as stderr_log:
        process = subprocess.Popen(
            command,
            cwd=ROOT,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            bufsize=1,
        )

        async def stream(source, log_file, terminal) -> None:
            while line := await asyncio.to_thread(source.readline):
                log_file.write(line)
                log_file.flush()
                terminal.write(line)
                terminal.flush()

        async def stream_worker_output() -> None:
            if process.stdout is None or process.stderr is None:
                return
            await asyncio.gather(
                stream(process.stdout, stdout_log, sys.stdout),
                stream(process.stderr, stderr_log, sys.stderr),
            )

        asyncio.run(stream_worker_output())
        returncode = process.wait()
    if returncode:
        msg = f"Worker failed ({returncode}); see {stderr_path}"
        raise RuntimeError(msg)


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
    """Run selected index/query stages, reusing every successful artifact."""
    dataset, workspace = Path(args.dataset).resolve(), Path(args.workspace).resolve()
    run_dir = Path(args.output).resolve()
    if run_dir.exists() and args.overwrite:
        shutil.rmtree(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)

    stage_names = [stage for stage, _, _ in STAGES]
    start_index = stage_names.index(args.start_stage)
    stop_index = stage_names.index(args.stop_after_stage)
    if start_index > stop_index:
        msg = "--start-stage must not come after --stop-after-stage"
        raise ValueError(msg)

    script = Path(__file__).resolve()
    for index, (stage, corpus_rel, question_paths) in enumerate(STAGES):
        if index < start_index or index > stop_index:
            continue
        stage_dir = run_dir / stage
        stage_dir.mkdir(exist_ok=True)
        index_result = stage_dir / "index_metrics.json"
        if index == 0 and not successful_result(index_result):
            recovery = successful_embedding_recovery(
                stage_dir / "embedding_recovery.json",
                args.namespace,
                run_dir / "lancedb",
            )
            if recovery is not None:
                write_json(
                    index_result,
                    recovered_base_index_metrics(
                        recovery=recovery,
                        corpus=dataset / corpus_rel,
                        stats_path=run_dir / "output" / "stats.json",
                    ),
                )
                print(
                    "Recognized successful embedding recovery; reusing the complete "
                    "01_base index."
                )
        if successful_result(index_result):
            print(f"Reusing completed index: {stage}")
        else:
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
            elif run_dir.exists() and (run_dir / "output" / "stats.json").exists():
                start_workflow = next_base_workflow(run_dir / "output" / "stats.json")
                index_command.extend(["--start-workflow", start_workflow])
                print(f"Resuming {stage} from workflow: {start_workflow}")
            run_worker(index_command, index_result)

        answers_path = stage_dir / "answers.jsonl"
        query_summary_path = stage_dir / "query_process_metrics.json"
        expected_questions = read_stage_questions(dataset, question_paths)
        if stage_queries_complete(answers_path, query_summary_path, expected_questions):
            print(f"Reusing completed queries: {stage}")
        else:
            if answers_path.exists() or query_summary_path.exists():
                print(
                    f"Existing query artifacts for {stage} do not cover all "
                    f"{len(expected_questions)} questions; rerunning queries."
                )
            run_worker([
                sys.executable,
                str(script),
                "_query-worker",
                "--workspace", str(workspace),
                "--run-dir", str(run_dir),
                "--namespace", args.namespace,
                "--stage", stage,
                "--dataset", str(dataset),
                "--question-files", *question_paths,
                "--result", str(answers_path),
                "--summary", str(query_summary_path),
            ], answers_path)

    completed_stages = [
        item for item in STAGES
        if successful_result(run_dir / item[0] / "index_metrics.json")
        and (run_dir / item[0] / "answers.jsonl").exists()
        and (run_dir / item[0] / "query_process_metrics.json").exists()
    ]
    all_answers: list[dict[str, Any]] = []
    stage_summaries = []
    for stage, _, _ in completed_stages:
        stage_dir = run_dir / stage
        answers = read_jsonl(stage_dir / "answers.jsonl")
        all_answers.extend(answers)
        stage_summaries.append({
            "stage": stage,
            "index": load_json(stage_dir / "index_metrics.json", {}),
            "index_llm": collect_token_metrics(run_dir / "metrics" / stage / "index"),
            "query_process": load_json(stage_dir / "query_process_metrics.json", {}),
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
    run.add_argument("--overwrite", action="store_true", help="Delete all prior results and start over")
    run.add_argument(
        "--start-stage", choices=[stage for stage, _, _ in STAGES], default=STAGES[0][0]
    )
    run.add_argument(
        "--stop-after-stage", choices=[stage for stage, _, _ in STAGES], default=STAGES[-1][0]
    )

    score = subparsers.add_parser("score-open", help="Apply completed OE judgments")
    score.add_argument("--run-dir", required=True)
    score.add_argument("--judgments", required=True)

    index = subparsers.add_parser("_index-worker")
    for name in ("workspace", "run-dir", "namespace", "stage", "corpus", "result"):
        index.add_argument(f"--{name}", required=True)
    index.add_argument("--update", action="store_true")
    index.add_argument("--start-workflow")

    query = subparsers.add_parser("_query-worker")
    for name in ("workspace", "run-dir", "namespace", "stage", "dataset", "result", "summary"):
        query.add_argument(f"--{name}", required=True)
    query.add_argument("--question-files", nargs="+", required=True)
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
