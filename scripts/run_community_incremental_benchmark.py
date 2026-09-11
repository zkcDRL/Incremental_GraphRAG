#!/usr/bin/env python3
# Copyright (c) 2026 Microsoft Corporation.
# Licensed under the MIT License
# ruff: noqa: S404, S603, T201, EM102, TRY003
"""Compare adaptive incremental updates with full community rebuilds."""

from __future__ import annotations

import argparse
import asyncio
import csv
import hashlib
import json
import math
import shutil
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd
from graphrag.api.index import build_index
from graphrag.api.query import local_search
from graphrag.config.enums import AsyncType, IndexingMethod
from graphrag.data_model.data_reader import DataReader
from graphrag_llm.completion.completion_factory import register_completion
from graphrag_storage import create_storage
from graphrag_storage.tables.neo4j_table_provider import Neo4jTableProvider
from graphrag_storage.tables.table_provider_factory import create_table_provider

from scripts.base_sse_completion import BaseSSECompletion
from scripts.prepare_community_benchmark_base import validate_documents
from scripts.run_incremental_benchmark import (
    QueryTimingCallbacks,
    configure_run,
    context_recall,
    documents_dataframe,
    json_safe,
    load_json,
    score_objective,
    write_json,
    write_jsonl,
)

ROOT = Path(__file__).resolve().parents[1]
DATASET = ROOT / "GraphRAG-Bench-community-incremental-v2"
WORKSPACE = ROOT / "rag_workspace"
DEFAULT_SOURCE_NAMESPACE = "graphrag-bench-incremental-164"


def positive_int(value: str) -> int:
    """Parse a positive command-line integer."""
    parsed = int(value)
    if parsed < 1:
        msg = "value must be a positive integer"
        raise argparse.ArgumentTypeError(msg)
    return parsed


@dataclass(frozen=True)
class StageDefinition:
    """Runtime paths for one cumulative comparison stage."""

    stage: int
    delta_path: Path
    question_path: Path
    cumulative_corpus: tuple[Path, ...]


_CORPUS_NAMES = (
    "00_base_1800.jsonl",
    "01_micro_local_30.jsonl",
    "02_near_region_30.jsonl",
    "03_far_existing_books_30.jsonl",
    "04_new_textbook_regions_30.jsonl",
)
STAGES = tuple(
    StageDefinition(
        stage=index,
        delta_path=DATASET / "corpus" / _CORPUS_NAMES[index],
        question_path=DATASET / "questions" / "cumulative" / f"stage_{index}.jsonl",
        cumulative_corpus=tuple(
            DATASET / "corpus" / name for name in _CORPUS_NAMES[: index + 1]
        ),
    )
    for index in range(1, 5)
)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    """Read JSON Lines."""
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def dataframe_fingerprint(frame: pd.DataFrame) -> dict[str, Any]:
    """Return a stable fingerprint for a Neo4j table round trip."""
    payload = frame.to_json(orient="records", date_format="iso", force_ascii=False)
    return {
        "rows": len(frame),
        "columns": list(frame.columns),
        "sha256": hashlib.sha256(payload.encode("utf-8")).hexdigest(),
    }


def question_prompt(item: dict[str, Any]) -> str:
    """Build the question without exposing any gold fields."""
    prompt = str(item["question"])
    if item.get("choices"):
        prompt += "\n" + "\n".join(
            f"{key}. {value}" for key, value in item["choices"].items()
        )
    instructions = {
        "FB": "Return only the filled answer.",
        "MC": "Return only the answer option letter.",
        "MS": "Return only all correct option letters.",
        "TF": "Return only True or False.",
        "OE": "Answer concisely using only the supplied evidence.",
    }
    return f"{prompt}\n\n{instructions.get(item['question_type'], instructions['OE'])}"


def community_assignments(frame: pd.DataFrame, level: int = 0) -> dict[str, str]:
    """Map entity IDs to community IDs at one hierarchy level."""
    assignments: dict[str, str] = {}
    if frame.empty:
        return assignments
    for _, row in frame.iterrows():
        if int(row.get("level", -1)) == level:
            assignments.update({
                str(entity_id): str(row["community"])
                for entity_id in row.get("entity_ids", [])
            })
    return assignments


def normalized_mutual_information(
    first: dict[str, str], second: dict[str, str]
) -> tuple[float | None, int]:
    """Calculate NMI from two categorical assignments without sklearn."""
    common = sorted(set(first) & set(second))
    if not common:
        return None, 0
    first_counts: dict[str, int] = {}
    second_counts: dict[str, int] = {}
    joint_counts: dict[tuple[str, str], int] = {}
    for entity_id in common:
        a, b = first[entity_id], second[entity_id]
        first_counts[a] = first_counts.get(a, 0) + 1
        second_counts[b] = second_counts.get(b, 0) + 1
        joint_counts[a, b] = joint_counts.get((a, b), 0) + 1
    total = len(common)
    mutual_information = 0.0
    for (a, b), count in joint_counts.items():
        mutual_information += (count / total) * math.log(
            count * total / (first_counts[a] * second_counts[b])
        )
    entropy_first = -sum(
        count / total * math.log(count / total) for count in first_counts.values()
    )
    entropy_second = -sum(
        count / total * math.log(count / total) for count in second_counts.values()
    )
    denominator = (entropy_first + entropy_second) / 2
    return (mutual_information / denominator if denominator else 1.0), total


def collect_stage_nmi(
    incremental_dir: Path,
    incremental_namespace: str,
    full_dir: Path,
    full_namespace: str,
    level: int = 0,
) -> tuple[float | None, int]:
    """Compare active community assignments for one stage."""

    async def read(namespace_dir: Path, namespace: str) -> pd.DataFrame:
        config = configure_run(
            WORKSPACE,
            namespace_dir,
            namespace,
            namespace_dir / "metrics" / "nmi",
        )
        provider = create_table_provider(
            config.table_provider, create_storage(config.output_storage)
        )
        if not isinstance(provider, Neo4jTableProvider):
            msg = "NMI comparison requires Neo4j"
            raise TypeError(msg)
        try:
            active = await provider.get_active_provider()
            return await active.read_dataframe("communities")
        finally:
            await provider.close()

    incremental = asyncio.run(read(incremental_dir, incremental_namespace))
    full = asyncio.run(read(full_dir, full_namespace))
    return normalized_mutual_information(
        community_assignments(incremental, level),
        community_assignments(full, level),
    )


def collect_report_metrics(metrics_dir: Path) -> dict[str, int | float]:
    """Aggregate only completion-model metrics emitted during report updates."""
    totals: dict[str, int | float] = {
        "report_llm_calls": 0,
        "report_input_tokens": 0,
        "report_output_tokens": 0,
        "report_total_tokens": 0,
    }
    for path in metrics_dir.glob("*.jsonl"):
        for row in read_jsonl(path):
            if "completion" not in str(row.get("id", "")):
                continue
            metrics = row.get("metrics", {})
            totals["report_llm_calls"] += metrics.get("successful_response_count", 0)
            totals["report_input_tokens"] += metrics.get("prompt_tokens", 0)
            totals["report_output_tokens"] += metrics.get("completion_tokens", 0)
            totals["report_total_tokens"] += metrics.get("total_tokens", 0)
    return totals


def copy_base_tables(
    source: Path,
    destination: Path,
    source_namespace: str,
    destination_namespace: str,
    *,
    resume: bool = False,
) -> None:
    """Copy Base tables, validating and skipping completed work when resumed."""
    print(f"Copying base data from namespace: {source_namespace}", flush=True)
    print(f"Preparing destination namespace: {destination_namespace}", flush=True)
    source_config = configure_run(
        WORKSPACE, source, source_namespace, source / "metrics" / "copy"
    )
    destination_config = configure_run(
        WORKSPACE,
        destination,
        destination_namespace,
        destination / "metrics" / "copy",
    )
    source_provider = create_table_provider(
        source_config.table_provider, create_storage(source_config.output_storage)
    )
    destination_provider = create_table_provider(
        destination_config.table_provider,
        create_storage(destination_config.output_storage),
    )
    if not isinstance(source_provider, Neo4jTableProvider) or not isinstance(
        destination_provider, Neo4jTableProvider
    ):
        msg = "Both comparison providers must be Neo4j providers"
        raise TypeError(msg)

    checkpoint_path = destination / "base_copy_checkpoint.json"

    async def copy() -> None:
        source_active = await source_provider.get_active_provider()
        source_documents = await source_active.read_dataframe("documents")
        validate_documents(
            source_documents, read_jsonl(DATASET / "corpus" / _CORPUS_NAMES[0])
        )
        source_version = source_active.vector_version or "legacy"
        source_identity = {
            "source_namespace": source_namespace,
            "source_version": source_version,
            "base_file": _CORPUS_NAMES[0],
            "documents": dataframe_fingerprint(source_documents),
        }
        checkpoint = load_json(checkpoint_path, {}) if resume else {}
        if (
            resume
            and checkpoint.get("status") == "complete"
            and checkpoint.get("source") == source_identity
            and checkpoint.get("destination_namespace") == destination_namespace
        ):
            print(
                f"[{destination_namespace}] Base copy checkpoint is complete; "
                "skipping all tables and graph projection.",
                flush=True,
            )
            return

        if resume:
            active_namespace = await destination_provider.get_active_version()
            if active_namespace not in {None, destination_namespace}:
                msg = (
                    f"[{destination_namespace}] cannot resume Base copy because an "
                    f"updated version is already active: {active_namespace}"
                )
                raise RuntimeError(msg)
            print(
                f"[{destination_namespace}] resume enabled; validating existing tables...",
                flush=True,
            )
        else:
            print(
                f"[{destination_namespace}] clearing destination namespace...",
                flush=True,
            )
            await destination_provider.clear_namespace_tree()

        checkpoint = {
            "status": "copying",
            "source": source_identity,
            "destination_namespace": destination_namespace,
            "tables": {},
            "projection": {},
        }
        write_json(checkpoint_path, checkpoint)
        table_names = (
            "entities",
            "relationships",
            "text_units",
            "documents",
            "communities",
            "community_reports",
        )
        destination_provider.defer_graph_projection()
        print(
            f"[{destination_namespace}] copying all tables inside Neo4j...",
            flush=True,
        )
        copied_counts = await destination_provider.copy_namespace_from(
            source_active,
            rebuild_projection=False,
        )
        missing_tables = set(table_names) - copied_counts.keys()
        if missing_tables:
            msg = (
                f"[{destination_namespace}] server-side copy omitted tables: "
                f"{sorted(missing_tables)}"
            )
            raise RuntimeError(msg)
        for table_number, table_name in enumerate(table_names, start=1):
            row_count = copied_counts[table_name]
            checkpoint["tables"][table_name] = {"rows": row_count}
            print(
                f"[{destination_namespace}] copied table "
                f"{table_number}/{len(table_names)}: {table_name}: {row_count} rows",
                flush=True,
            )
        write_json(checkpoint_path, checkpoint)
        print(f"[{destination_namespace}] rebuilding graph projection...", flush=True)
        counts = await destination_provider.rebuild_graph_projection()
        checkpoint["projection"] = counts
        print(
            f"[{destination_namespace}] base namespace ready; projection edges: {counts}",
            flush=True,
        )
        destination_vectors = destination / "lancedb"
        if destination_vectors.exists():
            shutil.rmtree(destination_vectors)
        destination_vectors.mkdir(parents=True, exist_ok=True)
        checkpoint["status"] = "complete"
        write_json(checkpoint_path, checkpoint)
        print(
            f"[{destination_namespace}] vector indexes will be rebuilt by the update run",
            flush=True,
        )

    async def copy_with_cleanup() -> None:
        try:
            await copy()
        finally:
            await source_provider.close()
            await destination_provider.close()

    asyncio.run(copy_with_cleanup())


async def run_update(
    *,
    method: str,
    stage: int,
    delta_path: Path,
    run_dir: Path,
    namespace: str,
    index_concurrency: int,
    embedding_batch_size: int,
    community_embedding_batch_size: int,
    embedding_concurrency: int,
    embedding_batch_max_tokens: int,
    vector_write_batch_size: int,
) -> dict[str, Any]:
    """Run one incremental update and return measured metadata."""
    metrics_dir = run_dir / "metrics" / f"stage_{stage}" / "index"
    config = configure_run(WORKSPACE, run_dir, namespace, metrics_dir)
    config.async_mode = AsyncType.AsyncIO
    config.concurrent_requests = index_concurrency
    register_completion("base_sse", BaseSSECompletion)
    report_model = config.get_completion_model_config(
        config.community_reports.completion_model_id
    )
    report_model.type = "base_sse"
    report_model.checkpoint_dir = str(run_dir / "completion_checkpoints")
    print(
        f"[{method}] Stage {stage}: starting update with "
        f"{len(documents_dataframe(delta_path))} new chunks",
        flush=True,
    )
    additional_context: dict[str, Any] = {
        "embedding_runtime": {
            "request_batch_size": embedding_batch_size,
            "community_request_batch_size": community_embedding_batch_size,
            "concurrency": embedding_concurrency,
            "batch_max_tokens": embedding_batch_max_tokens,
            "vector_write_batch_size": vector_write_batch_size,
        }
    }
    if method == "full":
        config.update.adaptive_community_update = False
        additional_context["force_full_community_rebuild"] = True
    started = time.perf_counter()
    try:
        results = await build_index(
            config,
            method=IndexingMethod.Standard,
            is_update_run=True,
            input_documents=documents_dataframe(delta_path),
            additional_context=additional_context,
            verbose=False,
        )
    finally:
        await BaseSSECompletion.close_clients()
    elapsed = time.perf_counter() - started
    errors = [str(result.error) for result in results if result.error is not None]
    result_state = results[-1].state if results else {}
    stats = load_json(run_dir / "output" / "stats.json", {}) or {}
    context = load_json(run_dir / "output" / "context.json", {}) or {}
    if not context and isinstance(result_state, dict):
        context = result_state
    history = context.get("community_update_history") or []
    update = history[-1] if history else {}
    if not update:
        strategy = context.get("incremental_update_community_strategy")
        if strategy:
            update = {
                "strategy": strategy,
                **(context.get("incremental_update_community_metrics") or {}),
            }
    report_ids = context.get("incremental_update_regenerated_community_report_ids", [])
    report_metrics = collect_report_metrics(metrics_dir)
    if errors:
        print(
            f"[{method}] Stage {stage}: update failed in {elapsed:.2f}s; "
            f"errors={'; '.join(errors)}",
            flush=True,
        )
    else:
        print(
            f"[{method}] Stage {stage}: update complete in {elapsed:.2f}s; "
            f"strategy={update.get('strategy', 'unknown')}; "
            f"reports={len(report_ids)}",
            flush=True,
        )
    result = {
        "stage": stage,
        "method": method,
        "new_chunk_count": len(documents_dataframe(delta_path)),
        "update_wall_time_s": elapsed,
        "workflow_metrics": stats.get("workflows", {}),
        "community_update": update,
        "community_strategy": update.get("strategy"),
        "reports_generated": len(report_ids),
        "index_concurrency": index_concurrency,
        "embedding_batch_size": embedding_batch_size,
        "community_embedding_batch_size": community_embedding_batch_size,
        "embedding_concurrency": embedding_concurrency,
        "embedding_batch_max_tokens": embedding_batch_max_tokens,
        "vector_write_batch_size": vector_write_batch_size,
        **report_metrics,
        "errors": errors,
    }
    if not errors:
        write_json(run_dir / f"stage_{stage}" / "update_metrics.json", result)
    return result


async def run_questions(
    *,
    method: str,
    stage: int,
    question_path: Path,
    run_dir: Path,
    namespace: str,
    vector_source_dir: Path | None = None,
    resume: bool = False,
) -> list[dict[str, Any]]:
    """Run cumulative questions and save raw answers for offline judging."""
    config = configure_run(
        WORKSPACE, run_dir, namespace, run_dir / "metrics" / f"stage_{stage}" / "query"
    )
    if vector_source_dir is not None:
        config.vector_store.db_uri = str((vector_source_dir / "lancedb").resolve())
    provider = create_table_provider(
        config.table_provider, create_storage(config.output_storage)
    )
    if not isinstance(provider, Neo4jTableProvider):
        msg = "Question evaluation requires Neo4j"
        raise TypeError(msg)
    try:
        active = await provider.get_active_provider()
        reader = DataReader(active)
        entity_frame = await reader.entities()
        questions = read_jsonl(question_path)
        stage_dir = run_dir / f"stage_{stage}"
        stage_dir.mkdir(parents=True, exist_ok=True)
        output = stage_dir / "question_results.jsonl"
        rows: list[dict[str, Any]] = (
            read_jsonl(output) if resume and output.exists() else []
        )
        expected_prefix = [str(item["qid"]) for item in questions[: len(rows)]]
        saved_prefix = [str(row.get("qid")) for row in rows]
        if (
            saved_prefix != expected_prefix
            or any(row.get("stage") != stage for row in rows)
            or any(row.get("method") != method for row in rows)
        ):
            msg = f"[{method}] Stage {stage}: saved question prefix is incompatible"
            raise ValueError(msg)
        print(
            f"[{method}] Stage {stage}: starting evaluation of "
            f"{len(questions)} questions; resuming after {len(rows)} completed",
            flush=True,
        )
        for question_number, item in enumerate(
            questions[len(rows) :], start=len(rows) + 1
        ):
            print(
                f"[{method}] Stage {stage}: question {question_number}/{len(questions)} "
                f"started ({item['qid']})",
                flush=True,
            )
            timing = QueryTimingCallbacks()
            started = time.perf_counter()
            request_metrics: dict[str, Any] = {}
            response, context = await local_search(
                config=config,
                entities=(
                    entity_frame
                    if isinstance(entity_frame, pd.DataFrame)
                    else pd.DataFrame()
                ),
                communities=pd.DataFrame(),
                community_reports=pd.DataFrame(),
                text_units=pd.DataFrame(),
                relationships=pd.DataFrame(),
                covariates=None,
                community_level=2,
                response_type="A concise answer",
                query=question_prompt(item),
                graph_context_loader=active.read_local_search_context,
                vector_store_version=getattr(active, "vector_version", None),
                metrics=request_metrics,
                callbacks=[timing],
            )
            total = time.perf_counter() - started
            request_metrics.update(timing.finish())
            evidence = [str(row["chunk_id"]) for row in item.get("gold_evidence", [])]
            hits, count, recall = context_recall(context, evidence)
            text = (
                response
                if isinstance(response, str)
                else json.dumps(response, ensure_ascii=False)
            )
            context_records = context if isinstance(context, dict) else {}
            sources = context_records.get("sources", [])
            context_entities = context_records.get("entities", [])
            context_relationships = context_records.get("relationships", [])
            rows.append({
                "stage": stage,
                "method": method,
                "qid": item["qid"],
                "answer": text,
                "correct": score_objective(
                    {
                        "question_type": item["question_type"],
                        "gold_answer": item["reference_answer"],
                    },
                    text,
                )
                if item["question_type"] in {"FB", "MC", "MS", "TF"}
                else None,
                "question_type": item["question_type"],
                "evaluation_cohort": item.get("evaluation_cohort"),
                "reference_answer": item["reference_answer"],
                "gold_claims": item.get("gold_claims", []),
                "gold_evidence": item.get("gold_evidence", []),
                "retrieved_text_unit_ids": [
                    str(row["id"])
                    for row in sources
                    if isinstance(row, dict) and "id" in row
                ],
                "retrieved_entity_ids": [
                    str(row["id"])
                    for row in context_entities
                    if isinstance(row, dict) and "id" in row
                ],
                "retrieved_relationship_ids": [
                    str(row["id"])
                    for row in context_relationships
                    if isinstance(row, dict) and "id" in row
                ],
                "evidence_recall_at_k": recall
                if item.get("gold_evidence_exhaustive", False)
                else None,
                "evidence_hits": hits,
                "evidence_count": count,
                "entity_recall_at_k": None,
                "relationship_recall_at_k": None,
                "latency_seconds": total,
                "timing": request_metrics,
                "context": json_safe(context),
            })
            write_jsonl(output, rows)
            print(
                f"[{method}] Stage {stage}: question {item['qid']} completed "
                f"({total:.2f}s; {len(rows)}/{len(questions)})",
                flush=True,
            )
        return rows
    finally:
        await provider.close()


def run_worker(command: list[str], result_path: Path) -> None:
    """Run a subprocess while retaining its logs."""
    completed = subprocess.run(
        command, cwd=ROOT, text=True, capture_output=True, check=False
    )
    result_path.with_suffix(".stdout.log").write_text(
        completed.stdout, encoding="utf-8"
    )
    result_path.with_suffix(".stderr.log").write_text(
        completed.stderr, encoding="utf-8"
    )
    if completed.returncode:
        raise RuntimeError(f"Worker failed ({completed.returncode})")


def orchestrate(args: argparse.Namespace) -> int:
    """Run both methods across four update stages."""
    source = args.source_run_dir.resolve()
    output = args.output.resolve()
    print("Community incremental benchmark starting", flush=True)
    print(f"Dataset: {DATASET}", flush=True)
    print(f"Output: {output}", flush=True)
    print(
        "Runtime limits: "
        f"index_concurrency={args.index_concurrency}, "
        f"embedding_batch_size={args.embedding_batch_size}, "
        "community_embedding_batch_size="
        f"{args.community_embedding_batch_size}, "
        f"embedding_concurrency={args.embedding_concurrency}, "
        f"embedding_batch_max_tokens={args.embedding_batch_max_tokens}, "
        f"vector_write_batch_size={args.vector_write_batch_size}",
        flush=True,
    )
    if output.exists() and args.overwrite:
        shutil.rmtree(output)
    output.mkdir(parents=True, exist_ok=True)
    methods = {"incremental": output / "incremental", "full": output / "full"}
    for method, method_dir in methods.items():
        print(f"Preparing {method} namespace: {args.namespace}-{method}", flush=True)
        method_dir.mkdir(exist_ok=True)
        copy_base_tables(
            source,
            method_dir,
            args.source_namespace,
            f"{args.namespace}-{method}",
            resume=args.resume,
        )

    base_dir = output / "base"
    base_dir.mkdir(exist_ok=True)
    print("===== Stage 0: published 1800-chunk Base =====", flush=True)
    base_questions = asyncio.run(
        run_questions(
            method="base",
            stage=0,
            question_path=DATASET / "questions" / "cumulative" / "stage_0.jsonl",
            run_dir=base_dir,
            namespace=args.source_namespace,
            vector_source_dir=source,
            resume=args.resume,
        )
    )
    base_correct = sum(bool(row["correct"]) for row in base_questions)
    summaries: list[dict[str, Any]] = [
        {
            "stage": 0,
            "method": "base",
            "new_chunk_count": 0,
            "update_wall_time_s": 0.0,
            "community_strategy": "published_base",
            "reports_generated": 0,
            "report_llm_calls": 0,
            "report_input_tokens": 0,
            "report_output_tokens": 0,
            "report_total_tokens": 0,
            "errors": [],
            "evaluated_questions": len(base_questions),
            "correct_questions": base_correct,
            "accuracy": base_correct / max(1, len(base_questions)),
        }
    ]
    for stage_def in STAGES:
        print(
            f"===== Stage {stage_def.stage}: {stage_def.delta_path.name} =====",
            flush=True,
        )
        for method, method_dir in methods.items():
            namespace = f"{args.namespace}-{method}"
            update_path = (
                method_dir / f"stage_{stage_def.stage}" / "update_metrics.json"
            )
            saved_update = load_json(update_path) if args.resume else None
            if (
                isinstance(saved_update, dict)
                and saved_update.get("stage") == stage_def.stage
                and saved_update.get("method") == method
                and not saved_update.get("errors")
            ):
                update = saved_update
                print(
                    f"[{method}] Stage {stage_def.stage}: completed update "
                    "checkpoint found; skipping update.",
                    flush=True,
                )
            else:
                update = asyncio.run(
                    run_update(
                        method=method,
                        stage=stage_def.stage,
                        delta_path=stage_def.delta_path,
                        run_dir=method_dir,
                        namespace=namespace,
                        index_concurrency=args.index_concurrency,
                        embedding_batch_size=args.embedding_batch_size,
                        community_embedding_batch_size=(
                            args.community_embedding_batch_size
                        ),
                        embedding_concurrency=args.embedding_concurrency,
                        embedding_batch_max_tokens=args.embedding_batch_max_tokens,
                        vector_write_batch_size=args.vector_write_batch_size,
                    )
                )
            if update["errors"]:
                raise RuntimeError(
                    f"[{method}] Stage {stage_def.stage} update failed; "
                    f"evaluation is not valid: {'; '.join(update['errors'])}"
                )
            questions = asyncio.run(
                run_questions(
                    method=method,
                    stage=stage_def.stage,
                    question_path=stage_def.question_path,
                    run_dir=method_dir,
                    namespace=namespace,
                    resume=args.resume,
                )
            )
            update["evaluated_questions"] = len(questions)
            update["correct_questions"] = sum(bool(row["correct"]) for row in questions)
            update["accuracy"] = update["correct_questions"] / max(1, len(questions))
            summaries.append(update)
        nmi_path = output / f"stage_{stage_def.stage}_nmi.json"
        saved_nmi = load_json(nmi_path) if args.resume else None
        if (
            isinstance(saved_nmi, dict)
            and saved_nmi.get("stage") == stage_def.stage
            and saved_nmi.get("community_level") == args.community_level
        ):
            nmi = saved_nmi.get("community_nmi")
            compared_nodes = int(saved_nmi.get("compared_node_count", 0))
            print(
                f"Stage {stage_def.stage}: completed NMI checkpoint found; skipping.",
                flush=True,
            )
        else:
            nmi, compared_nodes = collect_stage_nmi(
                methods["incremental"],
                f"{args.namespace}-incremental",
                methods["full"],
                f"{args.namespace}-full",
                level=args.community_level,
            )
            write_json(
                nmi_path,
                {
                    "stage": stage_def.stage,
                    "community_nmi": nmi,
                    "compared_node_count": compared_nodes,
                    "community_level": args.community_level,
                },
            )
        for update in summaries[-2:]:
            update["community_nmi"] = nmi
            update["compared_node_count"] = compared_nodes
            update["community_level"] = args.community_level

    result_dirs = {"base": base_dir, **methods}
    for method, method_dir in result_dirs.items():
        rows = [
            row
            for result_path in sorted(method_dir.glob("stage_*/question_results.jsonl"))
            for row in read_jsonl(result_path)
        ]
        write_jsonl(output / f"{method}_question_results.jsonl", rows)
        write_jsonl(
            output / f"{method}_open_questions_for_offline_judging.jsonl",
            [
                {
                    "stage": row["stage"],
                    "method": method,
                    "qid": row["qid"],
                    "question_type": row["question_type"],
                    "question": next(
                        item["question"]
                        for item in read_jsonl(
                            DATASET / f"questions/cumulative/stage_{row['stage']}.jsonl"
                        )
                        if item["qid"] == row["qid"]
                    ),
                    "reference_answer": row["reference_answer"],
                    "gold_claims": row["gold_claims"],
                    "answer": row["answer"],
                    "judge_correct": None,
                    "judge_score_0_to_1": None,
                    "judge_reason": None,
                }
                for row in rows
                if row["question_type"] == "OE"
            ],
        )
    write_json(
        output / "stage_metrics.json", {"stages": summaries, "dataset": str(DATASET)}
    )
    fieldnames = sorted({key for row in summaries for key in row})
    with (output / "stage_metrics.csv").open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(summaries)
    print(f"Comparison complete: {output / 'stage_metrics.json'}")
    return 0


def parser() -> argparse.ArgumentParser:
    """Build command-line parser."""
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument(
        "--source-run-dir",
        type=Path,
        default=WORKSPACE / "benchmark_results" / "incremental-164",
    )
    result.add_argument("--source-namespace", default=DEFAULT_SOURCE_NAMESPACE)
    result.add_argument(
        "--output",
        type=Path,
        default=WORKSPACE / "benchmark_results" / "community-incremental-v2",
    )
    result.add_argument("--namespace", default="graphrag-community-incremental-v2")
    output_mode = result.add_mutually_exclusive_group()
    output_mode.add_argument("--overwrite", action="store_true")
    output_mode.add_argument(
        "--resume",
        action="store_true",
        help="validate and reuse completed Base copies, updates, questions, and NMI",
    )
    result.add_argument("--community-level", type=int, default=0)
    result.add_argument("--index-concurrency", type=positive_int, default=4)
    result.add_argument("--embedding-batch-size", type=positive_int, default=4)
    result.add_argument(
        "--community-embedding-batch-size", type=positive_int, default=1
    )
    result.add_argument("--embedding-concurrency", type=positive_int, default=1)
    result.add_argument("--embedding-batch-max-tokens", type=positive_int, default=1024)
    result.add_argument("--vector-write-batch-size", type=positive_int, default=128)
    return result


if __name__ == "__main__":
    raise SystemExit(orchestrate(parser().parse_args()))
