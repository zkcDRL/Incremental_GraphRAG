#!/usr/bin/env python3
# Copyright (c) 2026 Microsoft Corporation.
# Licensed under the MIT License
# ruff: noqa: S404, S603, T201, EM102, TRY003
"""Compare adaptive incremental updates with full community rebuilds."""

from __future__ import annotations

import argparse
import asyncio
import csv
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
from graphrag.config.enums import IndexingMethod
from graphrag.data_model.data_reader import DataReader
from graphrag_storage import create_storage
from graphrag_storage.tables.neo4j_table_provider import Neo4jTableProvider
from graphrag_storage.tables.table_provider_factory import create_table_provider

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
DEFAULT_SOURCE_NAMESPACE = "graphrag-bench-incremental-101"


@dataclass(frozen=True)
class StageDefinition:
    """Runtime paths for one cumulative comparison stage."""

    stage: int
    delta_path: Path
    question_path: Path
    cumulative_corpus: tuple[Path, ...]


_CORPUS_NAMES = (
    "00_base_851.jsonl",
    "01_micro_local_50.jsonl",
    "02_near_region_50.jsonl",
    "03_far_existing_books_50.jsonl",
    "04_new_textbook_regions_50.jsonl",
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
) -> None:
    """Copy base logical tables and vectors into an isolated comparison namespace."""
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

    async def copy() -> None:
        source_active = await source_provider.get_active_provider()
        print(
            f"[{destination_namespace}] clearing destination namespace...", flush=True
        )
        await destination_provider.clear_namespace_tree()
        table_names = (
            "entities",
            "relationships",
            "text_units",
            "documents",
            "communities",
            "community_reports",
        )
        for table_number, table_name in enumerate(table_names, start=1):
            print(
                f"[{destination_namespace}] copying table {table_number}/{len(table_names)}: "
                f"{table_name}",
                flush=True,
            )
            table = await source_active.read_dataframe(table_name)
            await destination_provider.write_dataframe(table_name, table)
            print(
                f"[{destination_namespace}] copied {table_name}: {len(table)} rows",
                flush=True,
            )
        print(f"[{destination_namespace}] rebuilding graph projection...", flush=True)
        counts = await destination_provider.rebuild_graph_projection()
        print(
            f"[{destination_namespace}] base namespace ready; projection edges: {counts}",
            flush=True,
        )
        await source_provider.close()
        await destination_provider.close()

    asyncio.run(copy())
    source_vectors = source / "lancedb"
    destination_vectors = destination / "lancedb"
    if source_vectors.exists():
        print(f"[{destination_namespace}] copying vector indexes...", flush=True)
        shutil.copytree(source_vectors, destination_vectors, dirs_exist_ok=True)
        print(f"[{destination_namespace}] vector indexes ready", flush=True)


async def run_update(
    *,
    method: str,
    stage: int,
    delta_path: Path,
    run_dir: Path,
    namespace: str,
) -> dict[str, Any]:
    """Run one incremental update and return measured metadata."""
    metrics_dir = run_dir / "metrics" / f"stage_{stage}" / "index"
    config = configure_run(WORKSPACE, run_dir, namespace, metrics_dir)
    print(
        f"[{method}] Stage {stage}: starting update with "
        f"{len(documents_dataframe(delta_path))} new chunks",
        flush=True,
    )
    if method == "full":
        config.update.adaptive_community_update = False
        additional_context = {"force_full_community_rebuild": True}
    else:
        additional_context = {}
    started = time.perf_counter()
    results = await build_index(
        config,
        method=IndexingMethod.Standard,
        is_update_run=True,
        input_documents=documents_dataframe(delta_path),
        additional_context=additional_context,
        verbose=False,
    )
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
    return {
        "stage": stage,
        "method": method,
        "new_chunk_count": len(documents_dataframe(delta_path)),
        "update_wall_time_s": elapsed,
        "workflow_metrics": stats.get("workflows", {}),
        "community_update": update,
        "community_strategy": update.get("strategy"),
        "reports_generated": len(report_ids),
        **report_metrics,
        "errors": errors,
    }


async def run_questions(
    *,
    method: str,
    stage: int,
    question_path: Path,
    run_dir: Path,
    namespace: str,
) -> list[dict[str, Any]]:
    """Run cumulative questions and save raw answers for offline judging."""
    config = configure_run(
        WORKSPACE, run_dir, namespace, run_dir / "metrics" / f"stage_{stage}" / "query"
    )
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
        rows: list[dict[str, Any]] = []
        stage_dir = run_dir / f"stage_{stage}"
        stage_dir.mkdir(parents=True, exist_ok=True)
        output = stage_dir / "question_results.jsonl"
        print(
            f"[{method}] Stage {stage}: starting evaluation of "
            f"{len(questions)} questions",
            flush=True,
        )
        for question_number, item in enumerate(questions, start=1):
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
    if output.exists() and args.overwrite:
        shutil.rmtree(output)
    output.mkdir(parents=True, exist_ok=True)
    methods = {"incremental": output / "incremental", "full": output / "full"}
    for method, method_dir in methods.items():
        print(f"Preparing {method} namespace: {args.namespace}-{method}", flush=True)
        method_dir.mkdir(exist_ok=True)
        copy_base_tables(
            source, method_dir, args.source_namespace, f"{args.namespace}-{method}"
        )

    summaries: list[dict[str, Any]] = []
    for stage_def in STAGES:
        print(
            f"===== Stage {stage_def.stage}: {stage_def.delta_path.name} =====",
            flush=True,
        )
        for method, method_dir in methods.items():
            namespace = f"{args.namespace}-{method}"
            update = asyncio.run(
                run_update(
                    method=method,
                    stage=stage_def.stage,
                    delta_path=stage_def.delta_path,
                    run_dir=method_dir,
                    namespace=namespace,
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
                )
            )
            update["evaluated_questions"] = len(questions)
            update["correct_questions"] = sum(bool(row["correct"]) for row in questions)
            update["accuracy"] = update["correct_questions"] / max(1, len(questions))
            summaries.append(update)
        nmi, compared_nodes = collect_stage_nmi(
            methods["incremental"],
            f"{args.namespace}-incremental",
            methods["full"],
            f"{args.namespace}-full",
            level=args.community_level,
        )
        for update in summaries[-2:]:
            update["community_nmi"] = nmi
            update["compared_node_count"] = compared_nodes
            update["community_level"] = args.community_level

    for method, method_dir in methods.items():
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
        default=WORKSPACE / "benchmark_results" / "incremental-101",
    )
    result.add_argument("--source-namespace", default=DEFAULT_SOURCE_NAMESPACE)
    result.add_argument(
        "--output",
        type=Path,
        default=WORKSPACE / "benchmark_results" / "community-incremental-v2",
    )
    result.add_argument("--namespace", default="graphrag-community-incremental-v2")
    result.add_argument("--overwrite", action="store_true")
    result.add_argument("--community-level", type=int, default=0)
    return result


if __name__ == "__main__":
    raise SystemExit(orchestrate(parser().parse_args()))
