#!/usr/bin/env python3
# Copyright (c) 2026 Microsoft Corporation.
# Licensed under the MIT License
# ruff: noqa: T201, EM101, EM102, TRY003, TRY301
"""Extend a verified published Base with cold Leiden and full reports.

Run from the repository root with:
    .venv/bin/python -m scripts.prepare_community_benchmark_base

Uses rag_workspace model credentials without printing them. The source namespace
is read only; the target must already contain its verified copy. A completed base
is verified and reused, preventing duplicate ingestion on repeated invocations.
"""

import argparse
import asyncio
import faulthandler
import fcntl
import hashlib
import json
import time
from datetime import UTC, datetime
from pathlib import Path

import lancedb
import pandas as pd
from graphrag.api.index import build_index
from graphrag.callbacks.noop_workflow_callbacks import NoopWorkflowCallbacks
from graphrag.config.enums import AsyncType
from graphrag.index.workflows.generate_text_embeddings import EMBEDDING_FIELDS
from graphrag_llm.completion.completion_factory import register_completion
from graphrag_storage import create_storage
from graphrag_storage.tables.table_provider_factory import create_table_provider

from scripts.base_sse_completion import BaseSSECompletion
from scripts.run_incremental_benchmark import (
    configure_run,
    documents_dataframe,
    write_json,
    write_jsonl,
)

ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / "rag_workspace/benchmark_results/incremental-164"
BASE = ROOT / "GraphRAG-Bench-community-incremental-v2/corpus/00_base_1800.jsonl"
SOURCE = "graphrag-bench-incremental-101"
TARGET = "graphrag-bench-incremental-164"
TABLES = (
    "entities",
    "relationships",
    "text_units",
    "documents",
    "communities",
    "community_reports",
)


def fingerprint(frame: pd.DataFrame) -> str:
    """Hash rows with stable ordering, independent of database return ordering."""
    records = json.loads(frame.to_json(orient="records"))
    serialized = sorted(json.dumps(row, sort_keys=True) for row in records)
    return hashlib.sha256("\n".join(serialized).encode()).hexdigest()


def validate_documents(frame: pd.DataFrame, rows: list[dict]) -> None:
    """Require exact document IDs and text, including uniqueness."""
    expected = {str(row["chunk_id"]): str(row["text"]) for row in rows}
    actual = dict(zip(frame["id"].astype(str), frame["text"].astype(str), strict=True))
    if len(expected) != len(rows) or len(frame) != len(rows) or actual != expected:
        raise ValueError(
            "Document IDs, text, or counts do not match the expected corpus"
        )


class Progress(NoopWorkflowCallbacks):
    """Persist workflow progress without including model credentials or prompts."""

    def __init__(self) -> None:
        self.completed: list[str] = []
        self.started: dict[str, float] = {}

    def workflow_start(self, name: str, instance: object) -> None:
        """Record the workflow currently running."""
        BaseSSECompletion.failures.clear()
        self.started[name] = time.perf_counter()
        write_json(
            RUN / "base_progress.json",
            {
                "running": name,
                "completed": self.completed,
                "updated_at": datetime.now(UTC).isoformat(),
            },
        )
        print(f"START {name}", flush=True)

    def workflow_end(self, name: str, instance: object) -> None:
        """Record completion and elapsed time."""
        if BaseSSECompletion.failures:
            raise RuntimeError(
                f"{name}: model requests exhausted retries; refusing incomplete output"
            )
        self.completed.append(name)
        elapsed = time.perf_counter() - self.started[name]
        write_json(
            RUN / "base_progress.json",
            {
                "running": None,
                "completed": self.completed,
                "updated_at": datetime.now(UTC).isoformat(),
            },
        )
        print(f"DONE {name} ({elapsed:.1f}s)", flush=True)


async def verify_base(config, target, rows: list[dict]) -> dict:
    """Verify the published graph and matching LanceDB vector IDs."""
    active = await target.get_active_provider()
    if active.vector_version is None:
        raise ValueError("The expanded base has no published vector version")
    frames = {table: await active.read_dataframe(table) for table in TABLES}
    validate_documents(frames["documents"], rows)
    community_ids = set(frames["communities"]["community"].astype(int))
    reports = frames["community_reports"]
    if community_ids != set(reports["community"].astype(int)) or len(reports) != len(
        community_ids
    ):
        raise ValueError("Community reports do not cover every community exactly once")
    state = json.loads((RUN / "output/context.json").read_text())
    update = state["community_update_history"][-1]
    if update["strategy"] != "global_cold":
        raise ValueError("The expanded base was not built with global cold Leiden")
    if update["update_timestamp"] != active.vector_version:
        raise ValueError("Graph, vector, and context versions do not match")
    connection = lancedb.connect(config.vector_store.db_uri)
    vector_counts = {}
    for field in config.embed_text.names:
        schema = config.vector_store.index_schema[field]
        name = f"{schema.index_name}__graphrag_{active.vector_version}"
        table = connection.open_table(name)
        ids = table.to_arrow().column(schema.id_field).to_pylist()
        expected = set(frames[EMBEDDING_FIELDS[field].table_name]["id"].astype(str))
        if len(ids) != len(expected) or set(ids) != expected:
            raise ValueError(f"Incomplete vector index: {name}")
        vector_counts[name] = len(ids)
    return {
        "active_namespace": await target.get_active_version(),
        "vector_version": active.vector_version,
        "table_counts": {table: len(frame) for table, frame in frames.items()},
        "vector_counts": vector_counts,
        "community_update": update,
    }


async def run(check_only: bool = False, base_path: Path = BASE) -> None:
    """Validate the copied base, append exactly 80 documents, and verify publication."""
    rows = [
        json.loads(line) for line in base_path.read_text().splitlines() if line.strip()
    ]
    if len(rows) < 931 or len({row["chunk_id"] for row in rows}) != len(rows):
        raise ValueError("Expected at least 931 unique base chunks")
    config = configure_run(
        ROOT / "rag_workspace", RUN, TARGET, RUN / "metrics/base_extension"
    )
    config.update.adaptive_community_update = False
    config.async_mode = AsyncType.AsyncIO
    config.concurrent_requests = 4
    register_completion("base_sse", BaseSSECompletion)
    for model in config.completion_models.values():
        if (
            str(model.api_base).rstrip("/") != "https://www.micuapi.ai/v1"
            or model.model != "deepseek-v4-flash"
        ):
            raise ValueError(
                "Completion model differs from the authorized workspace configuration"
            )
        model.type = "base_sse"
        model.checkpoint_dir = str(RUN / "completion_checkpoints")
    if config.workflows:
        raise ValueError("Custom workflows are not supported for this base preparation")
    storage = create_storage(config.output_storage)
    source = create_table_provider(
        config.table_provider.model_copy(update={"namespace": SOURCE}),
        storage,
    )
    target = create_table_provider(config.table_provider, storage)
    summary = {
        "source_namespace": SOURCE,
        "namespace": TARGET,
        "base_path": str(base_path),
        "expected_documents": len(rows),
        "added_documents": 0,
        "errors": [],
        "status": "checking",
    }
    started = time.perf_counter()
    try:
        originals = {table: await source.read_dataframe(table) for table in TABLES}
        validate_documents(originals["documents"], rows[:851])
        hashes = {table: fingerprint(frame) for table, frame in originals.items()}
        active = await target.get_active_provider()
        current = await active.read_dataframe("documents")
        if len(current) == len(rows):
            summary.update(await verify_base(config, target, rows))
            summary.update(status="complete", reused_existing_base=True)
            print(
                f"The {len(rows)}-document base is already published and verified.",
                flush=True,
            )
        else:
            previous_count = len(current)
            if previous_count not in (851, 931) or previous_count >= len(rows):
                raise ValueError("Unexpected active base size; refusing ingestion")
            validate_documents(current, rows[:previous_count])
            if previous_count == 851:
                for table in TABLES:
                    if fingerprint(await active.read_dataframe(table)) != hashes[table]:
                        raise ValueError(f"Target is not an exact source copy: {table}")
            else:
                await verify_base(config, target, rows[:previous_count])
            summary["previous_documents"] = previous_count
            summary["previous_version"] = active.vector_version
            summary["added_documents"] = len(rows) - previous_count
            print(
                f"Verified base: {previous_count} documents; append exactly {len(rows) - previous_count}.",
                flush=True,
            )
            if check_only:
                print("Preflight passed; no model API was called.", flush=True)
                return
            delta = RUN / f"base_extension_{len(rows) - previous_count}.jsonl"
            write_jsonl(delta, rows[previous_count:])
            write_json(RUN / "base_source_fingerprints.json", hashes)
            summary["status"] = "building"
            write_json(RUN / "base_preparation.json", summary)
            results = await build_index(
                config,
                is_update_run=True,
                input_documents=documents_dataframe(delta),
                additional_context={"force_full_community_rebuild": True},
                callbacks=[Progress()],
            )
            summary["errors"] = [
                {"workflow": result.workflow, "error": str(result.error)}
                for result in results
                if result.error is not None
            ]
            if summary["errors"]:
                raise RuntimeError(f"Base preparation failed: {summary['errors']}")
            summary.update(await verify_base(config, target, rows))
            summary["status"] = "complete"
        for table in TABLES:
            if fingerprint(await source.read_dataframe(table)) != hashes[table]:
                raise ValueError(
                    f"Original namespace changed during execution: {table}"
                )
        summary["source_unchanged"] = True
    except Exception as error:
        summary["status"] = "failed"
        if not summary["errors"]:
            summary["errors"] = [{"error": str(error)}]
        raise
    finally:
        summary["elapsed_seconds"] = time.perf_counter() - started
        if not check_only:
            write_json(RUN / "base_preparation.json", summary)
        await BaseSSECompletion.close_clients()
        await source.close()
        await target.close()
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


def main() -> None:
    """Prevent concurrent local invocations and execute the requested mode."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", type=Path, default=BASE)
    parser.add_argument(
        "--check-only",
        action="store_true",
        help="Validate the base without calling models",
    )
    args = parser.parse_args()
    RUN.mkdir(parents=True, exist_ok=True)
    with (RUN / "prepare_base.lock").open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        faulthandler.enable()
        asyncio.run(run(args.check_only, args.base.resolve()))


if __name__ == "__main__":
    main()
