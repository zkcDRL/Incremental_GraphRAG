#!/usr/bin/env python3
# Copyright (c) 2026 Microsoft Corporation.
# Licensed under the MIT License
# ruff: noqa: T201, EM101, EM102, TRY003, TRY301
"""Resume an unpublished Base from a Neo4j building or failed candidate.

The candidate must already contain the merged graph and cold-Leiden communities.
Complete community reports are reused without an LLM call. Missing or invalid
reports are regenerated with response checkpoints. Completed graph projections
and vector fields are validated and reused; remaining vectors are rebuilt in
bounded model and storage batches. Publication happens only after strict validation.
"""

import argparse
import asyncio
import fcntl
import gc
import json
import time
from pathlib import Path
from typing import Any

import pandas as pd
from graphrag.cache.cache_key_creator import cache_key_creator
from graphrag.callbacks.noop_workflow_callbacks import NoopWorkflowCallbacks
from graphrag.config.embeddings import community_full_content_embedding
from graphrag.config.enums import AsyncType
from graphrag.data_model.data_reader import DataReader
from graphrag.index.workflows.create_community_reports import create_community_reports
from graphrag.index.workflows.generate_text_embeddings import (
    EMBEDDING_FIELDS,
    generate_text_embeddings,
)
from graphrag_cache import create_cache
from graphrag_llm.completion import create_completion
from graphrag_llm.completion.completion_factory import register_completion
from graphrag_llm.embedding import create_embedding
from graphrag_storage import create_storage
from graphrag_storage.tables.table_provider_factory import create_table_provider
from neo4j import AsyncGraphDatabase

from scripts.base_sse_completion import BaseSSECompletion
from scripts.prepare_community_benchmark_base import (
    ROOT,
    RUN,
    TABLES,
    TARGET,
    validate_documents,
    verify_base,
)
from scripts.run_incremental_benchmark import configure_run, write_json


def configure_recovery():
    """Load and constrain the model and storage configuration used by Base builds."""
    config = configure_run(
        ROOT / "rag_workspace", RUN, TARGET, RUN / "metrics/base_recovery"
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
    return config


def load_base_rows(base_path: Path) -> list[dict[str, Any]]:
    """Read a Base JSONL and require unique chunk IDs."""
    rows = [
        json.loads(line)
        for line in base_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    ids = [str(row["chunk_id"]) for row in rows]
    if not rows or len(ids) != len(set(ids)):
        raise ValueError("Base must contain at least one row and unique chunk IDs")
    return rows


def community_report_issues(
    communities: pd.DataFrame, reports: pd.DataFrame
) -> dict[str, list[Any]]:
    """Return report coverage and content problems without mutating either table."""
    expected = set(communities["community"].astype(int))
    if reports.empty or "community" not in reports.columns:
        return {
            "missing": sorted(expected),
            "extra": [],
            "duplicates": [],
            "invalid": [],
        }

    report_ids = reports["community"].astype(int)
    actual = set(report_ids)
    duplicates = sorted(report_ids[report_ids.duplicated()].unique().tolist())
    required = ("id", "title", "summary", "full_content", "full_content_json")
    invalid: list[int] = []
    if any(column not in reports.columns for column in required):
        invalid = sorted(actual & expected)
    else:
        for _, row in reports.iterrows():
            community_id = int(row["community"])
            try:
                values = [str(row[column]).strip() for column in required[:-1]]
                payload = json.loads(str(row["full_content_json"]))
                valid_payload = (
                    isinstance(payload, dict)
                    and bool(str(payload.get("title", "")).strip())
                    and bool(str(payload.get("summary", "")).strip())
                    and isinstance(payload.get("findings"), list)
                    and bool(payload["findings"])
                )
                if not all(values) or not valid_payload:
                    invalid.append(community_id)
            except (TypeError, ValueError, json.JSONDecodeError):
                invalid.append(community_id)
    return {
        "missing": sorted(expected - actual),
        "extra": sorted(actual - expected),
        "duplicates": duplicates,
        "invalid": sorted(set(invalid)),
    }


def require_complete_reports(
    communities: pd.DataFrame, reports: pd.DataFrame
) -> dict[str, int]:
    """Require one valid report per community and return validated counts."""
    issues = community_report_issues(communities, reports)
    if any(issues.values()):
        raise ValueError(f"Community reports are incomplete or invalid: {issues}")
    if len(reports) != len(communities):
        raise ValueError("Community report row count differs from community row count")
    return {"communities": len(communities), "community_reports": len(reports)}


async def generate_complete_reports(config, candidate, cache, max_passes: int):
    """Regenerate reports, reusing successful response checkpoints on every pass."""
    reader = DataReader(candidate)
    communities = await reader.communities()
    entities = await reader.entities()
    relationships = await reader.relationships()
    model_config = config.get_completion_model_config(
        config.community_reports.completion_model_id
    )
    model = create_completion(
        model_config,
        cache=cache.child(config.community_reports.model_instance_name),
        cache_key_creator=cache_key_creator,
    )
    prompts = config.community_reports.resolved_prompts()
    last_issues: dict[str, list[Any]] = {}
    for pass_number in range(1, max_passes + 1):
        BaseSSECompletion.failures.clear()
        started = time.perf_counter()
        reports = await create_community_reports(
            relationships=relationships,
            entities=entities,
            communities=communities,
            claims_input=None,
            callbacks=NoopWorkflowCallbacks(),
            model=model,
            tokenizer=model.tokenizer,
            prompt=prompts.graph_prompt,
            max_input_length=config.community_reports.max_input_length,
            max_report_length=config.community_reports.max_length,
            num_threads=config.concurrent_requests,
            async_type=config.async_mode,
        )
        last_issues = community_report_issues(communities, reports)
        print(
            f"REPORT PASS {pass_number}: generated={len(reports)}/{len(communities)} "
            f"issues={last_issues} elapsed={time.perf_counter() - started:.1f}s",
            flush=True,
        )
        if not any(last_issues.values()) and len(reports) == len(communities):
            await candidate.write_dataframe("community_reports", reports)
            return reports
    raise RuntimeError(
        f"Community reports remain incomplete after {max_passes} passes: {last_issues}"
    )


async def candidate_metadata(config, version: str) -> dict[str, str]:
    """Require an existing unpublished candidate in building or failed state."""
    driver = AsyncGraphDatabase.driver(
        config.table_provider.uri,
        auth=(config.table_provider.username, config.table_provider.password),
    )
    try:
        async with driver.session(database=config.table_provider.database) as session:
            result = await session.run(
                "MATCH (v:GraphRAGVersion {logical_namespace: $namespace, "
                "version: $version}) "
                "RETURN v.state AS state, v.storage_namespace AS storage_namespace",
                namespace=TARGET,
                version=version,
            )
            record = await result.single()
            if not record:
                raise ValueError("Recovery candidate does not exist")
            state = str(record["state"])
            if state not in {"building", "failed"}:
                raise ValueError(
                    f"Recovery candidate must be building or failed, not {state}"
                )
            return {
                "state": state,
                "storage_namespace": str(record["storage_namespace"]),
            }
    finally:
        await driver.close()


async def prepare_candidate_for_publish(
    config, version: str, expected_active_version: str
) -> str:
    """Move failed/building to building only if the active Base did not change."""
    driver = AsyncGraphDatabase.driver(
        config.table_provider.uri,
        auth=(config.table_provider.username, config.table_provider.password),
    )
    try:
        async with driver.session(database=config.table_provider.database) as session:
            result = await session.run(
                "MATCH (meta:GraphRAGNamespace {name: $namespace, "
                "active_version: $active_version}) "
                "MATCH (v:GraphRAGVersion {logical_namespace: $namespace, "
                "version: $version}) "
                "WHERE v.state IN ['building', 'failed'] "
                "WITH v, v.state AS previous_state "
                "SET v.state = 'building' RETURN previous_state",
                namespace=TARGET,
                version=version,
                active_version=expected_active_version,
            )
            record = await result.single()
            if not record:
                raise ValueError(
                    "Candidate state is invalid or the active Base changed during recovery"
                )
            return str(record["previous_state"])
    finally:
        await driver.close()


async def projection_counts(config, storage_namespace: str) -> dict[str, int]:
    """Read graph projection edge counts without rebuilding it again."""
    relations = (
        "RELATED",
        "MENTIONS",
        "EVIDENCES",
        "CONTAINS",
        "HAS_MEMBER",
        "INCLUDES_RELATIONSHIP",
        "CONTAINS_TEXT_UNIT",
        "REPORTS_ON",
    )
    driver = AsyncGraphDatabase.driver(
        config.table_provider.uri,
        auth=(config.table_provider.username, config.table_provider.password),
    )
    try:
        async with driver.session(database=config.table_provider.database) as session:
            counts = {}
            for relation in relations:
                result = await session.run(
                    "MATCH (:GraphRAGRecord {namespace: $namespace})"
                    f"-[edge:{relation}]->() RETURN count(edge) AS count",
                    namespace=storage_namespace,
                )
                record = await result.single()
                counts[relation] = int(record["count"]) if record else 0
            return counts
    finally:
        await driver.close()


def projection_checkpoint(version: str) -> dict[str, int] | None:
    """Load edge counts saved after a previously completed projection."""
    for path in (
        RUN / f"base_recovery_{version}.json",
        RUN / "base_recovery.json",
    ):
        if not path.exists():
            continue
        state = json.loads(path.read_text(encoding="utf-8"))
        counts = state.get("projection_counts")
        if state.get("recovered_version") == version and isinstance(counts, dict):
            return {str(name): int(count) for name, count in counts.items()}
    return None


async def source_ids(candidate, table_name: str) -> set[str]:
    """Read only source IDs through the bounded Neo4j row iterator."""
    ids: set[str] = set()
    async with candidate.open(table_name, truncate=False) as table:
        async for row in table:
            ids.add(str(row["id"]))
    return ids


def vector_index_status(config, candidate, field_name: str, expected: set[str]):
    """Return strict ID status for one candidate LanceDB index."""
    import lancedb

    schema = config.vector_store.index_schema[field_name]
    table_name = f"{schema.index_name}__graphrag_{candidate.vector_version}"
    connection = lancedb.connect(config.vector_store.db_uri)
    if table_name not in connection.table_names():
        return {
            "table": table_name,
            "count": 0,
            "complete": False,
            "missing": len(expected),
            "extra": 0,
        }

    table = connection.open_table(table_name)
    id_table = table.search().select([schema.id_field]).limit(None).to_arrow()
    ids = [str(value) for value in id_table.column(schema.id_field).to_pylist()]
    actual = set(ids)
    return {
        "table": table_name,
        "count": len(ids),
        "complete": len(ids) == len(expected) and actual == expected,
        "missing": len(expected - actual),
        "extra": len(actual - expected),
    }


def drop_vector_index(config, table_name: str) -> None:
    """Drop one incomplete candidate vector index without touching completed fields."""
    import lancedb

    connection = lancedb.connect(config.vector_store.db_uri)
    connection.drop_table(table_name, ignore_missing=True)


async def verify_candidate_vectors(config, candidate) -> dict[str, int]:
    """Validate candidate vector IDs before making its graph version active."""
    counts = {}
    for field_name in config.embed_text.names:
        source_table = EMBEDDING_FIELDS[field_name].table_name
        expected = await source_ids(candidate, source_table)
        status = vector_index_status(config, candidate, field_name, expected)
        if not status["complete"]:
            raise ValueError(f"Incomplete vector index: {status}")
        counts[str(status["table"])] = int(status["count"])
    return counts


def recovery_context(version: str) -> dict[str, Any]:
    """Preserve prior history and append the recovered cold-Leiden Base version."""
    current_path = RUN / "output/context.json"
    current = (
        json.loads(current_path.read_text(encoding="utf-8"))
        if current_path.exists()
        else {}
    )
    history = [
        item
        for item in current.get("community_update_history", [])
        if item.get("update_timestamp") != version
    ]
    history.append({
        "update_timestamp": version,
        "strategy": "global_cold",
        "recovered_candidate": True,
    })
    current.update(update_timestamp=version, community_update_history=history)
    return current


async def recover(
    version: str,
    base_path: Path,
    max_report_passes: int,
    force_report_regeneration: bool,
    check_only: bool,
    embedding_batch_size: int,
    embedding_concurrency: int,
    embedding_batch_max_tokens: int,
    vector_write_batch_size: int,
) -> None:
    """Validate, complete, publish, and verify one exact candidate version."""
    rows = load_base_rows(base_path)
    config = configure_recovery()
    storage = create_storage(config.output_storage)
    target = create_table_provider(config.table_provider, storage)
    summary: dict[str, Any] = {
        "namespace": TARGET,
        "recovered_version": version,
        "base_path": str(base_path),
        "expected_documents": len(rows),
        "status": "checking" if check_only else "recovering",
        "errors": [],
    }
    started = time.perf_counter()
    mutated = False
    try:
        metadata = await candidate_metadata(config, version)
        summary["candidate_state_before"] = metadata["state"]
        summary["candidate_storage_namespace"] = metadata["storage_namespace"]
        expected_storage_namespace = f"{TARGET}/__graphrag_versions__/{version}"
        if metadata["storage_namespace"] != expected_storage_namespace:
            raise ValueError(
                "Candidate storage namespace does not match its logical version"
            )
        candidate = await target.create_building_version(version)
        active = await target.get_active_provider()
        active_documents = await DataReader(active).documents()
        if len(active_documents) > len(rows):
            raise ValueError(
                "Active Base contains more documents than the recovery Base"
            )
        validate_documents(active_documents, rows[: len(active_documents)])
        summary.update(
            active_version_before=active.vector_version or "legacy",
            active_documents=len(active_documents),
            added_documents=len(rows) - len(active_documents),
        )

        candidate_documents = await DataReader(candidate).documents()
        validate_documents(candidate_documents, rows)
        existing_tables = {
            table for table in TABLES if await candidate.has(table)
        }
        required_graph_tables = set(TABLES) - {"community_reports"}
        missing_tables = sorted(required_graph_tables - existing_tables)
        if missing_tables:
            raise ValueError(f"Candidate is missing graph tables: {missing_tables}")
        table_counts: dict[str, int] = {}
        for table in existing_tables:
            async with candidate.open(table, truncate=False) as input_table:
                table_counts[table] = await input_table.length()
        communities = await candidate.read_dataframe("communities")
        if communities.empty:
            raise ValueError("Recovery candidate has no communities")
        reports = (
            await candidate.read_dataframe("community_reports")
            if "community_reports" in existing_tables
            else pd.DataFrame()
        )
        issues = community_report_issues(communities, reports)
        summary["candidate_table_counts_before"] = table_counts
        summary["report_issues_before"] = issues
        summary["reports_complete_before"] = not any(issues.values()) and len(
            reports
        ) == len(communities)
        if check_only:
            summary["status"] = "ready"
            print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
            return

        mutated = True
        if force_report_regeneration or not summary["reports_complete_before"]:
            print("Regenerating community reports with checkpoints.", flush=True)
            reports = await generate_complete_reports(
                config, candidate, create_cache(config.cache), max_report_passes
            )
            summary["report_action"] = "regenerated"
            counts = await projection_counts(config, metadata["storage_namespace"])
            summary["projection_action"] = "rebuilt_with_reports"
        else:
            require_complete_reports(communities, reports)
            summary["report_action"] = "reused"
            expected_counts = projection_checkpoint(version)
            counts = await projection_counts(config, metadata["storage_namespace"])
            if expected_counts is not None and counts == expected_counts:
                summary["projection_action"] = "reused"
                print("Graph projection matches its checkpoint; reusing it.", flush=True)
            else:
                summary["projection_action"] = "rebuilt"
                print("Graph projection is incomplete; rebuilding it.", flush=True)
                counts = await candidate.rebuild_graph_projection()
        summary["projection_counts"] = counts
        if counts.get("REPORTS_ON") != len(reports):
            raise ValueError(
                "Graph projection does not contain exactly one REPORTS_ON edge per report"
            )
        require_complete_reports(communities, reports)

        # The graph/report validation objects include large JSON fields. They are
        # no longer needed while embeddings stream from Neo4j, so release them
        # before Ollama loads its embedding model.
        del active_documents, candidate_documents, communities, reports, rows
        gc.collect()

        config.concurrent_requests = embedding_concurrency
        config.embed_text.batch_max_tokens = embedding_batch_max_tokens
        configured_fields = list(config.embed_text.names)
        vector_actions: dict[str, Any] = {}
        summary["vector_actions"] = vector_actions
        embedding_config = config.get_embedding_model_config(
            config.embed_text.embedding_model_id
        )
        embedding_model = create_embedding(
            embedding_config,
            cache=create_cache(config.cache).child(
                config.embed_text.model_instance_name
            ),
            cache_key_creator=cache_key_creator,
        )
        for field_name in configured_fields:
            field = EMBEDDING_FIELDS[field_name]
            expected_ids = await source_ids(candidate, field.table_name)
            before = vector_index_status(config, candidate, field_name, expected_ids)
            if before["complete"]:
                vector_actions[field_name] = {"action": "reused", **before}
                print(
                    f"Vector index {field_name} is complete ({before['count']} rows); "
                    "reusing it.",
                    flush=True,
                )
                continue

            drop_vector_index(config, str(before["table"]))
            request_batch_size = (
                1
                if field_name == community_full_content_embedding
                else embedding_batch_size
            )
            vector_actions[field_name] = {
                "action": "rebuilding",
                "previous": before,
                "request_batch_size": request_batch_size,
                "batch_max_tokens": embedding_batch_max_tokens,
                "write_batch_size": vector_write_batch_size,
            }
            print(
                f"Rebuilding {field_name}: request_batch_size={request_batch_size}, "
                f"batch_max_tokens={embedding_batch_max_tokens}, "
                f"write_batch_size={vector_write_batch_size}, "
                f"expected_rows={len(expected_ids)}.",
                flush=True,
            )
            config.embed_text.names = [field_name]
            config.embed_text.batch_size = request_batch_size
            await generate_text_embeddings(
                config=config,
                table_provider=candidate,
                callbacks=NoopWorkflowCallbacks(),
                model=embedding_model,
                tokenizer=embedding_model.tokenizer,
                vector_write_batch_size=vector_write_batch_size,
            )
            after = vector_index_status(config, candidate, field_name, expected_ids)
            if not after["complete"]:
                raise ValueError(f"Incomplete vector index after rebuild: {after}")
            vector_actions[field_name] = {"action": "rebuilt", **after}
            summary["elapsed_seconds"] = time.perf_counter() - started
            write_json(RUN / f"base_recovery_{version}.json", summary)
            print(
                f"Completed {field_name}: {after['count']} rows.",
                flush=True,
            )
        config.embed_text.names = configured_fields
        summary["vector_counts_before_publish"] = await verify_candidate_vectors(
            config, candidate
        )

        write_json(RUN / "output/context.json", recovery_context(version))
        expected_active = str(summary["active_version_before"])
        summary["candidate_state_for_publish"] = await prepare_candidate_for_publish(
            config, version, expected_active
        )
        try:
            await target.publish_version(version)
        except Exception:
            await target.mark_version_failed(version)
            raise

        summary.update(await verify_base(config, target, load_base_rows(base_path)))
        summary["status"] = "complete"
    except Exception as error:
        summary.update(status="failed", errors=[{"error": str(error)}])
        if mutated:
            await target.mark_version_failed(version)
        raise
    finally:
        summary["elapsed_seconds"] = time.perf_counter() - started
        if not check_only:
            write_json(RUN / "base_recovery.json", summary)
            write_json(RUN / f"base_recovery_{version}.json", summary)
            if summary["status"] == "complete":
                write_json(RUN / "base_preparation.json", summary)
        await BaseSSECompletion.close_clients()
        await target.close()
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


def main() -> None:
    """Parse one candidate and prevent concurrent Base preparation/recovery."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--version", required=True)
    parser.add_argument("--base", type=Path, required=True)
    parser.add_argument("--max-report-passes", type=int, default=4)
    parser.add_argument("--force-report-regeneration", action="store_true")
    parser.add_argument("--check-only", action="store_true")
    parser.add_argument(
        "--embedding-batch-size",
        type=int,
        default=4,
        help="Rows per non-community embedding request",
    )
    parser.add_argument(
        "--embedding-concurrency",
        type=int,
        default=1,
        help="Concurrent embedding requests; lower values reduce peak memory",
    )
    parser.add_argument(
        "--embedding-batch-max-tokens",
        type=int,
        default=1024,
        help="Maximum tokens per embedding snippet",
    )
    parser.add_argument(
        "--vector-write-batch-size",
        type=int,
        default=128,
        help="Completed vectors accumulated per LanceDB write",
    )
    args = parser.parse_args()
    if min(
        args.embedding_batch_size,
        args.embedding_concurrency,
        args.embedding_batch_max_tokens,
        args.vector_write_batch_size,
    ) < 1:
        parser.error("embedding and vector batch settings must be positive")
    RUN.mkdir(parents=True, exist_ok=True)
    with (RUN / "prepare_base.lock").open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        asyncio.run(
            recover(
                args.version,
                args.base.resolve(),
                args.max_report_passes,
                args.force_report_regeneration,
                args.check_only,
                args.embedding_batch_size,
                args.embedding_concurrency,
                args.embedding_batch_max_tokens,
                args.vector_write_batch_size,
            )
        )


if __name__ == "__main__":
    main()
