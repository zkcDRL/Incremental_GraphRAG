#!/usr/bin/env python3
# Copyright (c) 2026 Microsoft Corporation.
# Licensed under the MIT License
# ruff: noqa: T201, E402, EM101, EM102, TRY003, PERF401
"""Resume only text embedding from existing Neo4j GraphRAG tables."""

from __future__ import annotations

import argparse
import asyncio
import sys
import time
from pathlib import Path

from graphrag.cache.cache_key_creator import cache_key_creator
from graphrag.callbacks.noop_workflow_callbacks import NoopWorkflowCallbacks
from graphrag.index.workflows.generate_text_embeddings import generate_text_embeddings
from graphrag.logger.standard_logging import init_loggers
from graphrag_cache import create_cache
from graphrag_llm.embedding import create_embedding
from graphrag_storage import create_storage
from graphrag_storage.tables.neo4j_table_provider import Neo4jTableProvider
from graphrag_storage.tables.table_provider_factory import create_table_provider
from graphrag_vectors.vector_store_type import VectorStoreType

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.run_incremental_benchmark import (
    DEFAULT_WORKSPACE,
    configure_run,
    write_json,
)

DEFAULT_RUN_DIR = DEFAULT_WORKSPACE / "benchmark_results" / "incremental-101"


def clear_lancedb_indexes(config, version: str | None) -> list[str]:
    """Drop configured LanceDB tables so a forced run starts from no vectors."""
    if config.vector_store.type != VectorStoreType.LanceDB:
        msg = "--force-reembed currently supports only the LanceDB vector store"
        raise ValueError(msg)
    if config.vector_store.db_uri is None:
        msg = "LanceDB db_uri is required for --force-reembed"
        raise ValueError(msg)

    import lancedb

    connection = lancedb.connect(config.vector_store.db_uri)
    existing = set(connection.table_names())
    dropped: list[str] = []
    for schema in config.vector_store.index_schema.values():
        table_name = schema.index_name
        if version is not None:
            table_name = f"{table_name}__graphrag_{version}"
        if table_name in existing:
            connection.drop_table(table_name)
            dropped.append(table_name)
    return dropped


async def resume_embeddings(args: argparse.Namespace) -> int:
    """Read the active Neo4j tables and regenerate all configured vector indexes."""
    workspace = args.workspace.resolve()
    run_dir = args.run_dir.resolve()
    metrics_dir = run_dir / "metrics" / args.stage / "index"
    config = configure_run(workspace, run_dir, args.namespace, metrics_dir)
    if args.batch_size is not None:
        config.embed_text.batch_size = args.batch_size
    if args.concurrency is not None:
        config.concurrent_requests = args.concurrency

    model_config = config.get_embedding_model_config(config.embed_text.embedding_model_id)
    model_config = model_config.model_copy(deep=True)
    model_config.call_args = {**model_config.call_args, "timeout": args.request_timeout}

    init_loggers(config=config, verbose=args.verbose)
    provider = create_table_provider(
        config.table_provider, create_storage(config.output_storage)
    )
    if not isinstance(provider, Neo4jTableProvider):
        raise TypeError("Embedding recovery requires a Neo4j table provider")
    active_provider = await provider.get_active_provider()
    required_tables = {"entities", "community_reports", "text_units"}
    missing = []
    for table in sorted(required_tables):
        if not await active_provider.has(table):
            missing.append(table)
    if missing:
        raise RuntimeError(f"Neo4j is missing required tables: {', '.join(missing)}")

    cache = create_cache(config.cache)
    embedding_cache = cache.child(config.embed_text.model_instance_name)
    dropped_indexes: list[str] = []
    if args.force_reembed:
        dropped_indexes = clear_lancedb_indexes(
            config, getattr(active_provider, "vector_version", None)
        )
        await embedding_cache.clear()
        print(
            "Forced re-embedding: cleared the text embedding cache and dropped "
            f"{len(dropped_indexes)} vector indexes."
        )

    model = create_embedding(
        model_config,
        cache=embedding_cache,
        cache_key_creator=cache_key_creator,
    )
    probe = await asyncio.wait_for(
        model.embedding_async(input=["GraphRAG embedding dimension probe"]),
        timeout=args.request_timeout,
    )
    detected_dimension = len(probe.first_embedding)
    if detected_dimension <= 0:
        raise RuntimeError("Embedding dimension probe returned an empty vector")
    configured_dimension = config.vector_store.vector_size
    if configured_dimension != detected_dimension:
        print(
            f"Embedding dimension is {detected_dimension}; overriding configured "
            f"vector_size {configured_dimension}."
        )
    config.vector_store.vector_size = detected_dimension
    for schema in config.vector_store.index_schema.values():
        schema.vector_size = detected_dimension

    started = time.perf_counter()
    try:
        await generate_text_embeddings(
            config=config,
            table_provider=active_provider,
            callbacks=NoopWorkflowCallbacks(),
            model=model,
            tokenizer=model.tokenizer,
        )
    finally:
        await provider.close()
    elapsed = time.perf_counter() - started

    result_path = args.result or run_dir / args.stage / "embedding_recovery.json"
    write_json(result_path, {
        "stage": args.stage,
        "workflow": "generate_text_embeddings",
        "namespace": args.namespace,
        "source": "active Neo4j tables",
        "vector_store": str(config.vector_store.db_uri),
        "vector_size": detected_dimension,
        "force_reembed": args.force_reembed,
        "dropped_indexes": dropped_indexes,
        "elapsed_seconds": elapsed,
        "errors": [],
    })
    print(f"Embedding recovery complete in {elapsed:.2f}s: {result_path}")
    return 0


def parse_args() -> argparse.Namespace:
    """Parse embedding recovery options."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace", type=Path, default=DEFAULT_WORKSPACE)
    parser.add_argument("--run-dir", type=Path, default=DEFAULT_RUN_DIR)
    parser.add_argument("--namespace", default="graphrag-bench-incremental-101")
    parser.add_argument("--stage", default="01_base")
    parser.add_argument("--result", type=Path)
    parser.add_argument("--request-timeout", type=float, default=60.0)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--concurrency", type=int)
    parser.add_argument(
        "--force-reembed",
        action="store_true",
        help="Drop current vector indexes and clear the embedding cache first",
    )
    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args()


def main() -> int:
    """Run embedding recovery."""
    return asyncio.run(resume_embeddings(parse_args()))


if __name__ == "__main__":
    raise SystemExit(main())
