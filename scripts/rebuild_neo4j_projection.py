#!/usr/bin/env python3
# Copyright (c) 2026 Microsoft Corporation.
# Licensed under the MIT License
# ruff: noqa: T201, EM101, TRY003
"""Rebuild Neo4j graph projection edges from existing GraphRAG logical tables."""

from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path

from graphrag.config.load_config import load_config
from graphrag_storage import create_storage
from graphrag_storage.tables.neo4j_table_provider import Neo4jTableProvider
from graphrag_storage.tables.table_provider_factory import create_table_provider

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_WORKSPACE = ROOT / "rag_workspace"


async def rebuild(args: argparse.Namespace) -> int:
    """Resolve the active namespace and rebuild all graph-native edges."""
    config = load_config(root_dir=args.workspace.resolve())
    if args.run_dir is not None:
        config.output_storage.base_dir = str(args.run_dir.resolve() / "output")
    config.table_provider.namespace = args.namespace  # type: ignore[attr-defined]
    provider = create_table_provider(
        config.table_provider, create_storage(config.output_storage)
    )
    if not isinstance(provider, Neo4jTableProvider):
        raise TypeError("Projection recovery requires a Neo4j table provider")

    active_provider = await provider.get_active_provider()
    try:
        counts = await active_provider.rebuild_graph_projection()
    finally:
        await provider.close()

    print(f"Rebuilt Neo4j projection for namespace: {args.namespace}")
    print(json.dumps(counts, indent=2, ensure_ascii=False))
    required = ("RELATED", "MENTIONS")
    missing = [relation for relation in required if counts.get(relation, 0) == 0]
    if missing:
        print(f"Warning: required projection edges remain empty: {', '.join(missing)}")
        return 1
    return 0


def parse_args() -> argparse.Namespace:
    """Parse Neo4j projection recovery options."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace", type=Path, default=DEFAULT_WORKSPACE)
    parser.add_argument("--run-dir", type=Path)
    parser.add_argument("--namespace", default="graphrag-bench-incremental-101")
    return parser.parse_args()


def main() -> int:
    """Run projection recovery."""
    return asyncio.run(rebuild(parse_args()))


if __name__ == "__main__":
    raise SystemExit(main())
