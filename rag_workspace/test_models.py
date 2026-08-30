#!/usr/bin/env python3
# Copyright (c) 2026 Microsoft Corporation.
# Licensed under the MIT License
# ruff: noqa: T201, EM101, TRY003
"""Smoke-test the completion and embedding models configured for this workspace."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from graphrag.config.load_config import load_config
from graphrag_llm.completion import create_completion
from graphrag_llm.embedding import create_embedding
from graphrag_llm.utils import gather_completion_response


def test_completion(config) -> None:
    """Send a minimal deterministic completion request."""
    model_config = config.get_completion_model_config(
        config.extract_graph.completion_model_id
    )
    model = create_completion(model_config)
    response = model.completion(
        messages=[
            {"role": "system", "content": "Reply with exactly: OK"},
            {"role": "user", "content": "Connectivity test"},
        ],
        temperature=0,
        max_completion_tokens=16,
    )
    text = gather_completion_response(response).strip()
    if not text:
        raise RuntimeError("Completion endpoint returned an empty response.")
    print(f"[PASS] Completion model: {model_config.model}")
    print(f"       Response: {text[:120]}")


def test_embedding(config) -> None:
    """Request one embedding and validate its numeric vector."""
    model_config = config.get_embedding_model_config(
        config.embed_text.embedding_model_id
    )
    model = create_embedding(model_config)
    response = model.embedding(input=["GraphRAG connectivity test"])
    if not response.data or not response.data[0].embedding:
        raise RuntimeError("Embedding endpoint returned no vector data.")
    vector = response.data[0].embedding
    if not all(isinstance(value, (int, float)) for value in vector):
        raise RuntimeError("Embedding endpoint returned a non-numeric vector.")
    print(f"[PASS] Embedding model: {model_config.model}")
    print(f"       Dimensions: {len(vector)}")
    print(f"       First values: {vector[:3]}")


def main() -> int:
    """Load workspace configuration and run both endpoint checks."""
    parser = argparse.ArgumentParser(
        description="Test the GraphRAG completion and embedding model configuration."
    )
    parser.add_argument(
        "--root",
        type=Path,
        default=Path(__file__).resolve().parent,
        help="GraphRAG runtime root containing settings.yaml and .env.",
    )
    args = parser.parse_args()
    root = args.root.resolve()

    try:
        config = load_config(root_dir=root)
        print(f"Loaded configuration: {root / 'settings.yaml'}")
        test_completion(config)
        test_embedding(config)
    except Exception as error:  # noqa: BLE001
        print(f"[FAIL] {type(error).__name__}: {error}", file=sys.stderr)
        return 1

    print("All configured model endpoints are usable.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
