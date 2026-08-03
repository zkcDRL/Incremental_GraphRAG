# Copyright (c) 2026 Microsoft Corporation.
# Licensed under the MIT License
# ruff: noqa: I001, T201, TC003

"""Profile table reads, merges, summaries, and writes in an incremental update."""

from __future__ import annotations

import argparse
import asyncio
import json
from collections.abc import Awaitable, Callable
from dataclasses import asdict, dataclass
from pathlib import Path
from time import perf_counter
from typing import TypeVar

from graphrag_cache.memory_cache import MemoryCache

from graphrag.callbacks.noop_workflow_callbacks import NoopWorkflowCallbacks
from graphrag.config.load_config import load_config
from graphrag.data_model.data_reader import DataReader
from graphrag.index.operations.extract_graph.utils import filter_orphan_relationships
from graphrag.index.run.utils import get_update_table_providers
from graphrag.index.update.entities import _group_and_resolve_entities
from graphrag.index.update.relationships import _update_and_merge_relationships
from graphrag.index.workflows.extract_graph import get_summarized_entities_relationships
from graphrag.index.workflows.update_text_units import _update_and_merge_text_units

T = TypeVar("T")


@dataclass
class StageMeasurement:
    """Store elapsed time and row counts for one stage."""

    name: str
    elapsed_ms: float
    rows_before: int | None = None
    rows_delta: int | None = None
    rows_after: int | None = None
    skipped: bool = False


async def measure_async(
    name: str,
    operation: Callable[[], Awaitable[T]],
    *,
    rows_before: int | None = None,
    rows_delta: int | None = None,
    rows_after: int | None = None,
) -> tuple[T, StageMeasurement]:
    """Run and time an asynchronous operation."""
    started_at = perf_counter()
    result = await operation()
    return result, StageMeasurement(
        name=name,
        elapsed_ms=(perf_counter() - started_at) * 1000,
        rows_before=rows_before,
        rows_delta=rows_delta,
        rows_after=rows_after,
    )


def measure_sync(
    name: str,
    operation: Callable[[], T],
    *,
    rows_before: int | None = None,
    rows_delta: int | None = None,
    rows_after: int | None = None,
) -> tuple[T, StageMeasurement]:
    """Run and time a synchronous operation."""
    started_at = perf_counter()
    result = operation()
    return result, StageMeasurement(
        name=name,
        elapsed_ms=(perf_counter() - started_at) * 1000,
        rows_before=rows_before,
        rows_delta=rows_delta,
        rows_after=rows_after,
    )


async def profile_update(
    root_dir: Path,
    timestamp: str,
    summarize: bool,
    write_output: bool,
) -> list[StageMeasurement]:
    """Profile an extracted incremental batch."""
    config = load_config(root_dir=root_dir)
    output_provider, previous_provider, delta_provider = get_update_table_providers(
        config, timestamp
    )
    previous_reader = DataReader(previous_provider)
    delta_reader = DataReader(delta_provider)
    measurements: list[StageMeasurement] = []

    old_entities, stage = await measure_async("read_previous_entities", previous_reader.entities)
    measurements.append(stage)
    delta_entities, stage = await measure_async("read_delta_entities", delta_reader.entities)
    measurements.append(stage)
    (merged_entities, entity_id_mapping), stage = measure_sync(
        "merge_entities",
        lambda: _group_and_resolve_entities(old_entities, delta_entities.copy()),
        rows_before=len(old_entities),
        rows_delta=len(delta_entities),
    )
    stage.rows_after = len(merged_entities)
    measurements.append(stage)

    old_relationships, stage = await measure_async(
        "read_previous_relationships", previous_reader.relationships
    )
    measurements.append(stage)
    delta_relationships, stage = await measure_async(
        "read_delta_relationships", delta_reader.relationships
    )
    measurements.append(stage)
    merged_relationships, stage = measure_sync(
        "merge_relationships",
        lambda: filter_orphan_relationships(
            _update_and_merge_relationships(
                old_relationships.copy(), delta_relationships.copy()
            ),
            merged_entities,
        ),
        rows_before=len(old_relationships),
        rows_delta=len(delta_relationships),
    )
    stage.rows_after = len(merged_relationships)
    measurements.append(stage)

    old_text_units, stage = await measure_async(
        "read_previous_text_units", previous_reader.text_units
    )
    measurements.append(stage)
    delta_text_units, stage = await measure_async(
        "read_delta_text_units", delta_reader.text_units
    )
    measurements.append(stage)
    merged_text_units, stage = measure_sync(
        "merge_text_units",
        lambda: _update_and_merge_text_units(
            old_text_units.copy(), delta_text_units.copy(), entity_id_mapping
        ),
        rows_before=len(old_text_units),
        rows_delta=len(delta_text_units),
    )
    stage.rows_after = len(merged_text_units)
    measurements.append(stage)

    if summarize:
        from graphrag_llm.completion import create_completion

        model_config = config.get_completion_model_config(
            config.summarize_descriptions.completion_model_id
        )
        model = create_completion(model_config, cache=MemoryCache())
        prompts = config.summarize_descriptions.resolved_prompts()
        (merged_entities, merged_relationships), stage = await measure_async(
            "summarize_entity_and_relationship_descriptions",
            lambda: get_summarized_entities_relationships(
                extracted_entities=merged_entities,
                extracted_relationships=merged_relationships,
                callbacks=NoopWorkflowCallbacks(),
                model=model,
                max_summary_length=config.summarize_descriptions.max_length,
                max_input_tokens=config.summarize_descriptions.max_input_tokens,
                summarization_prompt=prompts.summarize_prompt,
                num_threads=config.concurrent_requests,
            ),
            rows_before=len(merged_entities) + len(merged_relationships),
            rows_after=len(merged_entities) + len(merged_relationships),
        )
        measurements.append(stage)
    else:
        measurements.append(
            StageMeasurement(
                name="summarize_entity_and_relationship_descriptions",
                elapsed_ms=0,
                skipped=True,
            )
        )

    if write_output:
        _, stage = await measure_async(
            "write_entities",
            lambda: output_provider.write_dataframe("entities", merged_entities),
            rows_after=len(merged_entities),
        )
        measurements.append(stage)
        _, stage = await measure_async(
            "write_relationships",
            lambda: output_provider.write_dataframe(
                "relationships", merged_relationships
            ),
            rows_after=len(merged_relationships),
        )
        measurements.append(stage)
        _, stage = await measure_async(
            "write_text_units",
            lambda: output_provider.write_dataframe("text_units", merged_text_units),
            rows_after=len(merged_text_units),
        )
        measurements.append(stage)

    for provider in (output_provider, previous_provider, delta_provider):
        await provider.close()
    return measurements


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True, help="GraphRAG 项目根目录")
    parser.add_argument(
        "--timestamp",
        required=True,
        help="Incremental batch timestamp produced by the update command.",
    )
    parser.add_argument(
        "--summarize",
        action="store_true",
        help="Call the LLM and measure entity and relationship description summaries.",
    )
    parser.add_argument(
        "--write-output",
        action="store_true",
        help="Write final tables and measure Neo4j write time. This replaces the current final index.",
    )
    parser.add_argument("--json", action="store_true", help="Print JSON output.")
    return parser.parse_args()


def main() -> None:
    """Run the profile and print the measurements."""
    args = parse_args()
    measurements = asyncio.run(
        profile_update(
            root_dir=args.root.resolve(),
            timestamp=args.timestamp,
            summarize=args.summarize,
            write_output=args.write_output,
        )
    )
    if args.json:
        print(json.dumps([asdict(item) for item in measurements], ensure_ascii=False))
        return

    print(f"{'阶段':<52} {'耗时(ms)':>12} {'旧行数':>10} {'增量行数':>10} {'结果行数':>10}")
    for item in measurements:
        if item.skipped:
            print(f"{item.name:<52} {'跳过':>12}")
            continue
        print(
            f"{item.name:<52} {item.elapsed_ms:>12.2f} "
            f"{item.rows_before if item.rows_before is not None else '-':>10} "
            f"{item.rows_delta if item.rows_delta is not None else '-':>10} "
            f"{item.rows_after if item.rows_after is not None else '-':>10}"
        )


if __name__ == "__main__":
    main()