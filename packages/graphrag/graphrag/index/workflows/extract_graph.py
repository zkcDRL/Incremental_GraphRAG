# Copyright (c) 2024 Microsoft Corporation.
# Licensed under the MIT License

"""A module containing run_workflow method definition."""

import logging
from typing import TYPE_CHECKING

import pandas as pd
from graphrag_llm.completion import create_completion

from graphrag.cache.cache_key_creator import cache_key_creator
from graphrag.callbacks.workflow_callbacks import WorkflowCallbacks
from graphrag.config.enums import AsyncType
from graphrag.config.models.graph_rag_config import GraphRagConfig
from graphrag.data_model.data_reader import DataReader
from graphrag.index.operations.extract_graph.extract_graph import (
    extract_graph as extractor,
)
from graphrag.index.operations.summarize_descriptions.summarize_descriptions import (
    summarize_descriptions,
)
from graphrag.index.typing.context import PipelineRunContext
from graphrag.index.typing.workflow import WorkflowFunctionOutput

if TYPE_CHECKING:
    from graphrag_llm.completion import LLMCompletion

logger = logging.getLogger(__name__)


async def run_workflow(
    config: GraphRagConfig,
    context: PipelineRunContext,
) -> WorkflowFunctionOutput:
    """All the steps to create the base entity graph."""
    logger.info("Workflow started: extract_graph")
    reader = DataReader(context.output_table_provider)
    text_units = await reader.text_units()

    extraction_model_config = config.get_completion_model_config(
        config.extract_graph.completion_model_id
    )
    extraction_prompts = config.extract_graph.resolved_prompts()
    extraction_model = create_completion(
        extraction_model_config,
        cache=context.cache.child(config.extract_graph.model_instance_name),
        cache_key_creator=cache_key_creator,
    )

    summarization_model_config = config.get_completion_model_config(
        config.summarize_descriptions.completion_model_id
    )
    summarization_prompts = config.summarize_descriptions.resolved_prompts()
    summarization_model = create_completion(
        summarization_model_config,
        cache=context.cache.child(config.summarize_descriptions.model_instance_name),
        cache_key_creator=cache_key_creator,
    )

    entities, relationships, raw_entities, raw_relationships = await extract_graph(
        text_units=text_units,
        callbacks=context.callbacks,
        extraction_model=extraction_model,
        extraction_prompt=extraction_prompts.extraction_prompt,
        entity_types=config.extract_graph.entity_types,
        max_gleanings=config.extract_graph.max_gleanings,
        extraction_num_threads=config.concurrent_requests,
        extraction_async_type=config.async_mode,
        summarization_model=summarization_model,
        max_summary_length=config.summarize_descriptions.max_length,
        max_input_tokens=config.summarize_descriptions.max_input_tokens,
        summarization_prompt=summarization_prompts.summarize_prompt,
        summarization_num_threads=config.concurrent_requests,
    )

    await context.output_table_provider.write_dataframe("entities", entities)
    await context.output_table_provider.write_dataframe("relationships", relationships)

    if config.snapshots.raw_graph:
        await context.output_table_provider.write_dataframe(
            "raw_entities", raw_entities
        )
        await context.output_table_provider.write_dataframe(
            "raw_relationships", raw_relationships
        )

    logger.info("Workflow completed: extract_graph")
    return WorkflowFunctionOutput(
        result={
            "entities": entities,
            "relationships": relationships,
        }
    )


async def extract_graph(
    text_units: pd.DataFrame,
    callbacks: WorkflowCallbacks,
    extraction_model: "LLMCompletion",
    extraction_prompt: str,
    entity_types: list[str],
    max_gleanings: int,
    extraction_num_threads: int,
    extraction_async_type: AsyncType,
    summarization_model: "LLMCompletion",
    max_summary_length: int,
    max_input_tokens: int,
    summarization_prompt: str,
    summarization_num_threads: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """All the steps to create the base entity graph."""
    # this returns a graph for each text unit, to be merged later
    extracted_entities, extracted_relationships = await extractor(
        text_units=text_units,
        callbacks=callbacks,
        text_column="text",
        id_column="id",
        model=extraction_model,
        prompt=extraction_prompt,
        entity_types=entity_types,
        max_gleanings=max_gleanings,
        num_threads=extraction_num_threads,
        async_type=extraction_async_type,
    )

    if len(extracted_entities) == 0:
        error_msg = "Graph Extraction failed. No entities detected during extraction."
        logger.error(error_msg)
        raise ValueError(error_msg)

    if len(extracted_relationships) == 0:
        error_msg = (
            "Graph Extraction failed. No relationships detected during extraction."
        )
        logger.error(error_msg)
        raise ValueError(error_msg)

    # copy these as is before any summarization
    raw_entities = extracted_entities.copy()
    raw_relationships = extracted_relationships.copy()

    entities, relationships = await get_summarized_entities_relationships(
        extracted_entities=extracted_entities,
        extracted_relationships=extracted_relationships,
        callbacks=callbacks,
        model=summarization_model,
        max_summary_length=max_summary_length,
        max_input_tokens=max_input_tokens,
        summarization_prompt=summarization_prompt,
        num_threads=summarization_num_threads,
    )

    return (entities, relationships, raw_entities, raw_relationships)


async def get_summarized_entities_relationships(
    extracted_entities: pd.DataFrame,
    extracted_relationships: pd.DataFrame,
    callbacks: WorkflowCallbacks,
    model: "LLMCompletion",
    max_summary_length: int,
    max_input_tokens: int,
    summarization_prompt: str,
    num_threads: int,
    changed_entity_titles: set[str] | None = None,
    changed_relationship_keys: set[tuple[str, str]] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Summarize all rows by default, or only changed rows for an update."""
    selective = changed_entity_titles is not None or changed_relationship_keys is not None
    if not selective:
        entities_to_summarize = extracted_entities
        relationships_to_summarize = extracted_relationships
    else:
        entity_titles = changed_entity_titles or set()
        relationship_keys = changed_relationship_keys or set()
        entities_to_summarize = extracted_entities[
            extracted_entities["title"].astype(str).isin(entity_titles)
        ]
        relationships_to_summarize = extracted_relationships[
            extracted_relationships.apply(
                lambda row: (str(row["source"]), str(row["target"])) in relationship_keys,
                axis=1,
            )
        ]

    entity_summaries, relationship_summaries = await summarize_descriptions(
        entities_df=entities_to_summarize,
        relationships_df=relationships_to_summarize,
        callbacks=callbacks,
        model=model,
        max_summary_length=max_summary_length,
        max_input_tokens=max_input_tokens,
        prompt=summarization_prompt,
        num_threads=num_threads,
    )

    if not selective:
        relationships = extracted_relationships.drop(columns=["description"]).merge(
            relationship_summaries, on=["source", "target"], how="left"
        )
        entities = extracted_entities.drop(columns=["description"]).merge(
            entity_summaries, on="title", how="left"
        )
        return entities, relationships

    # Rows outside the changed sets retain their existing finalized description.
    def existing_description(value: object) -> str:
        if isinstance(value, (list, tuple)):
            return str(value[0]) if value else ""
        return "" if value is None else str(value)

    entities = extracted_entities.copy()
    entity_summary_map = (
        dict(
            zip(
                entity_summaries["title"],
                entity_summaries["description"],
                strict=False,
            )
        )
        if "title" in entity_summaries.columns
        else {}
    )
    entities["description"] = entities.apply(
        lambda row: entity_summary_map.get(
            row["title"], existing_description(row["description"])
        ),
        axis=1,
    )

    relationships = extracted_relationships.copy()
    relationship_summary_map = {
        (row["source"], row["target"]): row["description"]
        for _, row in relationship_summaries.iterrows()
    }
    relationships["description"] = relationships.apply(
        lambda row: relationship_summary_map.get(
            (row["source"], row["target"]), existing_description(row["description"])
        ),
        axis=1,
    )
    return entities, relationships
