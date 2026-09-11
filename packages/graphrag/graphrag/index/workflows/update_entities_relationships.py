# Copyright (c) 2024 Microsoft Corporation.
# Licensed under the MIT License

"""A module containing run_workflow method definition."""

import logging

import pandas as pd
from graphrag_cache import Cache
from graphrag_llm.completion import create_completion
from graphrag_storage.tables.neo4j_table_provider import Neo4jTableProvider
from graphrag_storage.tables.table_provider import TableProvider

from graphrag.cache.cache_key_creator import cache_key_creator
from graphrag.callbacks.workflow_callbacks import WorkflowCallbacks
from graphrag.config.models.graph_rag_config import GraphRagConfig
from graphrag.data_model.data_reader import DataReader
from graphrag.index.operations.extract_graph.utils import (
    filter_orphan_relationships,
)
from graphrag.index.run.utils import get_update_table_providers
from graphrag.index.typing.context import PipelineRunContext
from graphrag.index.typing.workflow import WorkflowFunctionOutput
from graphrag.index.update.communities import changed_edge_keys
from graphrag.index.update.entities import _group_and_resolve_entities
from graphrag.index.update.relationships import _update_and_merge_relationships
from graphrag.index.workflows.extract_graph import get_summarized_entities_relationships

logger = logging.getLogger(__name__)


async def run_workflow(
    config: GraphRagConfig,
    context: PipelineRunContext,
) -> WorkflowFunctionOutput:
    """Update the entities and relationships from a incremental index run."""
    logger.info("Workflow started: update_entities_relationships")
    output_table_provider, previous_table_provider, delta_table_provider = (
        get_update_table_providers(
            config,
            context.state["update_timestamp"],
            context.final_output_table_provider,
        )
    )

    (
        merged_entities_df,
        merged_relationships_df,
        entity_id_mapping,
    ) = await _update_entities_and_relationships(
        previous_table_provider,
        delta_table_provider,
        output_table_provider,
        config,
        context.cache,
        context.callbacks,
    )

    context.state["incremental_update_merged_entities"] = merged_entities_df
    context.state["incremental_update_merged_relationships"] = merged_relationships_df
    context.state["incremental_update_entity_id_mapping"] = entity_id_mapping

    logger.info("Workflow completed: update_entities_relationships")
    return WorkflowFunctionOutput(result=None)


async def _update_entities_and_relationships(
    previous_table_provider: TableProvider,
    delta_table_provider: TableProvider,
    output_table_provider: TableProvider,
    config: GraphRagConfig,
    cache: Cache,
    callbacks: WorkflowCallbacks,
) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """Update Final Entities  and Relationships output."""
    if all(
        isinstance(provider, Neo4jTableProvider)
        for provider in (
            previous_table_provider,
            delta_table_provider,
            output_table_provider,
        )
    ):
        return await _update_neo4j_entities_and_relationships(
            previous_table_provider,
            delta_table_provider,
            output_table_provider,
            config,
            cache,
            callbacks,
        )

    old_entities = await DataReader(previous_table_provider).entities()
    delta_entities = await DataReader(delta_table_provider).entities()

    merged_entities_df, entity_id_mapping = _group_and_resolve_entities(
        old_entities, delta_entities
    )

    # Update Relationships
    old_relationships = await DataReader(previous_table_provider).relationships()
    delta_relationships = await DataReader(delta_table_provider).relationships()
    merged_relationships_df = _update_and_merge_relationships(
        old_relationships,
        delta_relationships,
    )

    merged_relationships_df = filter_orphan_relationships(
        merged_relationships_df, merged_entities_df
    )

    summarization_model_config = config.get_completion_model_config(
        config.summarize_descriptions.completion_model_id
    )
    prompts = config.summarize_descriptions.resolved_prompts()
    model = create_completion(
        summarization_model_config,
        cache=cache.child("summarize_descriptions"),
        cache_key_creator=cache_key_creator,
    )

    (
        merged_entities_df,
        merged_relationships_df,
    ) = await get_summarized_entities_relationships(
        extracted_entities=merged_entities_df,
        extracted_relationships=merged_relationships_df,
        callbacks=callbacks,
        model=model,
        max_summary_length=config.summarize_descriptions.max_length,
        max_input_tokens=config.summarize_descriptions.max_input_tokens,
        summarization_prompt=prompts.summarize_prompt,
        num_threads=config.concurrent_requests,
        changed_entity_titles=set(delta_entities["title"].astype(str)),
        changed_relationship_keys={
            (str(source), str(target))
            for source, target in zip(
                delta_relationships["source"],
                delta_relationships["target"],
                strict=False,
            )
        },
    )

    # Save the updated entities back to storage
    await output_table_provider.write_dataframe("entities", merged_entities_df)
    await output_table_provider.write_dataframe(
        "relationships", merged_relationships_df
    )

    return merged_entities_df, merged_relationships_df, entity_id_mapping


async def _update_neo4j_entities_and_relationships(
    previous: Neo4jTableProvider,
    delta: Neo4jTableProvider,
    output: Neo4jTableProvider,
    config: GraphRagConfig,
    cache: Cache,
    callbacks: WorkflowCallbacks,
) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """Merge only delta entity keys and their incident relationship rows."""
    delta_entities = await DataReader(delta).entities()
    delta_relationships = await DataReader(delta).relationships()
    entity_titles = set(delta_entities["title"].astype(str))
    old_entities = await previous.read_entities_by_titles(entity_titles)
    next_entity_id = (
        await output.max_numeric_property("entities", "human_readable_id") + 1
    )
    merged_entities, entity_id_mapping = _group_and_resolve_entities(
        old_entities,
        delta_entities,
        initial_human_readable_id=next_entity_id,
    )

    relationship_keys = {
        (str(source), str(target))
        for source, target in zip(
            delta_relationships["source"],
            delta_relationships["target"],
            strict=False,
        )
    }
    old_relationships = await previous.read_relationships_by_keys(relationship_keys)
    next_relationship_id = (
        await output.max_numeric_property("relationships", "human_readable_id") + 1
    )
    merged_relationships = _update_and_merge_relationships(
        old_relationships,
        delta_relationships,
        initial_human_readable_id=next_relationship_id,
    )
    merged_relationships = filter_orphan_relationships(
        merged_relationships, merged_entities
    )
    changed_keys = changed_edge_keys(old_relationships, merged_relationships)

    summarization_model_config = config.get_completion_model_config(
        config.summarize_descriptions.completion_model_id
    )
    prompts = config.summarize_descriptions.resolved_prompts()
    model = create_completion(
        summarization_model_config,
        cache=cache.child("summarize_descriptions"),
        cache_key_creator=cache_key_creator,
    )
    merged_entities, merged_relationships = await get_summarized_entities_relationships(
        extracted_entities=merged_entities,
        extracted_relationships=merged_relationships,
        callbacks=callbacks,
        model=model,
        max_summary_length=config.summarize_descriptions.max_length,
        max_input_tokens=config.summarize_descriptions.max_input_tokens,
        summarization_prompt=prompts.summarize_prompt,
        num_threads=config.concurrent_requests,
        changed_entity_titles=entity_titles,
        changed_relationship_keys=relationship_keys,
    )

    await output.upsert_dataframe("entities", merged_entities)
    await output.upsert_dataframe("relationships", merged_relationships)

    affected_titles = {str(value) for key in changed_keys for value in key}
    incident_relationships = await output.read_relationships_for_titles(affected_titles)
    degree_titles = (
        affected_titles
        | set(incident_relationships.get("source", pd.Series(dtype=str)).astype(str))
        | set(incident_relationships.get("target", pd.Series(dtype=str)).astype(str))
    )
    endpoint_entities = await output.read_entities_by_titles(degree_titles)
    degrees = {
        str(row["title"]): int(row["degree"])
        for _, row in endpoint_entities.iterrows()
    }
    affected_degrees = dict.fromkeys(affected_titles, 0)
    for source, target in incident_relationships[["source", "target"]].itertuples(
        index=False, name=None
    ):
        source = str(source)
        target = str(target)
        if source in affected_degrees:
            affected_degrees[source] += 1
        if target in affected_degrees:
            affected_degrees[target] += 1
    degrees.update(affected_degrees)
    if not incident_relationships.empty:
        incident_relationships["combined_degree"] = [
            degrees.get(str(source), 0) + degrees.get(str(target), 0)
            for source, target in zip(
                incident_relationships["source"],
                incident_relationships["target"],
                strict=False,
            )
        ]
        await output.upsert_dataframe("relationships", incident_relationships)

    affected_entities = endpoint_entities[
        endpoint_entities["title"].astype(str).isin(affected_titles)
    ].copy()
    if not affected_entities.empty:
        affected_entities["degree"] = (
            affected_entities["title"].astype(str).map(degrees).fillna(0).astype(int)
        )
        await output.upsert_dataframe("entities", affected_entities)

    # The candidate namespace is a complete server-side clone.  These frames are
    # intentionally limited to changed rows; the community workflow loads a full
    # graph only if strategy selection promotes the run to seeded/cold Leiden.
    context_state = {
        "changed_edge_keys": [list(value) for value in sorted(changed_keys)],
        "affected_entity_titles": sorted(affected_titles),
    }
    merged_entities.attrs["neo4j_incremental"] = context_state
    merged_relationships.attrs["neo4j_incremental"] = context_state
    return merged_entities, merged_relationships, entity_id_mapping
