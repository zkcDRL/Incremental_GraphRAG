# Copyright (c) 2024 Microsoft Corporation.
# Licensed under the MIT License

"""Selective community report update workflow."""

import logging

import pandas as pd
from graphrag_llm.completion import create_completion
from graphrag_storage.tables.table_provider import TableProvider

from graphrag.cache.cache_key_creator import cache_key_creator
from graphrag.config.models.graph_rag_config import GraphRagConfig
from graphrag.data_model.data_reader import DataReader
from graphrag.index.run.utils import get_update_table_providers
from graphrag.index.typing.context import PipelineRunContext
from graphrag.index.typing.workflow import WorkflowFunctionOutput
from graphrag.index.update.communities import _update_and_merge_community_reports
from graphrag.index.workflows.create_community_reports import create_community_reports

logger = logging.getLogger(__name__)


async def run_workflow(
    config: GraphRagConfig,
    context: PipelineRunContext,
) -> WorkflowFunctionOutput:
    """Reuse stable reports and regenerate changed communities and ancestors."""
    logger.info("Workflow started: update_community_reports")
    output_provider, previous_provider, _ = get_update_table_providers(
        config,
        context.state["update_timestamp"],
        context.final_output_table_provider,
    )
    output_reader = DataReader(output_provider)
    previous_reader = DataReader(previous_provider)
    communities = await output_reader.communities()
    old_reports = await previous_reader.community_reports()

    changed_ids = {
        int(value)
        for value in context.state.get("incremental_update_changed_community_ids", [])
    }
    parent_by_id = {
        int(row["community"]): int(row["parent"]) for _, row in communities.iterrows()
    }
    frontier = set(changed_ids)
    while frontier:
        parents = {
            parent_by_id.get(community_id, -1)
            for community_id in frontier
            if parent_by_id.get(community_id, -1) != -1
        } - changed_ids
        changed_ids.update(parents)
        frontier = parents

    existing_report_ids = set(old_reports["community"].astype(int))
    changed_ids.update(set(parent_by_id) - existing_report_ids)
    if not changed_ids:
        await output_provider.write_dataframe("community_reports", old_reports)
        return WorkflowFunctionOutput(result=old_reports)

    affected_communities = communities[
        communities["community"].astype(int).isin(changed_ids)
    ].copy()
    relationships = await output_reader.relationships()
    entities = await output_reader.entities()
    claims = None
    if config.extract_claims.enabled and await output_provider.has("covariates"):
        claims = await output_reader.covariates()

    model_config = config.get_completion_model_config(
        config.community_reports.completion_model_id
    )
    model = create_completion(
        model_config,
        cache=context.cache.child(config.community_reports.model_instance_name),
        cache_key_creator=cache_key_creator,
    )
    prompts = config.community_reports.resolved_prompts()
    regenerated = await create_community_reports(
        relationships=relationships,
        entities=entities,
        communities=affected_communities,
        claims_input=claims,
        callbacks=context.callbacks,
        model=model,
        tokenizer=model.tokenizer,
        prompt=prompts.graph_prompt,
        max_input_length=config.community_reports.max_input_length,
        max_report_length=config.community_reports.max_length,
        num_threads=config.concurrent_requests,
        async_type=config.async_mode,
    )
    reused = old_reports[
        ~old_reports["community"].astype(int).isin(changed_ids)
        & old_reports["community"].astype(int).isin(parent_by_id)
    ]
    merged_reports = pd.concat([reused, regenerated], ignore_index=True).sort_values(
        "community", kind="stable"
    )
    await output_provider.write_dataframe("community_reports", merged_reports)
    context.state["incremental_update_regenerated_community_report_ids"] = sorted(
        changed_ids
    )
    logger.info(
        "Regenerated %d community reports and reused %d",
        len(regenerated),
        len(reused),
    )
    logger.info("Workflow completed: update_community_reports")
    return WorkflowFunctionOutput(result=merged_reports)


async def _update_community_reports(
    previous_table_provider: TableProvider,
    delta_table_provider: TableProvider,
    output_table_provider: TableProvider,
    community_id_mapping: dict[int, int],
) -> pd.DataFrame:
    """Retain the former append-only report helper for API compatibility."""
    old_reports = await DataReader(previous_table_provider).community_reports()
    delta_reports = await DataReader(delta_table_provider).community_reports()
    merged = _update_and_merge_community_reports(
        old_reports, delta_reports, community_id_mapping
    )
    await output_table_provider.write_dataframe("community_reports", merged)
    return merged
