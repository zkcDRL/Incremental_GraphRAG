# Copyright (c) 2024 Microsoft Corporation.
# Licensed under the MIT License

"""Adaptive community update workflow."""

import logging

from graphrag_storage.tables.table_provider import TableProvider

from graphrag.config.models.graph_rag_config import GraphRagConfig
from graphrag.data_model.data_reader import DataReader
from graphrag.index.run.utils import get_update_table_providers
from graphrag.index.typing.context import PipelineRunContext
from graphrag.index.typing.workflow import WorkflowFunctionOutput
from graphrag.index.update.communities import (
    _update_and_merge_communities,
    update_communities_adaptively,
)

logger = logging.getLogger(__name__)


async def run_workflow(
    config: GraphRagConfig,
    context: PipelineRunContext,
) -> WorkflowFunctionOutput:
    """Select and execute the incremental community update strategy."""
    logger.info("Workflow started: update_communities")
    output_provider, previous_provider, _ = get_update_table_providers(
        config,
        context.state["update_timestamp"],
        context.final_output_table_provider,
    )
    previous_reader = DataReader(previous_provider)
    old_communities = await previous_reader.communities()
    old_entities = await previous_reader.entities()
    old_relationships = await previous_reader.relationships()
    merged_entities = context.state["incremental_update_merged_entities"]
    merged_relationships = context.state["incremental_update_merged_relationships"]

    result = update_communities_adaptively(
        old_communities,
        old_entities,
        old_relationships,
        merged_entities,
        merged_relationships,
        max_cluster_size=config.cluster_graph.max_cluster_size,
        use_lcc=config.cluster_graph.use_lcc,
        seed=config.cluster_graph.seed,
        local_edge_ratio=config.update.local_edge_ratio,
        local_community_ratio=config.update.local_community_ratio,
        cold_start_edge_ratio=config.update.cold_start_edge_ratio,
        community_hops=config.update.community_hops,
    )
    await output_provider.write_dataframe("communities", result.communities)

    metrics = result.metrics
    context.state["incremental_update_community_strategy"] = metrics.strategy.value
    context.state["incremental_update_community_metrics"] = {
        "changed_edge_count": metrics.changed_edge_count,
        "edge_union_count": metrics.edge_union_count,
        "affected_edge_ratio": metrics.affected_edge_ratio,
        "affected_community_ratio": metrics.affected_community_ratio,
        "direct_community_ids": list(metrics.direct_community_ids),
        "expanded_community_ids": list(metrics.expanded_community_ids),
    }
    context.state["incremental_update_changed_community_ids"] = list(
        result.changed_community_ids
    )
    context.state["incremental_update_community_id_mapping"] = {
        int(value): int(value) for value in result.communities["community"].astype(int)
    }

    logger.info(
        "Community strategy=%s changed_edges=%d/%d (%.4f) affected_communities=%.4f",
        metrics.strategy.value,
        metrics.changed_edge_count,
        metrics.edge_union_count,
        metrics.affected_edge_ratio,
        metrics.affected_community_ratio,
    )
    logger.info("Workflow completed: update_communities")
    return WorkflowFunctionOutput(result=result.communities)


async def _update_communities(
    previous_table_provider: TableProvider,
    delta_table_provider: TableProvider,
    output_table_provider: TableProvider,
) -> dict[int, int]:
    """Retain the former append-only helper for API compatibility."""
    old_communities = await DataReader(previous_table_provider).communities()
    delta_communities = await DataReader(delta_table_provider).communities()
    merged, mapping = _update_and_merge_communities(old_communities, delta_communities)
    await output_table_provider.write_dataframe("communities", merged)
    return mapping
