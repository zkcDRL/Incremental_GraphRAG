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
    CommunityDriftState,
    CommunityUpdateStrategy,
    _update_and_merge_communities,
    community_modularity,
    drift_threshold_reasons,
    relative_modularity_drop,
    root_membership_churn,
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
    result, drift_state, drift_reasons = _apply_drift_policy(
        result,
        context.state.get("community_drift_state"),
        old_communities,
        old_entities,
        old_relationships,
        merged_entities,
        merged_relationships,
        config,
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
    context.state["community_drift_state"] = drift_state.to_dict()
    context.state["incremental_update_drift_trigger_reasons"] = list(drift_reasons)

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


def _apply_drift_policy(
    result,
    persisted_state: dict | None,
    old_communities,
    old_entities,
    old_relationships,
    merged_entities,
    merged_relationships,
    config: GraphRagConfig,
):
    """Accumulate local drift and escalate to seeded/cold global Leiden."""
    previous = CommunityDriftState.from_dict(persisted_state)
    current_modularity = community_modularity(
        merged_relationships, result.communities, merged_entities
    )
    baseline = previous.baseline_modularity
    if baseline is None:
        baseline = community_modularity(
            old_relationships, old_communities, old_entities
        )
    membership_churn = root_membership_churn(old_communities, result.communities)

    # A local result that already exceeds the modularity threshold is promoted below.
    # Planned global seeded runs also need validation before they become the new baseline.
    if result.metrics.strategy == CommunityUpdateStrategy.GLOBAL_SEEDED:
        seeded_drop = relative_modularity_drop(baseline, current_modularity)
        if seeded_drop >= config.update.max_modularity_drop:
            logger.info(
                "Planned seeded Leiden remains below baseline; running cold Leiden"
            )
            result = _rerun_global(
                old_communities,
                old_entities,
                old_relationships,
                merged_entities,
                merged_relationships,
                config,
                CommunityUpdateStrategy.GLOBAL_COLD,
            )
            current_modularity = community_modularity(
                merged_relationships, result.communities, merged_entities
            )
            return result, CommunityDriftState(
                baseline_modularity=current_modularity,
                current_modularity=current_modularity,
            ), ("seeded_modularity_drop",)

    if result.metrics.strategy == CommunityUpdateStrategy.LOCAL:
        drift_state = CommunityDriftState(
            consecutive_local_updates=previous.consecutive_local_updates + 1,
            cumulative_edge_change_ratio=min(
                1.0,
                previous.cumulative_edge_change_ratio
                + result.metrics.affected_edge_ratio,
            ),
            cumulative_membership_churn=min(
                1.0,
                previous.cumulative_membership_churn + membership_churn,
            ),
            baseline_modularity=baseline,
            current_modularity=current_modularity,
            modularity_drop=relative_modularity_drop(baseline, current_modularity),
        )
        reasons = drift_threshold_reasons(
            drift_state,
            max_consecutive_local_updates=config.update.max_consecutive_local_updates,
            max_cumulative_edge_change_ratio=(
                config.update.max_cumulative_edge_change_ratio
            ),
            max_cumulative_membership_churn=(
                config.update.max_cumulative_membership_churn
            ),
            max_modularity_drop=config.update.max_modularity_drop,
        )
        if reasons:
            logger.info("Cumulative drift triggered global seeded Leiden: %s", reasons)
            result = _rerun_global(
                old_communities,
                old_entities,
                old_relationships,
                merged_entities,
                merged_relationships,
                config,
                CommunityUpdateStrategy.GLOBAL_SEEDED,
            )
            seeded_modularity = community_modularity(
                merged_relationships, result.communities, merged_entities
            )
            seeded_drop = relative_modularity_drop(baseline, seeded_modularity)
            if seeded_drop >= config.update.max_modularity_drop:
                logger.info(
                    "Seeded Leiden modularity remains below baseline; running cold Leiden"
                )
                reasons = (*reasons, "seeded_modularity_drop")
                result = _rerun_global(
                    old_communities,
                    old_entities,
                    old_relationships,
                    merged_entities,
                    merged_relationships,
                    config,
                    CommunityUpdateStrategy.GLOBAL_COLD,
                )
                seeded_modularity = community_modularity(
                    merged_relationships, result.communities, merged_entities
                )
            return result, CommunityDriftState(
                baseline_modularity=seeded_modularity,
                current_modularity=seeded_modularity,
            ), reasons
        return result, drift_state, reasons

    # Any planned global run establishes a fresh baseline and clears cumulative drift.
    return result, CommunityDriftState(
        baseline_modularity=current_modularity,
        current_modularity=current_modularity,
    ), ()


def _rerun_global(
    old_communities,
    old_entities,
    old_relationships,
    merged_entities,
    merged_relationships,
    config: GraphRagConfig,
    strategy: CommunityUpdateStrategy,
):
    """Force adaptive clustering into the requested global strategy."""
    if strategy == CommunityUpdateStrategy.GLOBAL_SEEDED:
        local_edge_ratio = 0.0
        cold_start_edge_ratio = 1.0
    else:
        local_edge_ratio = 0.0
        cold_start_edge_ratio = -1.0
    return update_communities_adaptively(
        old_communities,
        old_entities,
        old_relationships,
        merged_entities,
        merged_relationships,
        max_cluster_size=config.cluster_graph.max_cluster_size,
        use_lcc=config.cluster_graph.use_lcc,
        seed=config.cluster_graph.seed,
        local_edge_ratio=local_edge_ratio,
        local_community_ratio=0.0,
        cold_start_edge_ratio=cold_start_edge_ratio,
        community_hops=config.update.community_hops,
    )


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
