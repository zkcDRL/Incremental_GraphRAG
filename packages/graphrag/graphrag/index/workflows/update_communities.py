# Copyright (c) 2024 Microsoft Corporation.
# Licensed under the MIT License

"""Adaptive community update workflow."""

import logging

import pandas as pd
from graphrag_storage.tables.neo4j_table_provider import Neo4jTableProvider
from graphrag_storage.tables.table_provider import TableProvider

from graphrag.config.models.graph_rag_config import GraphRagConfig
from graphrag.data_model.data_reader import DataReader
from graphrag.index.run.utils import get_update_table_providers
from graphrag.index.typing.context import PipelineRunContext
from graphrag.index.typing.workflow import WorkflowFunctionOutput
from graphrag.index.update.communities import (
    CommunityDriftState,
    CommunityUpdateMetrics,
    CommunityUpdateResult,
    CommunityUpdateStrategy,
    _changed_community_ids,
    _rebuild_children,
    _reconcile_ids,
    _root_title_mapping,
    _starting_communities,
    _update_and_merge_communities,
    community_modularity,
    drift_threshold_reasons,
    relative_modularity_drop,
    root_membership_churn,
    select_strategy,
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
    merged_entities = context.state["incremental_update_merged_entities"]
    merged_relationships = context.state["incremental_update_merged_relationships"]
    targeted = merged_relationships.attrs.get("neo4j_incremental")
    if (
        isinstance(output_provider, Neo4jTableProvider)
        and isinstance(previous_provider, Neo4jTableProvider)
        and isinstance(targeted, dict)
    ):
        (
            result,
            drift_state,
            drift_reasons,
            full_replacement,
        ) = await _update_neo4j_communities(
            previous_provider,
            output_provider,
            targeted,
            context.state.get("community_drift_state"),
            config,
        )
        if full_replacement:
            await output_provider.write_dataframe("communities", result.communities)
    else:
        previous_reader = DataReader(previous_provider)
        old_communities = await previous_reader.communities()
        old_entities = await previous_reader.entities()
        old_relationships = await previous_reader.relationships()
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
            adaptive_update_enabled=config.update.adaptive_community_update,
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
    history = context.state.setdefault("community_update_history", [])
    history.append({
        "update_timestamp": context.state["update_timestamp"],
        "strategy": metrics.strategy.value,
        "changed_edge_count": metrics.changed_edge_count,
        "edge_union_count": metrics.edge_union_count,
        "affected_edge_ratio": metrics.affected_edge_ratio,
        "affected_community_ratio": metrics.affected_community_ratio,
        "changed_community_count": len(result.changed_community_ids),
        "drift_trigger_reasons": list(drift_reasons),
        **drift_state.to_dict(),
    })

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


async def _update_neo4j_communities(
    previous: Neo4jTableProvider,
    output: Neo4jTableProvider,
    targeted: dict,
    persisted_drift: dict | None,
    config: GraphRagConfig,
) -> tuple[CommunityUpdateResult, CommunityDriftState, tuple[str, ...], bool]:
    """Plan from delta rows and load a bounded graph for a local Leiden update."""
    changed_edges = {
        tuple(str(item) for item in value)
        for value in targeted.get("changed_edge_keys", [])
    }
    affected_titles = {
        str(value) for value in targeted.get("affected_entity_titles", [])
    }
    edge_union_count = await output.row_count("relationships")
    affected_edge_ratio = (
        len(changed_edges) / edge_union_count if edge_union_count else 0.0
    )

    if not changed_edges:
        metrics = CommunityUpdateMetrics(
            strategy=CommunityUpdateStrategy.LOCAL,
            changed_edge_count=0,
            edge_union_count=edge_union_count,
            affected_edge_ratio=0.0,
            direct_community_ids=(),
            expanded_community_ids=(),
            affected_community_ratio=0.0,
        )
        empty = await previous.read_communities_by_ids(set())
        return (
            CommunityUpdateResult(empty, metrics, ()),
            CommunityDriftState.from_dict(persisted_drift),
            (),
            False,
        )

    old_endpoint_entities = await previous.read_entities_by_titles(affected_titles)
    old_endpoint_ids = set(
        old_endpoint_entities.get("id", pd.Series(dtype=str)).astype(str)
    )
    direct_roots = await previous.read_communities_for_entity_ids(
        old_endpoint_ids, level=0
    )
    direct_ids = set(direct_roots.get("community", pd.Series(dtype=int)).astype(int))
    root_count = await previous.count_rows_with_value("communities", "level", 0)
    affected_community_ratio = len(direct_ids) / root_count if root_count else 0.0
    strategy = (
        CommunityUpdateStrategy.GLOBAL_COLD
        if root_count == 0
        else select_strategy(
            affected_edge_ratio,
            affected_community_ratio,
            local_edge_ratio=config.update.local_edge_ratio,
            local_community_ratio=config.update.local_community_ratio,
            cold_start_edge_ratio=config.update.cold_start_edge_ratio,
            adaptive_update_enabled=config.update.adaptive_community_update,
        )
    )
    if strategy != CommunityUpdateStrategy.LOCAL:
        return await _run_neo4j_global_update(
            previous,
            output,
            config,
            persisted_drift,
        )

    expanded_ids = set(direct_ids)
    frontier = set(direct_ids)
    for _ in range(config.update.community_hops):
        if not frontier:
            break
        frontier_rows = await previous.read_communities_by_ids(frontier)
        frontier_roots = (
            frontier_rows[frontier_rows["level"].astype(int) == 0]
            if not frontier_rows.empty
            else frontier_rows
        )
        member_ids = {
            str(entity_id)
            for values in frontier_roots.get("entity_ids", pd.Series(dtype=object))
            for entity_id in values
        }
        member_entities = await previous.read_entities_by_ids(member_ids)
        member_titles = set(
            member_entities.get("title", pd.Series(dtype=str)).astype(str)
        )
        incident = await output.read_relationships_for_titles(member_titles)
        neighbor_titles = set(
            incident.get("source", pd.Series(dtype=str)).astype(str)
        ) | set(incident.get("target", pd.Series(dtype=str)).astype(str))
        neighbor_entities = await previous.read_entities_by_titles(neighbor_titles)
        neighbor_ids = set(
            neighbor_entities.get("id", pd.Series(dtype=str)).astype(str)
        )
        neighbor_roots = await previous.read_communities_for_entity_ids(
            neighbor_ids, level=0
        )
        next_ids = (
            set(neighbor_roots.get("community", pd.Series(dtype=int)).astype(int))
            - expanded_ids
        )
        expanded_ids.update(next_ids)
        frontier = next_ids

    root_rows = await previous.read_communities_by_ids(expanded_ids)
    if not root_rows.empty:
        root_rows = root_rows[root_rows["level"].astype(int) == 0]
    region_old_ids = {
        str(entity_id)
        for values in root_rows.get("entity_ids", pd.Series(dtype=object))
        for entity_id in values
    }
    root_entities = await previous.read_entities_by_ids(region_old_ids)
    region_titles = affected_titles | set(
        root_entities.get("title", pd.Series(dtype=str)).astype(str)
    )
    local_entities = await output.read_entities_by_titles(region_titles)
    local_relationships = await output.read_relationships_for_titles(
        region_titles, induced=True
    )
    edge_nodes = set(
        local_relationships.get("source", pd.Series(dtype=str)).astype(str)
    ) | set(local_relationships.get("target", pd.Series(dtype=str)).astype(str))
    if edge_nodes:
        local_entities = local_entities[
            local_entities["title"].astype(str).isin(edge_nodes)
        ].copy()
    local_entity_ids = set(local_entities.get("id", pd.Series(dtype=str)).astype(str))
    old_candidates = await previous.read_communities_for_entity_ids(local_entity_ids)
    title_to_root, _ = _root_title_mapping(root_rows, root_entities)

    from graphrag.index.workflows.create_communities import (
        build_communities_dataframe,
    )

    local_new = build_communities_dataframe(
        local_entities,
        local_relationships,
        max_cluster_size=config.cluster_graph.max_cluster_size,
        use_lcc=False,
        seed=config.cluster_graph.seed,
        starting_communities=_starting_communities(edge_nodes, title_to_root),
    )
    max_old_id = await previous.max_numeric_property("communities", "community")
    local_final = _rebuild_children(
        _reconcile_ids(local_new, old_candidates, max_old_id)
    )
    old_candidate_ids = set(
        old_candidates.get("community", pd.Series(dtype=int)).astype(int)
    )
    final_ids = set(local_final.get("community", pd.Series(dtype=int)).astype(int))
    changed_ids = set(_changed_community_ids(old_candidates, local_final))
    changed_ids.update(old_candidate_ids - final_ids)
    await output.delete_rows_by_ids(
        "communities",
        old_candidates.get("id", pd.Series(dtype=str)).astype(str),
    )
    await output.upsert_dataframe("communities", local_final)

    metrics = CommunityUpdateMetrics(
        strategy=CommunityUpdateStrategy.LOCAL,
        changed_edge_count=len(changed_edges),
        edge_union_count=edge_union_count,
        affected_edge_ratio=affected_edge_ratio,
        direct_community_ids=tuple(sorted(direct_ids)),
        expanded_community_ids=tuple(sorted(expanded_ids)),
        affected_community_ratio=affected_community_ratio,
    )
    result = CommunityUpdateResult(
        local_final,
        metrics,
        tuple(sorted(changed_ids)),
    )

    previous_state = CommunityDriftState.from_dict(persisted_drift)
    local_old_root_members = (
        sum(
            len(values)
            for values in old_candidates[
                old_candidates.get("level", pd.Series(dtype=int)).astype(int) == 0
            ].get("entity_ids", pd.Series(dtype=object))
        )
        if not old_candidates.empty
        else 0
    )
    local_churn = root_membership_churn(old_candidates, local_final)
    total_root_members = await previous.root_community_membership_count()
    global_churn = (
        local_churn * local_old_root_members / total_root_members
        if total_root_members
        else 0.0
    )
    drift_state = CommunityDriftState(
        consecutive_local_updates=previous_state.consecutive_local_updates + 1,
        cumulative_edge_change_ratio=min(
            1.0,
            previous_state.cumulative_edge_change_ratio + affected_edge_ratio,
        ),
        cumulative_membership_churn=min(
            1.0,
            previous_state.cumulative_membership_churn + global_churn,
        ),
        baseline_modularity=previous_state.baseline_modularity,
        current_modularity=previous_state.current_modularity,
        modularity_drop=previous_state.modularity_drop,
    )
    reasons = drift_threshold_reasons(
        drift_state,
        max_consecutive_local_updates=config.update.max_consecutive_local_updates,
        max_cumulative_edge_change_ratio=config.update.max_cumulative_edge_change_ratio,
        max_cumulative_membership_churn=(config.update.max_cumulative_membership_churn),
        max_modularity_drop=config.update.max_modularity_drop,
    )
    if reasons:
        global_result, global_state, global_reasons, _ = await _run_neo4j_global_update(
            previous,
            output,
            config,
            persisted_drift,
            forced_strategy=CommunityUpdateStrategy.GLOBAL_SEEDED,
        )
        return (
            global_result,
            global_state,
            tuple(dict.fromkeys((*reasons, *global_reasons))),
            True,
        )
    return result, drift_state, reasons, False


async def _run_neo4j_global_update(
    previous: Neo4jTableProvider,
    output: Neo4jTableProvider,
    config: GraphRagConfig,
    persisted_drift: dict | None,
    forced_strategy: CommunityUpdateStrategy | None = None,
) -> tuple[CommunityUpdateResult, CommunityDriftState, tuple[str, ...], bool]:
    """Stream complete tables only after strategy selection requires a global run."""
    old_communities = await previous.read_dataframe_streaming("communities")
    old_entities = await previous.read_dataframe_streaming("entities")
    old_relationships = await previous.read_dataframe_streaming("relationships")
    merged_entities = await output.read_dataframe_streaming("entities")
    merged_relationships = await output.read_dataframe_streaming("relationships")
    if forced_strategy is None:
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
            adaptive_update_enabled=config.update.adaptive_community_update,
        )
    else:
        result = _rerun_global(
            old_communities,
            old_entities,
            old_relationships,
            merged_entities,
            merged_relationships,
            config,
            forced_strategy,
        )
    result, drift_state, drift_reasons = _apply_drift_policy(
        result,
        persisted_drift,
        old_communities,
        old_entities,
        old_relationships,
        merged_entities,
        merged_relationships,
        config,
    )
    return result, drift_state, drift_reasons, True


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
            return (
                result,
                CommunityDriftState(
                    baseline_modularity=current_modularity,
                    current_modularity=current_modularity,
                ),
                ("seeded_modularity_drop",),
            )

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
            return (
                result,
                CommunityDriftState(
                    baseline_modularity=seeded_modularity,
                    current_modularity=seeded_modularity,
                ),
                reasons,
            )
        return result, drift_state, reasons

    # Any planned global run establishes a fresh baseline and clears cumulative drift.
    return (
        result,
        CommunityDriftState(
            baseline_modularity=current_modularity,
            current_modularity=current_modularity,
        ),
        (),
    )


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
