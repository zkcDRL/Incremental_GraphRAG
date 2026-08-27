# Copyright (c) 2024 Microsoft Corporation.
# Licensed under the MIT License

"""Adaptive community operations for incremental indexing."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import Any
from uuid import uuid4

import numpy as np
import pandas as pd


class CommunityUpdateStrategy(StrEnum):
    """Available incremental community clustering strategies."""

    LOCAL = "local"
    GLOBAL_SEEDED = "global_seeded"
    GLOBAL_COLD = "global_cold"


@dataclass(frozen=True)
class CommunityUpdateMetrics:
    """Measurements and selected scope for one community update."""

    strategy: CommunityUpdateStrategy
    changed_edge_count: int
    edge_union_count: int
    affected_edge_ratio: float
    direct_community_ids: tuple[int, ...]
    expanded_community_ids: tuple[int, ...]
    affected_community_ratio: float


@dataclass(frozen=True)
class CommunityUpdateResult:
    """Final communities and update measurements."""

    communities: pd.DataFrame
    metrics: CommunityUpdateMetrics
    changed_community_ids: tuple[int, ...]


def _edge_key(row: pd.Series) -> tuple[str, str]:
    source, target = str(row["source"]), str(row["target"])
    return (source, target) if source <= target else (target, source)


def _stable_value(value: Any) -> Any:
    if isinstance(value, (list, tuple, set, np.ndarray)):
        return tuple(sorted(str(item) for item in value))
    if isinstance(value, dict):
        return tuple(sorted((str(key), _stable_value(item)) for key, item in value.items()))
    if pd.isna(value):
        return None
    return value


def _edge_fingerprints(edges: pd.DataFrame) -> dict[tuple[str, str], tuple[Any, ...]]:
    if edges.empty:
        return {}
    fields = [
        field
        for field in ("weight", "description", "text_unit_ids")
        if field in edges.columns
    ]
    fingerprints: dict[tuple[str, str], tuple[Any, ...]] = {}
    for _, row in edges.iterrows():
        fingerprints[_edge_key(row)] = tuple(_stable_value(row[field]) for field in fields)
    return fingerprints


def changed_edge_keys(
    old_relationships: pd.DataFrame, merged_relationships: pd.DataFrame
) -> set[tuple[str, str]]:
    """Return added, removed, or content/weight changed undirected edges."""
    old = _edge_fingerprints(old_relationships)
    merged = _edge_fingerprints(merged_relationships)
    return {key for key in old.keys() | merged.keys() if old.get(key) != merged.get(key)}


def select_strategy(
    affected_edge_ratio: float,
    affected_community_ratio: float,
    *,
    local_edge_ratio: float,
    local_community_ratio: float,
    cold_start_edge_ratio: float,
) -> CommunityUpdateStrategy:
    """Select local, warm-started global, or cold global clustering."""
    if (
        affected_edge_ratio < local_edge_ratio
        and affected_community_ratio < local_community_ratio
    ):
        return CommunityUpdateStrategy.LOCAL
    if affected_edge_ratio <= cold_start_edge_ratio:
        return CommunityUpdateStrategy.GLOBAL_SEEDED
    return CommunityUpdateStrategy.GLOBAL_COLD


def _root_title_mapping(
    old_communities: pd.DataFrame, old_entities: pd.DataFrame
) -> tuple[dict[str, int], dict[int, set[str]]]:
    id_to_title = dict(
        zip(old_entities["id"].astype(str), old_entities["title"].astype(str), strict=False)
    )
    title_to_root: dict[str, int] = {}
    root_to_titles: dict[int, set[str]] = {}
    roots = old_communities[old_communities["level"].astype(int) == 0]
    for _, row in roots.iterrows():
        community_id = int(row["community"])
        titles = {
            id_to_title[str(entity_id)]
            for entity_id in row["entity_ids"]
            if str(entity_id) in id_to_title
        }
        root_to_titles[community_id] = titles
        for title in titles:
            title_to_root[title] = community_id
    return title_to_root, root_to_titles


def _expand_communities(
    direct_ids: set[int],
    relationships: pd.DataFrame,
    title_to_root: dict[str, int],
    hops: int,
) -> set[int]:
    adjacency: dict[int, set[int]] = {}
    for _, row in relationships.iterrows():
        source = title_to_root.get(str(row["source"]))
        target = title_to_root.get(str(row["target"]))
        if source is None or target is None or source == target:
            continue
        adjacency.setdefault(source, set()).add(target)
        adjacency.setdefault(target, set()).add(source)

    expanded = set(direct_ids)
    frontier = set(direct_ids)
    for _ in range(hops):
        frontier = set().union(*(adjacency.get(item, set()) for item in frontier)) - expanded
        expanded.update(frontier)
        if not frontier:
            break
    return expanded


def _starting_communities(
    nodes: set[str], title_to_root: dict[str, int]
) -> dict[str, int]:
    next_id = max(title_to_root.values(), default=-1) + 1
    result: dict[str, int] = {}
    for node in sorted(nodes):
        if node in title_to_root:
            result[node] = title_to_root[node]
        else:
            result[node] = next_id
            next_id += 1
    return result


def _jaccard(left: Any, right: Any) -> float:
    left_set = {str(value) for value in left}
    right_set = {str(value) for value in right}
    union = left_set | right_set
    return len(left_set & right_set) / len(union) if union else 1.0


def _reconcile_ids(
    new_communities: pd.DataFrame,
    old_candidates: pd.DataFrame,
    max_old_id: int,
) -> pd.DataFrame:
    """Greedily retain old IDs for maximum same-level member overlap."""
    from graphrag.index.workflows.create_communities import remap_community_ids

    if new_communities.empty:
        return new_communities

    candidates: list[tuple[float, int, int]] = []
    for _, new_row in new_communities.iterrows():
        same_level = old_candidates[
            old_candidates["level"].astype(int) == int(new_row["level"])
        ]
        for _, old_row in same_level.iterrows():
            score = _jaccard(new_row["entity_ids"], old_row["entity_ids"])
            if score > 0:
                candidates.append(
                    (score, int(new_row["community"]), int(old_row["community"]))
                )

    matched_new: set[int] = set()
    matched_old: set[int] = set()
    mapping: dict[int, int] = {}
    for _, new_id, old_id in sorted(candidates, key=lambda item: (-item[0], item[1], item[2])):
        if new_id not in matched_new and old_id not in matched_old:
            mapping[new_id] = old_id
            matched_new.add(new_id)
            matched_old.add(old_id)

    next_id = max_old_id + 1
    for new_id in sorted(new_communities["community"].astype(int).unique()):
        if new_id not in mapping:
            mapping[new_id] = next_id
            next_id += 1

    old_uuid = {
        int(row["community"]): str(row["id"]) for _, row in old_candidates.iterrows()
    }
    uuid_mapping = {
        mapped_id: old_uuid.get(mapped_id, str(uuid4())) for mapped_id in mapping.values()
    }
    return remap_community_ids(new_communities, mapping, uuid_mapping)


def _rebuild_children(communities: pd.DataFrame) -> pd.DataFrame:
    result = communities.copy()
    children_by_parent = (
        result[result["parent"].astype(int) != -1]
        .groupby("parent")["community"]
        .apply(lambda values: sorted({int(value) for value in values}))
        .to_dict()
    )
    result["children"] = result["community"].astype(int).apply(
        lambda community: children_by_parent.get(community, [])
    )
    result["human_readable_id"] = result["community"].astype(int)
    result["title"] = "Community " + result["community"].astype(int).astype(str)
    return result.sort_values(["level", "community"], kind="stable").reset_index(drop=True)


def _changed_community_ids(
    old_communities: pd.DataFrame, new_communities: pd.DataFrame
) -> tuple[int, ...]:
    old_by_id = {int(row["community"]): row for _, row in old_communities.iterrows()}
    changed: set[int] = set()
    for _, row in new_communities.iterrows():
        community_id = int(row["community"])
        old = old_by_id.get(community_id)
        if old is None or any(
            _stable_value(old[field]) != _stable_value(row[field])
            for field in ("parent", "children", "entity_ids", "relationship_ids", "text_unit_ids")
        ):
            changed.add(community_id)
    return tuple(sorted(changed))


def update_communities_adaptively(
    old_communities: pd.DataFrame,
    old_entities: pd.DataFrame,
    old_relationships: pd.DataFrame,
    merged_entities: pd.DataFrame,
    merged_relationships: pd.DataFrame,
    *,
    max_cluster_size: int,
    use_lcc: bool,
    seed: int | None,
    local_edge_ratio: float,
    local_community_ratio: float,
    cold_start_edge_ratio: float,
    community_hops: int,
) -> CommunityUpdateResult:
    """Apply the configured adaptive clustering strategy."""
    from graphrag.index.workflows.create_communities import build_communities_dataframe

    changed_edges = changed_edge_keys(old_relationships, merged_relationships)
    edge_union_count = len(
        set(_edge_fingerprints(old_relationships))
        | set(_edge_fingerprints(merged_relationships))
    )
    affected_edge_ratio = len(changed_edges) / edge_union_count if edge_union_count else 0.0
    changed_endpoints = {node for edge in changed_edges for node in edge}

    if old_communities.empty:
        strategy = CommunityUpdateStrategy.GLOBAL_COLD
        direct_ids: set[int] = set()
        expanded_ids: set[int] = set()
        affected_community_ratio = 0.0
        title_to_root: dict[str, int] = {}
        root_to_titles: dict[int, set[str]] = {}
    else:
        title_to_root, root_to_titles = _root_title_mapping(old_communities, old_entities)
        direct_ids = {
            title_to_root[title] for title in changed_endpoints if title in title_to_root
        }
        root_count = len(root_to_titles)
        affected_community_ratio = len(direct_ids) / root_count if root_count else 0.0
        strategy = select_strategy(
            affected_edge_ratio,
            affected_community_ratio,
            local_edge_ratio=local_edge_ratio,
            local_community_ratio=local_community_ratio,
            cold_start_edge_ratio=cold_start_edge_ratio,
        )
        expanded_ids = _expand_communities(
            direct_ids, merged_relationships, title_to_root, community_hops
        )

    max_old_id = (
        int(old_communities["community"].astype(int).max())
        if not old_communities.empty
        else -1
    )

    if strategy == CommunityUpdateStrategy.LOCAL and not changed_edges:
        final = old_communities.copy()
    elif strategy == CommunityUpdateStrategy.LOCAL:
        region_titles = set(changed_endpoints)
        for community_id in expanded_ids:
            region_titles.update(root_to_titles.get(community_id, set()))
        local_edges = merged_relationships[
            merged_relationships["source"].astype(str).isin(region_titles)
            & merged_relationships["target"].astype(str).isin(region_titles)
        ].copy()
        edge_nodes = set(local_edges["source"].astype(str)) | set(
            local_edges["target"].astype(str)
        )
        local_entities = merged_entities[
            merged_entities["title"].astype(str).isin(edge_nodes)
        ].copy()
        local_new = build_communities_dataframe(
            local_entities,
            local_edges,
            max_cluster_size=max_cluster_size,
            use_lcc=False,
            seed=seed,
            starting_communities=_starting_communities(edge_nodes, title_to_root),
        )
        old_region_ids = {
            str(entity_id)
            for _, row in old_communities.iterrows()
            if {str(value) for value in row["entity_ids"]}
            & set(local_entities["id"].astype(str))
            for entity_id in row["entity_ids"]
        }
        old_candidates = old_communities[
            old_communities["entity_ids"].apply(
                lambda values: bool({str(value) for value in values} & old_region_ids)
            )
        ]
        preserved = old_communities.drop(index=old_candidates.index)
        reconciled = _reconcile_ids(local_new, old_candidates, max_old_id)
        final = pd.concat([preserved, reconciled], ignore_index=True)
    else:
        all_nodes = set(merged_relationships["source"].astype(str)) | set(
            merged_relationships["target"].astype(str)
        )
        starting = (
            _starting_communities(all_nodes, title_to_root)
            if strategy == CommunityUpdateStrategy.GLOBAL_SEEDED
            else None
        )
        rebuilt = build_communities_dataframe(
            merged_entities,
            merged_relationships,
            max_cluster_size=max_cluster_size,
            use_lcc=use_lcc,
            seed=seed,
            starting_communities=starting,
        )
        final = _reconcile_ids(rebuilt, old_communities, max_old_id)

    final = _rebuild_children(final)
    changed_ids = _changed_community_ids(old_communities, final)
    metrics = CommunityUpdateMetrics(
        strategy=strategy,
        changed_edge_count=len(changed_edges),
        edge_union_count=edge_union_count,
        affected_edge_ratio=affected_edge_ratio,
        direct_community_ids=tuple(sorted(direct_ids)),
        expanded_community_ids=tuple(sorted(expanded_ids)),
        affected_community_ratio=affected_community_ratio,
    )
    return CommunityUpdateResult(final, metrics, changed_ids)


# Kept for callers that still use the legacy append-only merge helper.
def _update_and_merge_communities(
    old_communities: pd.DataFrame,
    delta_communities: pd.DataFrame,
) -> tuple[pd.DataFrame, dict[int, int]]:
    """Merge delta communities with non-overlapping IDs (legacy compatibility)."""
    from graphrag.index.workflows.create_communities import remap_community_ids

    old_max = old_communities["community"].fillna(0).astype(int).max()
    mapping = {
        value: value + old_max + 1
        for value in delta_communities["community"].dropna().astype(int)
    }
    mapping[-1] = -1
    remapped = remap_community_ids(delta_communities, mapping)
    return pd.concat([old_communities, remapped], ignore_index=True), mapping


def _update_and_merge_community_reports(
    old_community_reports: pd.DataFrame,
    delta_community_reports: pd.DataFrame,
    community_id_mapping: dict[int, int],
) -> pd.DataFrame:
    """Retain the former append-only report merge helper for compatibility."""
    from graphrag.data_model.schemas import COMMUNITY_REPORTS_FINAL_COLUMNS

    old = old_community_reports.copy()
    delta = delta_community_reports.copy()
    for frame in (old, delta):
        if "size" not in frame.columns:
            frame["size"] = None
        if "period" not in frame.columns:
            frame["period"] = None
    delta["community"] = delta["community"].astype(int).apply(
        lambda value: community_id_mapping.get(value, value)
    )
    delta["parent"] = delta["parent"].astype(int).apply(
        lambda value: community_id_mapping.get(value, value)
    )
    merged = pd.concat([old, delta], ignore_index=True)
    merged["community"] = merged["community"].astype(int)
    merged["human_readable_id"] = merged["community"]
    return merged.loc[:, COMMUNITY_REPORTS_FINAL_COLUMNS]
