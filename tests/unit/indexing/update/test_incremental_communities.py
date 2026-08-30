# Copyright (C) 2026 Microsoft Corporation.
# Licensed under the MIT License

"""Tests for adaptive incremental community strategy helpers."""

import pandas as pd
import pytest
from graphrag.index.update.communities import (
    CommunityDriftState,
    CommunityUpdateStrategy,
    _expand_communities,
    _reconcile_ids,
    changed_edge_keys,
    drift_threshold_reasons,
    relative_modularity_drop,
    root_membership_churn,
    select_strategy,
)


def test_selects_local_only_when_both_ratios_are_below_thresholds():
    assert (
        select_strategy(
            0.009,
            0.049,
            local_edge_ratio=0.01,
            local_community_ratio=0.05,
            cold_start_edge_ratio=0.10,
        )
        == CommunityUpdateStrategy.LOCAL
    )
    assert (
        select_strategy(
            0.009,
            0.05,
            local_edge_ratio=0.01,
            local_community_ratio=0.05,
            cold_start_edge_ratio=0.10,
        )
        == CommunityUpdateStrategy.GLOBAL_SEEDED
    )


def test_selects_seeded_at_one_and_ten_percent_boundaries():
    for edge_ratio in (0.01, 0.10):
        assert (
            select_strategy(
                edge_ratio,
                0.0,
                local_edge_ratio=0.01,
                local_community_ratio=0.05,
                cold_start_edge_ratio=0.10,
            )
            == CommunityUpdateStrategy.GLOBAL_SEEDED
        )


def test_selects_cold_above_ten_percent():
    assert (
        select_strategy(
            0.10001,
            0.0,
            local_edge_ratio=0.01,
            local_community_ratio=0.05,
            cold_start_edge_ratio=0.10,
        )
        == CommunityUpdateStrategy.GLOBAL_COLD
    )


def test_changed_edges_detect_additions_and_evidence_changes():
    old = pd.DataFrame([
        {
            "source": "B",
            "target": "A",
            "weight": 1.0,
            "description": "old",
            "text_unit_ids": ["t1"],
        }
    ])
    merged = pd.DataFrame([
        {
            "source": "A",
            "target": "B",
            "weight": 1.0,
            "description": "old",
            "text_unit_ids": ["t1", "t2"],
        },
        {
            "source": "B",
            "target": "C",
            "weight": 1.0,
            "description": "new",
            "text_unit_ids": ["t2"],
        },
    ])
    assert changed_edge_keys(old, merged) == {("A", "B"), ("B", "C")}


def test_expands_affected_communities_by_one_hop():
    relationships = pd.DataFrame([
        {"source": "A", "target": "B"},
        {"source": "B", "target": "C"},
        {"source": "C", "target": "D"},
    ])
    title_to_root = {"A": 1, "B": 2, "C": 3, "D": 4}
    assert _expand_communities({1}, relationships, title_to_root, 1) == {1, 2}
    assert _expand_communities({1}, relationships, title_to_root, 2) == {1, 2, 3}


def _community_row(
    community: int, level: int, entity_ids: list[str], row_id: str
) -> dict:
    return {
        "id": row_id,
        "human_readable_id": community,
        "community": community,
        "level": level,
        "parent": -1,
        "children": [],
        "title": f"Community {community}",
        "entity_ids": entity_ids,
        "relationship_ids": [f"r-{community}"],
        "text_unit_ids": [f"t-{community}"],
        "period": "2026-01-01",
        "size": len(entity_ids),
    }


def test_reconcile_ids_preserves_best_jaccard_match_and_uuid():
    old = pd.DataFrame([
        _community_row(10, 0, ["a", "b", "c"], "old-10"),
        _community_row(20, 0, ["x", "y"], "old-20"),
    ])
    new = pd.DataFrame([
        _community_row(0, 0, ["a", "b", "c", "d"], "new-0"),
        _community_row(1, 0, ["x", "y"], "new-1"),
    ])
    reconciled = _reconcile_ids(new, old, 20)
    assert set(reconciled["community"]) == {10, 20}
    assert reconciled.set_index("community").loc[10, "id"] == "old-10"
    assert reconciled.set_index("community").loc[20, "id"] == "old-20"


def test_drift_thresholds_trigger_at_configured_boundaries():
    state = CommunityDriftState(
        consecutive_local_updates=20,
        cumulative_edge_change_ratio=0.10,
        cumulative_membership_churn=0.10,
        baseline_modularity=0.5,
        current_modularity=0.495,
        modularity_drop=0.01,
    )
    assert drift_threshold_reasons(
        state,
        max_consecutive_local_updates=20,
        max_cumulative_edge_change_ratio=0.10,
        max_cumulative_membership_churn=0.10,
        max_modularity_drop=0.01,
    ) == (
        "consecutive_local_updates",
        "cumulative_edge_change_ratio",
        "cumulative_membership_churn",
        "modularity_drop",
    )


def test_drift_state_round_trip_preserves_persistent_values():
    state = CommunityDriftState(
        consecutive_local_updates=3,
        cumulative_edge_change_ratio=0.04,
        cumulative_membership_churn=0.02,
        baseline_modularity=0.45,
        current_modularity=0.44,
        modularity_drop=0.022,
    )
    assert CommunityDriftState.from_dict(state.to_dict()) == state


def test_relative_modularity_drop_is_non_negative():
    assert relative_modularity_drop(0.5, 0.495) == pytest.approx(0.01)
    assert relative_modularity_drop(0.5, 0.51) == pytest.approx(0.0)
    assert relative_modularity_drop(None, 0.4) == pytest.approx(0.0)


def test_root_membership_churn_counts_only_existing_members():
    old = pd.DataFrame([
        _community_row(10, 0, ["a", "b"], "old-10"),
        _community_row(20, 0, ["c", "d"], "old-20"),
    ])
    new = pd.DataFrame([
        _community_row(10, 0, ["a", "b", "c"], "new-10"),
        _community_row(20, 0, ["d", "e"], "new-20"),
    ])
    assert root_membership_churn(old, new) == pytest.approx(0.25)
