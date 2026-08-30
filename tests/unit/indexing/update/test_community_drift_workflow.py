# Copyright (C) 2026 Microsoft Corporation.
# Licensed under the MIT License

"""Tests for cumulative drift escalation in the community workflow."""

from types import SimpleNamespace

import pandas as pd
import pytest
from graphrag.index.update.communities import (
    CommunityUpdateMetrics,
    CommunityUpdateResult,
    CommunityUpdateStrategy,
)
from graphrag.index.workflows import update_communities as workflow


def _result(strategy: CommunityUpdateStrategy) -> CommunityUpdateResult:
    return CommunityUpdateResult(
        communities=pd.DataFrame(),
        metrics=CommunityUpdateMetrics(
            strategy=strategy,
            changed_edge_count=0,
            edge_union_count=100,
            affected_edge_ratio=0.0,
            direct_community_ids=(),
            expanded_community_ids=(),
            affected_community_ratio=0.0,
        ),
        changed_community_ids=(),
    )


def _config():
    return SimpleNamespace(
        update=SimpleNamespace(
            max_consecutive_local_updates=20,
            max_cumulative_edge_change_ratio=0.10,
            max_cumulative_membership_churn=0.10,
            max_modularity_drop=0.01,
            community_hops=1,
        ),
        cluster_graph=SimpleNamespace(max_cluster_size=10, use_lcc=False, seed=42),
    )


def test_twentieth_local_update_escalates_to_seeded_and_resets(monkeypatch):
    seeded = _result(CommunityUpdateStrategy.GLOBAL_SEEDED)
    monkeypatch.setattr(workflow, "community_modularity", lambda *_args: 0.5)
    monkeypatch.setattr(workflow, "root_membership_churn", lambda *_args: 0.0)
    monkeypatch.setattr(workflow, "_rerun_global", lambda *_args: seeded)

    result, state, reasons = workflow._apply_drift_policy(  # noqa: SLF001
        _result(CommunityUpdateStrategy.LOCAL),
        {
            "consecutive_local_updates": 19,
            "baseline_modularity": 0.5,
        },
        pd.DataFrame(),
        pd.DataFrame(),
        pd.DataFrame(),
        pd.DataFrame(),
        pd.DataFrame(),
        _config(),
    )

    assert result.metrics.strategy == CommunityUpdateStrategy.GLOBAL_SEEDED
    assert reasons == ("consecutive_local_updates",)
    assert state.consecutive_local_updates == 0
    assert state.cumulative_edge_change_ratio == pytest.approx(0.0)


def test_seeded_modularity_drop_escalates_to_cold(monkeypatch):
    cold = _result(CommunityUpdateStrategy.GLOBAL_COLD)
    modularities = iter([0.49, 0.48])
    monkeypatch.setattr(
        workflow, "community_modularity", lambda *_args: next(modularities)
    )
    monkeypatch.setattr(workflow, "root_membership_churn", lambda *_args: 0.0)
    monkeypatch.setattr(workflow, "_rerun_global", lambda *_args: cold)

    result, state, reasons = workflow._apply_drift_policy(  # noqa: SLF001
        _result(CommunityUpdateStrategy.GLOBAL_SEEDED),
        {"baseline_modularity": 0.5},
        pd.DataFrame(),
        pd.DataFrame(),
        pd.DataFrame(),
        pd.DataFrame(),
        pd.DataFrame(),
        _config(),
    )

    assert result.metrics.strategy == CommunityUpdateStrategy.GLOBAL_COLD
    assert reasons == ("seeded_modularity_drop",)
    assert state.baseline_modularity == pytest.approx(0.48)
