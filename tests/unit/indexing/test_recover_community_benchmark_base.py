# Copyright (c) 2026 Microsoft Corporation.
# Licensed under the MIT License

import json

import pandas as pd
import pytest

from scripts.recover_community_benchmark_base import (
    candidate_metadata,
    community_report_issues,
    load_base_rows,
    require_complete_reports,
)


def valid_report(community: int) -> dict:
    return {
        "id": f"report-{community}",
        "community": community,
        "title": "Title",
        "summary": "Summary",
        "full_content": "Full report",
        "full_content_json": json.dumps({
            "title": "Title",
            "summary": "Summary",
            "findings": [{"summary": "Finding", "explanation": "Evidence"}],
        }),
    }


def test_load_base_rows_accepts_arbitrary_size_and_rejects_duplicates(tmp_path):
    base = tmp_path / "base.jsonl"
    base.write_text(
        "\n".join(
            json.dumps({"chunk_id": chunk_id, "text": "text"})
            for chunk_id in ("a", "b")
        )
    )
    assert len(load_base_rows(base)) == 2
    base.write_text(
        "\n".join(json.dumps({"chunk_id": "a", "text": "text"}) for _ in range(2))
    )
    with pytest.raises(ValueError, match="unique chunk IDs"):
        load_base_rows(base)


def test_report_validation_requires_exact_coverage_and_nonempty_content():
    communities = pd.DataFrame({"community": [1, 2]})
    reports = pd.DataFrame([valid_report(1), valid_report(2)])
    assert require_complete_reports(communities, reports) == {
        "communities": 2,
        "community_reports": 2,
    }

    incomplete = pd.DataFrame([valid_report(1), valid_report(1), valid_report(3)])
    issues = community_report_issues(communities, incomplete)
    assert issues["missing"] == [2]
    assert issues["extra"] == [3]
    assert issues["duplicates"] == [1]

    invalid = reports.copy()
    invalid.loc[0, "full_content"] = ""
    assert community_report_issues(communities, invalid)["invalid"] == [1]


class FakeResult:
    def __init__(self, state):
        self.state = state

    async def single(self):
        if self.state is None:
            return None
        return {
            "state": self.state,
            "storage_namespace": f"target/versions/{self.state}",
        }


class FakeSession:
    def __init__(self, state):
        self.state = state

    async def __aenter__(self):
        return self

    async def __aexit__(self, *_args):
        return None

    async def run(self, *_args, **_kwargs):
        return FakeResult(self.state)


class FakeDriver:
    def __init__(self, state):
        self.state = state

    def session(self, **_kwargs):
        return FakeSession(self.state)

    async def close(self):
        return None


@pytest.mark.parametrize("state", ["building", "failed"])
async def test_candidate_metadata_accepts_unpublished_states(monkeypatch, state):
    config = type(
        "Config",
        (),
        {
            "table_provider": type(
                "Provider",
                (),
                {
                    "uri": "bolt://test",
                    "username": "u",
                    "password": "p",
                    "database": "neo4j",
                },
            )()
        },
    )()
    monkeypatch.setattr(
        "scripts.recover_community_benchmark_base.AsyncGraphDatabase.driver",
        lambda *_args, **_kwargs: FakeDriver(state),
    )
    assert (await candidate_metadata(config, "version"))["state"] == state


async def test_candidate_metadata_rejects_active_state(monkeypatch):
    config = type(
        "Config",
        (),
        {
            "table_provider": type(
                "Provider",
                (),
                {
                    "uri": "bolt://test",
                    "username": "u",
                    "password": "p",
                    "database": "neo4j",
                },
            )()
        },
    )()
    monkeypatch.setattr(
        "scripts.recover_community_benchmark_base.AsyncGraphDatabase.driver",
        lambda *_args, **_kwargs: FakeDriver("active"),
    )
    with pytest.raises(ValueError, match="building or failed"):
        await candidate_metadata(config, "version")
