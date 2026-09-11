# Copyright (C) 2026 Microsoft
# Licensed under the MIT License

"""Focused tests for the Neo4j-native incremental update path."""

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pandas as pd
import pytest
from graphrag.index.run.run_pipeline import _copy_previous_output
from graphrag.index.workflows import update_communities, update_entities_relationships
from graphrag_storage.tables.neo4j_table_provider import Neo4jTableProvider


@pytest.mark.asyncio
async def test_previous_snapshot_uses_server_side_copy_without_projection():
    source = object.__new__(Neo4jTableProvider)
    destination = object.__new__(Neo4jTableProvider)
    source_read = AsyncMock(side_effect=AssertionError("must not read through Python"))
    source.read_dataframe = source_read
    destination.defer_graph_projection = Mock()
    destination.copy_namespace_from = AsyncMock(return_value={"entities": 1})

    await _copy_previous_output(source, destination)

    destination.defer_graph_projection.assert_called_once_with()
    destination.copy_namespace_from.assert_awaited_once_with(
        source, rebuild_projection=False
    )
    source_read.assert_not_awaited()


class _TargetedProvider:
    def __init__(self, tables: dict[str, pd.DataFrame]) -> None:
        self.tables = {name: frame.copy(deep=True) for name, frame in tables.items()}
        self.full_reads: list[str] = []
        self.entity_title_queries: list[set[str]] = []
        self.relationship_key_queries: list[set[tuple[str, str]]] = []

    async def read_dataframe(self, table_name: str) -> pd.DataFrame:
        self.full_reads.append(table_name)
        return self.tables[table_name].copy(deep=True)

    async def read_entities_by_titles(self, titles: set[str]) -> pd.DataFrame:
        self.entity_title_queries.append(set(titles))
        frame = self.tables["entities"]
        return frame[frame["title"].astype(str).isin(titles)].copy()

    async def read_relationships_by_keys(
        self, keys: set[tuple[str, str]]
    ) -> pd.DataFrame:
        self.relationship_key_queries.append(set(keys))
        frame = self.tables["relationships"]
        selected = frame.apply(
            lambda row: (str(row["source"]), str(row["target"])) in keys,
            axis=1,
        )
        return frame[selected].copy()

    async def read_relationships_for_titles(self, titles: set[str]) -> pd.DataFrame:
        frame = self.tables["relationships"]
        return frame[
            frame["source"].astype(str).isin(titles)
            | frame["target"].astype(str).isin(titles)
        ].copy()

    async def relationship_degree_counts(self, titles: set[str]) -> dict[str, int]:
        message = "targeted workflow must derive degree from incident relationships"
        raise AssertionError(message)

    async def max_numeric_property(self, table_name: str, property_name: str) -> int:
        frame = self.tables[table_name]
        return int(frame[property_name].max()) if not frame.empty else -1

    async def upsert_dataframe(self, table_name: str, rows: pd.DataFrame) -> None:
        existing = self.tables[table_name]
        retained = existing[~existing["id"].astype(str).isin(rows["id"].astype(str))]
        self.tables[table_name] = pd.concat([retained, rows], ignore_index=True).loc[
            :, existing.columns
        ]


def _entity(entity_id: str, human_id: int, title: str, text_units: list[str]) -> dict:
    return {
        "id": entity_id,
        "human_readable_id": human_id,
        "title": title,
        "type": "PERSON",
        "description": title,
        "text_unit_ids": text_units,
        "frequency": len(text_units),
        "degree": 0,
    }


def _relationship(
    relationship_id: str,
    human_id: int,
    source: str,
    target: str,
    text_units: list[str],
) -> dict:
    return {
        "id": relationship_id,
        "human_readable_id": human_id,
        "source": source,
        "target": target,
        "description": f"{source}-{target}",
        "weight": 1.0,
        "combined_degree": 0,
        "text_unit_ids": text_units,
    }


@pytest.mark.asyncio
async def test_entity_merge_reads_only_delta_keys_and_incident_relationships(
    monkeypatch,
):
    old_tables = {
        "entities": pd.DataFrame([
            _entity("old-alice", 0, "Alice", ["old-tu"]),
            _entity("old-carol", 1, "Carol", ["old-tu"]),
            _entity("old-x", 2, "Unrelated X", ["other-tu"]),
            _entity("old-y", 3, "Unrelated Y", ["other-tu"]),
        ]),
        "relationships": pd.DataFrame([
            _relationship("old-ac", 0, "Alice", "Carol", ["old-tu"]),
            _relationship("old-xy", 1, "Unrelated X", "Unrelated Y", ["other-tu"]),
        ]),
    }
    delta_tables = {
        "entities": pd.DataFrame([
            _entity("delta-alice", 0, "Alice", ["new-tu"]),
            _entity("delta-bob", 1, "Bob", ["new-tu"]),
        ]),
        "relationships": pd.DataFrame([
            _relationship("delta-ab", 0, "Alice", "Bob", ["new-tu"]),
        ]),
    }
    previous = _TargetedProvider(old_tables)
    delta = _TargetedProvider(delta_tables)
    output = _TargetedProvider(old_tables)

    async def retain_descriptions(**kwargs):
        await asyncio.sleep(0)
        return kwargs["extracted_entities"], kwargs["extracted_relationships"]

    monkeypatch.setattr(
        update_entities_relationships,
        "get_summarized_entities_relationships",
        retain_descriptions,
    )
    monkeypatch.setattr(
        update_entities_relationships,
        "create_completion",
        lambda *args, **kwargs: object(),
    )
    config = SimpleNamespace(
        summarize_descriptions=SimpleNamespace(
            completion_model_id="summarizer",
            max_length=500,
            max_input_tokens=4000,
            resolved_prompts=lambda: SimpleNamespace(summarize_prompt="prompt"),
        ),
        concurrent_requests=1,
        get_completion_model_config=lambda model_id: SimpleNamespace(id=model_id),
    )
    cache = SimpleNamespace(child=lambda name: name)

    (
        entities,
        relationships,
        mapping,
    ) = await update_entities_relationships._update_neo4j_entities_and_relationships(  # noqa: SLF001
        previous,
        delta,
        output,
        config,
        cache,
        SimpleNamespace(),
    )

    assert previous.full_reads == []
    assert previous.entity_title_queries == [{"Alice", "Bob"}]
    assert previous.relationship_key_queries == [{("Alice", "Bob")}]
    assert mapping == {"delta-alice": "old-alice"}
    assert set(entities["title"]) == {"Alice", "Bob"}
    assert set(
        relationships[["source", "target"]].itertuples(index=False, name=None)
    ) == {("Alice", "Bob")}
    assert set(output.tables["entities"]["title"]) == {
        "Alice",
        "Bob",
        "Carol",
        "Unrelated X",
        "Unrelated Y",
    }
    unrelated = output.tables["relationships"].query(
        "source == 'Unrelated X' and target == 'Unrelated Y'"
    )
    assert int(unrelated.iloc[0]["combined_degree"]) == 0
    assert relationships.attrs["neo4j_incremental"]["changed_edge_keys"] == [
        ["Alice", "Bob"]
    ]


def _community(
    community_id: int,
    entity_ids: list[str],
    relationship_ids: list[str],
) -> dict:
    return {
        "id": f"community-{community_id}",
        "human_readable_id": community_id,
        "community": community_id,
        "level": 0,
        "parent": -1,
        "children": [],
        "title": f"Community {community_id}",
        "entity_ids": entity_ids,
        "relationship_ids": relationship_ids,
        "text_unit_ids": [f"tu-{community_id}"],
        "period": "2026-09-11",
        "size": len(entity_ids),
    }


class _CommunityProvider(_TargetedProvider):
    def __init__(self, tables: dict[str, pd.DataFrame]) -> None:
        super().__init__(tables)
        self.community_queries: list[set[int]] = []
        self.deleted_community_rows: list[str] = []

    async def row_count(self, table_name: str) -> int:
        return len(self.tables[table_name])

    async def count_rows_with_value(
        self, table_name: str, property_name: str, value: int
    ) -> int:
        frame = self.tables[table_name]
        return int((frame[property_name].astype(int) == value).sum())

    async def read_communities_for_entity_ids(
        self, entity_ids: set[str], *, level: int | None = None
    ) -> pd.DataFrame:
        frame = self.tables["communities"]
        selected = frame["entity_ids"].apply(
            lambda values: bool(set(map(str, values)) & set(entity_ids))
        )
        result = frame[selected].copy()
        if level is not None:
            result = result[result["level"].astype(int) == level].copy()
        return result

    async def read_communities_by_ids(
        self, community_ids: set[int]
    ) -> pd.DataFrame:
        self.community_queries.append(set(community_ids))
        frame = self.tables["communities"]
        return frame[frame["community"].astype(int).isin(community_ids)].copy()

    async def read_entities_by_ids(self, entity_ids: set[str]) -> pd.DataFrame:
        frame = self.tables["entities"]
        return frame[frame["id"].astype(str).isin(entity_ids)].copy()

    async def read_relationships_for_titles(
        self, titles: set[str], *, induced: bool = False
    ) -> pd.DataFrame:
        frame = self.tables["relationships"]
        source = frame["source"].astype(str).isin(titles)
        target = frame["target"].astype(str).isin(titles)
        return frame[source & target if induced else source | target].copy()

    async def root_community_membership_count(self) -> int:
        roots = self.tables["communities"]
        roots = roots[roots["level"].astype(int) == 0]
        return sum(len(values) for values in roots["entity_ids"])

    async def delete_rows_by_ids(
        self, table_name: str, row_ids: pd.Series
    ) -> None:
        values = set(row_ids.astype(str))
        self.deleted_community_rows.extend(sorted(values))
        frame = self.tables[table_name]
        self.tables[table_name] = frame[
            ~frame["id"].astype(str).isin(values)
        ].copy()

    async def read_dataframe_streaming(self, table_name: str) -> pd.DataFrame:
        message = f"local update must not stream {table_name}"
        raise AssertionError(message)


@pytest.mark.asyncio
async def test_local_community_update_never_loads_full_graph(monkeypatch):
    tables = {
        "entities": pd.DataFrame([
            _entity("a", 0, "A", ["tu-a"]),
            _entity("b", 1, "B", ["tu-b"]),
            _entity("c", 2, "C", ["tu-c"]),
            _entity("d", 3, "D", ["tu-d"]),
        ]),
        "relationships": pd.DataFrame([
            _relationship("r-ab", 0, "A", "B", ["tu-a"]),
            _relationship("r-cd", 1, "C", "D", ["tu-c"]),
        ]),
        "communities": pd.DataFrame([
            _community(10, ["a", "b"], ["r-ab"]),
            _community(20, ["c", "d"], ["r-cd"]),
        ]),
    }
    previous = _CommunityProvider(tables)
    output = _CommunityProvider(tables)
    rebuilt = pd.DataFrame([_community(0, ["a", "b"], ["r-ab"])])
    monkeypatch.setattr(
        "graphrag.index.workflows.create_communities.build_communities_dataframe",
        lambda *_args, **_kwargs: rebuilt,
    )
    config = SimpleNamespace(
        update=SimpleNamespace(
            local_edge_ratio=1.1,
            local_community_ratio=1.1,
            cold_start_edge_ratio=2.0,
            adaptive_community_update=True,
            community_hops=0,
            max_consecutive_local_updates=100,
            max_cumulative_edge_change_ratio=1.1,
            max_cumulative_membership_churn=1.1,
            max_modularity_drop=1.1,
        ),
        cluster_graph=SimpleNamespace(max_cluster_size=10, use_lcc=False, seed=42),
    )

    result, _, reasons, full_replacement = (
        await update_communities._update_neo4j_communities(  # noqa: SLF001
            previous,
            output,
            {
                "changed_edge_keys": [["A", "B"]],
                "affected_entity_titles": ["A", "B"],
            },
            None,
            config,
        )
    )

    assert result.metrics.strategy.value == "local"
    assert result.metrics.edge_union_count == 2
    assert result.metrics.direct_community_ids == (10,)
    assert set(result.communities["community"]) == {10}
    assert set(output.tables["communities"]["community"]) == {10, 20}
    assert output.deleted_community_rows == ["community-10"]
    assert reasons == ()
    assert full_replacement is False
