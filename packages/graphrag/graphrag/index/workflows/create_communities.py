# Copyright (c) 2024 Microsoft Corporation.
# Licensed under the MIT License

"""A module containing run_workflow method definition."""

import logging
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any, cast
from uuid import uuid4

import numpy as np
import pandas as pd
from graphrag_storage.tables.table import Table

from graphrag.config.models.graph_rag_config import GraphRagConfig
from graphrag.data_model.data_reader import DataReader
from graphrag.data_model.schemas import COMMUNITIES_FINAL_COLUMNS
from graphrag.index.operations.cluster_graph import Communities, cluster_graph
from graphrag.index.typing.context import PipelineRunContext
from graphrag.index.typing.workflow import WorkflowFunctionOutput

if TYPE_CHECKING:
    from collections.abc import Mapping

logger = logging.getLogger(__name__)


async def run_workflow(
    config: GraphRagConfig,
    context: PipelineRunContext,
) -> WorkflowFunctionOutput:
    """All the steps to transform final communities."""
    logger.info("Workflow started: create_communities")
    reader = DataReader(context.output_table_provider)
    relationships = await reader.relationships()

    async with (
        context.output_table_provider.open("entities") as entities_table,
        context.output_table_provider.open("communities") as communities_table,
    ):
        sample_rows = await create_communities(
            communities_table,
            entities_table,
            relationships,
            max_cluster_size=config.cluster_graph.max_cluster_size,
            use_lcc=config.cluster_graph.use_lcc,
            seed=config.cluster_graph.seed,
        )

    logger.info("Workflow completed: create_communities")
    return WorkflowFunctionOutput(result=sample_rows)


async def create_communities(
    communities_table: Table,
    entities_table: Table,
    relationships: pd.DataFrame,
    max_cluster_size: int,
    use_lcc: bool,
    seed: int | None = None,
) -> list[dict[str, Any]]:
    """Build communities and stream rows to the output table."""
    entity_rows = [row async for row in entities_table]
    entities = pd.DataFrame(entity_rows)
    output = build_communities_dataframe(
        entities,
        relationships,
        max_cluster_size=max_cluster_size,
        use_lcc=use_lcc,
        seed=seed,
    )

    sample_rows: list[dict[str, Any]] = []
    for row in output.to_dict("records"):
        row = _sanitize_row(cast("dict[str, Any]", row))
        await communities_table.write(row)
        if len(sample_rows) < 5:
            sample_rows.append(row)
    return sample_rows


def build_communities_dataframe(
    entities: pd.DataFrame,
    relationships: pd.DataFrame,
    *,
    max_cluster_size: int,
    use_lcc: bool,
    seed: int | None = None,
    starting_communities: dict[str, int] | None = None,
) -> pd.DataFrame:
    """Build finalized community rows from entity and relationship DataFrames."""
    clusters = cluster_graph(
        relationships,
        max_cluster_size,
        use_lcc,
        seed=seed,
        starting_communities=starting_communities,
    )
    return build_communities_from_clusters(entities, relationships, clusters)


def build_communities_from_clusters(
    entities: pd.DataFrame,
    relationships: pd.DataFrame,
    clusters: Communities,
) -> pd.DataFrame:
    """Build finalized community rows from a precomputed hierarchy."""
    if not clusters:
        return pd.DataFrame(columns=COMMUNITIES_FINAL_COLUMNS)

    title_to_entity_id = dict(
        zip(entities["title"].astype(str), entities["id"].astype(str), strict=False)
    )
    communities = pd.DataFrame(
        clusters, columns=pd.Index(["level", "community", "parent", "title"])
    ).explode("title")
    communities["community"] = communities["community"].astype(int)

    entity_map = communities[["community", "title"]].copy()
    entity_map["entity_id"] = entity_map["title"].map(title_to_entity_id)
    entity_ids = (
        entity_map.dropna(subset=["entity_id"])
        .groupby("community")
        .agg(entity_ids=("entity_id", list))
        .reset_index()
    )

    level_results: list[pd.DataFrame] = []
    for level in communities["level"].unique():
        level_comms = communities[communities["level"] == level]
        with_source = relationships.merge(
            level_comms, left_on="source", right_on="title", how="inner"
        )
        with_both = with_source.merge(
            level_comms, left_on="target", right_on="title", how="inner"
        )
        intra = with_both[with_both["community_x"] == with_both["community_y"]]
        if intra.empty:
            continue
        grouped = (
            intra.explode("text_unit_ids")
            .groupby(["community_x", "parent_x"])
            .agg(
                relationship_ids=("id", list),
                text_unit_ids=("text_unit_ids", list),
            )
            .reset_index()
        )
        grouped["level"] = level
        level_results.append(grouped)

    if not level_results:
        return pd.DataFrame(columns=COMMUNITIES_FINAL_COLUMNS)

    all_grouped = pd.concat(level_results, ignore_index=True).rename(
        columns={"community_x": "community", "parent_x": "parent"}
    )
    all_grouped["relationship_ids"] = all_grouped["relationship_ids"].apply(
        lambda values: sorted(set(values))
    )
    all_grouped["text_unit_ids"] = all_grouped["text_unit_ids"].apply(
        lambda values: sorted({value for value in values if pd.notna(value)})
    )

    final_communities = all_grouped.merge(entity_ids, on="community", how="inner")
    final_communities["id"] = [str(uuid4()) for _ in range(len(final_communities))]
    final_communities["human_readable_id"] = final_communities["community"]
    final_communities["title"] = "Community " + final_communities[
        "community"
    ].astype(str)
    final_communities["parent"] = final_communities["parent"].astype(int)
    parent_grouped = cast(
        "pd.DataFrame",
        final_communities.groupby("parent").agg(children=("community", "unique")),
    )
    final_communities = final_communities.merge(
        parent_grouped, left_on="community", right_on="parent", how="left"
    )
    final_communities["children"] = final_communities["children"].apply(
        lambda value: value if isinstance(value, np.ndarray) else []  # type: ignore
    )
    final_communities["period"] = datetime.now(timezone.utc).date().isoformat()
    final_communities["size"] = final_communities["entity_ids"].apply(len)
    return final_communities.loc[:, COMMUNITIES_FINAL_COLUMNS]


def remap_community_ids(
    communities: pd.DataFrame,
    id_mapping: "Mapping[int, int]",
    uuid_mapping: "Mapping[int, str] | None" = None,
) -> pd.DataFrame:
    """Apply a complete community ID mapping and rebuild hierarchy metadata."""
    result = communities.copy()
    result["community"] = result["community"].astype(int).map(id_mapping)
    result["parent"] = result["parent"].astype(int).apply(
        lambda value: -1 if value == -1 else id_mapping[value]
    )
    result["children"] = result["children"].apply(
        lambda values: [id_mapping[int(value)] for value in values]
    )
    if uuid_mapping:
        result["id"] = result.apply(
            lambda row: uuid_mapping.get(int(row["community"]), row["id"]), axis=1
        )
    result["human_readable_id"] = result["community"]
    result["title"] = "Community " + result["community"].astype(str)
    return result.loc[:, COMMUNITIES_FINAL_COLUMNS]


def _sanitize_row(row: dict[str, Any]) -> dict[str, Any]:
    """Convert numpy types to native Python types for table serialization."""
    sanitized = {}
    for key, value in row.items():
        if isinstance(value, np.ndarray):
            sanitized[key] = value.tolist()
        elif isinstance(value, np.integer):
            sanitized[key] = int(value)
        elif isinstance(value, np.floating):
            sanitized[key] = float(value)
        else:
            sanitized[key] = value
    return sanitized
