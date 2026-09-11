# Copyright (c) 2024 Microsoft Corporation.
# Licensed under the MIT License

"""A module containing run_workflow method definition."""

import logging

import numpy as np
import pandas as pd
from graphrag_storage.tables.neo4j_table_provider import Neo4jTableProvider
from graphrag_storage.tables.table_provider import TableProvider

from graphrag.config.models.graph_rag_config import GraphRagConfig
from graphrag.data_model.data_reader import DataReader
from graphrag.index.run.utils import get_update_table_providers
from graphrag.index.typing.context import PipelineRunContext
from graphrag.index.typing.workflow import WorkflowFunctionOutput

logger = logging.getLogger(__name__)


async def run_workflow(
    config: GraphRagConfig,
    context: PipelineRunContext,
) -> WorkflowFunctionOutput:
    """Update the text units from a incremental index run."""
    logger.info("Workflow started: update_text_units")
    output_table_provider, previous_table_provider, delta_table_provider = (
        get_update_table_providers(
            config,
            context.state["update_timestamp"],
            context.final_output_table_provider,
        )
    )
    entity_id_mapping = context.state["incremental_update_entity_id_mapping"]

    if isinstance(output_table_provider, Neo4jTableProvider) and isinstance(
        delta_table_provider, Neo4jTableProvider
    ):
        merged_text_units = await DataReader(delta_table_provider).text_units()
        merged_text_units = _normalize_reference_columns(merged_text_units)
        if entity_id_mapping:
            merged_text_units["entity_ids"] = merged_text_units["entity_ids"].apply(
                lambda values: [entity_id_mapping.get(value, value) for value in values]
            )
        initial_id = (
            await output_table_provider.max_numeric_property(
                "text_units", "human_readable_id"
            )
            + 1
        )
        merged_text_units["human_readable_id"] = np.arange(
            initial_id, initial_id + len(merged_text_units)
        )
        await output_table_provider.upsert_dataframe("text_units", merged_text_units)
    else:
        merged_text_units = await _update_text_units(
            previous_table_provider,
            delta_table_provider,
            output_table_provider,
            entity_id_mapping,
        )

    context.state["incremental_update_merged_text_units"] = merged_text_units

    logger.info("Workflow completed: update_text_units")
    return WorkflowFunctionOutput(result=None)


async def _update_text_units(
    previous_table_provider: TableProvider,
    delta_table_provider: TableProvider,
    output_table_provider: TableProvider,
    entity_id_mapping: dict,
) -> pd.DataFrame:
    """Update the text units output."""
    old_text_units = await DataReader(previous_table_provider).text_units()
    delta_text_units = await DataReader(delta_table_provider).text_units()
    old_text_units = _normalize_reference_columns(old_text_units)
    delta_text_units = _normalize_reference_columns(delta_text_units)
    merged_text_units = _update_and_merge_text_units(
        old_text_units, delta_text_units, entity_id_mapping
    )

    await output_table_provider.write_dataframe("text_units", merged_text_units)

    return merged_text_units


def _normalize_reference_columns(text_units: pd.DataFrame) -> pd.DataFrame:
    """Normalize nullable list references from legacy or interrupted indexes."""
    normalized = text_units.copy()
    for column in ("entity_ids", "relationship_ids", "covariate_ids"):
        if column in normalized.columns:
            normalized[column] = normalized[column].apply(
                lambda value: value if isinstance(value, (list, tuple)) else []
            )
    return normalized


def _update_and_merge_text_units(
    old_text_units: pd.DataFrame,
    delta_text_units: pd.DataFrame,
    entity_id_mapping: dict,
) -> pd.DataFrame:
    """Update and merge text units.

    Parameters
    ----------
    old_text_units : pd.DataFrame
        The old text units.
    delta_text_units : pd.DataFrame
        The delta text units.
    entity_id_mapping : dict
        The entity id mapping.

    Returns
    -------
    pd.DataFrame
        The updated text units.
    """
    # Look for entity ids in entity_ids and replace them with the corresponding id in the mapping
    if entity_id_mapping:
        delta_text_units["entity_ids"] = delta_text_units["entity_ids"].apply(
            lambda x: [entity_id_mapping.get(i, i) for i in x] if x is not None else x
        )

    if "human_readable_id" not in old_text_units.columns:
        old_text_units = old_text_units.copy()
        old_text_units["human_readable_id"] = np.arange(len(old_text_units))
    initial_id = (
        int(old_text_units["human_readable_id"].max()) + 1
        if not old_text_units.empty
        else 0
    )
    delta_text_units["human_readable_id"] = np.arange(
        initial_id, initial_id + len(delta_text_units)
    )
    # Merge the final text units
    return pd.concat([old_text_units, delta_text_units], ignore_index=True)
