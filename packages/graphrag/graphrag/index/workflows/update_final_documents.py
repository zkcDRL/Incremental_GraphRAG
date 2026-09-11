# Copyright (c) 2024 Microsoft Corporation.
# Licensed under the MIT License

"""A module containing run_workflow method definition."""

import logging

import numpy as np
from graphrag_storage.tables.neo4j_table_provider import Neo4jTableProvider

from graphrag.config.models.graph_rag_config import GraphRagConfig
from graphrag.data_model.data_reader import DataReader
from graphrag.index.run.utils import get_update_table_providers
from graphrag.index.typing.context import PipelineRunContext
from graphrag.index.typing.workflow import WorkflowFunctionOutput
from graphrag.index.update.incremental_index import concat_dataframes

logger = logging.getLogger(__name__)


async def run_workflow(
    config: GraphRagConfig,
    context: PipelineRunContext,
) -> WorkflowFunctionOutput:
    """Update the documents from a incremental index run."""
    logger.info("Workflow started: update_final_documents")
    output_table_provider, previous_table_provider, delta_table_provider = (
        get_update_table_providers(
            config,
            context.state["update_timestamp"],
            context.final_output_table_provider,
        )
    )

    if isinstance(output_table_provider, Neo4jTableProvider) and isinstance(
        delta_table_provider, Neo4jTableProvider
    ):
        final_documents = await DataReader(delta_table_provider).documents()
        initial_id = (
            await output_table_provider.max_numeric_property(
                "documents", "human_readable_id"
            )
            + 1
        )
        final_documents["human_readable_id"] = np.arange(
            initial_id, initial_id + len(final_documents)
        )
        await output_table_provider.upsert_dataframe("documents", final_documents)
    else:
        final_documents = await concat_dataframes(
            "documents",
            previous_table_provider,
            delta_table_provider,
            output_table_provider,
        )

    context.state["incremental_update_final_documents"] = final_documents

    logger.info("Workflow completed: update_final_documents")
    return WorkflowFunctionOutput(result=None)
