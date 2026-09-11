# Copyright (C) 2026 Microsoft
# Licensed under the MIT License

"""Verb test for the update_text_embeddings workflow."""

from unittest.mock import AsyncMock, patch

from graphrag.config.embeddings import all_embeddings
from graphrag.index.workflows.update_text_embeddings import (
    run_workflow,
)

from tests.unit.config.utils import get_default_graphrag_config

from .util import create_test_context


async def test_update_text_embeddings():
    """Verify update_text_embeddings produces embedding tables.

    Mocks get_update_table_providers to return the test context's
    output_table_provider, simulating the merged tables written by
    upstream update workflows.
    """
    context = await create_test_context(
        storage=[
            "documents",
            "relationships",
            "text_units",
            "entities",
            "community_reports",
        ]
    )
    context.state["update_timestamp"] = "20260220-000000"

    config = get_default_graphrag_config()
    llm_settings = config.get_embedding_model_config(
        config.embed_text.embedding_model_id
    )
    llm_settings.type = "mock"
    llm_settings.mock_responses = [1.0] * 3072

    config.embed_text.names = list(all_embeddings)
    config.snapshots.embeddings = True

    with patch(
        "graphrag.index.workflows.update_text_embeddings.get_update_table_providers",
    ) as mock_providers:
        mock_providers.return_value = (
            context.output_table_provider,
            None,
            None,
        )
        await run_workflow(config, context)

    parquet_files = context.output_storage.keys()
    for field in all_embeddings:
        assert f"embeddings.{field}.parquet" in parquet_files

    entity_embeddings = await context.output_table_provider.read_dataframe(
        "embeddings.entity_description"
    )
    assert len(entity_embeddings.columns) == 2
    assert "id" in entity_embeddings.columns
    assert "embedding" in entity_embeddings.columns


async def test_update_text_embeddings_honors_runtime_limits():
    """Use benchmark-specific embedding limits without reducing other LLM work."""
    context = await create_test_context(storage=[])
    context.state.update({
        "update_timestamp": "20260220-000000",
        "additional_context": {
            "embedding_runtime": {
                "request_batch_size": 4,
                "community_request_batch_size": 1,
                "batch_max_tokens": 1024,
                "concurrency": 1,
                "vector_write_batch_size": 128,
            }
        },
    })
    config = get_default_graphrag_config()
    model_config = config.get_embedding_model_config(
        config.embed_text.embedding_model_id
    )
    model_config.type = "mock"
    model_config.mock_responses = [1.0]
    generate = AsyncMock()

    with (
        patch(
            "graphrag.index.workflows.update_text_embeddings.get_update_table_providers",
            return_value=(context.output_table_provider, None, None),
        ),
        patch(
            "graphrag.index.workflows.update_text_embeddings.generate_text_embeddings",
            generate,
        ),
    ):
        await run_workflow(config, context)

    kwargs = generate.await_args.kwargs
    assert kwargs["request_batch_sizes"]["community_full_content"] == 1
    assert all(
        size == 4
        for name, size in kwargs["request_batch_sizes"].items()
        if name != "community_full_content"
    )
    assert kwargs["batch_max_tokens"] == 1024
    assert kwargs["num_threads"] == 1
    assert kwargs["vector_write_batch_size"] == 128
