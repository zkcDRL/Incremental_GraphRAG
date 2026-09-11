# Copyright (C) 2026 Microsoft
# Licensed under the MIT License

"""A module containing run_workflow method definition."""

import logging

from graphrag_llm.embedding import create_embedding

from graphrag.cache.cache_key_creator import cache_key_creator
from graphrag.config.embeddings import community_full_content_embedding
from graphrag.config.models.graph_rag_config import GraphRagConfig
from graphrag.index.run.utils import get_update_table_providers
from graphrag.index.typing.context import PipelineRunContext
from graphrag.index.typing.workflow import WorkflowFunctionOutput
from graphrag.index.workflows.generate_text_embeddings import (
    generate_text_embeddings,
)

logger = logging.getLogger(__name__)


def _positive_runtime_int(runtime: dict, name: str, default: int) -> int:
    """Read a positive integer supplied by an indexing entry point."""
    value = runtime.get(name, default)
    if not isinstance(value, int) or isinstance(value, bool) or value < 1:
        msg = f"embedding_runtime.{name} must be a positive integer"
        raise ValueError(msg)
    return value


async def run_workflow(
    config: GraphRagConfig,
    context: PipelineRunContext,
) -> WorkflowFunctionOutput:
    """Update text embeddings for an incremental index run."""
    logger.info("Workflow started: update_text_embeddings")

    output_table_provider, _, _ = get_update_table_providers(
        config,
        context.state["update_timestamp"],
        context.final_output_table_provider,
    )

    model_config = config.get_embedding_model_config(
        config.embed_text.embedding_model_id
    )
    model = create_embedding(
        model_config,
        cache=context.cache.child(config.embed_text.model_instance_name),
        cache_key_creator=cache_key_creator,
    )
    tokenizer = model.tokenizer

    runtime = (context.state.get("additional_context") or {}).get(
        "embedding_runtime", {}
    )
    if not isinstance(runtime, dict):
        msg = "additional_context.embedding_runtime must be an object"
        raise TypeError(msg)
    request_batch_size = _positive_runtime_int(
        runtime, "request_batch_size", config.embed_text.batch_size
    )
    community_request_batch_size = _positive_runtime_int(
        runtime,
        "community_request_batch_size",
        request_batch_size,
    )
    batch_max_tokens = _positive_runtime_int(
        runtime, "batch_max_tokens", config.embed_text.batch_max_tokens
    )
    num_threads = _positive_runtime_int(
        runtime, "concurrency", config.concurrent_requests
    )
    vector_write_batch_size = _positive_runtime_int(
        runtime,
        "vector_write_batch_size",
        request_batch_size * num_threads,
    )
    logger.info(
        "Update embedding runtime: batch_size=%d, community_batch_size=%d, "
        "batch_max_tokens=%d, concurrency=%d, vector_write_batch_size=%d",
        request_batch_size,
        community_request_batch_size,
        batch_max_tokens,
        num_threads,
        vector_write_batch_size,
    )

    await generate_text_embeddings(
        config=config,
        table_provider=output_table_provider,
        callbacks=context.callbacks,
        model=model,
        tokenizer=tokenizer,
        vector_write_batch_size=vector_write_batch_size,
        request_batch_sizes={
            name: (
                community_request_batch_size
                if name == community_full_content_embedding
                else request_batch_size
            )
            for name in config.embed_text.names
        },
        batch_max_tokens=batch_max_tokens,
        num_threads=num_threads,
    )

    logger.info("Workflow completed: update_text_embeddings")
    return WorkflowFunctionOutput(result=None)
