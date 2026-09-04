# Copyright (c) 2024 Microsoft Corporation.
# Licensed under the MIT License

"""LocalSearch implementation."""

import logging
import time
from collections.abc import AsyncGenerator, AsyncIterator
from typing import TYPE_CHECKING, Any

from graphrag_llm.tokenizer import Tokenizer
from graphrag_llm.utils import CompletionMessagesBuilder

from graphrag.callbacks.query_callbacks import QueryCallbacks
from graphrag.prompts.query.local_search_system_prompt import (
    LOCAL_SEARCH_SYSTEM_PROMPT,
)
from graphrag.query.context_builder.builders import LocalContextBuilder
from graphrag.query.context_builder.conversation_history import (
    ConversationHistory,
)
from graphrag.query.structured_search.base import BaseSearch, SearchResult

if TYPE_CHECKING:
    from graphrag_llm.completion import LLMCompletion
    from graphrag_llm.types import LLMCompletionChunk

logger = logging.getLogger(__name__)


class LocalSearch(BaseSearch[LocalContextBuilder]):
    """Search orchestration for local search mode."""

    def __init__(
        self,
        model: "LLMCompletion",
        context_builder: LocalContextBuilder,
        tokenizer: Tokenizer | None = None,
        system_prompt: str | None = None,
        response_type: str = "multiple paragraphs",
        callbacks: list[QueryCallbacks] | None = None,
        model_params: dict[str, Any] | None = None,
        context_builder_params: dict | None = None,
    ):
        super().__init__(
            model=model,
            context_builder=context_builder,
            tokenizer=tokenizer,
            model_params=model_params,
            context_builder_params=context_builder_params or {},
        )
        self.system_prompt = system_prompt or LOCAL_SEARCH_SYSTEM_PROMPT
        self.callbacks = callbacks or []
        self.response_type = response_type

    async def search(
        self,
        query: str,
        conversation_history: ConversationHistory | None = None,
        **kwargs,
    ) -> SearchResult:
        """Build local search context that fits a single context window and generate answer for the user query."""
        start_time = time.time()
        search_prompt = ""
        llm_calls, prompt_tokens, output_tokens = {}, {}, {}
        context_result = self.context_builder.build_context(
            query=query,
            conversation_history=conversation_history,
            **kwargs,
            **self.context_builder_params,
        )
        llm_calls["build_context"] = context_result.llm_calls
        prompt_tokens["build_context"] = context_result.prompt_tokens
        output_tokens["build_context"] = context_result.output_tokens

        logger.debug("GENERATE ANSWER: %s. QUERY: %s", start_time, query)
        try:
            if "drift_query" in kwargs:
                drift_query = kwargs["drift_query"]
                search_prompt = self.system_prompt.format(
                    context_data=context_result.context_chunks,
                    response_type=self.response_type,
                    global_query=drift_query,
                    followups=kwargs.get("k_followups", 0),
                )
            else:
                search_prompt = self.system_prompt.format(
                    context_data=context_result.context_chunks,
                    response_type=self.response_type,
                )

            messages_builder = (
                CompletionMessagesBuilder()
                .add_system_message(search_prompt)
                .add_user_message(query)
            )

            full_response = ""

            response: AsyncIterator[
                LLMCompletionChunk
            ] = await self.model.completion_async(
                messages=messages_builder.build(),
                stream=True,
                **self.model_params,
            )  # type: ignore

            async for chunk in response:
                response_text = chunk.choices[0].delta.content or ""
                full_response += response_text
                for callback in self.callbacks:
                    callback.on_llm_new_token(response_text)

            llm_calls["response"] = 1
            prompt_tokens["response"] = len(self.tokenizer.encode(search_prompt))
            output_tokens["response"] = len(self.tokenizer.encode(full_response))

            for callback in self.callbacks:
                callback.on_context(context_result.context_records)

            return SearchResult(
                response=full_response,
                context_data=context_result.context_records,
                context_text=context_result.context_chunks,
                completion_time=time.time() - start_time,
                llm_calls=sum(llm_calls.values()),
                prompt_tokens=sum(prompt_tokens.values()),
                output_tokens=sum(output_tokens.values()),
                llm_calls_categories=llm_calls,
                prompt_tokens_categories=prompt_tokens,
                output_tokens_categories=output_tokens,
            )

        except Exception:
            logger.exception("Exception in _asearch")
            return SearchResult(
                response="",
                context_data=context_result.context_records,
                context_text=context_result.context_chunks,
                completion_time=time.time() - start_time,
                llm_calls=1,
                prompt_tokens=len(self.tokenizer.encode(search_prompt)),
                output_tokens=0,
            )

    async def stream_search(
        self,
        query: str,
        conversation_history: ConversationHistory | None = None,
        metrics: dict[str, int | float] | None = None,
    ) -> AsyncGenerator:
        """Build local search context that fits a single context window and generate answer for the user query."""
        start_time = time.perf_counter()

        context_started = time.perf_counter()
        context_result = self.context_builder.build_context(
            query=query,
            conversation_history=conversation_history,
            **self.context_builder_params,
        )
        context_completed = time.perf_counter()
        if metrics is not None:
            metrics["context_build_seconds"] = context_completed - context_started
        logger.debug("GENERATE ANSWER: %s. QUERY: %s", start_time, query)
        search_prompt = self.system_prompt.format(
            context_data=context_result.context_chunks, response_type=self.response_type
        )

        messages_builder = (
            CompletionMessagesBuilder()
            .add_system_message(search_prompt)
            .add_user_message(query)
        )

        for callback in self.callbacks:
            callback.on_context(context_result.context_records)

        llm_request_started = time.perf_counter()
        response: AsyncIterator[LLMCompletionChunk] = await self.model.completion_async(
            messages=messages_builder.build(),
            stream=True,
            metrics=metrics,
            **self.model_params,
        )  # type: ignore
        response_stream_opened = time.perf_counter()
        first_token_at: float | None = None
        output_chunks = 0

        async for chunk in response:
            if first_token_at is None:
                first_token_at = time.perf_counter()
            response_text = chunk.choices[0].delta.content or ""
            output_chunks += 1
            for callback in self.callbacks:
                callback.on_llm_new_token(response_text)
            yield response_text

        completed_at = time.perf_counter()
        if metrics is not None:
            first_token_or_end = first_token_at or completed_at
            metrics["llm_stream_open_seconds"] = (
                response_stream_opened - llm_request_started
            )
            metrics["llm_time_to_first_token_seconds"] = (
                first_token_or_end - llm_request_started
            )
            metrics["llm_stream_seconds"] = completed_at - first_token_or_end
            metrics["llm_total_seconds"] = completed_at - llm_request_started
            metrics["local_search_total_seconds"] = completed_at - start_time
            metrics["output_chunks"] = output_chunks
