# Copyright (c) 2026 Microsoft Corporation.
# Licensed under the MIT License
# ruff: noqa: T201, EM101, EM102, TRY003, TRY301
"""Bounded, checkpointed SSE transport for the one-off benchmark base build.

Uses the configured OpenAI-compatible endpoint via the installed OpenAI SDK.
Avoids LiteLLM's expensive partial-stream reconstruction on provider stream errors.
"""

import asyncio
import hashlib
import json
from pathlib import Path
from typing import ClassVar

from graphrag_llm.completion.lite_llm_completion import LiteLLMCompletion
from graphrag_llm.utils import create_completion_response, structure_completion_response
from openai import AsyncOpenAI


class TruncatedResponseError(ValueError):
    """The provider stopped generation at its output length limit."""


class BaseSSECompletion(LiteLLMCompletion):
    """Keep graph workflows unchanged while bounding and caching complete responses."""

    failures: ClassVar[list[str]] = []
    clients: ClassVar[list[AsyncOpenAI]] = []
    _structured_retry_reminder: ClassVar[str] = (
        "Your previous response could not be parsed. Return only one valid JSON object "
        "that exactly follows the requested schema. Do not use Markdown fences, headings, "
        "comments, or prose outside the JSON. Keep findings concise so the full response "
        "remains valid JSON."
    )

    def __init__(self, *, checkpoint_dir: str, **kwargs):
        super().__init__(**kwargs)
        self.checkpoints = Path(checkpoint_dir)
        self.checkpoints.mkdir(parents=True, exist_ok=True)
        self.client = AsyncOpenAI(
            api_key=self._model_config.api_key,
            base_url=self._model_config.api_base,
            timeout=120,
            max_retries=0,
        )
        self.clients.append(self.client)
        self._completion_async = self._bounded_completion

    async def _bounded_completion(
        self, *, messages, response_format=None, metrics=None, **kwargs
    ):
        args = {**self._model_config.call_args, **kwargs}
        args.pop("stream", None)
        args.pop("response_format_json_object", None)
        args.pop("mock_response", None)
        args.setdefault("max_tokens", 8192)
        args.update(model=self._model_config.model, messages=messages, stream=True)
        if response_format is not None:
            args["response_format"] = {"type": "json_object"}
        key_data = {
            "endpoint": self._model_config.api_base,
            "args": args,
            "schema": response_format.model_json_schema() if response_format else None,
        }
        key = hashlib.sha256(json.dumps(key_data, sort_keys=True).encode()).hexdigest()
        checkpoint = self.checkpoints / f"{key}.json"
        if checkpoint.exists():
            try:
                content = json.loads(checkpoint.read_text())["content"]
                if not content.strip():
                    raise ValueError("Empty cached response")
                if response_format:
                    content = self._validated_structured_content(
                        content, response_format
                    )
            except (ValueError, TypeError, KeyError):
                checkpoint.unlink()
            else:
                if metrics is not None:
                    metrics["cached_response_count"] = 1
                return create_completion_response(content)
        max_attempts = 8 if response_format is not None else 4
        retry_args = dict(args)
        for attempt in range(1, max_attempts + 1):
            try:
                async with asyncio.timeout(
                    min(600, 180 * retry_args["max_tokens"] / args["max_tokens"])
                ):
                    content, usage = await self._read_stream(
                        self._args_for_attempt(retry_args, response_format, attempt)
                    )
                if not content.strip():
                    raise ValueError("Empty SSE response")
                if response_format:
                    content = self._validated_structured_content(
                        content, response_format
                    )
                temporary = checkpoint.with_suffix(".tmp")
                temporary.write_text(
                    json.dumps({"content": content}, ensure_ascii=False)
                )
                temporary.replace(checkpoint)
                if metrics is not None:
                    metrics.update(successful_response_count=1)
                    if usage:
                        metrics.update(
                            prompt_tokens=usage.prompt_tokens,
                            completion_tokens=usage.completion_tokens,
                            total_tokens=usage.total_tokens,
                        )
                return create_completion_response(content)
            except Exception as error:
                print(
                    f"SSE retry {attempt}/{max_attempts} [{key[:12]}]: {type(error).__name__}",
                    flush=True,
                )
                if attempt == max_attempts:
                    self.failures.append(key)
                    raise
                if isinstance(error, TruncatedResponseError):
                    current_limit = retry_args["max_tokens"]
                    retry_args["max_tokens"] = max(
                        current_limit, min(32768, current_limit * 2)
                    )
                    print(
                        f"SSE length retry [{key[:12]}]: max_tokens={retry_args['max_tokens']}",
                        flush=True,
                    )
                await asyncio.sleep(min(attempt * 2, 10))
        raise RuntimeError("Unreachable retry state")

    def _args_for_attempt(self, args, response_format, attempt):
        """Append a narrowly scoped JSON reminder after a structured-response failure."""
        if response_format is None or attempt == 1:
            return args
        retry_args = dict(args)
        messages = list(args["messages"])
        messages.append({"role": "user", "content": self._structured_retry_reminder})
        retry_args["messages"] = messages
        return retry_args

    @staticmethod
    def _validated_structured_content(content, response_format):
        """Validate JSON and accept only a safely removable Markdown code fence."""
        try:
            parsed = structure_completion_response(content, response_format)
        except json.JSONDecodeError:
            stripped = content.strip()
            if not (stripped.startswith("```") and stripped.endswith("```")):
                raise
            lines = stripped.splitlines()
            if len(lines) < 3:
                raise
            normalized = "\n".join(lines[1:-1]).strip()
            parsed = structure_completion_response(normalized, response_format)
            content = normalized
        if hasattr(parsed, "findings"):
            for field in ("title", "summary", "rating_explanation"):
                value = getattr(parsed, field, None)
                if not isinstance(value, str) or not value.strip():
                    raise ValueError(f"Empty community report field: {field}")
            if not parsed.findings:
                raise ValueError("Empty community report findings")
            for finding in parsed.findings:
                if not finding.summary.strip() or not finding.explanation.strip():
                    raise ValueError("Empty community report finding")
        return content

    async def _read_stream(self, args):
        pieces = []
        length = 0
        count = 0
        finish = None
        usage = None
        stream = await self.client.chat.completions.create(**args)
        async with stream:
            async for chunk in stream:
                count += 1
                if count > 20000:
                    raise ValueError("SSE chunk limit exceeded")
                if chunk.usage:
                    usage = chunk.usage
                if not chunk.choices:
                    continue
                choice = chunk.choices[0]
                if choice.finish_reason:
                    finish = choice.finish_reason
                    if finish == "length":
                        raise TruncatedResponseError("Incomplete SSE response: length")
                    if finish != "stop":
                        raise ValueError(f"Incomplete SSE response: {finish}")
                piece = choice.delta.content or ""
                length += len(piece)
                if length > 1_000_000:
                    raise ValueError("SSE content limit exceeded")
                pieces.append(piece)
        if finish != "stop":
            raise ValueError("SSE ended without a stop marker")
        return "".join(pieces), usage

    @classmethod
    async def close_clients(cls):
        """Close all API clients after the base build."""
        for client in cls.clients:
            await client.close()
        cls.clients.clear()
