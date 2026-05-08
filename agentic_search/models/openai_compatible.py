from __future__ import annotations

import logging
import os
from typing import Any, Dict, List, Optional

import httpx
from openai import OpenAI
from tenacity import before_sleep_log, retry, stop_after_attempt, wait_exponential

from agentic_search.models.base import BaseModel
from agentic_search.utils.image_utils import image_input_to_data_url

logger = logging.getLogger(__name__)


_OPENAI_DEFAULT_BASE_URL = "https://api.openai.com/v1"
_REASONING_MODEL_PREFIXES = ("o1", "o3", "o4", "gpt-5")


class OpenAICompatibleModel(BaseModel):
    """
    OpenAI-compatible chat model client.

    Works with OpenAI and most OpenAI-compatible endpoints such as vLLM,
    LM Studio, Ollama's OpenAI-compatible server, DashScope-compatible Qwen,
    and other self-hosted gateways.

    Configure via environment variables or keyword arguments:
        OPENAI_API_KEY  - API key. Required for the official OpenAI endpoint.
                          For local compatible endpoints, "EMPTY" is accepted.
        OPENAI_BASE_URL - Base URL, defaults to https://api.openai.com/v1

    Useful keyword arguments:
        api_key, base_url, timeout, http2, default_params, extra_body,
        reasoning_effort, enable_thinking
    """

    def __init__(self, model_name_or_path: str = "gpt-4o", http_client=None, **kwargs) -> None:
        super().__init__(model_name_or_path, **kwargs)
        self.model = model_name_or_path
        self.base_url = str(kwargs.get("base_url") or os.getenv("OPENAI_BASE_URL") or _OPENAI_DEFAULT_BASE_URL).rstrip("/")
        self.api_key = str(kwargs.get("api_key") or os.getenv("OPENAI_API_KEY") or "")

        # The OpenAI Python SDK requires a non-empty api_key even for local endpoints.
        # Keep the official endpoint strict, but make local/custom endpoints painless.
        if not self.api_key:
            if self.base_url == _OPENAI_DEFAULT_BASE_URL:
                raise EnvironmentError(
                    "OPENAI_API_KEY environment variable is required for the official OpenAI endpoint. "
                    "For local OpenAI-compatible endpoints, set OPENAI_BASE_URL and optionally "
                    "OPENAI_API_KEY=EMPTY."
                )
            self.api_key = "EMPTY"

        timeout_seconds = float(kwargs.get("timeout", 300.0))
        if http_client:
            self.httpx_client = http_client
        else:
            self.httpx_client = httpx.Client(
                http2=bool(kwargs.get("http2", True)),
                timeout=httpx.Timeout(timeout_seconds, read=timeout_seconds),
            )
        self.client = OpenAI(
            api_key=self.api_key,
            base_url=self.base_url,
            http_client=self.httpx_client,
        )
        self.default_params: Dict[str, Any] = dict(kwargs.get("default_params") or {})

    def _build_messages(self, text: str, images: Optional[List[Any]] = None) -> List[Dict[str, Any]]:
        if not images:
            return [{"role": "user", "content": text}]
        content: List[Dict[str, Any]] = [{"type": "text", "text": text}]
        for image in images:
            data_url = image_input_to_data_url(image)
            content.append({"type": "image_url", "image_url": {"url": data_url}})
        return [{"role": "user", "content": content}]

    @staticmethod
    def _supports_reasoning_effort(model_name: str) -> bool:
        normalized = model_name.lower()
        return normalized.startswith(_REASONING_MODEL_PREFIXES)

    def _build_chat_params(self, messages: List[Dict[str, Any]], **kwargs) -> Dict[str, Any]:
        model_name = kwargs.get("model", self.model)
        params: Dict[str, Any] = {
            "model": model_name,
            "messages": messages,
        }
        params.update(self.default_params)

        passthrough_keys = {
            "temperature",
            "top_p",
            "max_tokens",
            "max_completion_tokens",
            "stop",
            "presence_penalty",
            "frequency_penalty",
            "seed",
            "stream",
            "response_format",
            "tools",
            "tool_choice",
        }
        for key in passthrough_keys:
            if key in kwargs and kwargs[key] is not None:
                params[key] = kwargs[key]

        # Do not send reasoning_effort by default. Many OpenAI-compatible endpoints
        # reject unknown parameters. Send it only when explicitly requested and the
        # model family is known to support it.
        reasoning_effort = kwargs.get("reasoning_effort")
        if reasoning_effort is not None and self._supports_reasoning_effort(str(model_name)):
            params["reasoning_effort"] = reasoning_effort

        extra_body = dict(kwargs.get("extra_body") or {})
        if "enable_thinking" in kwargs:
            extra_body["enable_thinking"] = kwargs["enable_thinking"]
        elif "qwen" in str(model_name).lower() and kwargs.get("disable_qwen_thinking", True):
            # DashScope/Qwen OpenAI-compatible endpoints support this field. If a
            # different Qwen-compatible endpoint rejects it, call with
            # disable_qwen_thinking=False or pass your own extra_body.
            extra_body.setdefault("enable_thinking", False)

        if extra_body:
            params["extra_body"] = extra_body

        return params

    @retry(
        wait=wait_exponential(multiplier=2, min=4, max=60),
        stop=stop_after_attempt(8),
        before_sleep=before_sleep_log(logger, logging.WARNING),
        reraise=True,
    )
    def response(self, messages: List[Dict[str, Any]], **kwargs) -> str:
        params = self._build_chat_params(messages, **kwargs)
        response = self.client.chat.completions.create(**params)
        return response.choices[0].message.content or ""

    def generate_response(self, text: str, images: Optional[List[Any]] = None, **kwargs) -> str:
        messages = self._build_messages(text=text, images=images)
        return self.response(messages, **kwargs)
