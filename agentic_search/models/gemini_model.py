from __future__ import annotations

import logging
import os
from typing import Any, Dict, List, Optional

import google.generativeai as genai
from tenacity import before_sleep_log, retry, stop_after_attempt, wait_exponential

from agentic_search.models.base import BaseModel
from agentic_search.utils.image_utils import image_input_to_data_url

logger = logging.getLogger(__name__)


class GeminiModel(BaseModel):
    """
    Google Gemini model using the official google-generativeai SDK.

    Configure via environment variable:
        GOOGLE_API_KEY – get yours at https://aistudio.google.com
    """

    def __init__(self, model_name_or_path: str = "gemini-2.5-pro", **kwargs) -> None:
        super().__init__(model_name_or_path, **kwargs)
        api_key = os.getenv("GOOGLE_API_KEY")
        if not api_key:
            raise EnvironmentError(
                "GOOGLE_API_KEY environment variable is required. "
                "Get your key at https://aistudio.google.com"
            )
        genai.configure(api_key=api_key)
        self.model_name = model_name_or_path
        self._model = genai.GenerativeModel(model_name_or_path)

    def _build_contents(self, text: str, images: Optional[List[Any]] = None) -> List[Dict[str, Any]]:
        """Build content parts for the Gemini API from text and optional images."""
        parts: List[Any] = [{"text": text}]
        for image in images or []:
            data_url = image_input_to_data_url(image)
            header, b64 = data_url.split(",", 1)
            mime_type = header.split(";")[0].replace("data:", "")
            image_bytes = __import__("base64").b64decode(b64)
            parts.append({
                "mime_type": mime_type,
                "data": image_bytes,
            })
        return [{"role": "user", "parts": parts}]

    @retry(
        wait=wait_exponential(multiplier=2, min=4, max=60),
        stop=stop_after_attempt(8),
        before_sleep=before_sleep_log(logger, logging.WARNING),
        reraise=True,
    )
    def response(self, messages: List[Dict[str, Any]], **kwargs) -> str:
        """Send messages to the Gemini model and return the text response."""
        generation_config = genai.GenerationConfig(
            max_output_tokens=kwargs.get("max_output_tokens", 8192),
        )
        response = self._model.generate_content(
            messages,
            generation_config=generation_config,
            request_options={"timeout": kwargs.get("timeout", 120)},
        )
        try:
            return response.text
        except ValueError as e:
            # response.text raises ValueError if the response was blocked
            block_reason = getattr(response, "prompt_feedback", None)
            raise ValueError(
                f"Gemini response was blocked or empty. "
                f"Block reason: {block_reason}. "
                f"Candidates: {response.candidates}"
            ) from e

    def generate_response(self, text: str, images: Optional[List[Any]] = None, **kwargs) -> str:
        messages = self._build_contents(text=text, images=images)
        return self.response(messages, **kwargs)
