from __future__ import annotations

import os
from pathlib import Path
from typing import Callable, Dict, Iterable, Optional

from agentic_search.exceptions import ModelLoadError
from agentic_search.models.base import BaseModel

ModelLoader = Callable[..., BaseModel]
_CUSTOM_BACKENDS: Dict[str, ModelLoader] = {}


def register_model_backend(name: str, loader: ModelLoader, *, overwrite: bool = False) -> None:
    """Register a custom model backend for load_model(..., backend=name).

    The loader can be either a BaseModel subclass or any callable with the
    signature loader(model_name_or_path: str, **kwargs) -> BaseModel.
    """
    normalized = name.strip().lower()
    if not normalized:
        raise ValueError("backend name must be non-empty")
    if normalized in _builtin_backend_names() and not overwrite:
        raise ValueError(f"'{name}' is a built-in backend name; pass overwrite=True to replace it")
    if normalized in _CUSTOM_BACKENDS and not overwrite:
        raise ValueError(f"backend '{name}' is already registered")
    _CUSTOM_BACKENDS[normalized] = loader


def available_model_backends() -> Iterable[str]:
    """Return built-in and currently registered custom backend names."""
    return sorted(set(_builtin_backend_names()) | set(_CUSTOM_BACKENDS))


def _builtin_backend_names() -> set[str]:
    return {
        "openai",
        "gpt",
        "api",
        "openai_compatible",
        "gemini",
        "qwen",
        "local_qwen",
        "local",
    }


def _load_local_qwen(model_name_or_path: str, **kwargs) -> BaseModel:
    from agentic_search.models.qwen_local import LocalQwenModel
    return LocalQwenModel(model_name_or_path, **kwargs)


def _load_gemini(model_name_or_path: str, **kwargs) -> BaseModel:
    from agentic_search.models.gemini_model import GeminiModel
    return GeminiModel(model_name_or_path, **kwargs)


def _load_openai(model_name_or_path: str, **kwargs) -> BaseModel:
    from agentic_search.models.openai_compatible import OpenAICompatibleModel
    return OpenAICompatibleModel(model_name_or_path, **kwargs)


def _load_custom_backend(backend: str, model_name_or_path: str, **kwargs) -> BaseModel:
    loader = _CUSTOM_BACKENDS[backend]
    model = loader(model_name_or_path, **kwargs)
    if not isinstance(model, BaseModel):
        raise ModelLoadError(
            f"Custom backend '{backend}' returned {type(model)!r}, expected a BaseModel instance."
        )
    return model


def load_model(model_name_or_path: str, backend: Optional[str] = None, **kwargs) -> BaseModel:
    backend = (backend or "auto").lower()
    name = model_name_or_path.lower()
    is_local_path = Path(model_name_or_path).exists()

    if backend in _CUSTOM_BACKENDS:
        return _load_custom_backend(backend, model_name_or_path, **kwargs)

    if backend in {"qwen", "local_qwen", "local"}:
        return _load_local_qwen(model_name_or_path, **kwargs)
    if backend in {"gemini"}:
        return _load_gemini(model_name_or_path, **kwargs)
    if backend in {"openai", "gpt", "api", "openai_compatible"}:
        return _load_openai(model_name_or_path, **kwargs)

    if backend != "auto":
        available = ", ".join(available_model_backends())
        raise ModelLoadError(f"Unknown backend '{backend}'. Available backends: {available}")

    if is_local_path or "qwen" in name:
        return _load_local_qwen(model_name_or_path, **kwargs)
    if "gemini" in name:
        return _load_gemini(model_name_or_path, **kwargs)
    if any(token in name for token in ["gpt", "o1", "o3", "o4", "claude"]):
        return _load_openai(model_name_or_path, **kwargs)

    if os.getenv("AGENTIC_SEARCH_DEFAULT_BACKEND") == "gemini":
        return _load_gemini(model_name_or_path, **kwargs)
    return _load_openai(model_name_or_path, **kwargs)
