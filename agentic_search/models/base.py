from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, List, Optional

from agentic_search.types import ImageLike


class BaseModel(ABC):
    def __init__(self, model_name_or_path: str, **kwargs):
        self.model_name_or_path = model_name_or_path
        self.kwargs = kwargs

    @abstractmethod
    def generate_response(self, text: str, images: Optional[List[ImageLike]] = None, **kwargs) -> str:
        raise NotImplementedError
