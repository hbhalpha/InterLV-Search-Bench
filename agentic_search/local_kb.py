from __future__ import annotations

import importlib
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np


def _l2_normalize(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    if x.ndim == 1:
        denom = np.linalg.norm(x)
        return x if denom == 0 else x / denom
    denom = np.linalg.norm(x, axis=1, keepdims=True)
    denom[denom == 0] = 1.0
    return x / denom


def _topk(scores: np.ndarray, k: int, mask: Optional[np.ndarray] = None) -> List[int]:
    if scores.size == 0:
        return []
    working = np.asarray(scores, dtype=np.float32).copy()
    if mask is not None:
        mask = np.asarray(mask, dtype=bool)
        if mask.shape != working.shape:
            raise ValueError(f"mask shape {mask.shape} does not match scores shape {working.shape}")
        if not mask.any():
            return []
        working[~mask] = -np.inf
    finite = np.isfinite(working)
    if not finite.any():
        return []
    k = max(1, min(int(k), int(finite.sum())))
    idx = np.argpartition(-working, k - 1)[:k]
    idx = idx[np.argsort(-working[idx])]
    return idx.tolist()


def _truncate(text: str, max_chars: int) -> str:
    text = text or ""
    if max_chars <= 0 or len(text) <= max_chars:
        return text
    return text[:max_chars] + " ..."


def _load_json_or_jsonl(path: Path) -> List[dict]:
    text = path.read_text(encoding="utf-8").strip()
    if not text:
        return []
    if path.suffix.lower() == ".jsonl":
        return [json.loads(line) for line in text.splitlines() if line.strip()]
    data = json.loads(text)
    if isinstance(data, list):
        return data
    if isinstance(data, dict) and "records" in data and isinstance(data["records"], list):
        return data["records"]
    raise ValueError(f"Expected a JSON list, JSONL file, or object with a records list: {path}")


@dataclass
class KBRecord:
    entity_id: str
    image_path: str
    text: str


class Qwen3VLEmbedderWrapper:
    """
    Thin wrapper around a user-provided multimodal embedding class.

    Configure the class path via env:
      AGENTIC_SEARCH_QWEN_EMBEDDER_CLASS=your_module.path:YourEmbedder

    The embedder class must expose:
      process(list[dict]) -> list[tensor|np.ndarray]
    where each item is one of:
      {"text": ...}
      {"image": ...}
      {"image": ..., "text": ...}
    """

    def __init__(self, model_name_or_path: str):
        self.model_name_or_path = model_name_or_path
        class_path = os.getenv(
            "AGENTIC_SEARCH_QWEN_EMBEDDER_CLASS",
            "src.models.qwen3_vl_embedding:Qwen3VLEmbedder",
        )
        try:
            module_name, class_name = class_path.split(":", 1)
            module = importlib.import_module(module_name)
            embedder_cls = getattr(module, class_name)
        except Exception as exc:
            raise RuntimeError(
                "Failed to import the local KB embedder class. Set "
                "AGENTIC_SEARCH_QWEN_EMBEDDER_CLASS to 'module.path:ClassName'. "
                "The class must implement process(list[dict]) -> list[array/tensor]. "
                f"Tried: {class_path!r}."
            ) from exc

        torch_dtype_name = os.getenv("QWEN_DTYPE", "float16").strip().lower()
        try:
            import torch
            torch_dtype = {
                "float16": torch.float16,
                "fp16": torch.float16,
                "bfloat16": torch.bfloat16,
                "bf16": torch.bfloat16,
                "float32": torch.float32,
                "fp32": torch.float32,
                "auto": "auto",
            }.get(torch_dtype_name, torch.float16)
            self.model = embedder_cls(model_name_or_path=model_name_or_path, torch_dtype=torch_dtype)
        except TypeError:
            self.model = embedder_cls(model_name_or_path=model_name_or_path)

    def _to_numpy(self, x) -> np.ndarray:
        try:
            import torch
            if isinstance(x, torch.Tensor):
                return x.detach().cpu().numpy().astype(np.float32)
        except Exception:
            pass
        return np.asarray(x, dtype=np.float32)

    def embed_queries(self, queries: List[Dict[str, Any]]) -> List[np.ndarray]:
        outs = self.model.process(queries)
        return [self._to_numpy(x).reshape(-1) for x in outs]

    def embed_texts(self, texts: Sequence[str]) -> List[np.ndarray]:
        return self.embed_queries([{"text": t} for t in texts])

    def embed_images(self, image_paths: Sequence[str]) -> List[np.ndarray]:
        return self.embed_queries([{"image": p} for p in image_paths])


def _stack_or_empty(vecs: List[np.ndarray], dim: int = 0) -> np.ndarray:
    if vecs:
        return np.stack(vecs, axis=0).astype(np.float32)
    return np.zeros((0, dim), dtype=np.float32)


class LocalKBIndex:
    def __init__(
        self,
        records: List[KBRecord],
        text_vectors: np.ndarray,
        image_vectors: np.ndarray,
    ):
        self.records = records
        text_vectors = np.asarray(text_vectors, dtype=np.float32)
        image_vectors = np.asarray(image_vectors, dtype=np.float32)
        if len(records) != len(text_vectors) or len(records) != len(image_vectors):
            raise ValueError("records, text_vectors, and image_vectors must have the same length")
        self.image_available = np.linalg.norm(image_vectors, axis=1) > 0 if image_vectors.size else np.zeros(len(records), dtype=bool)
        self.text_vectors = _l2_normalize(text_vectors)
        self.image_vectors = _l2_normalize(image_vectors)

    @classmethod
    def from_json_records(cls, rows: List[dict], embedder: Qwen3VLEmbedderWrapper, batch_size: int = 8) -> "LocalKBIndex":
        records: List[KBRecord] = []
        texts: List[str] = []
        images: List[str] = []

        for row in rows:
            entity_id = str(row["id"])
            image_path = str(row.get("image_path") or "")
            text = str(row.get("text") or "")
            records.append(KBRecord(entity_id=entity_id, image_path=image_path, text=text))
            texts.append(text)
            images.append(image_path)

        text_vecs: List[np.ndarray] = []
        for i in range(0, len(records), batch_size):
            text_vecs.extend(embedder.embed_texts(texts[i:i + batch_size]))

        if text_vecs:
            dim = int(text_vecs[0].reshape(-1).shape[0])
        else:
            dim = 0

        image_vecs: List[Optional[np.ndarray]] = [None] * len(records)
        image_jobs: List[Tuple[int, str]] = [
            (i, p) for i, p in enumerate(images) if p and Path(p).exists()
        ]
        for start in range(0, len(image_jobs), batch_size):
            batch = image_jobs[start:start + batch_size]
            idxs = [x[0] for x in batch]
            paths = [x[1] for x in batch]
            for idx, vec in zip(idxs, embedder.embed_images(paths)):
                image_vecs[idx] = vec

        zero = np.zeros(dim, dtype=np.float32)
        final_image_vecs = [
            np.asarray(v, dtype=np.float32).reshape(-1) if v is not None else zero.copy()
            for v in image_vecs
        ]

        return cls(
            records=records,
            text_vectors=_stack_or_empty([np.asarray(v).reshape(-1) for v in text_vecs], dim=dim),
            image_vectors=_stack_or_empty(final_image_vecs, dim=dim),
        )

    @classmethod
    def load(cls, index_dir: str | Path) -> "LocalKBIndex":
        index_dir = Path(index_dir)
        meta_rows = _load_json_or_jsonl(index_dir / "records.jsonl")
        records = [KBRecord(entity_id=str(r["id"]), image_path=str(r.get("image_path") or ""), text=str(r.get("text") or "")) for r in meta_rows]
        text_vectors = np.load(index_dir / "text_vectors.npy")
        image_vectors = np.load(index_dir / "image_vectors.npy")
        return cls(records=records, text_vectors=text_vectors, image_vectors=image_vectors)

    def save(self, index_dir: str | Path) -> None:
        index_dir = Path(index_dir)
        index_dir.mkdir(parents=True, exist_ok=True)

        with (index_dir / "records.jsonl").open("w", encoding="utf-8") as f:
            for r in self.records:
                f.write(json.dumps({"id": r.entity_id, "image_path": r.image_path, "text": r.text}, ensure_ascii=False) + "\n")

        np.save(index_dir / "text_vectors.npy", self.text_vectors.astype(np.float32))
        np.save(index_dir / "image_vectors.npy", self.image_vectors.astype(np.float32))

    def _format_rows(self, idxs: List[int], scores: np.ndarray, score_key: str, max_text_chars: int, include_image_path: bool = False) -> List[Dict[str, Any]]:
        out: List[Dict[str, Any]] = []
        for rank, i in enumerate(idxs, 1):
            r = self.records[i]
            row = {
                "rank": rank,
                "entity_id": r.entity_id,
                score_key: float(scores[i]),
                "has_image": bool(r.image_path),
                "text": _truncate(r.text, max_text_chars),
            }
            if include_image_path:
                row["image_path"] = r.image_path
            out.append(row)
        return out

    def search_text_docs(self, query_vec: np.ndarray, top_k: int = 10, max_text_chars: int = 1200, include_image_path: bool = False) -> List[Dict[str, Any]]:
        q = _l2_normalize(query_vec)
        scores = self.text_vectors @ q
        idxs = _topk(scores, top_k)
        return self._format_rows(idxs, scores, "score_t2t", max_text_chars, include_image_path=include_image_path)

    def search_text_to_images(self, query_vec: np.ndarray, top_k: int = 10, max_text_chars: int = 1200, include_image_path: bool = False) -> List[Dict[str, Any]]:
        q = _l2_normalize(query_vec)
        scores = self.image_vectors @ q
        idxs = _topk(scores, top_k, mask=self.image_available)
        return self._format_rows(idxs, scores, "score_t2i", max_text_chars, include_image_path=include_image_path)

    def search_image_to_images(self, query_vec: np.ndarray, top_k: int = 10, max_text_chars: int = 1200, include_image_path: bool = False) -> List[Dict[str, Any]]:
        q = _l2_normalize(query_vec)
        scores = self.image_vectors @ q
        idxs = _topk(scores, top_k, mask=self.image_available)
        return self._format_rows(idxs, scores, "score_i2i", max_text_chars, include_image_path=include_image_path)


class LocalKBManager:
    def __init__(self, embedder: Qwen3VLEmbedderWrapper, index: LocalKBIndex):
        self.embedder = embedder
        self.index = index

    @classmethod
    def from_env(cls) -> "LocalKBManager":
        embed_model = os.getenv("AGENTIC_SEARCH_LOCAL_EMBED_MODEL")
        index_dir = os.getenv("AGENTIC_SEARCH_LOCAL_INDEX_DIR")
        if not embed_model:
            raise RuntimeError("AGENTIC_SEARCH_LOCAL_EMBED_MODEL is required for local retrieval.")
        if not index_dir:
            raise RuntimeError("AGENTIC_SEARCH_LOCAL_INDEX_DIR is required for local retrieval.")
        embedder = Qwen3VLEmbedderWrapper(embed_model)
        index = LocalKBIndex.load(index_dir)
        return cls(embedder=embedder, index=index)

    def text_search(self, query: str, top_k: int = 10, max_text_chars: int = 1200, include_image_path: bool = False) -> List[Dict[str, Any]]:
        q = self.embedder.embed_texts([query])[0]
        return self.index.search_text_docs(q, top_k=top_k, max_text_chars=max_text_chars, include_image_path=include_image_path)

    def text_to_image_search(self, query: str, top_k: int = 10, max_text_chars: int = 1200, include_image_path: bool = False) -> List[Dict[str, Any]]:
        q = self.embedder.embed_texts([query])[0]
        return self.index.search_text_to_images(q, top_k=top_k, max_text_chars=max_text_chars, include_image_path=include_image_path)

    def image_search(self, image_path: str, top_k: int = 10, max_text_chars: int = 1200, include_image_path: bool = False) -> List[Dict[str, Any]]:
        q = self.embedder.embed_images([image_path])[0]
        return self.index.search_image_to_images(q, top_k=top_k, max_text_chars=max_text_chars, include_image_path=include_image_path)
