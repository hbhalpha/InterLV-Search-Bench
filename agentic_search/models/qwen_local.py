from __future__ import annotations

from typing import Any, List, Optional

from agentic_search.models.base import BaseModel
from agentic_search.utils.image_utils import download_image_to_local


class LocalQwenModel(BaseModel):
    """
    Local Qwen multimodal wrapper.

    Notes:
    - Designed for Qwen2.5-VL style local inference.
    - Keeps the outside API minimal: load once, call generate_response(text, images=...).
    - Internals can be swapped later without changing downstream usage.
    """

    def __init__(self, model_name_or_path: str, device_map: str = "auto", torch_dtype: str = "auto", max_new_tokens: int = 1024, **kwargs):
        super().__init__(model_name_or_path, **kwargs)
        self.device_map = device_map
        self.torch_dtype = torch_dtype
        self.max_new_tokens = max_new_tokens
        try:
            from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration  # type: ignore
        except Exception as e:
            raise ImportError("LocalQwenModel requires transformers with Qwen2.5-VL support.") from e
        self.processor = AutoProcessor.from_pretrained(model_name_or_path, trust_remote_code=True)
        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_name_or_path,
            device_map=device_map,
            torch_dtype=torch_dtype,
            trust_remote_code=True,
        )

    def _normalize_images(self, images: Optional[List[Any]]) -> List[str]:
        local_paths: List[str] = []
        for image in images or []:
            if isinstance(image, str) and (image.startswith("http://") or image.startswith("https://") or image.startswith("file://")):
                local_path, _ = download_image_to_local(image, save_dir="/tmp/agentic_search_images")
                local_paths.append(local_path)
            elif isinstance(image, str):
                local_paths.append(image)
            else:
                try:
                    from PIL import Image
                    if isinstance(image, Image.Image):
                        tmp_path = f"/tmp/agentic_search_images/pil_{abs(hash(id(image)))}.png"
                        image.save(tmp_path)
                        local_paths.append(tmp_path)
                    else:
                        raise ValueError(f"Unsupported image input: {type(image)}")
                except Exception as e:
                    raise ValueError(f"Unsupported image input: {type(image)}") from e
        return local_paths

    def generate_response(self, text: str, images: Optional[List[Any]] = None, **kwargs) -> str:
        try:
            from qwen_vl_utils import process_vision_info  # type: ignore
        except Exception:
            process_vision_info = None

        local_images = self._normalize_images(images)
        if local_images:
            messages = [
                {
                    "role": "user",
                    "content": ([{"type": "text", "text": text}] + [{"type": "image", "image": p} for p in local_images]),
                }
            ]
        else:
            messages = [{"role": "user", "content": [{"type": "text", "text": text}]}]

        if hasattr(self.processor, "apply_chat_template"):
            prompt = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        else:
            prompt = text

        if process_vision_info and local_images:
            image_inputs, video_inputs = process_vision_info(messages)
            inputs = self.processor(text=[prompt], images=image_inputs, videos=video_inputs, padding=True, return_tensors="pt")
        else:
            pil_images = None
            if local_images:
                from PIL import Image
                pil_images = [Image.open(p).convert("RGB") for p in local_images]
            inputs = self.processor(text=[prompt], images=pil_images, padding=True, return_tensors="pt")

        inputs = {k: v.to(self.model.device) if hasattr(v, "to") else v for k, v in inputs.items()}
        generated = self.model.generate(**inputs, max_new_tokens=kwargs.get("max_new_tokens", self.max_new_tokens))
        trimmed = []
        for input_ids, output_ids in zip(inputs["input_ids"], generated):
            trimmed.append(output_ids[len(input_ids):])
        text_outputs = self.processor.batch_decode(trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False)
        return text_outputs[0] if text_outputs else ""
