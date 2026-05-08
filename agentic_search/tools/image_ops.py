from __future__ import annotations

import hashlib
from pathlib import Path
from typing import List

from PIL import Image

from agentic_search.tools.base import BaseSkill
from agentic_search.types import SkillResult
from agentic_search.utils.image_utils import SAVE_IMAGE_DIR, download_image_to_local


def _bbox_1000_to_pixels(bbox: List[float], width: int, height: int) -> List[int]:
    """
    Convert normalized [0, 1000] bbox to pixel bbox.
    bbox format: [x1, y1, x2, y2]
    """
    if len(bbox) != 4:
        raise ValueError(f"bbox must have 4 elements, got: {bbox}")

    x1, y1, x2, y2 = bbox

    # Clamp to valid range
    x1 = max(0.0, min(1000.0, float(x1)))
    y1 = max(0.0, min(1000.0, float(y1)))
    x2 = max(0.0, min(1000.0, float(x2)))
    y2 = max(0.0, min(1000.0, float(y2)))

    # Ensure correct order
    if x1 > x2:
        x1, x2 = x2, x1
    if y1 > y2:
        y1, y2 = y2, y1

    # [0,1000] -> pixel
    px1 = int(round(x1 / 1000.0 * width))
    py1 = int(round(y1 / 1000.0 * height))
    px2 = int(round(x2 / 1000.0 * width))
    py2 = int(round(y2 / 1000.0 * height))

    # Pixel boundary protection
    px1 = max(0, min(width, px1))
    py1 = max(0, min(height, py1))
    px2 = max(0, min(width, px2))
    py2 = max(0, min(height, py2))

    if px1 == px2 or py1 == py2:
        raise ValueError(
            f"Converted bbox has zero area: input={bbox}, "
            f"pixel_bbox={[px1, py1, px2, py2]}, image_size={(width, height)}"
        )

    return [px1, py1, px2, py2]


class ImageCropSkill(BaseSkill):
    name = "image_crop"
    description = (
        "Crop an image by bbox [x1, y1, x2, y2]. "
        "bbox uses normalized coordinates in [0, 1000], not pixels. "
        "Args: image, bbox, save_dir(optional)."
    )

    def run(self, **kwargs) -> SkillResult:
        image = kwargs["image"]
        bbox = kwargs["bbox"]
        save_dir = kwargs.get("save_dir", SAVE_IMAGE_DIR)
        Path(save_dir).mkdir(parents=True, exist_ok=True)

        try:
            if isinstance(image, str) and (
                image.startswith("http://")
                or image.startswith("https://")
                or image.startswith("file://")
            ):
                local_path, _ = download_image_to_local(image, save_dir=save_dir)
                pil = Image.open(local_path).convert("RGB")
            elif isinstance(image, str):
                pil = Image.open(image).convert("RGB")
            elif isinstance(image, Image.Image):
                pil = image.convert("RGB")
            else:
                raise ValueError(f"Unsupported image input: {type(image)}")

            width, height = pil.size
            pixel_bbox = _bbox_1000_to_pixels(bbox, width=width, height=height)

            cropped = pil.crop(tuple(pixel_bbox))
            hash_input = f"{str(image)}|{tuple(pixel_bbox)}"
            file_hash = hashlib.sha256(hash_input.encode()).hexdigest()[:16]
            out_path = str(Path(save_dir) / f"crop_{file_hash}.jpg")
            cropped.save(out_path)

            return SkillResult(
                ok=True,
                skill_name=self.name,
                output={
                    "cropped_image_path": out_path,
                    "bbox_1000": bbox,
                    "bbox_pixels": pixel_bbox,
                    "image_size": [width, height],
                },
            )
        except Exception as e:
            return SkillResult(ok=False, skill_name=self.name, output=None, error=repr(e))
