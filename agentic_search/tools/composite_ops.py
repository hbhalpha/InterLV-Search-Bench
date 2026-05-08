from __future__ import annotations

from dataclasses import asdict
from typing import Any, Dict, List

from agentic_search.tools.base import BaseSkill
from agentic_search.types import SkillResult


def _ensure_bbox_list(bbox):
    """
    Normalize bbox input:
    - [x1,y1,x2,y2]
    - [[x1,y1,x2,y2], [..], ...]
    """
    if bbox is None:
        return []
    if isinstance(bbox, list) and len(bbox) == 4 and all(isinstance(x, (int, float)) for x in bbox):
        return [bbox]
    if isinstance(bbox, list) and bbox and isinstance(bbox[0], list):
        return bbox
    return []


class CropAndSearchSkill(BaseSkill):
    """
    Composite tool: equivalent to clip + reverse-image-search
    """
    name = "crop_and_search"
    description = (
        "Crop local region(s) from an image and then run reverse-image search on the crop(s). "
        "Args: image, bbox, goal(optional), hl(optional), gl(optional)."
    )

    def __init__(self, registry):
        self.registry = registry

    def run(self, **kwargs) -> SkillResult:
        image = kwargs.get("image") or kwargs.get("image_id") or kwargs.get("image_url")
        boxes = _ensure_bbox_list(kwargs.get("bbox"))
        goal = kwargs.get("goal", "")
        hl = kwargs.get("hl", "en")
        gl = kwargs.get("gl", "us")

        if not image:
            return SkillResult(ok=False, skill_name=self.name, output=None, error="crop_and_search requires image")
        if not boxes:
            return SkillResult(ok=False, skill_name=self.name, output=None, error="crop_and_search requires bbox")

        results: List[Dict[str, Any]] = []
        overall_ok = True
        errors: List[str] = []

        for box in boxes:
            crop_res = self.registry.run("image_crop", image=image, bbox=box)
            crop_dict = asdict(crop_res)

            one = {
                "bbox": box,
                "crop": crop_dict,
                "lens": None,
            }

            if not crop_res.ok:
                overall_ok = False
                errors.append(f"crop failed for bbox={box}: {crop_res.error}")
                results.append(one)
                continue

            crop_path = crop_res.output.get("cropped_image_path")
            if not crop_path:
                overall_ok = False
                errors.append(f"crop returned no cropped_image_path for bbox={box}")
                results.append(one)
                continue

            # Prefer local -> upload -> lens chain
            lens_res = self.registry.run(
                "search_image_from_local",
                local_path=crop_path,
                hl=hl,
                gl=gl,
            )
            lens_dict = asdict(lens_res)
            one["lens"] = lens_dict

            if not lens_res.ok:
                overall_ok = False
                errors.append(f"lens failed for bbox={box}: {lens_res.error}")

            results.append(one)

        return SkillResult(
            ok=overall_ok,
            skill_name=self.name,
            output={
                "image": image,
                "goal": goal,
                "results": results,
            },
            error="; ".join(errors) if errors else None,
        )
