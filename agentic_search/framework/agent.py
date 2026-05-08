from __future__ import annotations

import os
from dataclasses import asdict
from enum import Enum
from typing import Any, Dict, List, Optional

from PIL import Image

from agentic_search.framework.result import AgentRunResult, StepTrace
from agentic_search.framework.state import AgentState
from agentic_search.models.base import BaseModel
from agentic_search.parsing.actions import parse_actions
from agentic_search.prompts import DEFAULT_SYSTEM_PROMPT, build_user_prompt, build_memory_update_prompt
from agentic_search.tools.registry import SkillRegistry
from agentic_search.types import ParsedAction
from agentic_search.utils.text_utils import to_pretty_json


class AgentMode(str, Enum):
    FIXED_DEPTH = "fixed_depth"
    UNTIL_DONE = "until_done"


class AgenticSearchFramework:
    def __init__(
        self,
        model: BaseModel,
        skill_registry: SkillRegistry,
        mode: AgentMode = AgentMode.UNTIL_DONE,
        max_iters: int = 8,
        system_prompt: str = DEFAULT_SYSTEM_PROMPT,
        max_actions_per_interact: int = 2,
        use_running_memory: bool = False,
    ):
        self.model = model
        self.skill_registry = skill_registry
        self.mode = mode
        self.max_iters = max_iters
        self.system_prompt = system_prompt
        self.max_actions_per_interact = max_actions_per_interact
        self.use_running_memory = use_running_memory
        self._image_search_count: int = 0

    def _strip_local_paths(self, obj: Any) -> Any:
        if isinstance(obj, dict):
            out = {}
            for k, v in obj.items():
                if k in {"local_path", "cropped_image_path", "screenshot_path", "image_path"}:
                    continue
                out[k] = self._strip_local_paths(v)
            return out
        if isinstance(obj, list):
            return [self._strip_local_paths(x) for x in obj]
        return obj

    def _render_trace_text(self, observations: List[Dict], max_items: int = 8) -> str:
        if not observations:
            return "No observations yet."
        chunks = [to_pretty_json(self._strip_local_paths(item)) for item in observations[-max_items:]]
        return "\n\n".join(chunks)

    def _render_image_summary(self, image_store: List[Dict[str, Any]]) -> str:
        if not image_store:
            return "No attached images available yet."

        lines: List[str] = []
        for item in image_store:
            parts = [f"- {item['image_id']}"]

            if item.get("origin_query"):
                parts.append(f"from {item.get('source', 'image_search')} query={item['origin_query']!r}")
            if item.get("rank") is not None:
                parts.append(f"rank={item['rank']}")
            if item.get("title"):
                parts.append(f"title={item['title']}")
            if item.get("page_url"):
                parts.append(f"page_url={item['page_url']}")
            if item.get("source") == "image_crop" and item.get("parent_image_id"):
                parts.append(f"crop_of={item['parent_image_id']}")
            if item.get("source") == "browse_web_page" and item.get("truncated") is not None:
                parts.append(f"truncated={item['truncated']}")

            lines.append("; ".join(parts))
        return "\n".join(lines)

    def _is_valid_model_image(self, path: str) -> bool:
        try:
            if not path or not os.path.exists(path):
                return False
            with Image.open(path) as im:
                im.verify()
            return True
        except Exception as e:
            print(f"[skip invalid model image] path={path} error={repr(e)}", flush=True)
            return False

    def _collect_model_images(self, image_store: List[Dict[str, Any]]) -> tuple:
        """
        Returns:
            (images, image_id_list)
        """
        images: List[str] = []
        image_id_list: List[str] = []
        for item in image_store:
            local_path = item.get("local_path")
            if local_path and self._is_valid_model_image(local_path):
                images.append(local_path)
                image_id_list.append(item.get("image_id", "unknown"))
        if image_store and not images:
            print(f"[WARNING] All {len(image_store)} images failed validation, none will be sent to model.", flush=True)
        return images, image_id_list

    def _build_prompt(
        self,
        query: str,
        observations: List[Dict],
        image_store: List[Dict[str, Any]],
        remaining_budget: int,
        image_id_list: Optional[List[str]] = None,
        running_memory: str = "",
    ) -> str:
        obs_window = 3 if self.use_running_memory else 1
        trace_text = self._render_trace_text(observations, max_items=obs_window)
        user_prompt = build_user_prompt(
            query=query,
            trace_text=trace_text,
            skill_descriptions=self.skill_registry.specs(),
            running_memory=running_memory,
        )

        image_text = self._render_image_summary(image_store)

        if image_id_list:
            mapping_lines = ["The following images are passed to the model as visual inputs:"]
            for pos_idx, img_id in enumerate(image_id_list, 1):
                mapping_lines.append(f"- Visual input position {pos_idx} -> {img_id}")
            image_position_text = "\n".join(mapping_lines)
        else:
            image_position_text = (
                "These attached images are also passed to the model as visual inputs in the same order.\n"
                "Image handles preserve original image-search rank when available.\n"
                "For example, if only rank 1, 3, and 5 were downloaded successfully,\n"
                "the available handles may be img_1, img_3, and img_5."
            )

        budget_text = f"""
Budget rules:
- Remaining interaction budget (search_depth / max_iters): {remaining_budget}
- In a single interaction, at most {self.max_actions_per_interact} action blocks will be executed.
- If you emit more than {self.max_actions_per_interact} actions, they will be split into multiple interactions.
- Each executed batch consumes 1 unit of the remaining budget before you see any new results again.
- Before calling tools, think about what you already know from the question and current observations.
- Use your own reasoning first to choose the next highest-value action.
- Avoid unnecessary searches. If current evidence already identifies the next hop, infer it and search the next step directly.
- Use <done> only when you are answering the original question directly from the current question plus the gathered observations/images.
- When you output <done>, answer based on the information already shown above. Do not output meta commentary about needing more searches or more batches.
""".strip()

        if self._is_local_search_mode():
            extra = f"""
Attached images available this turn:
{image_text}

{image_position_text}

{budget_text}

Local-mode image handle rules:
- If local_text_to_image_search or local_image_search returns an image_id such as kb_1, that image is attached to future model calls when the file is readable.
- To inspect a region of an attached image, use: <clip>{{"image":"kb_1","bbox":[x1,y1,x2,y2]}}</clip>
- To search the local KB using an attached image, use: <query>{{"skill":"local_image_search","image":"kb_1","top_k":5}}</query>
- Do not call web-only tools such as web_search, image_search, lens_search, browse_web_page, or search_image_from_local in local mode.

Do not mention local file paths in your answer.
""".strip()
        else:
            extra = f"""
Attached images available this turn:
{image_text}

{image_position_text}

{budget_text}

If you need visual evidence from a webpage (layout, poster, rendered text, page design),
or if fetch_webpage_text was not sufficient, use:
<tool name="browse_web_page">{{"url":"https://..."}}</tool>

You may refer to attached images by handle:
- for crop: <clip>{{"image":"img_1","bbox":[x1,y1,x2,y2]}}</clip>
- for reverse image search from a local image:
  <tool name="search_image_from_local">{{"image":"img_1"}}</tool>
- for lens search via handle:
  <query>{{"skill":"lens_search","image":"img_1"}}</query>

The same handle mechanism also applies to webpage screenshots such as web_1, web_2, etc.

Do not mention local file paths in your answer.
""".strip()

        return f"{self.system_prompt}\n\n{user_prompt}\n\n{extra}"

    def _update_running_memory(self, state: "AgentState", new_observations: List[Dict]) -> str:
        """Update running memory by calling the model."""
        new_obs_text = "\n\n".join(
            to_pretty_json(self._strip_local_paths(obs)) for obs in new_observations
        )

        prompt = build_memory_update_prompt(
            query=state.query,
            old_memory=state.running_memory,
            new_observations_text=new_obs_text,
        )

        new_memory = self.model.generate_response(prompt)

        if hasattr(self.model, "normalize_output"):
            new_memory = self.model.normalize_output(new_memory)

        state.running_memory = new_memory.strip()
        return state.running_memory

    def _next_image_id(self, image_store: List[Dict[str, Any]]) -> str:
        return f"img_{len(image_store) + 1}"

    def _next_browse_image_id(self, image_store: List[Dict[str, Any]]) -> str:
        existing = {x.get("image_id") for x in image_store}
        idx = 1
        image_id = f"web_{idx}"
        while image_id in existing:
            idx += 1
            image_id = f"web_{idx}"
        return image_id

    def _lookup_image(self, image_store: List[Dict[str, Any]], image_ref: str) -> Optional[Dict[str, Any]]:
        for item in image_store:
            if item.get("image_id") == image_ref:
                return item
        return None

    def _make_crop_image_id(self, image_store: List[Dict[str, Any]], parent_image_ref: str) -> str:
        existing_ids = {x.get("image_id") for x in image_store}
        image_id = f"{parent_image_ref}_crop"
        suffix = 1
        while image_id in existing_ids:
            suffix += 1
            image_id = f"{parent_image_ref}_crop_{suffix}"
        return image_id

    def _is_local_search_mode(self) -> bool:
        return self.skill_registry.exists("local_text_search")

    def _next_local_kb_image_id(self, image_store: List[Dict[str, Any]]) -> str:
        existing_ids = {x.get("image_id") for x in image_store}
        idx = 1
        image_id = f"kb_{idx}"
        while image_id in existing_ids:
            idx += 1
            image_id = f"kb_{idx}"
        return image_id

    def _attach_local_kb_images(
        self,
        result_dict: Dict[str, Any],
        image_store: List[Dict[str, Any]],
        *,
        skill_name: str,
    ) -> Dict[str, Any]:
        """Register private local KB image paths as image handles and remove paths from observations."""
        output = result_dict.get("output")
        if not isinstance(output, dict):
            return result_dict
        rows = output.get("results")
        if not isinstance(rows, list):
            return result_dict

        registered_images: List[Dict[str, Any]] = []
        for row in rows:
            if not isinstance(row, dict):
                continue
            image_path = row.pop("image_path", "") or ""
            if not image_path:
                continue

            if not os.path.exists(image_path):
                row["image_available"] = False
                continue

            image_id = self._next_local_kb_image_id(image_store)
            stored = {
                "image_id": image_id,
                "local_path": image_path,
                "remote_url": "",
                "page_url": "",
                "title": str(row.get("entity_id", "")),
                "rank": row.get("rank"),
                "origin_query": output.get("query", ""),
                "source": skill_name,
                "entity_id": row.get("entity_id"),
            }
            image_store.append(stored)
            row["image_id"] = image_id
            row["image_available"] = True
            registered_images.append(
                {
                    "image_id": image_id,
                    "rank": stored["rank"],
                    "entity_id": stored["entity_id"],
                    "source": stored["source"],
                }
            )

        if registered_images:
            output["registered_images"] = registered_images
        return result_dict

    def _execute_action(self, action: ParsedAction, image_store: List[Dict[str, Any]]) -> Dict:
        try:
            if action.action_type == "query":
                payload = dict(action.payload)
                skill_name = payload.pop("skill", "web_search")

                if skill_name in {"local_text_search", "local_text_to_image_search", "local_image_search"}:
                    if skill_name == "local_image_search" and "image" in payload:
                        image_ref = str(payload["image"])
                        item = self._lookup_image(image_store, image_ref)
                        if item:
                            payload["image"] = item["local_path"]
                    payload["include_image_path"] = True

                if skill_name == "lens_search" and "image" in payload and "image_url" not in payload:
                    image_ref = str(payload.pop("image"))
                    item = self._lookup_image(image_store, image_ref)
                    if not item:
                        raise ValueError(f"Unknown image handle: {image_ref}")

                    if item.get("remote_url"):
                        payload["image_url"] = item["remote_url"]
                        result = self.skill_registry.run("lens_search", **payload)
                        return {
                            "action_type": action.action_type,
                            "skill": "lens_search",
                            "result": asdict(result),
                        }
                    else:
                        result = self.skill_registry.run(
                            "search_image_from_local",
                            local_path=item["local_path"],
                        )
                        return {
                            "action_type": action.action_type,
                            "skill": "search_image_from_local",
                            "result": asdict(result),
                        }

                result = self.skill_registry.run(skill_name, **payload)
                result_dict = asdict(result)

                if skill_name in {"local_text_search", "local_text_to_image_search", "local_image_search"}:
                    result_dict = self._attach_local_kb_images(result_dict, image_store, skill_name=skill_name)

                if skill_name == "image_search" and result.ok and isinstance(result.output, dict):
                    downloaded = result.output.get("downloaded_images", []) or []
                    registered_images: List[Dict[str, Any]] = []

                    self._image_search_count += 1
                    existing_ids = {x.get("image_id") for x in image_store}

                    for item in downloaded:
                        rank = item.get("rank")
                        if rank is not None:
                            if self._image_search_count == 1:
                                image_id = f"img_{rank}"
                            else:
                                image_id = f"img_s{self._image_search_count}_{rank}"
                        else:
                            image_id = self._next_image_id(image_store)

                        if image_id in existing_ids:
                            print(f"[WARNING] image_id '{image_id}' already exists in image_store, skipping.", flush=True)
                            continue

                        stored = {
                            "image_id": image_id,
                            "local_path": item.get("local_path", ""),
                            "remote_url": item.get("remote_url", ""),
                            "page_url": item.get("page_url", ""),
                            "title": item.get("title", ""),
                            "rank": rank,
                            "origin_query": result.output.get("query", ""),
                            "source": "image_search",
                        }
                        image_store.append(stored)
                        existing_ids.add(image_id)

                        registered_images.append(
                            {
                                "image_id": image_id,
                                "rank": stored["rank"],
                                "origin_query": stored["origin_query"],
                                "title": stored["title"],
                                "page_url": stored["page_url"],
                                "remote_url": stored["remote_url"],
                                "source": stored["source"],
                            }
                        )

                    result_dict["output"]["registered_images"] = registered_images

                return {
                    "action_type": action.action_type,
                    "skill": skill_name,
                    "result": result_dict,
                }

            if action.action_type == "tool":
                payload = dict(action.payload)
                tool_name = action.tool_name or ""

                if tool_name in {"local_text_search", "local_text_to_image_search", "local_image_search"}:
                    if tool_name == "local_image_search" and "image" in payload:
                        image_ref = str(payload["image"])
                        item = self._lookup_image(image_store, image_ref)
                        if item:
                            payload["image"] = item["local_path"]
                    payload["include_image_path"] = True

                if tool_name == "search_image_from_local" and "image" in payload and "local_path" not in payload:
                    image_ref = str(payload.pop("image"))
                    item = self._lookup_image(image_store, image_ref)
                    if not item:
                        raise ValueError(f"Unknown image handle: {image_ref}")
                    payload["local_path"] = item["local_path"]

                result = self.skill_registry.run(tool_name, **payload)
                result_dict = asdict(result)

                if tool_name in {"local_text_search", "local_text_to_image_search", "local_image_search"}:
                    result_dict = self._attach_local_kb_images(result_dict, image_store, skill_name=tool_name)

                if tool_name == "browse_web_page" and result.ok and isinstance(result.output, dict):
                    screenshot_path = result.output.get("screenshot_path")
                    if screenshot_path:
                        image_id = self._next_browse_image_id(image_store)

                        new_item = {
                            "image_id": image_id,
                            "local_path": screenshot_path,
                            "remote_url": "",
                            "page_url": result.output.get("final_url") or result.output.get("url", ""),
                            "title": result.output.get("title", ""),
                            "rank": None,
                            "origin_query": "",
                            "source": "browse_web_page",
                            "truncated": result.output.get("truncated", False),
                        }
                        image_store.append(new_item)

                        result_dict["output"]["registered_image"] = {
                            "image_id": image_id,
                            "title": new_item["title"],
                            "page_url": new_item["page_url"],
                            "source": new_item["source"],
                            "truncated": result.output.get("truncated", False),
                            "orig_image_size": result.output.get("orig_image_size"),
                            "final_image_size": result.output.get("final_image_size"),
                        }

                return {
                    "action_type": action.action_type,
                    "skill": tool_name,
                    "result": result_dict,
                }

            if action.action_type == "code":
                result = self.skill_registry.run("python_eval", code=action.content)
                return {
                    "action_type": action.action_type,
                    "skill": "python_eval",
                    "result": asdict(result),
                }

            if action.action_type == "clip":
                payload = dict(action.payload)
                if "image" not in payload:
                    raise ValueError("<clip> payload must contain 'image'.")

                image_ref = str(payload["image"])
                source_item = self._lookup_image(image_store, image_ref)

                if source_item:
                    payload["image"] = source_item["local_path"]

                result = self.skill_registry.run("image_crop", **payload)
                result_dict = asdict(result)

                if result.ok and isinstance(result.output, dict) and result.output.get("cropped_image_path"):
                    crop_path = result.output["cropped_image_path"]
                    image_id = self._make_crop_image_id(image_store, image_ref)

                    new_item = {
                        "image_id": image_id,
                        "local_path": crop_path,
                        "remote_url": source_item.get("remote_url", "") if source_item else "",
                        "page_url": source_item.get("page_url", "") if source_item else "",
                        "title": f"crop_of_{image_ref}",
                        "rank": None,
                        "origin_query": source_item.get("origin_query", "") if source_item else "",
                        "source": "image_crop",
                        "parent_image_id": image_ref,
                    }
                    image_store.append(new_item)

                    result_dict["output"]["registered_image"] = {
                        "image_id": image_id,
                        "title": new_item["title"],
                        "page_url": new_item["page_url"],
                        "remote_url": new_item["remote_url"],
                        "source": new_item["source"],
                        "parent_image_id": image_ref,
                    }

                return {
                    "action_type": action.action_type,
                    "skill": "image_crop",
                    "result": result_dict,
                }

            if action.action_type == "done":
                return {"action_type": "done", "answer": action.content}

            return {"action_type": "text", "content": action.content}

        except Exception as e:
            return {
                "action_type": action.action_type,
                "skill": action.tool_name or action.payload.get("skill") or action.action_type,
                "error": repr(e),
            }

    def run(self, query: str) -> AgentRunResult:
        state = AgentState(query=query)
        trace: List[StepTrace] = []
        image_store: List[Dict[str, Any]] = []

        budget_used = 0
        pending_actions: List[ParsedAction] = []
        current_prompt: str = ""
        current_model_output: str = ""
        current_batch_index: int = 0

        while budget_used < self.max_iters and not state.done:
            if not pending_actions:
                image_inputs, image_id_list = self._collect_model_images(image_store)
                remaining_budget = self.max_iters - budget_used
                current_prompt = self._build_prompt(
                    query=state.query,
                    observations=state.observations,
                    image_store=image_store,
                    remaining_budget=remaining_budget,
                    image_id_list=image_id_list,
                    running_memory=state.running_memory,
                )

                raw_model_output = self.model.generate_response(
                    current_prompt,
                    images=image_inputs if image_inputs else None,
                )

                current_model_output = (
                    self.model.normalize_output(raw_model_output)
                    if hasattr(self.model, "normalize_output")
                    else raw_model_output
                )

                pending_actions = parse_actions(current_model_output)
                current_batch_index = 0

                if not pending_actions:
                    break

            batch_actions = pending_actions[: self.max_actions_per_interact]
            pending_actions = pending_actions[self.max_actions_per_interact :]
            current_batch_index += 1
            budget_used += 1

            executed: List[Dict] = []
            for action in batch_actions:
                result = self._execute_action(action, image_store=image_store)
                executed.append(result)

                if result.get("action_type") == "done":
                    state.final_answer = result.get("answer", "")
                    state.done = True
                    pending_actions = []
                    break
                else:
                    state.observations.append(result)

            # === running memory update ===
            if self.use_running_memory and not state.done:
                new_obs_for_memory = [r for r in executed if r.get("action_type") != "done"]
                if new_obs_for_memory:
                    self._update_running_memory(state, new_obs_for_memory)

            if current_batch_index == 1:
                trace_model_output = current_model_output
            else:
                trace_model_output = "[CONTINUATION OF PREVIOUS MODEL OUTPUT; ACTIONS SPLIT BY BUDGET]"

            trace.append(
                StepTrace(
                    iteration=budget_used,
                    prompt=current_prompt,
                    model_output=trace_model_output,
                    parsed_actions=[asdict(a) for a in batch_actions],
                    observations=executed,
                    running_memory=state.running_memory,
                )
            )

            if state.done:
                break

            if self.mode == AgentMode.FIXED_DEPTH and budget_used >= self.max_iters:
                break

        if pending_actions and not state.done and budget_used >= self.max_iters:
            state.observations.append(
                {
                    "action_type": "system",
                    "warning": (
                        f"Budget exhausted before executing all pending actions. "
                        f"{len(pending_actions)} action(s) were left unexecuted."
                    ),
                }
            )

        if not state.final_answer:
            state.final_answer = self._fallback_answer(state, image_store=image_store)

        return AgentRunResult(
            final_answer=state.final_answer,
            done=state.done,
            iterations=len(trace),
            trace=trace,
        )

    def _fallback_answer(self, state: AgentState, image_store: Optional[List[Dict[str, Any]]] = None) -> str:
        obs_text = self._render_trace_text(state.observations, max_items=12)
        image_text = self._render_image_summary(image_store or [])

        memory_section = ""
        if self.use_running_memory and state.running_memory:
            memory_section = f"\nAccumulated memory:\n{state.running_memory}\n"

        prompt = (
            "Given the following gathered observations and attached images, answer the original user question directly and briefly.\n\n"
            "Important:\n"
            "- Answer the original question from the information already shown above.\n"
            "- Use the current observations and images plus your reasoning.\n"
            "- Do not ask for more searches, more browsing, or more batches.\n\n"
            f"Question:\n{state.query}\n\n"
            f"{memory_section}"
            f"Available attached images:\n{image_text}\n\n"
            f"Observations:\n{obs_text}"
        )
        image_inputs, _ = self._collect_model_images(image_store or [])
        return self.model.generate_response(
            prompt,
            images=image_inputs if image_inputs else None,
        )

    def answer(self, query: str) -> str:
        return self.run(query).final_answer
