import re
import jax.numpy as jnp
from typing import Tuple

from action.base import Action, ActionInterface
from models.models import QwenVLModel


class QwenVLActionModel(ActionInterface, QwenVLModel):
    """
    ActionModel implentation for QwenVL based models. It also supports derivatives such as OSAtlas
    """

    capabilities: list[str] = ["image", "text"]

    def __init__(self, model_name: str, *args, **kwargs):
        super().__init__(model_name, *args, **kwargs)

    def action(self, sys_prompt, user_prompt, *args, **kwargs):
        messages, processed_output_text = super()._call(
            sys_prompt, user_prompt, *args, **kwargs
        )

        processed_output_text = next(iter(processed_output_text), None)

        if not processed_output_text:
            raise RuntimeError(
                "Something went wrong while generating the action and no output was given by the model"
            )

        return self.parse_action(messages, processed_output_text)

    def parse_action(self, prompt: list[dict[str, str]], model_response: str):
        reasoning_pattern = (
            r"<\|context_analysis_begin\|>(.*?)<\|context_analysis_end\|>"
        )
        action_name_pattern = r"<\|action_begin\|>(.*?)<\|action_end\|>"
        action_target_pattern = r"\[(.*)\]"

        reasoning_match = re.search(reasoning_pattern, model_response, re.DOTALL)
        action_name_match = re.search(action_name_pattern, model_response, re.DOTALL)
        action_target_match = re.search(
            action_target_pattern, model_response, re.DOTALL
        )

        reasoning_content = (
            reasoning_match.group(1).strip() if reasoning_match else None
        )
        action_name_content = (
            action_name_match.group(1).strip() if action_name_match else None
        )
        action_target_content = (
            action_target_match.group(1).strip() if action_target_match else None
        )

        if reasoning_content is None:
            return Action(
                prompt, action_target_content, model_response, action_name_content
            )

        reasoning = re.split(r"\n\d+\.\s", reasoning_content)

        return Action(
            prompt,
            action_target_content,
            model_response,
            action=action_name_content,
            reasoning=reasoning,
        )


class AtlasActionmodel(QwenVLActionModel, QwenVLModel):
    def parse_action(self, prompt: list[dict[str, str]], model_response: str):
        object_ref_pattern = r"<\|object_ref_start\|>(.*?)<\|object_ref_end\|>"
        box_pattern = r"<\|box_start\|>(.*?)<\|box_end\|>"

        object_ref_match = re.search(object_ref_pattern, model_response, re.DOTALL)
        box_match = re.search(box_pattern, model_response, re.DOTALL)

        object_ref_content = (
            object_ref_match.group(1).strip() if object_ref_match else None
        )
        box_content = box_match.group(1).strip() if box_match else None
        coords: Tuple[int, int]
        if box_content:
            num_pattern = r"(\d+).*?(\d+)"  # Number then closest number to it
            nums = re.findall(num_pattern, box_content)
            element_bbox = jnp.asarray([(int(x), int(y)) for x, y in nums])
            coords = tuple(jnp.mean(element_bbox, axis=0).tolist())

        return Action(prompt, object_ref_content, model_response, coords=coords)
