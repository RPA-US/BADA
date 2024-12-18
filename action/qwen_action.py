import re
from typing import List, Tuple

from action.base import Action, BaseActionModel
from models.models import QwenVLModel


class QwenVLActionModel(BaseActionModel, QwenVLModel):
    """
    ActionModel implentation for QwenVL based models. It also supports derivatives such as OSAtlas
    """

    capabilities: list[str] = ["image", "text"]

    def __init__(self, model_name: str, quantization_bits: int = None):
        BaseActionModel.__init__(self, model_name)
        QwenVLModel.__init__(self, model_name, quantization_bits)

    def action(self, sys_prompt, user_prompt, *args, **kwargs):
        messages, processed_output_text = super().inference(
            sys_prompt, user_prompt, *args, **kwargs
        )

        processed_output_text = next(iter(processed_output_text), None)

        if not processed_output_text:
            raise RuntimeError(
                "Something went wrong while generating the action and no output was given by the model"
            )

        return self.parse_action(messages, processed_output_text)

    def parse_action(self, prompt: list[dict[str, str]], action: str):
        reasoning_pattern = (
            r"<\|context_analysis_begin\|>(.*?)<\|context_analysis_end\|>"
        )
        object_ref_pattern = r"<\|object_ref_start\|>(.*?)<\|object_ref_end\|>"

        reasoning_match = re.search(reasoning_pattern, action, re.DOTALL)
        object_ref_match = re.search(object_ref_pattern, action, re.DOTALL)

        reasoning_content = (
            reasoning_match.group(1).strip() if reasoning_match else None
        )
        object_ref_content = (
            object_ref_match.group(1).strip() if object_ref_match else None
        )

        if reasoning_content is None:
            return Action(prompt, object_ref_content, action)

        reasoning = re.split(r"\n\d+\.\s", reasoning_content)

        return Action(prompt, object_ref_content, action, reasoning=reasoning)


class AtlasActionmodel(QwenVLActionModel, QwenVLModel):
    def parse_action(self, prompt: list[dict[str, str]], action: str):
        print(action)
        object_ref_pattern = r"<\|object_ref_start\|>(.*?)<\|object_ref_end\|>"
        box_pattern = r"<\|box_start\|>(.*?)<\|box_end\|>"

        object_ref_match = re.search(object_ref_pattern, action, re.DOTALL)
        box_match = re.search(box_pattern, action, re.DOTALL)

        object_ref_content = (
            object_ref_match.group(1).strip() if object_ref_match else None
        )
        box_content = box_match.group(1).strip() if box_match else None
        element_bbox: List[Tuple[int, int]]
        if box_content:
            num_pattern = r"(\d+).*?(\d+)"  # Number then closest number to it
            nums = re.findall(num_pattern, box_content)
            element_bbox = [(int(x), int(y)) for x, y in nums]

        return Action(prompt, object_ref_content, action, coords=element_bbox)
