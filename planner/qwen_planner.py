import re

from models.models import QwenVLModel
from planner.base import BasePlanner, Plan


class QwenVLPlanner(BasePlanner, QwenVLModel):
    """
    Planner implentation for QwenVL based models. It also supports derivatives such as OSAtlas
    """

    def __init__(self, model_name: str):
        super().__init__(model_name)

    def plan(self, sys_prompt, user_prompt, *args, **kwargs):
        messages, processed_output_text = super().inference(
            sys_prompt, user_prompt, *args, **kwargs
        )

        processed_output_text = next(iter(processed_output_text), None)

        if not processed_output_text:
            raise RuntimeError(
                "Something went wrong while generating the plan and no output was given by the model"
            )

        return self.parse_plan(messages, processed_output_text)

    def parse_plan(self, prompt: list[dict[str, str]], plan: str):
        reasoning_pattern = r"<\|reasoning_begin\|>(.*?)<\|reasoning_end\|>"
        steps_pattern = r"<\|steps_begin\|>(.*?)<\|steps_end\|>"

        reasoning_match = re.search(reasoning_pattern, plan, re.DOTALL)
        steps_match = re.search(steps_pattern, plan, re.DOTALL)

        reasoning_content = (
            reasoning_match.group(1).strip() if reasoning_match else None
        )
        steps_content = steps_match.group(1).strip() if steps_match else None

        if reasoning_content is None:
            return Plan(prompt, plan, None, steps_content)

        reasoning_dict = {}
        sections = re.split(r"\n\d+\.\s", reasoning_content)

        for section in sections[1:]:
            lines = section.strip().split("\n- ")
            key = lines[0].strip()
            values = [line.strip() for line in lines[1:]]
            reasoning_dict[key] = values

        return Plan(prompt, plan, reasoning_dict, steps_content)
