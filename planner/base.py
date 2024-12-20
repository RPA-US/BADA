from abc import abstractmethod
from typing import Optional


class Plan:
    """
    Representation of a plan given by a planner using a structured format
    """

    def __init__(
        self,
        prompt: list[dict[str, str]],
        raw: str,
        steps: list[str],
        reasoning: Optional[dict[str, list[str]]] = None,
    ):
        self.prompt = prompt
        self.raw = raw
        self.reasoning = reasoning
        self.steps = steps

    def __str__(self):
        return f"""
Model Reasoning: {self.reasoning}
---
Provided steps: {self.steps}
"""

    def to_str_extended(self):
        return f"""
Prompt: {self.prompt}
---
Raw Output: {self.raw}
---
Model Reasoning: {self.reasoning}
---
Provided steps: {self.steps}
"""


class PlannerInterface:
    @abstractmethod
    def plan(self, sys_prompt, user_prompt, *args, **kwargs) -> Plan:
        """
        Creates a plan for a current state of the desktop

        @param sys_prompt: The system prompt to give to the model
        @param user_prompt: User textual prompt for the model

        @returns Plan object

        Additionally kwargs can be provided to include extra messages with the user role, such as an image.


        """

        pass

    @abstractmethod
    def parse_plan(self, prompt: list[dict[str, str]], plan: str) -> Plan:
        """
        Given a plan in string format, it parses it and returns a Plan object

        @param prompt: The prompt given to the model
        @param plan: Output of plan method

        @returns Plan object
        """
        pass
