from typing import Dict, List, Optional, NamedTuple
from enum import Enum, auto

######
# BASE TYPES
######


class Action:
    """
    Representation of an action given by a action model using a structured format
    """

    def __init__(
        self,
        prompt: List[dict[str, str]],
        action_target: str,
        raw: str,
        action: Optional[str] = None,
        reasoning: Optional[Dict[str, list[str]] | str] = None,
        coords: Optional[List[float | list[float]]] = None,
        key: Optional[str] = None,
    ):
        if coords and key:
            raise ValueError("Coords and Key are mutually exclusive")
        self.prompt = prompt
        self.action = action
        self.action_target = action_target
        self.raw = raw
        self.reasoning = reasoning  # optional
        self.coords = coords  # optional
        self.key = key  # optional

    def __str__(self):
        return f"""
Model Reasoning: {self.reasoning}
---
Provided action: {self.action} {self.action_target} {self.key if self.key else self.coords if self.coords else ""}
"""

    def to_str_extended(self):
        return f"""
Prompt: {self.prompt}
---
Raw Output: {self.raw}
---
Model Reasoning: {self.reasoning}
---
Provided action: {self.action} {self.action_target} {self.key if self.key else self.coords if self.coords else ""}
"""


class ActionResult(Enum):
    FAIL = auto()
    PENDING = auto()
    SUCCESS = auto()

    def __str__(self):
        return self.name


class ActionExecution(NamedTuple):
    action: Action
    result: ActionResult


class History:
    """
    Stores past actions for an ongoing task execution and its corresponding results
    """

    def __init__(
        self,
        actions: List[Action] = [],
        results: List[ActionExecution] = [],
    ):
        self.actions = actions
        self.results = results

    def __iter__(self):
        return self.results

    def __str__(self):
        return "\n".join(
            [
                f"Executed {action} with result {result}"
                for action, result in self.results
            ]
        )

    @property
    def last_result(self):
        last = next(iter(self.results[-1:]), None)
        return last.result if last else None

    def append(self, action: Action, result: ActionExecution):
        self.actions.append(action)
        self.results.append(ActionExecution(action, result))


######
# INTERFACES
#####


class ActionInterface:
    def action(self, sys_prompt, user_prompt, *args, **kwargs) -> Action:
        """
        Creates an action for a current state of the desktop given a action and an action to execute

        @param sys_prompt: The system prompt to give to the model
        @param user_prompt: User textual prompt for the model

        @returns Action object

        Additionally kwargs can be provided to include extra messages with the user role, such as an image.
        """

        pass

    def parse_action(self, action: str) -> Action:
        """
        Given a action in string format, it parses it and returns a Action object

        @param action: Output of action method

        @returns Action object
        """
        pass
