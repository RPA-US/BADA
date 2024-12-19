from action.base import Action, History, ActionResult
from action.qwen_action import AtlasActionmodel, QwenVLActionModel
from planner.base import Plan
from planner.qwen_planner import QwenVLPlanner
from prompts.action_prompts import SYS_PROMPT_MID as MIDDLEMAN
from prompts.planner_prompts import SYS_PROMPT_COT as PLANNER_COT


# TODO: take image as bytes
def take_action(
    subtask: str,
    history: History,
    image_path: str,
    task: str,
    plan: Plan,
    context: str,
    task_description: str,
) -> None:
    """
    Performs an action on the current screen given an instruction

    @param subtask: Subtask at hand from the original plan
    @param history: Actions and results until this point
    @image_path: Path to the image upon which the interaction will happen
    @task: Objective
    @param plan: Plan layed out by the planner beforehand
    @context: Bussiness context
    @task_description: Detailed description of the task at hand, from a process POV
    """

    middle_model = QwenVLActionModel("Qwen/Qwen2-VL-7B-Instruct-GPTQ-Int4")

    prompt = f"""
    **task**: {task}
    **plan**: {plan.steps}.
    **Plan reasoning**: {plan.reasoning}

    **history**:
    {history}

    **result of the last executed action**: {history.last_result}

    **task description**:
    {task_description}

    **context description**:
    {context}

    **current subtask**: {subtask}
    """
    action: Action = middle_model.action(
        MIDDLEMAN,
        prompt,
        image=image_path,
    )

    action_model = AtlasActionmodel("OS-Copilot/OS-Atlas-Base-7B")
    grounding: Action = action_model.action(
        None,
        f'In this UI screenshot, what is the position of the element corresponding to the command "{action.action_target}" (with bbox)?',
        image=image_path,
    )
    grounding.action = action.action
    grounding.action_target = action.action_target
    print("Grounding: ", grounding)

    history.append(grounding, ActionResult.PENDING)


def plan_task(
    task: str,
    image_path: str,
    context: str,
    task_description: str,
) -> Plan:
    """
    Plans ahead the steps to carry out to complete the given task

    @param task: Task to complete on the user's computer
    @image_path: Path to the image upon which the interaction will happen
    @context: Bussiness context
    @task_description: Detailed description of the task at hand, from a process POV

    @returns plan: Plan object
    """
    # prompt = """
    # **Task Description**: Register a client with email \"example@email.com\" and password \"password123\".
    # **Contextual Information**:
    # - Users must have first sent an email to the company attaching their NIF, which must be included in the registration
    # - Once registered, we respond back to the email sent by the user confirming that the registration was done correctly
    # """

    # This prompt now resembles a process description
    planner = QwenVLPlanner("Qwen/Qwen2-VL-7B-Instruct-GPTQ-Int4")

    prompt = f"""
    **Task Description**: {task}.
    **Contextual Information**:
    {task_description}
    """

    plan: Plan = planner.plan(
        PLANNER_COT.format(context=context),
        prompt,
        image=image_path,
    )

    return plan


if __name__ == "__main__":
    task: str = (
        'Register a client with email "example@email.com" and password "password123"'
    )

    context: str = """
    - The organization operates in a legal advisory setting.
    - Users are registered in the Odoo system.
    - Chrome is used as the main browser
    - Gmail is used as the email client
    """

    task_description: str = """
    - First, check email from user to see if a NIF was sent
    - If it was, register user, else respond to email and stop process
    - Register the user in Odoo
    - Send email back to user confirming registration
    """

    history = History()

    plan = plan_task(
        task,
        image_path=".resources/A_720p.png",
        context=context,
        task_description=task_description,
    )

    take_action(
        plan.steps[0],
        history,
        image_path=".resources/A_720p.png",
        task=task,
        plan=plan,
        context=context,
        task_description=task_description,
    )

    print("\n-----------------------\n", history)
