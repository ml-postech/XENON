from typing import List, Optional

import torch
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info

from ..util.prompt import language_action_to_subgoal

from .base_model import BasePlanningModel

prompt_decomposed_plan = """For an item name, you need to make a plan using examples.
"""

#################### Our context-aware reasoning prompt ####################
description_prompt = """Given a Minecraft game image, describe nearby Minecraft objects, like tree, grass, cobblestone, etc.

[Example]
"There is a large tree with dark green leaves surrounding the area."
"The image shows a dark, cave-like environment in Minecraft. The player is digging downwards. There are no visible trees or grass in this particular view."
"The image shows a dark, narrow tunnel made of stone blocks. The player is digging downwards."

[Your turn]
Describe the given image, simply and clearly like the examples."""

context_aware_reasoning_prompt = """
Given <task> and <visual_description>, determine if the player needs intervention to achieve the goal. If intervention is needed, suggest a task that the player should perform.
I will give you examples.

[Example]
<task>: chop tree
<visual_description>: There is a large tree with dark green leaves surrounding the area.
<goal>: logs
<reasoning>:
{{
    "need_intervention": false,
    "thoughts": "The player can see a tree and can chop it down to get logs.",
    "task": "",
}}

[Example]
<task>: chop tree
<visual_description>: The image shows a dirt block in Minecraft. There is a tree in the image, but it is too far from here.
<goal>: logs
<reasoning>:
{{
    "need_intervention": true,
    "thoughts": "The player is far from trees. The player needs to move to the trees.",
    "task": "explore to find trees",
}}

[Example]
<task>: dig down to mine iron_ore
<visual_description>: The image shows a dark, narrow tunnel made of stone blocks. The player is digging downwards.
<goal>: iron_ore
<reasoning>:
{{
    "need_intervention": false,
    "thoughts": "The player is already digging down and is likely to find iron ore.",
    "task": "",
}}

[Your turn]
Here is the <task>, <visual_description>, and <goal>.
You MUST output the <reasoning> in JSON format.
<task>: {task}
<visual_description>: {visual_description}
<goal>: {goal}
<reasoning>:
"""


def is_path(path):
    if len(path) == 2:
        return True
    else:
        return False


class PlanningModel(BasePlanningModel):

    def __init__(self, model_path: str = "Qwen/Qwen2.5-VL-7B-Instruct", device_id: int = 1,
                 system_prompt: Optional[str] = None) -> None:
        self.device = f"cuda:{device_id}"

        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,
        )
        self.model = self.model.to(self.device).eval()
        self.processor = AutoProcessor.from_pretrained(model_path)

    def decomposed_plan(
        self,
        waypoint: str,
        images: str | List[str],
        similar_wp_sg_dict: dict | None = None,
        failed_sg_list_for_wp: List[str] | None = None,
    ):
        images = None
        prompt = prompt_decomposed_plan

        if similar_wp_sg_dict is not None and len(similar_wp_sg_dict) > 0:
            prompt += "I will give you examples of which plans are needed to achieve an item.\n"
            for similar_wp, sg_str in similar_wp_sg_dict.items():
                prompt += f"""[Example]
<item name>
{similar_wp}
<task planning>
{sg_str}

"""
        else:
            # similar waypoints are not available
            # That is, it does not use memory
            pass

        if "log" not in waypoint:
            # waypoint is not logs
            language_actions = ["dig down and mine", "craft", "smelt"]
            for failed_sg_str in failed_sg_list_for_wp:
                if "mine" in failed_sg_str:
                    language_actions.remove("dig down and mine")
                elif "craft" in failed_sg_str:
                    language_actions.remove("craft")
                elif "smelt" in failed_sg_str:
                    language_actions.remove("smelt")
            language_action_options = [f"{action} {waypoint}" for action in language_actions]
            if len(language_action_options) == 0:
                language_action_options = [f"dig down and mine {waypoint}", f"craft {waypoint}", f"smelt {waypoint}"]
        else:
            # waypoint is logs
            language_actions = ["chop a tree", "craft logs", "smelt logs"]
            for failed_sg_str in failed_sg_list_for_wp:
                if "mine" in failed_sg_str or "chop" in failed_sg_str:
                    language_actions.remove("chop a tree")
                elif "craft" in failed_sg_str:
                    language_actions.remove("craft logs")
                elif "smelt" in failed_sg_str:
                    language_actions.remove("smelt logs")
            language_action_options = language_actions
            if len(language_action_options) == 0:
                language_action_options = [f"chop a tree", f"craft {waypoint}", f"smelt {waypoint}"]

        language_subgoal_options = []
        i = 1
        for action in language_action_options:
            _, subgoal = language_action_to_subgoal(action, waypoint)
            language_subgoal_options.append(f"{i}. {subgoal}")
            i += 1
        options_str = "\n".join(language_subgoal_options)

        prompt += f"""
[Your turn]
Here is <item name>, you MUST output <task planning> in JSON format.
You can make <task planning> by selecting an option from below:
{options_str}

<item name>
{waypoint}
<task planning>
"""

        print(f"====\n{prompt}\n====")
        return self._inference(prompt, None), prompt


    def context_aware_reasoning(
        self,
        task: str,
        goal: str,
        image_path: str,
    ):
        visual_description = self._inference(description_prompt, image_path)

        new_reasoning_prompt = context_aware_reasoning_prompt.format(
            task=task,
            visual_description=visual_description,
            goal=goal,
        )
        reasoning = self._inference(new_reasoning_prompt, None)
        return reasoning, visual_description
    


    # From Optimus-1

    # def retrieve(
    #     self,
    #     task: str,
    #     image_path: str,
    # ):
    #     return self._inference(retrieve_prompt.format(task=task), image_path)


#     def replan(
#         self,
#         task: str,
#         image_path: str,
#         error_info: str | None = None,
#         examples: str | None = None,
#         graph_summary: str | None = None,
#     ):
#         logic1 = ""
#         if examples is None or examples == "":
#             logic1 = """craft 1 crafting_table summary:
# 1. log: need 1
# 2. planks: need 4
# 3. crafting_table: need 1"""
#             examples = """<task>: craft wooden_pickaxe.
# <error>: missing material: {"crafting_table": 1}.
# <replan>: 
# {
#     "step 1": {"task": "chop tree", "goal": ["logs", 1]},
#     "step 2": {"task": "craft planks", "goal": ["planks", 4]},
#     "step 3": {"task": "craft crafting table", "goal": ["crafting_table", 1]
# }
# """

#         if logic1 == "":
#             logic1 = graph_summary

#         if graph_summary is None or graph_summary == "":
#             prompt = non_reflection_replan_prompt.format(
#                 task1=task,
#                 logic1=logic1,
#                 example=examples.strip(),
#                 error=error_info,
#             )
#         else:
#             prompt = replan_prompt.format(
#                 task1=task,
#                 logic1=logic1,
#                 logic=graph_summary.strip(),  # type: ignore
#                 example=examples.strip(),
#                 error=error_info,
#             )

#         return self._inference(prompt, image_path)


    # def planning(
    #     self,
    #     task: str,
    #     images: str | List[str],
    #     example: str | None = None,
    #     visual_info: str | None = None,
    #     graph: str | None = None,
    # ):
    #     if visual_info is None and graph is None:
    #         prompt = no_reflection_plan_prompt.format(task=task, example=example)
    #     else:
    #         prompt = plan_prompt.format(
    #             task=task,
    #             example=example,
    #             visual=visual_info,
    #             graph=graph,
    #         )
    #     print(f"====\n{prompt}\n====")
    #     return self._inference(prompt, images)

    # def reflection(
    #     self,
    #     task: str,
    #     done_path: List[str],
    #     continue_path: List[str],
    #     replan_path: List[str],
    #     image_path: List[str],
    # ):
    #     is_done, is_continue, is_replan = (
    #         is_path(done_path),
    #         is_path(continue_path),
    #         is_path(replan_path),
    #     )
    #     prompt = reflection_systerm.format(task=task)
    #     imgs = []
    #     if is_done or is_continue or is_replan:
    #         prompt += "\n" + reflection_examples
    #         if is_done:
    #             prompt += f"\n<done>:\n{self.IMAGE_TAG} {self.IMAGE_TAG}"
    #             imgs += done_path

    #         if is_continue:
    #             prompt += f"\n<continue>:\n{self.IMAGE_TAG} {self.IMAGE_TAG}"
    #             imgs += continue_path

    #         if is_replan:
    #             prompt += f"\n<replan>:\n{self.IMAGE_TAG} {self.IMAGE_TAG}"
    #             imgs += replan_path

    #     imgs += image_path
    #     prompt += reflection_prompt

    #     return self._inference(prompt, imgs)


    def _inference(self, instruction: str, images: str | List[str] = None) -> str:
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": instruction},
                    {"type": "image", "image": images},
                ]
            },
        ]

        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        if images is None:
            inputs = self.processor(
                text=[text],
                padding=True,
                return_tensors="pt",
            )
        else:
            image_inputs, video_inputs = process_vision_info(messages)
            inputs = self.processor(
                text=[text],
                images=image_inputs,
                videos=video_inputs,
                padding=True,
                return_tensors="pt",
            )
        inputs = inputs.to(self.model.device)

        generated_ids = self.model.generate(**inputs, max_new_tokens=512)
        generated_ids_trimmed = [
            out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        response = self.processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )

        return response[0]
