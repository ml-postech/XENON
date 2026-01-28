import logging
import time
from typing import List
import gc

import torch

# from ..models.qwen_2_5_planning import PlanningModel as QwenPlanningModel
from ..models.qwen_vl_planning import PlanningModel as QwenVLPlanningModel
# from ..models.deepseek_vl_planning import PlanningModel as DeepSeekPlanningModel
# from ..models.gpt4_planning import PlanningModel as GPT4PlanningModel
from ..models.steve_action_model import ActionModel as SteveActionModel

logger = logging.getLogger(__name__)


# PLAN_MODEL_PATH = {
#     "deepseek-ai/deepseek-vl-7b-chat": "deepseek-ai/deepseek-vl-7b-chat",
#     "Qwen/Qwen2.5-7B-Instruct": "Qwen/Qwen2.5-7B-Instruct",
# }


class Agent:
    gpt_v: bool = False

    def __init__(
        self,
        plan_with_gpt: bool = True,
        plan_model: str | None = None,
        in_model: str = "checkpoints/vpt/2x.model",
        in_weights: str = "checkpoints/steve1/steve1.weights",
        prior_weights: str = "checkpoints/steve1/steve1_prior.pt",
    ) -> None:
        self.plan_with_gpt = plan_with_gpt
        if plan_with_gpt:
            self.gpt_v = True
            self.plan_model = GPT4PlanningModel()
            logger.info("gpt4o as planning model")
            self.reflection_model = self.plan_model
        else:
            if plan_model == "Qwen/Qwen2.5-VL-7B-Instruct":
                logger.info("Using Qwen-2.5-VL for planning.")
                self.plan_model = QwenVLPlanningModel(plan_model)
            # if plan_model == "Qwen/Qwen2.5-7B-Instruct":
            #     logger.info("Using Qwen-2.5 for planning.")
            #     self.plan_model = QwenPlanningModel(plan_model)
            # elif plan_model == "deepseek-ai/deepseek-vl-7b-chat":
            #     logger.info("Using DeepSeek-VL for planning.")
            #     self.plan_model = DeepSeekPlanningModel(plan_model)
            # elif plan_model == "Qwen/Qwen2.5-VL-7B-Instruct":
            #     logger.info("Using Qwen-2.5-VL for planning.")
            #     self.plan_model = QwenVLPlanningModel(plan_model)
            else:
                raise ValueError(f"Unknown plan model: {plan_model}")
            self.reflection_model = self.plan_model

        logger.info("Using Steve-1 for action.")
        self.action_model = SteveActionModel(
            in_model=in_model,
            in_weights=in_weights,
            prior_weights=prior_weights,
        )

        logger.info("Agent initialized.")

    def retrieve(
        self,
        task: str,
        rgb_obs: str,
    ):
        """rgb_obs: [img1, img2, ...]"""
        assert self.plan_model is not None, "The plan model is not initialized."

        plans_retrieve = self.plan_model.retrieve(task, rgb_obs)
        return plans_retrieve
    
    def decomposed_plan(
        self,
        waypoint: str,
        rgb_obs: str,
        similar_wp_sg_dict: dict | None = None,
        failed_sg_list_for_wp: List[str] | None = None,
    ):
        """rgb_obs: [img1, img2, ...]"""
        assert self.plan_model is not None, "The plan model is not initialized."
        decomposed_plan, prompt = self.plan_model.decomposed_plan(
            waypoint, rgb_obs, similar_wp_sg_dict, failed_sg_list_for_wp
        )
        return decomposed_plan, prompt

    def context_aware_reasoning(
        self,
        task: str,
        goal: str,
        rgb_obs: str
    ):
        """rgb_obs: [img1, img2, ...]"""
        assert self.plan_model is not None, "The plan model is not initialized."
        context_aware_reasoning, visual_description = self.plan_model.context_aware_reasoning(
            task, goal, rgb_obs
        )
        return context_aware_reasoning, visual_description

    def plan(
        self,
        task: str,
        rgb_obs: str,
        example: str | None = None,
        visual_info: str | None = None,
        graph: str | None = None,
    ):
        """rgb_obs: [img1, img2, ...]"""
        assert self.plan_model is not None, "The plan model is not initialized."
        if self.plan_with_gpt:
            if self.gpt_v:
                plan = self.plan_model.planning(
                    task, rgb_obs, example, visual_info, graph
                )
            else:
                plan = self.plan_model.planning(task, example)
        else:
            plan = self.plan_model.planning(task, rgb_obs, example, visual_info, graph)
        return plan
    
    def fix_json_format(
        self,
        errorneous_planning: str,
        rgb_obs: str,
    ):
        # print("fix_json_format")
        # print(errorneous_planning)
        # print(rgb_obs)
        assert self.plan_model is not None, "The plan model is not initialized."
        if self.plan_with_gpt:
            if self.gpt_v:
                fixed_json = self.plan_model.fix_json_format(
                    errorneous_planning, rgb_obs
                )
            else:
                fixed_json = self.plan_model.fix_json_format(errorneous_planning)
        else:
            fixed_json = self.plan_model.fix_json_format(errorneous_planning, rgb_obs)
        return fixed_json

    def action(self, instruction: str, rgb_obs: List[str]):
        return self.action_model.action(instruction, rgb_obs)

    def replan(
        self,
        task: str,
        rgb_obs: str,
        error_info: str | None = None,
        examples: str | None = None,
        graph_summary: str | None = None,
    ):
        assert self.plan_model is not None, "The plan model is not initialized."
        if self.plan_with_gpt:
            # TODO: change GPT plan_model function names to replan
            if self.gpt_v:
                replan = self.plan_model.replan(
                    task, rgb_obs, error_info, examples, graph_summary
                )  # type: ignore
            else:
                replan = self.plan_model.replan(task, error_info)
        else:
            replan = self.plan_model.replan(
                task, rgb_obs, error_info, examples, graph_summary
            )
        return replan

    def reflection(
        self,
        task: str,
        old_obs: str,
        current_obs: str,
        done_img_path: List[str] | None = None,
        cont_img_path: List[str] | None = None,
        replan_img_path: List[str] | None = None,
    ):
        assert self.reflection_model is not None, "The plan model is not initialized."
        if done_img_path is None:
            done_img_path = []
        if cont_img_path is None:
            cont_img_path = []
        if replan_img_path is None:
            replan_img_path = []
        reflection = self.reflection_model.reflection(
            task, done_img_path, cont_img_path, replan_img_path, [old_obs, current_obs]
        )
        return reflection


class AgentFactory:
    _agent: Agent | None = None
    _args = None

    @staticmethod
    def get_agent(
        plan_with_gpt: bool = False,
        plan_model: str | None ="Qwen/Qwen2.5-VL-7B-Instruct",
        in_model: str = "checkpoints/vpt/2x.model",
        in_weights: str = "checkpoints/steve1/steve1.weights",
        prior_weights: str = "checkpoints/steve1/steve1_prior.pt",
    ):
        if AgentFactory._agent is None:
            AgentFactory._agent = Agent(
                plan_with_gpt=plan_with_gpt,
                plan_model=plan_model,
                in_model=in_model,
                in_weights=in_weights,
                prior_weights=prior_weights,
            )
            AgentFactory._args = (
                plan_with_gpt,
                plan_model,
                in_model,
                in_weights,
                prior_weights,
            )
        return AgentFactory._agent
    
    @staticmethod
    def set_args(
        plan_with_gpt: bool = False,
        plan_model: str | None ="Qwen/Qwen2.5-VL-7B-Instruct",
        in_model: str = "checkpoints/vpt/2x.model",
        in_weights: str = "checkpoints/steve1/steve1.weights",
        prior_weights: str = "checkpoints/steve1/steve1_prior.pt",
    ):
        AgentFactory._args = (
            plan_with_gpt,
            plan_model,
            in_model,
            in_weights,
            prior_weights,
        )

    @staticmethod
    def reset():
        if AgentFactory._agent is not None:
            del AgentFactory._agent
            torch.cuda.empty_cache()
            AgentFactory._agent = None
            time.sleep(1)
            gc.collect()
            torch.cuda.empty_cache()
            time.sleep(5)
        return AgentFactory.get_agent(
            plan_with_gpt=AgentFactory._args[0],
            plan_model=AgentFactory._args[1],
            in_model=AgentFactory._args[2],
            in_weights=AgentFactory._args[3],
            prior_weights=AgentFactory._args[4],
        )
