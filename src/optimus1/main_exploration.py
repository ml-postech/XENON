import copy
import json
import logging
import os
import re
import shutil
import time
from typing import Any, Dict
import sys
import random

import hydra
import shortuuid
from omegaconf import DictConfig, OmegaConf
from rich.progress import Progress, TaskID, TimeElapsedColumn
# import wandb

import random
import numpy as np
import torch
import transformers

from optimus1.env import CustomEnvWrapper, env_make, register_custom_env
from optimus1.helper import NewHelper
from optimus1.memories import KnowledgeGraph as OracleGraph
from optimus1.memories import DecomposedMemory
from optimus1.memories import HypothesizedRecipeGraph

from optimus1.monitor import Monitors, StepMonitor, SuccessMonitor
from optimus1.util import (
    ServerAPI,
    base64_to_img,
    get_evaluate_task,
    get_evaluate_task_and_goal,
    get_logger,
    pretty_result,
    render_subgoal,
    max_pickaxe_level,
    render_context_aware_reasoning,
)


MINUTE = 1200
visual_info = ""


def get_recipe_data_mining(new_item_dict, curr_inventory, logger):
    if len(new_item_dict) == 0:
        logger.error("env.inventory_new_item is True, but env.inventory_new_item_what() is empty.")
        return []

    pickaxe_level = max_pickaxe_level(curr_inventory)

    recipe_data = []
    for item_name in new_item_dict.keys():
        tmp_name = copy.deepcopy(item_name)
        if 'log' in tmp_name:
            tmp_name = 'logs'
        elif 'planks' in tmp_name:
            tmp_name = 'planks'
        elif 'sapling' in tmp_name:
            tmp_name = 'saplings'
        elif 'coal' in tmp_name:
            tmp_name = 'coals'

        data = {
            "item_name": tmp_name,
            "output_qty": 1,
            "ingredients": dict(),
            "required_pickaxe": pickaxe_level,
            "is_crafting": False,
        }
        recipe_data.append(data)
    return recipe_data


def call_planner_with_retry(
    cfg: DictConfig,
    obs: Dict[str, Any],
    wp: str,
    wp_num: int,
    similar_wp_sg_dict: dict,
    failed_sg_list: list,
    hydra_path: str,
    run_uuid: str,
    logger: logging.Logger,
):
    attempts = 0
    max_retries = 3
    subgoal, sg_str = [], ""
    while attempts < max_retries:
        attempts += 1

        logger.info(f"Attempt: {attempts}, Just before get_decomposed_plan: ")
        logger.info(f"waypoint: {wp}")
        logger.info(f"similar_wp_sg_dict: {json.dumps(similar_wp_sg_dict)}")
        logger.info(f"failed_sg_list: {str(failed_sg_list)}")
        logger.info(f"Starting get_decomposed_plan ...\n")

        try:
            sg_str, prompt = ServerAPI.get_decomposed_plan(
                cfg["server"],
                obs,
                waypoint=wp,
                similar_wp_sg_dict=similar_wp_sg_dict,
                failed_sg_list_for_wp=failed_sg_list,
                hydra_path=hydra_path,
                run_uuid=run_uuid
            )

            logger.info(f'prompt before render_subgoal at attempt {attempts}')
            logger.info(f"{prompt}\n")
            logger.info(f'sg_str before render_subgoal at attempt {attempts}')
            logger.info(f"{sg_str}\n")

            tmp_subgoal, _, render_error = render_subgoal(copy.deepcopy(sg_str), wp_num)
            if render_error is None:
                break

            logger.warning(f"get_decomposed_plan at attempt {attempts} failed. Error message: {render_error}")
            if attempts >= max_retries:
                logger.error("Max retries reached. Could not fetch get_decomposed_plan.")
                return [], "", "max_tries_get_decomposed_plan"

        except Exception as e:
            logger.info(f"Error in get_decomposed_plan: {e}")
            if attempts >= max_retries:
                logger.error("Max retries reached. Could not fetch get_decomposed_plan.")
                return [], "", "max_tries_get_decomposed_plan"
            continue

    subgoal, language_action_str, _ = render_subgoal(sg_str, wp_num)

    return subgoal, language_action_str, None


def retrieve_waypoints(
    hypothesized_recipe_graph: HypothesizedRecipeGraph,
    item: str,
    number: int = 1,
    cur_inventory: dict = dict()
) -> str:
    item = item.lower().replace(" ", "_")
    item = item.replace("logs", "log")

    _cur_inventory = copy.deepcopy(cur_inventory)
    if item in _cur_inventory:
        del _cur_inventory[item]

    pretty_result, ordered_text, ordered_item, ordered_item_quantity = \
         hypothesized_recipe_graph.compile(item.replace(" ", "_"), number, _cur_inventory)
    return pretty_result


def make_plan(
    original_final_goal: str,
    inventory: dict,
    action_memory: DecomposedMemory,
    hypothesized_recipe_graph: HypothesizedRecipeGraph,
    topK: int,
    cfg: DictConfig,

    logger: logging.Logger,

    # needed for VLM call using Optimus-1's code
    obs: Dict[str, Any],
    hydra_path: str,
    run_uuid: str,
):
    wp_list_str = retrieve_waypoints(hypothesized_recipe_graph, original_final_goal, 1, inventory)
    logger.info(f"In make_plan")
    logger.info(f"wp_list_str: {wp_list_str}")
    first_wp_str = wp_list_str.splitlines()[1] # 0th line is 'craft 1 <goal> summary:'

    wp = first_wp_str.split('.')[1].split(':')[0].strip()
    if 'log' in wp:
        wp = 'logs'

    wp_num = int(first_wp_str.split('.')[1].split('need')[1].strip())

    is_succeeded, sg_str = action_memory.is_succeeded_waypoint(wp)

    logger.info(f"In make_plan")
    logger.info(f"waypoint: {wp}, waypoint_num: {wp_num}")
    logger.info(f"is_succeeded: {str(is_succeeded)}")

    if is_succeeded:
        subgoal, language_action_str, _ = render_subgoal(sg_str, wp_num)
        return wp, subgoal, language_action_str, None

    else:
        logger.info(f"No success experience for waypoint: {wp}, so, call planner to generate a subgoal.")

        similar_wp_sg_dict = action_memory.retrieve_similar_succeeded_waypoints(wp, topK)
        failed_sg_list = action_memory.retrieve_failed_subgoals(wp) # could be empty list, i.e., []

        subgoal, language_action_str, error_message = call_planner_with_retry(
            cfg, obs, wp, wp_num, similar_wp_sg_dict, failed_sg_list, hydra_path, run_uuid, logger
        )

        return wp, subgoal, language_action_str, error_message


def check_waypoint_item_obtained(new_item_dict, waypoint, logger):
    if len(new_item_dict) == 0:
        logger.error("env.inventory_new_item is True, but env.inventory_new_item_what() is empty.")
        return False

    for new_item_name in new_item_dict.keys():
        if "log" in waypoint and "log" in new_item_name:
            return True
        elif "planks" in waypoint and "planks" in new_item_name:
            return True
        elif "coal" in waypoint and "coal" in new_item_name:
            return True
        elif waypoint == new_item_name:
            return True

    return False


# our intrinsic goal selection strategy
def feasibility_min_count_frontier(hypothesized_recipe_graph: HypothesizedRecipeGraph, logger: logging.Logger, cfg: DictConfig):
    logger.info(f"In feasibility_min_count_frontier()")
    hypothesized_recipe_graph.load_and_init_all_recipes()
    frontier_item_names = hypothesized_recipe_graph.find_frontiers()
    exploration_count_dict = hypothesized_recipe_graph.get_exploration_count_all_hypothesized()
    level_dict = hypothesized_recipe_graph.calculate_level_all_hypothesized()

    frontier_exploration_count_dict = {}
    frontier_level_dict = {}

    for item_name in frontier_item_names:
        frontier_exploration_count_dict[item_name] = exploration_count_dict[item_name]
        frontier_level_dict[item_name] = level_dict[item_name]

    sorted_item_names = sorted(
        frontier_item_names,
        key=lambda item: (
            frontier_exploration_count_dict[item], # fewer exploration count is better
            - (1 / frontier_level_dict[item]) # higher feasibility score is better
        )
    )

    selected_int_goal = hypothesized_recipe_graph.select_non_conflicting_goal(sorted_item_names)

    if frontier_exploration_count_dict[selected_int_goal] > 1:
        hypothesized_recipe_graph.update_hypothesis(selected_int_goal)
    return selected_int_goal


def select_int_goal(hypothesized_recipe_graph: HypothesizedRecipeGraph, logger: logging.Logger, cfg: DictConfig):
    prefix = cfg.get("prefix")

    int_goal = feasibility_min_count_frontier(hypothesized_recipe_graph, logger, cfg)

    return int_goal


# def log_wandb_intermediately(experienced_item_names, verified_int_goal_names, explored_int_goals, num_new_verified_items):
#     experienced_item_names = list(set(experienced_item_names))
#     verified_int_goal_names = list(set(verified_int_goal_names))
#     explored_int_goals_names = list(set(explored_int_goals))

#     num_experienced_items = len(experienced_item_names)
#     num_verified_int_goals = len(verified_int_goal_names)
#     num_explored_int_goals = len(explored_int_goals_names)

#     wandb.log({
#         "num_experienced_items": num_experienced_items,
#         "num_verified_int_goals": num_verified_int_goals,
#         "num_explored_int_goals": num_explored_int_goals,
#         "num_new_verified_items": num_new_verified_items,
#     })


def exploration_do(
    cfg: DictConfig,
    env: CustomEnvWrapper,
    logger: logging.Logger,
    monitors: Monitors,
    reset_obs: Dict[str, Any],
    run_uuid: str
):
    prefix = cfg.get("prefix")
    logger.info(f"[yellow]In exploration_do(), prefix: {prefix}[/yellow]")

    oracle_knowledge_graph = OracleGraph()
    helper = NewHelper(env, oracle_knowledge_graph, prefix)
    obs = reset_obs

    # image_to_log = wandb.Image(obs["pov"], caption=f"Observation at step 0")
    # wandb.log({
    #     f"obs/0": image_to_log,
    # })
    env_status = env.get_status()
    loc = env_status["location_stats"]
    initial_xpos, initial_ypos, initial_zpos = loc["xpos"].item(), loc["ypos"].item(), loc["zpos"].item()
    # wandb.config.update({
    #     "initial_xpos": initial_xpos,
    #     "initial_ypos": initial_ypos,
    #     "initial_zpos": initial_zpos,
    # }, allow_val_change=True)

    hydra_path = hydra.core.hydra_config.HydraConfig.get().runtime.output_dir

    # MineRL is unstable, so check env_malmo_logger for 'Exception' periodically
    env_malmo_logger_port = env.instances[0]._target_port - 9000
    env_malmo_logger_path = os.path.join(hydra_path.split('logs')[0], 'logs', f'mc_{env_malmo_logger_port}.log')

    if not os.path.exists(env_malmo_logger_path):
        logger.error(f"env_malmo_logger_path: {env_malmo_logger_path} does not exist.")
        return "env_malmo_logger_error", None, None, None, None
    else:
        with open(env_malmo_logger_path, 'r', encoding='utf-8') as file:
            content = file.read()
        if 'Exception' in content:
            return "env_malmo_logger_error", None, None, None, None

        logger.info(f"normal! env_malmo_logger_path: {env_malmo_logger_path} exists.")

    status = ""

    int_goal = None
    waypoint = ""
    language_action_str = "" # mine/craft/smelt {waypoint}
    subgoal = None # {"task": language_action_str, "goal": [waypoint, 1]}
    subgoal_done = False

    explored_int_goals = []

    int_goal_steps = 0
    waypoint_steps = 0

    topK = cfg["memory"]["topK"]
    plan_failure_threshold = cfg["memory"]["plan_failure_threshold"]

    hypothesized_recipe_graph = HypothesizedRecipeGraph(cfg, logger)
    action_memory = DecomposedMemory(cfg, logger)

    _verified_items = list(set(copy.deepcopy(hypothesized_recipe_graph.verified_item_names)))
    _hypothesized_items = list(set(copy.deepcopy(hypothesized_recipe_graph.hypothesized_item_names)))
    _frontier_items = list(set(copy.deepcopy(hypothesized_recipe_graph.frontier_item_names)))
    _inadmissible_items = list(set(copy.deepcopy(hypothesized_recipe_graph.inadmissible_item_names)))

    logger.info(f"Verified items: {_verified_items}")
    logger.info(f"Hypothesized items: {_hypothesized_items}")
    logger.info(f"Frontier items: {_frontier_items}")
    logger.info(f"Inadmissible items: {_inadmissible_items}")

    num_initial_verified_items = len(_verified_items)
    num_initial_hypothesized_items = len(_hypothesized_items)
    num_initial_frontier_items = len(_frontier_items)
    num_initial_inadmissible_items = len(_inadmissible_items)

    experienced_item_names = []
    verified_int_goal_names = []
    num_new_verified_items = 0

    failed_waypoints = []

    num_reasoning_intervention = 0
    step_waypoint_obtained = 0

    with Progress(
        *Progress.get_default_columns(),
        TimeElapsedColumn(),
        "{task.completed} of {task.total}",
        expand=True,
    ) as pbar:
        num_step = pbar.add_task("[cyan]Running...", total=env.timeout)

        progress = 0
        game_over = False

        while not game_over:
            if int_goal is None:
                int_goal = select_int_goal(hypothesized_recipe_graph, logger, cfg)
                explored_int_goals.append(int_goal)
                explored_int_goals = list(set(explored_int_goals))
                # wandb.config.update(
                #     {"explored_int_goals": explored_int_goals},
                #     allow_val_change=True
                # )
                subgoal = None
                logger.info(f"New intrinsic goal: {int_goal}")
                logger.info(f"Current all crafting resources {hypothesized_recipe_graph.crafting_resources}")
                logger.info(f"Recipe from graph of intrinsic goal {int_goal}: {str(hypothesized_recipe_graph.get_recipe(int_goal))}\n")
                # if int_goal in hypothesized_recipe_graph.exploration_count_dict.keys():
                #     logger.info(f"Exploration count of intrinsic goal {int_goal}: {hypothesized_recipe_graph.exploration_count_dict[int_goal]}\n")
                # else:
                #     logger.info(f"intrinsic goal {int_goal} is already verified item.\n")

            if subgoal is None:
                env_status = env.get_status()
                inventory = env_status["inventory"]
                waypoint, subgoal, language_action_str, error_message = make_plan(
                    int_goal,
                    inventory,
                    action_memory,
                    hypothesized_recipe_graph,
                    topK,
                    cfg,
                    logger,
                    obs,
                    hydra_path,
                    run_uuid
                )
                if error_message is not None:
                    logger.error(f"Error message: {error_message}")
                    status = "cannot generate plan"
                    action_memory.save_success_failure(waypoint, "", is_success=False)
                    break

                subgoal_done = False
                logger.info(f"After make_plan()")
                logger.info(f"[yellow]Waypoint: {waypoint}, Subgoal: {subgoal}[/yellow]")


            current_sg = subgoal
            current_sg_prompt, current_sg_target = copy.deepcopy(current_sg["task"]), copy.deepcopy(current_sg["goal"])
            if current_sg_target[0] == "log":
                current_sg_target[0] = "logs"

            temp_sg_prompt = current_sg_prompt
            if "punch" in current_sg_prompt:
                current_sg_prompt = current_sg_prompt.replace("punch", "chop")
            op = current_sg_prompt.split(" ")[0]

            if "create" in current_sg_prompt:
                op = "craft"

            logger.info(f"[yellow]Subgoal Prompt: {current_sg_prompt}, Subgoal Target: {current_sg_target}[/yellow]")

            if op in ["craft", "smelt"] or "smelt" in current_sg_prompt:
                if not env.can_change_hotbar:
                    env.can_change_hotbar = True
                if not env.can_open_inventory:
                    env.can_open_inventory = True
                helper.reset(current_sg_prompt, pbar, num_step, logger)
                helper_done, info = helper.step(current_sg_prompt, current_sg_target)
                steps = helper.get_task_steps(current_sg_prompt)

                env.can_open_inventory = False
                env.can_change_hotbar = False

                monitors.update(f"{current_sg_prompt}_{progress}", helper_done, steps)
                if helper_done:
                    logger.info(f"[green]{current_sg_prompt} Success[/green]!")
                    progress += 1
                    subgoal_done = True

                    if "pickaxe" in waypoint:
                        env.can_change_hotbar = True
                        env.can_open_inventory = True
                        tmp_prompt = f"equip {waypoint}"
                        tmp_sg_target = [waypoint, 1]
                        helper.reset(tmp_prompt, pbar, num_step, logger)
                        _, _ = helper.step(tmp_prompt, tmp_sg_target)
                        env.can_open_inventory = False
                        env.can_change_hotbar = False

                    if "ingredients" in info:
                        # NOTE: Currently, environment emits succeeded recipes when crafting/smelting trials are succeeded.
                        # I choose this way because it is easy to implement.
                        # It might be seen as cheating, but this is same with learning recipes just using state transition data.
                        # We can track which items are used and consumed, and which items are crafted/smelted.
                        recipe_data_crafting_smelting = {
                            "item_name": info["item_name"],
                            "output_qty": info["output_qty"],
                            "ingredients": info["ingredients"],
                            "required_pickaxe": info["required_pickaxe"],
                            "is_crafting": info["is_crafting"],
                        }
                        if info["item_name"] not in experienced_item_names:
                            flag_new = hypothesized_recipe_graph.save_verified_recipe_data(recipe_data_crafting_smelting)
                            experienced_item_names.append(info["item_name"])
                            logger.info(f"New experienced item!")
                            logger.info(f"recipe_data_crafting_smelting: {recipe_data_crafting_smelting}")

                            if flag_new:
                                num_new_verified_items += 1
                                # log_wandb_intermediately(experienced_item_names, verified_int_goal_names,
                                #                             explored_int_goals, num_new_verified_items)

                else:
                    assert (
                        info is not None
                    ), "info should not be None! Because equip/craft/smelt failed!"
                    env.can_open_inventory = False
                    env.can_change_hotbar = False

                    if not os.path.exists(env_malmo_logger_path):
                        logger.error(f"env_malmo_logger_path: {env_malmo_logger_path} does not exist.")
                        return "env_malmo_logger_error", None, None, None, None

                    with open(env_malmo_logger_path, 'r', encoding='utf-8') as file:
                        content = file.read()
                    if 'Exception' in content:
                        return "env_malmo_logger_error", None, None, None, None

                    fail_env_step = (env.num_steps >= int(cfg["env"]["max_minutes"])*MINUTE)
                    fail_monitor_step = (monitors.all_steps()  >= int(cfg["env"]["max_minutes"])*MINUTE)

                    if fail_env_step or fail_monitor_step:
                        game_over = True
                        status = "timeout_programmatic"
                        failed_waypoints.append(waypoint)
                        break

                    logger.warning(f"[red]:warning: {current_sg_prompt} failed... Beacuse {info}[/red]")
                    logger.warning("[red]programmatic operation is failed, so replan is needed[/red]")

                    if 'error_msg' not in info:
                        logger.warning(f'fail for unkown reason. info: {info}')
                        continue
                    if not ("cannot find a recipe" in info['error_msg'] or "missing material" in info['error_msg']):
                        logger.warning(f'fail for unkown reason. info: {info}')
                        continue

                    failed_waypoints.append(waypoint)
                    action_memory.save_success_failure(waypoint, language_action_str, is_success=False)
                    subgoal = None

                    wp_total_failure_counts = action_memory.retrieve_total_failed_counts(waypoint)
                    if wp_total_failure_counts <= -plan_failure_threshold * 3:
                        logger.warning(f"{waypoint} failed {abs(wp_total_failure_counts)} times, so increment exploration count of {waypoint}.")
                        hypothesized_recipe_graph.increment_count(waypoint, prefix)

                        # reset success failure history of the changed items from the action_memory
                        recipe_revised_items = hypothesized_recipe_graph.get_recipe_revised_items()
                        for item in recipe_revised_items:
                            action_memory.reset_success_failure_history(item)
                        hypothesized_recipe_graph.reset_recipe_revised_items()

                        int_goal = None
                        hypothesized_recipe_graph.free_exploring_goal()

                    # NOTE: if a same waypoint is failed multiple times, then end this episode
                    # MineRL environment is not stable, so sometimes it fails to craft item even if it has enough materials
                    # if failed_waypoints.count(waypoint) >= 10:
                    #     status = "failed"
                    #     break
                    if failed_waypoints.count(waypoint) >= 10:
                        status = "failed"
                        break

            else:
                # op is not in ["craft", "smelt", "equip"]
                step_waypoint_obtained = env.num_steps
                current_sg_prompt = copy.deepcopy(temp_sg_prompt)

                while True:
                    env._only_once = True

                    action = ServerAPI.get_action(
                        cfg["server"], obs, current_sg_prompt, step=env.num_steps,
                        hydra_path=hydra_path, run_uuid=run_uuid
                    )
                    obs, reward, game_over, info = env.step(action, current_sg_target)
                    pbar.update(num_step, advance=1)
                    monitors.update(f"{temp_sg_prompt}_{progress}", env.current_subgoal_finish)

                    if env.inventory_new_item():
                        new_item_dict = env.inventory_new_item_what()
                        env_status = env.get_status()
                        curr_inventory = env_status["inventory"]
                        recipe_data_mining = get_recipe_data_mining(new_item_dict, curr_inventory, logger)
                        for recipe_data in recipe_data_mining:
                            if recipe_data["item_name"] not in experienced_item_names:
                                flag_new = hypothesized_recipe_graph.save_verified_recipe_data(recipe_data)
                                experienced_item_names.append(recipe_data["item_name"])
                                logger.info(f"New experienced item!")
                                logger.info(f"recipe_data: {recipe_data}")

                                if flag_new:
                                    num_new_verified_items += 1
                                    # log_wandb_intermediately(experienced_item_names, verified_int_goal_names,
                                    #                          explored_int_goals, num_new_verified_items)

                    if env.num_steps % (MINUTE * 3) == 0:
                        if not os.path.exists(env_malmo_logger_path):
                            logger.error(f"env_malmo_logger_path: {env_malmo_logger_path} does not exist.")
                            return "env_malmo_logger_error", None, None, None, None

                        with open(env_malmo_logger_path, 'r', encoding='utf-8') as file:
                            content = file.read()
                        if 'Exception' in content:
                            return "env_malmo_logger_error", None, None, None, None

                    if game_over:
                        logger.warning("[red]:warning: Timeout![/red]")
                        status = "timeout_non_programmatic"
                        failed_waypoints.append(waypoint)
                        break

                    if env.current_subgoal_finish:
                        # sg is achieved
                        logger.info(f"[green]{current_sg_prompt} Success :smile: [/green]!")
                        progress += 1
                        steps = monitors.get_steps(current_sg_prompt)
                        subgoal_done = True
                        break

            # current_sg is done
            if subgoal_done:
                env_status = env.get_status()
                inventory = env_status["inventory"]

                waypoint_success = env.check_waypoint_finish([waypoint, 1])

                action_memory.save_success_failure(waypoint, language_action_str, is_success=waypoint_success)
                if waypoint_success:
                    logger.info(f"[green]Achieved waypoint {waypoint}[/green]")

                    if waypoint == int_goal:
                        logger.info(f"Intrinsic goal {int_goal} reached!")
                        hypothesized_recipe_graph.free_exploring_goal()
                        verified_int_goal_names.append(waypoint)
                        int_goal = None
                else:
                    logger.info(f"[red]Subgoal is done, but failed to achieve waypoint {waypoint}[/red]")
                    failed_waypoints.append(waypoint)
                subgoal = None

        logger.info(f"After an episode, experienced_item_names: {experienced_item_names}")

        if not os.path.exists(env_malmo_logger_path):
            logger.error(f"env_malmo_logger_path: {env_malmo_logger_path} does not exist.")
            return "env_malmo_logger_error", None, None, None, None
        with open(env_malmo_logger_path, 'r', encoding='utf-8') as file:
            content = file.read()
        if 'Exception' in content:
            return "env_malmo_logger_error", None, None, None, None

        # end of while loop. game is done.
        if game_over:
            action_memory.save_success_failure(waypoint, language_action_str, is_success=False)
            wp_total_failure_counts = action_memory.retrieve_total_failed_counts(waypoint)

            if wp_total_failure_counts <= -plan_failure_threshold * 3:
                logger.warning(f"{waypoint} failed {abs(wp_total_failure_counts)} times, so increment exploration count of {waypoint}.")
                hypothesized_recipe_graph.increment_count(waypoint, prefix)
                action_memory.reset_success_failure_history(waypoint)

                # reset success failure history of the changed items from the action_memory
                recipe_revised_items = hypothesized_recipe_graph.get_recipe_revised_items()
                for item in recipe_revised_items:
                    action_memory.reset_success_failure_history(item)
                hypothesized_recipe_graph.reset_recipe_revised_items()

        if env.api_thread is not None and env.api_thread_is_alive():
            env.api_thread.join()

        # wandb.log({
        #     "env_num_steps": env.num_steps,
        #     "num_reasoning_intervention": num_reasoning_intervention,
        # })

    hypothesized_recipe_graph.free_exploring_goal()

    _verified_items = list(set(copy.deepcopy(hypothesized_recipe_graph.verified_item_names)))
    _hypothesized_items = list(set(copy.deepcopy(hypothesized_recipe_graph.hypothesized_item_names)))
    _frontier_items = list(set(copy.deepcopy(hypothesized_recipe_graph.frontier_item_names)))
    _inadmissible_items = list(set(copy.deepcopy(hypothesized_recipe_graph.inadmissible_item_names)))
    experienced_item_names = list(set(experienced_item_names))
    verified_int_goal_names = list(set(verified_int_goal_names))

    logger.info(f"After exploration\n")
    logger.info(f"experienced_item_names: {experienced_item_names}")
    logger.info(f"verified_int_goal_names: {verified_int_goal_names}\n")

    logger.info(f"After exploration\n")
    logger.info(f"Verified items: {_verified_items}")
    logger.info(f"Hypothesized items: {_hypothesized_items}")
    logger.info(f"Frontier items: {_frontier_items}")
    logger.info(f"Inadmissible items: {_inadmissible_items}")

    num_final_verified_items = len(_verified_items)
    num_final_hypothesized_items = len(_hypothesized_items)
    num_final_frontier_items = len(_frontier_items)
    num_final_inadmissible_items = len(_inadmissible_items)
    num_experienced_items = len(experienced_item_names)
    num_verified_int_goals = len(verified_int_goal_names)

    # wandb.log({
    #     "num_initial_verified": num_initial_verified_items,
    #     "num_initial_hypothesized": num_initial_hypothesized_items,
    #     "num_initial_frontier": num_initial_frontier_items,
    #     "num_initial_inadmissible": num_initial_inadmissible_items,
    #     "num_final_verified": num_final_verified_items,
    #     "num_final_hypothesized": num_final_hypothesized_items,
    #     "num_final_frontier": num_final_frontier_items,
    #     "num_final_inadmissible": num_final_inadmissible_items,
    #     "num_experienced_items": num_experienced_items,
    #     "num_verified_int_goals": num_verified_int_goals,
    #     "num_explored_int_goals": len(explored_int_goals),
    # })

    hypothesized_recipe_graph.increment_num_episodes_save_memory()

    return status, monitors.all_steps(), [], [], []


@hydra.main(version_base=None, config_path="conf", config_name="evaluate")
def main(cfg: DictConfig):
    register_custom_env(cfg)

    logger = get_logger(__name__)

    seed = int(cfg["seed"])
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    transformers.set_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    prefix = cfg.get("prefix")
    # wandb.init(project=prefix, entity="", config=OmegaConf.to_container(cfg, resolve=True), save_code=True)

    logger.info(f"main_exploration.py is executed.")

    is_fixed_memory = cfg["memory"]["is_fixed"]
    logger.info(f"is_fixed_memory: {is_fixed_memory}")
    if not is_fixed_memory: # if growing memory
        logger.info(f"Growing memory")
        cfg["memory"]["waypoint_to_sg"]["save_path"] = cfg["memory"]["waypoint_to_sg"]["path"]
    else:
        logger.info("Fixed memory. Only a few experiences are used, and the memory doesn't grow.")

    prefix = cfg.get("prefix")
    logger.info(f"prefix: {prefix}\n")

    env = env_make(cfg["env"]["name"], cfg, logger)

    if cfg["task"]["interactive"] and cfg["type"] != "headless":
        raise NotImplementedError("Not implemented yet!")

    logger.info(OmegaConf.to_yaml(cfg))

    times = cfg["env"]["times"]

    monitors = []
    for run_t in range(times):
        try:
            ServerAPI._reset(cfg["server"])
            logger.info("[red]env & server reset...[/red] ")
            obs = env.reset()

        except Exception as e:
            logger.error(f"Error during reset: {e}")
            # wandb.finish(exit_code=1)
            sys.exit(1)

        logger.info("Done of reset of env and server")

        hydra_path = hydra.core.hydra_config.HydraConfig.get().runtime.output_dir
        run_uuid = shortuuid.uuid()
        logger.info(f"trial: {run_t}, hydra_path: {hydra_path}, run_uuid: {run_uuid}\n\n")

        # wandb.config.update(
        #     {"hydra_path": hydra_path, "run_uuid": run_uuid},
        #     allow_val_change=True
        # )

        current_monitos = Monitors([SuccessMonitor(), StepMonitor()])

        # wandb.config.update({
        #     "is_fixed_memory": bool(is_fixed_memory),
        #     "biome": cfg["env"]["prefer_biome"],
        #     "prefix": prefix,
        # }, allow_val_change=True)

        status, steps, completed_subgoals, failed_subgoals, failed_waypoints = exploration_do(
            cfg, env, logger, current_monitos, obs, run_uuid
        )

        if status == "env_malmo_logger_error":
            logger.error("env_malmo_logger_error")
            # wandb.finish(exit_code=1)
            sys.exit(1)

        biome = cfg["env"]["prefer_biome"]

        monitors.append(current_monitos)

        logger.info(f"Summary: {current_monitos.get_metric()}")

        logger.info(f"Done of trial: {run_t}, hydra_path: {hydra_path}, run_uuid: {run_uuid}")

        img_dir = os.path.join(hydra_path, run_uuid, "imgs")
        shutil.rmtree(img_dir)

        # wandb.finish()

    env.close()
    all_steps = 0
    for monitor in monitors:
        logger.info(monitor.get_metric())
        all_steps += monitor.all_steps()
    logger.info(f" All Steps: {all_steps}")

    exit(0)


if __name__ == "__main__":
    main()
