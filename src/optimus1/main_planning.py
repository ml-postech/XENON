import copy
import json
import logging
import os
import re
import shutil
import time
from typing import Any, Dict
import sys

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
from optimus1.memories import DecomposedMemory
from optimus1.memories import KnowledgeGraph as OracleGraph

from optimus1.monitor import Monitors, StepMonitor, SuccessMonitor
from optimus1.util import (
    ServerAPI,
    base64_to_img,
    get_evaluate_task,
    get_evaluate_task_and_goal,
    get_logger,
    pretty_result,
    render_subgoal,
    render_context_aware_reasoning
)


MINUTE = 1200
visual_info = ""

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
    waypoint_generator: OracleGraph,
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
        waypoint_generator.compile(item.replace(" ", "_"), number, _cur_inventory)
    return pretty_result


def make_plan(
    original_final_goal: str,
    inventory: dict,
    action_memory: DecomposedMemory,
    waypoint_generator: OracleGraph,
    topK: int,
    cfg: DictConfig,

    logger: logging.Logger,

    # needed for VLM call using Optimus-1's code
    obs: Dict[str, Any],
    hydra_path: str,
    run_uuid: str,
):
    wp_list_str = retrieve_waypoints(waypoint_generator, original_final_goal, 1, inventory)
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
        logger.info(f"No success experience for waypoint: {wp}, so, call planner to generate a plan.")

        similar_wp_sg_dict = action_memory.retrieve_similar_succeeded_waypoints(wp, topK)
        failed_sg_list = action_memory.retrieve_failed_subgoals(wp) # could be empty list, i.e., []

        subgoal, language_action_str, error_message = call_planner_with_retry(
            cfg, obs, wp, wp_num, similar_wp_sg_dict, failed_sg_list, hydra_path, run_uuid, logger
        )

        return wp, subgoal, language_action_str, error_message


# cfg, obs, current_sg_prompt, waypoint, hydra_path, run_uuid, logger
def call_reasoning_with_retry(
    cfg: DictConfig,
    obs: Dict[str, Any],
    current_sg_prompt: str,
    waypoint: str,
    hydra_path: str,
    run_uuid: str,
    logger: logging.Logger,
):
    attempts = 0
    max_retries = 3
    reasoning, visual_description = "", ""
    while attempts < max_retries:
        attempts += 1

        logger.info(f"Attempt: {attempts}, Just before get_context_aware_reasoning: ")
        logger.info(f"current_sg_prompt: {current_sg_prompt}")
        logger.info(f"waypoint: {waypoint}")
        logger.info(f"Starting get_context_aware_reasoning ...\n")

        try:
            reasoning, visual_description = ServerAPI.get_context_aware_reasoning(
                cfg["server"],
                obs,
                current_sg_prompt,
                waypoint,
                hydra_path=hydra_path,
                run_uuid=run_uuid
            )
            tmp_dict, render_error = render_context_aware_reasoning(copy.deepcopy(reasoning))
            if render_error is None:
                break

            logger.warning(f"get_context_aware_reasoning at attempt {attempts} failed. Error message: {render_error}")
            if attempts >= max_retries:
                logger.error("Max retries reached. Could not fetch get_context_aware_reasoning.")
                return dict(), "", "max_tries_get_context_aware_reasoning"
        
        except Exception as e:
            logger.info(f"Error in get_context_aware_reasoning: {e}")
            if attempts >= max_retries:
                logger.error("Max retries reached. Could not fetch get_context_aware_reasoning.")
                return dict(), "", "max_tries_get_context_aware_reasoning"
            continue
    
    reasoning_dict, render_error = render_context_aware_reasoning(reasoning)
    return reasoning_dict, visual_description, render_error


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


def new_agent_do(
    cfg: DictConfig,
    env: CustomEnvWrapper,
    logger: logging.Logger,
    monitors: Monitors,
    reset_obs: Dict[str, Any],
    action_memory: DecomposedMemory,
    original_task: str,
    original_final_goal: str,
    run_uuid: str
):
    prefix = cfg.get("prefix")
    logger.info(f"[yellow]In agent_do(), prefix: {prefix}[/yellow]")

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

    logger.info(f"[yellow]original_final_goal: {original_final_goal}[/yellow]")

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
    original_final_goal_success = False

    waypoint = ""
    subgoal = None
    language_action_str = ""
    subgoal_done = False
    topK = cfg["memory"]["topK"]
    waypoint_generator = OracleGraph() # OracleGraph knows all recipes accurately.

    completed_subgoals = []
    completed_waypoints = []
    failed_subgoals = []
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
            if subgoal is None:
                # check if original_final_goal is achieved
                original_final_goal_success = env.check_original_goal_finish([original_final_goal, 1])
                if original_final_goal_success:
                    logger.info(f"[green]Original Goal: {original_final_goal} is achieved![/green]")
                    status = "success"
                    break

                env_status = env.get_status()
                inventory = env_status["inventory"]
                waypoint, subgoal, language_action_str, error_message = make_plan(
                    original_final_goal,
                    inventory,
                    action_memory,
                    waypoint_generator,
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
                    failed_subgoals = [f"achieve {waypoint}"]
                    break

                subgoal_done = False
                logger.info(f"After make_plan()")
                logger.info(f"[yellow]Waypoint: {waypoint}, Subgoal: {subgoal}[/yellow]")

            current_sg = subgoal
            current_sg_prompt, current_sg_target = copy.deepcopy(current_sg["task"]), copy.deepcopy(current_sg["goal"])
            if current_sg_target[0] == "log":
                current_sg_target[0] = "logs"

            temp_sg_prompt = copy.deepcopy(current_sg_prompt)
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
                sg_done, info = helper.step(current_sg_prompt, current_sg_target)
                steps = helper.get_task_steps(current_sg_prompt)

                env.can_open_inventory = False
                env.can_change_hotbar = False

                monitors.update(f"{current_sg_prompt}_{progress}", sg_done, steps)
                if sg_done:
                    logger.info(f"[green]{current_sg_prompt} Success[/green]!")
                    progress += 1
                    completed_subgoals.append(current_sg)
                    subgoal_done = True

                    if "pickaxe" in waypoint:
                        env.can_change_hotbar = True
                        env.can_open_inventory = True
                        tmp_prompt = f"equip {waypoint}"
                        tmp_sg_target = [waypoint, 1]
                        helper.reset(tmp_prompt, pbar, num_step, logger)
                        sg_done, info = helper.step(tmp_prompt, tmp_sg_target)
                        env.can_open_inventory = False
                        env.can_change_hotbar = False
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
                        failed_subgoals = [current_sg]
                        failed_waypoints.append(waypoint)
                        break

                    if 'error_msg' not in info:
                        logger.warning(f'fail for unkown reason. info: {info}')
                        continue
                    if not ("cannot find a recipe" in info['error_msg'] or "missing material" in info['error_msg']):
                        logger.warning(f'fail for unkown reason. info: {info}')
                        continue

                    failed_waypoints.append(waypoint)
                    action_memory.save_success_failure(waypoint, language_action_str, is_success=False)
                    subgoal = None

                    # NOTE: if a same waypoint is failed multiple times, then end this episode
                    # MineRL environment is not stable, so sometimes it fails to craft item even if it has enough materials
                    if failed_waypoints.count(waypoint) >= 3:
                        status = "failed"
                        failed_subgoals = [current_sg]
                        break

                    continue
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


                    # if current waypoint item is not obtained over a MINUTE, then do get_context_aware_reasoning.
                    if env.inventory_change():
                        new_item_dict = env.inventory_change_what()
                        is_waypoint_obtained = check_waypoint_item_obtained(new_item_dict, waypoint, logger)
                        if is_waypoint_obtained:
                            step_waypoint_obtained = env.num_steps
                            current_sg_prompt = copy.deepcopy(temp_sg_prompt)
                    if env.num_steps - step_waypoint_obtained >= MINUTE:
                        current_sg_prompt = copy.deepcopy(temp_sg_prompt)
                        logger.info(f"Current timestep: {env.num_steps}. Calling get_context_aware_reasoning ...")
                        reasoning_dict, visual_description, render_error = call_reasoning_with_retry(
                            cfg, obs, temp_sg_prompt, waypoint, hydra_path, run_uuid, logger
                        )
                        if render_error is not None:
                            logger.error(f"Error message: {render_error}")
                            status = "cannot generate reasoning"
                            failed_subgoals = [f"achieve {waypoint}"]
                            break

                        logger.info(f"visual_description: {visual_description}")
                        logger.info(f"reasoning_dict: {str(reasoning_dict)}")
                        step_waypoint_obtained = env.num_steps

                        if reasoning_dict["need_intervention"]:
                            current_sg_prompt = reasoning_dict["task"]
                            logger.info(f"New prompt for STEVE-1: {current_sg_prompt}. timestep: {env.num_steps}\n\n")
                            num_reasoning_intervention += 1
                            # image_to_log = wandb.Image(obs["pov"], caption=f"Observation at step {env.num_steps}")
                            # wandb.log({
                            #     f"obs/{env.num_steps}": image_to_log,
                            #     "env_num_steps": env.num_steps,
                            #     "num_reasoning_intervention": num_reasoning_intervention,
                            # })

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
                        failed_subgoals = [current_sg]
                        failed_waypoints.append(waypoint)
                        break

                    if env.current_subgoal_finish:
                        # sg is achieved
                        logger.info(f"[green]{temp_sg_prompt} Success :smile: [/green]!")
                        progress += 1
                        steps = monitors.get_steps(temp_sg_prompt)
                        completed_subgoals.append(current_sg)
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
                    completed_waypoints.append(waypoint)
                else:
                    logger.info(f"[red]Subgoal is done, but failed to achieve waypoint {waypoint}[/red]")
                    failed_waypoints.append(waypoint)
                subgoal = None


        if not os.path.exists(env_malmo_logger_path):
            logger.error(f"env_malmo_logger_path: {env_malmo_logger_path} does not exist.")
            return "env_malmo_logger_error", None, None, None, None

        with open(env_malmo_logger_path, 'r', encoding='utf-8') as file:
            content = file.read()
        if 'Exception' in content:
            return "env_malmo_logger_error", None, None, None, None

        # end of while loop. game is done.
        if not original_final_goal_success:
            action_memory.save_success_failure(waypoint, language_action_str, is_success=False)

        if env.api_thread is not None and env.api_thread_is_alive():
            env.api_thread.join()

        # wandb.log({
        #     "env_num_steps": env.num_steps,
        #     "num_reasoning_intervention": num_reasoning_intervention,
        # })

    return status, monitors.all_steps(), completed_subgoals, failed_subgoals, failed_waypoints


@hydra.main(version_base=None, config_path="conf", config_name="evaluate")
def main(cfg: DictConfig):
    register_custom_env(cfg)

    logger = get_logger(__name__)

    benchmark = ""
    if "wooden" in cfg["env"]["name"].lower():
        benchmark = "wooden"
    elif "redstone" in cfg["env"]["name"].lower():
        benchmark = "redstone"
    elif "armor" in cfg["env"]["name"].lower():
        benchmark = "armor"
    elif "stone" in cfg["env"]["name"].lower():
        benchmark = "stone"
    elif "iron" in cfg["env"]["name"].lower():
        benchmark = "iron"
    elif "golden" in cfg["env"]["name"].lower():
        benchmark = "golden"
    elif "diamond" in cfg["env"]["name"].lower():
        benchmark = "diamond"
    cfg["benchmark"] = benchmark

    seed = int(cfg["seed"])
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    transformers.set_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    # wandb.init(project="", entity="", config=OmegaConf.to_container(cfg, resolve=True), save_code=True)

    logger.info(f"main_ours_planning.py is executed.")

    logger.info(f"benchmark: {benchmark}")
    logger.info(f"cfg['benchmark']: {cfg['benchmark']}")

    is_fixed_memory = cfg["memory"]["is_fixed"]
    logger.info(f"is_fixed_memory: {is_fixed_memory}")
    if not is_fixed_memory: # if growing memory
        logger.info(f"Growing memory")
        # Save path = retrieve path
        cfg["memory"]["waypoint_to_sg"]["save_path"] = cfg["memory"]["waypoint_to_sg"]["path"]
    else:
        logger.info("Fixed memory. Only a few experiences are used, and the memory doesn't grow.")

    prefix = cfg.get("prefix")
    logger.info(f"prefix: {prefix}\n")

    env = env_make(cfg["env"]["name"], cfg, logger)

    action_memory = DecomposedMemory(cfg, logger)

    if cfg["task"]["interactive"] and cfg["type"] != "headless":
        raise NotImplementedError("Not implemented yet!")

    running_tasks, running_goals = get_evaluate_task_and_goal(cfg)

    if len(running_tasks) == 0:
        logger.error("No tasks to evaluate.")
        # wandb.finish(exit_code=1)
        sys.exit(1)

    logger.info(f"Running Tasks: {running_tasks}")
    logger.info(OmegaConf.to_yaml(cfg))

    times = cfg["env"]["times"]
    for task, goal in zip(running_tasks, running_goals):
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
            logger.info(f"trial: {run_t}, goal: {goal}, hydra_path: {hydra_path}, run_uuid: {run_uuid}\n\n")

            visual_info = ""
            environment = cfg["env"]["prefer_biome"]

            logger.info("goal, environment. start")
            logger.info(f"goal: {goal}, environment: {environment}")
            logger.info("goal, environment. end")

            # wandb.config.update(
            #     {"task": task.replace(' ', '_').lower(), "goal": goal.replace(' ', '_').lower(),
            #      "hydra_path": hydra_path, "run_uuid": run_uuid, "benchmark": benchmark},
            #     allow_val_change=True
            # )

            action_memory.current_environment = environment

            current_monitos = Monitors([SuccessMonitor(), StepMonitor()])

            # wandb.config.update({
            #     "is_fixed_memory": bool(is_fixed_memory),
            #     "biome": cfg["env"]["prefer_biome"],
            #     "prefix": prefix,
            # }, allow_val_change=True)

            status, steps, completed_subgoals, failed_subgoals, failed_waypoints = new_agent_do(
                cfg, env, logger, current_monitos, obs, action_memory, task, goal, run_uuid
            )

            if status == "env_malmo_logger_error":
                logger.error("env_malmo_logger_error")
                # wandb.finish(exit_code=1)
                sys.exit(1)

            failed_waypoints = list(set(failed_waypoints))
            failed_waypoints.sort()

            done_final_task = completed_subgoals[-1]["task"] if len(completed_subgoals) > 0 else None
            biome = cfg["env"]["prefer_biome"]

            video_file = env.save_video(task, status, is_sub_task=False,
                                        actual_done_final_task=done_final_task, biome=biome, run_uuid=run_uuid)

            status_detailed = copy.deepcopy(status)
            status = "failed" if status != "success" else status

            video_path = ""
            if video_file is not None:
                video_file.join()
                video_path = video_file.get_result()
                if not video_path:
                    video_path = ""

            current_planning = completed_subgoals + failed_subgoals
            t = action_memory.save_plan(
                task,
                visual_info,
                goal,
                status,
                current_planning,
                steps,
                run_uuid,
                video_path,
                environment=environment,
            )

            monitors.append(current_monitos)

            logger.info(f"completed_subgoals: {str(completed_subgoals)}\n")
            logger.info(f"failed_subgoals: {str(failed_subgoals)}\n")
            logger.info(f"Summary: {current_monitos.get_metric()}")

            result_file_name = f"{prefix}_{task.replace(' ', '_').lower()}_{cfg['exp_num']:003}_{status}_{biome}_{run_uuid[:4]}.json"
            result_data = {
                "run_uuid": run_uuid,
                "seed": seed,
                "prefix": prefix,
                "benchmark": benchmark,
                "task": task.replace(' ', '_').lower(),
                "goal": goal.replace(' ', '_').lower(),
                "exp_num": cfg["exp_num"],
                "biome": biome,
                "is_fixed_memory": bool(is_fixed_memory),
                "max_minutes": cfg["env"]["max_minutes"],
                "success": bool(status=="success"),
                "status_detailed": status_detailed,
                "video_file": video_path,
                "steps": steps,
                "minutes": round(steps / MINUTE, 2),
                "metrics": current_monitos.get_metric(),
                "completed_subgoals": completed_subgoals,
                "completed_plans": completed_subgoals, # backward compatibility
                "failed_subgoals": failed_subgoals,
                "remain_plans": failed_subgoals, # backward compatibility
                "all_subgoals": current_planning,
                "all_plans": current_planning, # backward compatibility
                "failed_waypoints": failed_waypoints,
            }

            result_file_path = os.path.join(hydra_path, result_file_name)
            with open(result_file_path, 'w') as f:
                json.dump(result_data, f, indent=2)

            # with open(f"{wandb.run.dir}/result.json", "w") as f:
            #     json.dump(result_data, f, indent=2)
            #     wandb.save(f"result.json")

            # wandb.log({
            #     "success": int(bool(status=="success")),
            #     "total_steps": steps,
            #     "total_failed_waypoints": len(failed_waypoints),
            #     "total_minutes": round(steps / MINUTE, 2),
            # })

            os.makedirs(cfg["results"]["path"], exist_ok=True)
            result_file_path = os.path.join(cfg["results"]["path"], result_file_name)
            with open(result_file_path, 'w') as f:
                json.dump(result_data, f, indent=2)

            pretty_result(
                task, current_monitos.get_metric(), 1, steps=current_monitos.all_steps()
            )

            t.join()
            logger.info(f"Done of trial: {run_t}, task: {task}, hydra_path: {hydra_path}, run_uuid: {run_uuid}")

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
