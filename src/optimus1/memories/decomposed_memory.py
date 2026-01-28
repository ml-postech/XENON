import copy
import json
import logging
import os
import threading
from typing import Any, Dict, List

import shortuuid
from omegaconf import DictConfig
from thefuzz import process
import torch
import torch.nn.functional as F
from sentence_transformers import SentenceTransformer

from ..util.prompt import render_replan_example, language_action_to_subgoal
from ..util.thread import MultiThreadServerAPI

from .relative_graph import KnowledgeGraph

# from ..models.steve1.config import DEVICE, MINECLIP_CONFIG
# from ..models.steve1.utils.mineclip_agent_env_utils import load_mineclip_wconfig
# from ..models.steve1.mineclip import MineCLIP

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

import fcntl
import time

def read_json_data_with_shared_lock(file_path):
    with open(file_path, "r") as fp:
        fcntl.flock(fp, fcntl.LOCK_SH)
        data = json.load(fp)
        fcntl.flock(fp, fcntl.LOCK_UN)
    return data


class DecomposedMemory:
    current_environment: str = ""

    # crafting_graph: KnowledgeGraph # Just for backward compatibility

    _lock: threading.Lock = threading.Lock()

    def __init__(
        self,
        cfg: DictConfig,
        logger: logging.Logger | None = None,
        life_long_learning: bool = False,
    ) -> None:
        self.cfg = cfg
        self.version = self.cfg["version"]

        self.logger = logger

        self.plan_failure_threshold = int(cfg["memory"]["plan_failure_threshold"])
        # self.logger.info(f"Plan failure threshold: {self.plan_failure_threshold}")

        self.root_path = self.cfg["memory"]["path"] # src/optimus1/memories/v1

        os.makedirs(self.root_path, exist_ok=True)

        self.wp_to_sg_dir_path = os.path.join(
            self.root_path, self.cfg["memory"]["waypoint_to_sg"]["path"]
        )
        os.makedirs(self.wp_to_sg_dir_path, exist_ok=True)

        self.plan_dir_save_path = os.path.join(
            self.root_path, self.cfg["memory"]["decomposed_plan"]["save_path"]
        )
        os.makedirs(self.plan_dir_save_path, exist_ok=True)

        # self.mineclip = load_mineclip_wconfig()
        self.bert_encoder = SentenceTransformer('all-MiniLM-L6-v2', device=DEVICE)

        self._prepare_waypoint_embeddings()
        self._prepare_succeeded_waypoints()

        # self.crafting_graph = KnowledgeGraph( # Just for backward compatibility
        #     life_long_learning=life_long_learning,
        # )

    def _prepare_waypoint_embeddings(self):
        lst_dir = os.listdir(self.wp_to_sg_dir_path)
        with self._lock:
            for file_name in lst_dir:
                if not file_name.endswith(".json"):
                    continue

                waypoint = file_name.replace(".json", "")

                wp_embed = torch.tensor(self.bert_encoder.encode(waypoint)).unsqueeze(0).to(DEVICE)

                torch.save(wp_embed, os.path.join(self.wp_to_sg_dir_path, f"{waypoint}.pt"))
                # print(f'saved {os.path.join(self.wp_to_sg_dir_path, f"{waypoint}.pt")}')

    def _prepare_succeeded_waypoints(self):
        # get succeeded waypoint (item names) into self.succeeded_waypoints
        lst_dir = os.listdir(self.wp_to_sg_dir_path)

        self.succeeded_waypoints = []

        for file_name in lst_dir:
            if not file_name.endswith(".json"):
                continue

            waypoint = file_name.replace(".json", "")
            is_succeeded, _ = self.is_succeeded_waypoint(waypoint)
            if is_succeeded:
                self.succeeded_waypoints.append(waypoint)


    def save_success_failure(self, waypoint, action_str, is_success):
        self._save_success_failure(waypoint, action_str, is_success)
        # thread = MultiThreadServerAPI(
        #     self._save_success_failure,
        #     args=(
        #         waypoint,
        #         action_str,
        #         is_success,
        #     ),
        # )
        # thread.start()
        # return thread

    def _save_success_failure(self, waypoint, action_str, is_success):

        json_file_name = f"{waypoint}.json"
        lst_dir = os.listdir(self.wp_to_sg_dir_path)

        new_success = False

        # There is no {waypoint}.json file
        if json_file_name not in lst_dir:
            with self._lock:
                with open(os.path.join(self.wp_to_sg_dir_path, json_file_name), "w+") as fp:
                    fcntl.flock(fp, fcntl.LOCK_EX)

                    wp_file_data = dict()
                    wp_file_data['action'] = dict()
                    wp_file_data['action'][action_str] = {'success': 1, 'failure': 0} if is_success else {'success': 0, 'failure': 1}
                    new_success = is_success

                    fp.seek(0)
                    fp.truncate()
                    json.dump(wp_file_data, fp, indent=2)

                    fcntl.flock(fp, fcntl.LOCK_UN)

                wp_embed = torch.tensor(self.bert_encoder.encode(waypoint)).unsqueeze(0).to(DEVICE)

                torch.save(wp_embed, os.path.join(self.wp_to_sg_dir_path, f"{waypoint}.pt"))

            if new_success:
                self._prepare_waypoint_embeddings()
            self._prepare_succeeded_waypoints()

            return

        # There is {waypoint}.json file
        with self._lock:
            with open(os.path.join(self.wp_to_sg_dir_path, json_file_name), "r+") as fp:
                fcntl.flock(fp, fcntl.LOCK_EX)
                wp_file_data = json.load(fp)

                if action_str in wp_file_data['action'].keys():
                    if is_success:
                        wp_file_data['action'][action_str]['success'] += 1
                    else:
                        wp_file_data['action'][action_str]['failure'] += 1
                else:
                    wp_file_data['action'][action_str] = {'success': 1, 'failure': 0} if is_success else {'success': 0, 'failure': 1}

                fp.seek(0)
                fp.truncate()
                json.dump(wp_file_data, fp, indent=2)
                fcntl.flock(fp, fcntl.LOCK_UN)

            if wp_file_data['action'][action_str]['success'] == 1 and is_success:
                new_success = True

        if new_success:
            self._prepare_waypoint_embeddings()
        self._prepare_succeeded_waypoints()

        return


    def is_succeeded_waypoint(self, waypoint):
        json_file_name = f"{waypoint}.json"
        lst_dir = os.listdir(self.wp_to_sg_dir_path)

        if json_file_name not in lst_dir:
            return False, None
        
        if not os.path.exists(os.path.join(self.wp_to_sg_dir_path, json_file_name)):
            return False, None

        succeeded_action_lists = []

        wp_file_data = read_json_data_with_shared_lock(os.path.join(self.wp_to_sg_dir_path, json_file_name))

        for action, action_history in wp_file_data['action'].items():
            if action_history['success'] > 0 and (action_history['success'] - action_history['failure']) > -self.plan_failure_threshold:
                succeeded_action_lists.append([action, action_history['success'] - action_history['failure']])

        if len(succeeded_action_lists) > 0:
            succeeded_action_lists = sorted(succeeded_action_lists, key=lambda x: x[1], reverse=True)
            _, succeeded_subgoal_str = language_action_to_subgoal(succeeded_action_lists[0][0], waypoint)
            return True, succeeded_subgoal_str
        else:
            return False, None


    def retrieve_similar_succeeded_waypoints(self, waypoint, topK=3):
        # 1. for succeeded waypoint $wp^{success} \in M$, calculate $similarity(BERT^{text}(wp^{unseen}), BERT^{text}(wp^{success}))$.
        # 2. select top-k $wp^{success} \in M$ which are most similar to the $wp^{unseen}$.
        # 3. retrieve subgoals for the top-k $wp^{success}$, making $\{(wp_i^{success}, sg_i^{success})\}_{i=1}^k$.

        sorted_succeeded_waypoints = sorted(self.succeeded_waypoints)
        embedding_tensors = [torch.load(os.path.join(self.wp_to_sg_dir_path, f'{name}.pt')) for name in sorted_succeeded_waypoints]
        embedding_matrix = torch.cat(embedding_tensors, dim=0)
        embedding_matrix = embedding_matrix.to(DEVICE)

        # wp_embedding = self.mineclip.encode_text(waypoint)

        wp_embedding = torch.tensor(self.bert_encoder.encode(waypoint)).unsqueeze(0).to(DEVICE)

        cosine_similarities = torch.matmul(embedding_matrix, wp_embedding.T).squeeze()

        topK_values, topK_indices = torch.topk(cosine_similarities, topK)
        top_succeeded_waypoints = [sorted_succeeded_waypoints[i] for i in topK_indices.tolist()]

        wp_sg_dict = dict()

        for succeeded_wp in top_succeeded_waypoints:
            _, sg = self.is_succeeded_waypoint(succeeded_wp)
            wp_sg_dict[succeeded_wp] = sg

        return wp_sg_dict

    def retrieve_failed_subgoals(self, waypoint):
        json_file_name = f"{waypoint}.json"
        lst_dir = os.listdir(self.wp_to_sg_dir_path)

        if json_file_name not in lst_dir:
            return []
        
        failed_subgoal_lists = []
        wp_file_data = read_json_data_with_shared_lock(os.path.join(self.wp_to_sg_dir_path, json_file_name))

        for action, action_history in wp_file_data['action'].items():
            if (action_history['success'] - action_history['failure']) <= -self.plan_failure_threshold:
                _, failed_subgoal_str = language_action_to_subgoal(action, waypoint)
                failed_subgoal_lists.append(failed_subgoal_str)

        return failed_subgoal_lists

    def retrieve_total_failed_counts(self, waypoint):
        json_file_name = f"{waypoint}.json"
        lst_dir = os.listdir(self.wp_to_sg_dir_path)

        if json_file_name not in lst_dir:
            return 0
        
        total_failure_counts = 0
        wp_file_data = read_json_data_with_shared_lock(os.path.join(self.wp_to_sg_dir_path, json_file_name))

        for action, action_history in wp_file_data['action'].items():
            action_failure_count = action_history['success'] - action_history['failure']
            total_failure_counts += action_failure_count

        return total_failure_counts

    def reset_success_failure_history(self, item_name):
        is_succeeded, _ = self.is_succeeded_waypoint(item_name)
        if is_succeeded:
            return

        json_file_name = f"{item_name}.json"
        json_path = os.path.join(self.wp_to_sg_dir_path, json_file_name)
        if not os.path.exists(json_path):
            return

        with self._lock:
            with open(json_path, "r+") as fp:
                fcntl.flock(fp, fcntl.LOCK_EX)
                wp_file_data = json.load(fp)

                for action in wp_file_data['action'].keys():
                    wp_file_data['action'][action] = {'success': 0, 'failure': 0}

                fp.seek(0)
                fp.truncate()
                json.dump(wp_file_data, fp, indent=2)
                fcntl.flock(fp, fcntl.LOCK_UN)

        self.logger.info(f"Reset success/failure history for {item_name}")


    # Just for backward compatibility
    def save_plan(
        self,
        task: str,
        visual_info: str,
        goal: str,
        status: str,
        planning: List[Dict[str, Any]],
        steps: int | float,
        run_uuid: str,
        video_path: str = "",
        environment: str = "none",
    ):
        thread = MultiThreadServerAPI(
            self._save_plan,
            args=(
                task,
                visual_info,
                goal,
                status,
                planning,
                steps,
                run_uuid,
                video_path,
                environment,
            ),
        )
        thread.start()
        return thread

    def _save_plan(
        self,
        task: str,
        visual_info: str,
        goal: str,
        status: str,
        planning: List[Dict[str, Any]],
        steps: int | float,
        run_uuid: str,
        video_path: str = "",
        environment: str = "none",
    ):
        assert status in [
            "success",
            "failed",
        ], "status should be one of success, failed"
        file_name = self.cfg["memory"]["decomposed_plan"]["file"].replace(
            "<task>", task.replace(" ", "_")
        )

        memory_path = self.plan_dir_save_path.replace("<status>", status)
        os.makedirs(memory_path, exist_ok=True)

        memory_file = os.path.join(memory_path, file_name)

        if self.logger:
            self.logger.info(
                f"[hot_pink]store plan of {task} to {memory_file} :smile:[/hot_pink]"
            )
        with self._lock:
            if os.path.exists(memory_file):
                with open(memory_file, "r") as fp:
                    memory = json.load(fp)
            else:
                memory = {"plan": []}

        with self._lock, open(memory_file, "w") as fp:
            memory["plan"].append(
                {
                    "id": run_uuid,
                    "environment": environment,
                    "visual_info": visual_info,
                    "goal": goal,
                    "video": video_path,
                    "planning": planning,
                    "status": status,
                    "steps": steps,
                }
            )
            json.dump(memory, fp, indent=2)

