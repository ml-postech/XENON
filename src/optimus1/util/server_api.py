from typing import Any, Dict, List
import argparse

import numpy as np
import requests
from omegaconf import DictConfig

from .image import img2base64, img_lst2base64
from .thread import MultiThreadServerAPI


class ServerAPI:
    @staticmethod
    def _reset(server_cfg: DictConfig):
        res = requests.get(
            f"{server_cfg['url']}:{server_cfg['port']}/reset",
            timeout=600, # 10 minutes
        )
        if res.status_code != 200:
            raise RuntimeError(f"Failed to reset: {res.text}")

    @staticmethod
    def reset(server_cfg: DictConfig):
        thread = MultiThreadServerAPI(ServerAPI._reset, (server_cfg,))
        thread.start()
        return thread

        # res = requests.get(
        #     f"{server_cfg['url']}:{server_cfg['port']}/reset",
        #     timeout=server_cfg["timeout"],
        # )
        # if res.status_code != 200:
        #     raise RuntimeError(f"Failed to reset: {res.text}")
        
        # reset_response = res.json()["response"]
        # return reset_response

    @staticmethod
    def get_retrieval(
        server_cfg: DictConfig,
        obs: Dict[str, Any],
        task2plan: str | None = "gp4 plan task",
        error_info: str | None = None,
        hydra_path: str = None,
        run_uuid: str = None,
    ) -> str:
        if task2plan is None:
            raise ValueError("task2plan is None")
        b = img2base64(obs["pov"])
        plan_type = "retrieval"
        data = {
            "rgb_images": [{"image": b}],
            "task_or_instruction": task2plan,
            "error_info": error_info,
            "type": plan_type,
            "hydra_path": hydra_path,
            "run_uuid": run_uuid,
        }
        res = requests.post(
            f"{server_cfg['url']}:{server_cfg['port']}/chat",
            json=data,
            timeout=server_cfg["timeout"],
        )
        if res.status_code != 200:
            raise RuntimeError(f"Failed to get plan: {res.text}")
        plan = res.json()["response"]
        return plan
    
    @staticmethod
    def get_decomposed_plan(
        server_cfg: DictConfig,
        obs: Dict[str, Any],
        waypoint: str,
        similar_wp_sg_dict: Dict[Any, Any] = None,
        failed_sg_list_for_wp: List[str] = None,
        hydra_path: str = None,
        run_uuid: str = None,
    ) -> str:
        if waypoint is None or waypoint == "":
            raise ValueError(f"waypoint is None or ''")
        b = img2base64(obs["pov"])
        type = "decomposed_plan"
        data = {
            "rgb_images": [{"image": b}],
            "waypoint": waypoint,
            "task_or_instruction": waypoint,
            "similar_wp_sg_dict": similar_wp_sg_dict,
            "failed_sg_list_for_wp": failed_sg_list_for_wp,
            "type": type,
            "hydra_path": hydra_path,
            "run_uuid": run_uuid,
        }
        res = requests.post(
            f"{server_cfg['url']}:{server_cfg['port']}/chat",
            json=data,
            timeout=server_cfg["timeout"],
        )
        if res.status_code != 200:
            raise RuntimeError(f"Failed to get plan: {res.text}")
        plan = res.json()["response"]
        prompt = res.json()["message"]
        return plan, prompt

    @staticmethod
    def get_context_aware_reasoning(
        server_cfg: DictConfig,
        obs: Dict[str, Any],
        task: str,
        goal: str,
        hydra_path: str = None,
        run_uuid: str = None,
    ) -> str:
        if task is None or task == "":
            raise ValueError(f"task is None or ''")
        b = img2base64(obs["pov"])
        type = "context_aware_reasoning"
        data = {
            "rgb_images": [{"image": b}],
            "task_or_instruction": task,
            "goal": goal,
            "type": type,
            "hydra_path": hydra_path,
            "run_uuid": run_uuid,
        }
        res = requests.post(
            f"{server_cfg['url']}:{server_cfg['port']}/chat",
            json=data,
            timeout=server_cfg["timeout"],
        )
        if res.status_code != 200:
            raise RuntimeError(f"Failed to get context_aware_reasoning: {res.text}")
        reasoning = res.json()["response"]
        visual_description = res.json()["message"]
        return reasoning, visual_description

    @staticmethod
    def get_plan(
        server_cfg: DictConfig,
        obs: Dict[str, Any],
        task2plan: str | None = "gp4 plan task",
        error_info: str | None = None,
        example: str | None = None,
        graph: str | None = None,
        visual_info: str | None = None,
        hydra_path: str = None,
        run_uuid: str = None,
    ) -> str:
        if task2plan is None:
            raise ValueError("task2plan is None")
        b = img2base64(obs["pov"])
        plan_type = "plan" if error_info is None else "replan"
        # plan_type = "plan"
        # error_info = error_info if error_info is not None else ""
        data = {
            "rgb_images": [{"image": b}],
            "task_or_instruction": task2plan,
            "error_info": error_info,
            "example": example,
            "graph": graph,
            "type": plan_type,
            "visual_info": visual_info,
            "hydra_path": hydra_path,
            "run_uuid": run_uuid,
        }
        res = requests.post(
            f"{server_cfg['url']}:{server_cfg['port']}/chat",
            json=data,
            timeout=server_cfg["timeout"],
        )
        if res.status_code != 200:
            raise RuntimeError(f"Failed to get plan: {res.text}")
        plan = res.json()["response"]
        # print(plan)
        return plan
    
    @staticmethod
    def get_fixed_json(
        server_cfg: DictConfig,
        obs: Dict[str, Any],
        task2plan: str | None = "gp4 plan task",
        errorneous_planning: str | None = "",
        hydra_path: str = None,
        run_uuid: str = None,
    ):
        if task2plan is None:
            raise ValueError("task2plan is None")
        b = img2base64(obs["pov"])
        type = "fixjson"
        data = {
            "rgb_images": [{"image": b}],
            "task_or_instruction": task2plan,
            "errorneous_planning": errorneous_planning,
            "type": type,
            "hydra_path": hydra_path,
            "run_uuid": run_uuid,
        }
        res = requests.post(
            f"{server_cfg['url']}:{server_cfg['port']}/chat",
            json=data,
            timeout=server_cfg["timeout"],
        )
        if res.status_code != 200:
            raise RuntimeError(f"Failed to get plan: {res.text}")
        plan = res.json()["response"]
        return plan

    @staticmethod
    def get_action(
        server_cfg: Dict[str, Any],
        obs: Dict[str, Any],
        task: str | None,
        step: int = 0,
        hydra_path: str = None,
        run_uuid: str = None,
    ) -> Dict[str, np.ndarray] | List[Dict[str, np.ndarray]]:
        """
        Sends a request to a server to get an action based on the given observation and task.

        Args:
            server_cfg (Dict[str, Any]): A dictionary containing server configuration parameters.
            obs (Dict[str, Any]): The observation data.
            task (str): The task to perform.
            step (int, optional): The current step. Defaults to 0.

        Returns:
            Dict[str, np.ndarray]: A dictionary containing the action to perform.

        Raises:
            RuntimeError: If the request to the server fails.
        """
        if task is None:
            raise ValueError("task is None")
        b = img2base64(obs["pov"])
        data = {
            "rgb_images": [{"image": b}],
            "task_or_instruction": task,
            "type": "action",
            "current_step": step,
            "hydra_path": hydra_path,
            "run_uuid": run_uuid,
        }
        res = requests.post(
            f"{server_cfg['url']}:{server_cfg['port']}/chat",
            json=data,
            timeout=server_cfg["timeout"],
        )
        if res.status_code != 200:
            raise RuntimeError(f"Failed to get action: {res.text}")
        action = res.json()["response"]
        if isinstance(action, dict):
            for k, v in action.items():
                action[k] = np.array(v)
        elif isinstance(action, list):
            for ac in action:
                for k, v in ac.items():
                    ac[k] = np.array(v)
        return action

    @staticmethod
    def _get_reflection(
        server_cfg: DictConfig,
        obs: Dict[str, Any],
        done_imgs: List[str] | None = None,
        cont_imgs: List[str] | None = None,
        replan_imgs: List[str] | None = None,
        task2reflection: str | None = "gpt4 reflection task",
        step: int = 0,
        hydra_path: str = None,
        run_uuid: str = None,
    ) -> tuple[str, str]:
        if task2reflection is None:
            raise ValueError("task2reflection is None")

        # status = ["done", "continue", "replan"]

        b = img2base64(obs["pov"])

        data = {
            "rgb_images": [{"image": b}],
            "task_or_instruction": task2reflection,
            "type": "reflection",
            "current_step": step,
            "done_imgs": img_lst2base64(done_imgs) if done_imgs else None,
            "cont_imgs": img_lst2base64(cont_imgs) if cont_imgs else None, # read image and convert to string data
            "replan_imgs": img_lst2base64(replan_imgs) if replan_imgs else None,
            "hydra_path": hydra_path,
            "run_uuid": run_uuid,
        }
        res = requests.post(
            f"{server_cfg['url']}:{server_cfg['port']}/chat",
            json=data,
            timeout=server_cfg["timeout"],
        )
        res.raise_for_status()
        # if res.status_code != 200:
        #     raise RuntimeError(f"Failed to get action: {res.text}")
        response = res.json()
        res, appendix = response["response"], response["appendix"]
        # assert response in status, f"Invalid response: {response}"
        return res, appendix

    @staticmethod
    def get_reflection(
        server_cfg: DictConfig,
        obs: Dict[str, Any],
        done_imgs: List[str] | None = None,
        cont_imgs: List[str] | None = None,
        replan_imgs: List[str] | None = None,
        task2reflection: str | None = "gpt4 reflection task",
        step: int = 0,
        hydra_path: str = None,
        run_uuid: str = None,
    ):
        thread = MultiThreadServerAPI(
            ServerAPI._get_reflection,
            (server_cfg, obs, done_imgs, cont_imgs, replan_imgs, task2reflection, step, hydra_path, run_uuid),
        )
        # thread.daemon = True
        thread.start()
        return thread
    
    @staticmethod
    def shutdown_server(
        url: str, port: int, timeout: int = 60
    ):
        res = requests.post(
            f"{url}:{port}/shutdown",  # Use the /shutdown endpoint
            timeout=timeout,
        )
        if res.status_code != 200:
            raise RuntimeError(f"Failed to shutdown server: {res.text}")
        return res

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Shutdown the running web server.")
    parser.add_argument(
        "--port",
        type=int,
        required=True,
        help="Port number of the server to shut down",
    )
    parser.add_argument(
        "--url",
        type=str,
        default="http://127.0.0.1",
        help="URL of the server (default: http://127.0.0.1)",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=60,
        help="Request timeout in seconds (default: 60)",
    )
    args = parser.parse_args()

    res = ServerAPI.shutdown_server(args.url, args.port, args.timeout)
    print(res.text)
