import os
import sys
import signal
import time
import argparse
import base64
import random
import numpy as np
import torch
import transformers

from fastapi import FastAPI
import uvicorn

from optimus1.server.agent import AgentFactory
from optimus1.server.api.request import MCRequest, MCResponse
from optimus1.server.api.utils import base64_to_image, base64lst2img_path

import time

app = FastAPI()
agent = None


def _img2base64(img_path: str):
    with open(img_path, "rb") as f:
        img = base64.b64encode(f.read())
    return img.decode("utf-8")


def _filter_task_obs(task: str, image_root: str) -> str:
    """
    Filter the task observations based on the given task.

    Args:
        task (str): The task to filter the observations for.

    Returns:
        str: The path of the first image that matches the given task.

    """
    task = task.replace(" ", "_")
    task_imgs = [img for img in os.listdir(image_root) if ".jpg" in img and task in img]
    task_imgs.sort(key=lambda x: int(x.split("_")[-1].replace(".jpg", "")))
    return os.path.join(image_root, task_imgs[0])


def stop_server():
    print("Stopping server...")
    os.kill(os.getpid(), signal.SIGINT)  # Graceful shutdown using SIGINT


@app.post("/shutdown")
def shutdown():
    stop_server()
    return {"message": "Server is stopping..."}


@app.get("/reset")
def reset() -> MCResponse:
    global agent
    agent = AgentFactory.reset()
    print("agent reset")
    return MCResponse(response="reset done")


@app.post("/chat")
def chat(req: MCRequest) -> MCResponse:
    global agent

    # print(f'req.type: {req.type}')

    if req.type is None:
        req.type = "plan"
    
    # Save current obs (bytes) to image file, and return the path
    # print(req)

    hydra_path = req.hydra_path
    run_uuid = req.run_uuid

    image_root = os.path.join(hydra_path, run_uuid)
    image_root = os.path.join(image_root, "imgs")
    rgb_obs = base64_to_image(
        req.rgb_images,
        image_root=image_root,
        task=req.task_or_instruction,
        step=req.current_step,
    )
    # print(f"HERE req.type: {req.type}")
    match req.type:
        case "decomposed_plan":
            retry = 0
            while True:
                try:
                    plans, prompt = agent.decomposed_plan(
                        req.waypoint,
                        rgb_obs[-1],
                        req.similar_wp_sg_dict,
                        req.failed_sg_list_for_wp,
                    )
                    response = MCResponse(response=plans, message=prompt)
                    break
                except:
                    retry += 1
                    print("connection error, retry: ", retry)
        case "context_aware_reasoning":
            retry = 0
            while True:
                try:
                    reasoning, visual_description = agent.context_aware_reasoning(
                        req.task_or_instruction,
                        req.goal,
                        rgb_obs[-1],
                    )
                    response = MCResponse(response=reasoning, message=visual_description)
                    break
                except:
                    retry += 1
                    print("connection error, retry: ", retry)
        case "retrieval":
            retry = 0
            while True:
                try:
                    plans_retrieval = agent.retrieve(
                        req.task_or_instruction,
                        rgb_obs[-1],
                    )
                    response = MCResponse(response=plans_retrieval)
                    break
                except:
                    retry += 1
                    print("connection error while retrieval, retry: ", retry)
        case "plan":
            retry = 0
            while True:
                try:
                    plans = agent.plan(
                        req.task_or_instruction,
                        rgb_obs[-1],
                        req.example,
                        req.visual_info,
                        req.graph,
                    )
                    response = MCResponse(response=plans)
                    break
                except:
                    retry += 1
                    print("connection error, retry: ", retry)
        case "fixjson":
            retry = 0
            # print(f"HERE!!!@#!%$@%!")
            # print(f"HERE!!! req.errorneous_planning: {req.errorneous_planning}")
            # print(f"rgb_obs[-1]: {rgb_obs[-1]}")
            while retry < 10:
                try:
                    fixed_json = agent.fix_json_format(
                        req.errorneous_planning, rgb_obs[-1]
                    )
                    response = MCResponse(response=fixed_json)
                    break
                except:
                    retry += 1
                    print("connection error, retry: ", retry)

        case "action":

            start = time.perf_counter()
            minrl_action = agent.action(req.task_or_instruction, rgb_obs)
            end = time.perf_counter()
            # print(end - start, " s")  # 0.04s
            response = MCResponse(response=minrl_action)
            # print(response)
        case "reflection":
            # old_obs: path of the obs when the current task is given
            old_obs = _filter_task_obs(req.task_or_instruction, image_root)
            print(f"old_obs {old_obs} current step {req.current_step}")
            retry = 0

            done_imgs, cont_imgs, replan_imgs = (
                req.done_imgs,
                req.cont_imgs, # str data (bytes) of the images
                req.replan_imgs,
            )
            done, cont, replan = (
                base64lst2img_path(done_imgs, image_root), # save image data (bytes) to file and return path
                base64lst2img_path(cont_imgs, image_root),
                base64lst2img_path(replan_imgs, image_root),
            )
            while retry < 10:
                try:
                    # NOTE: Can VLM determine the progress only using 2 images (current obs, old obs)?
                    reflection = agent.reflection(
                        req.task_or_instruction,
                        old_obs, # obs when the current task is given
                        rgb_obs[-1], # current obs
                        done_img_path=done,
                        cont_img_path=cont,
                        replan_img_path=replan,
                    )
                    print(f"{old_obs} <-> {rgb_obs[-1]}: {reflection}")
                    response = MCResponse(
                        response=reflection, appendix=_img2base64(old_obs)
                    )
                    break
                except:
                    retry += 1
                    time.sleep(1)
                    print("connection error while reflection, retry: ", retry)
        case "replan":
            retry = 0
            while retry < 10:
                try:
                    replan = agent.replan(
                        req.task_or_instruction,
                        rgb_obs[-1],
                        req.error_info,
                        req.example,
                        req.graph,
                    )
                    response = MCResponse(response=replan)
                    print(replan)
                    break
                except Exception as e:
                    retry += 1
                    time.sleep(1)
                    print(f"connection error while replan {e}, retry: {retry}")
        case _:
            response = MCResponse(message=f"{req.type} not support...", status_code=400)
    return response


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Start FastAPI server with custom AgentFactory configuration."
    )
    parser.add_argument("--plan_with_gpt", action="store_true", help="Use GPT for planning.")
    parser.add_argument("--plan_model", type=str, default="Qwen/Qwen2.5-VL-7B-Instruct", help="Model for planning.")
    parser.add_argument("--in_model", type=str, default="checkpoints/vpt/2x.model", help="Model for input.")
    parser.add_argument("--in_weights", type=str, default="checkpoints/steve1/steve1.weights", help="Weights for input.")
    parser.add_argument("--prior_weights", type=str, default="checkpoints/steve1/steve1_prior.pt", help="Weights for prior.")
    parser.add_argument("--port", type=int, default=12345, help="Port to run the server on.")
    parser.add_argument("--seed", type=int, default=0)

    args = parser.parse_args()
    print("Starting server...")
    print(f'args: {args}')

    seed = int(args.seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    transformers.set_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    AgentFactory.set_args(
        plan_with_gpt=args.plan_with_gpt,
        plan_model=args.plan_model,
        in_model=args.in_model,
        in_weights=args.in_weights,
        prior_weights=args.prior_weights,
    )

    uvicorn.run(app, host="0.0.0.0", port=args.port)
