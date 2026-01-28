import copy
import json
import re
from typing import Any, Dict, List

PlanList = List[Dict[str, Any]]



def language_action_to_subgoal(action, waypoint) -> str:
    subgoal = {
        "task": action,
        "goal": [waypoint, 1],
    }
    subgoal_str = json.dumps(subgoal)

    return subgoal, subgoal_str


def render_subgoal(subgoal: str, wp_num: int = 1) -> str:
    subgoal = subgoal.replace("<task planning>:", "<task planning>").replace("**", "")
    sep_str = "<task planning>"

    temp = subgoal.split(sep_str)[-1].strip()
    if "```json" in temp:
        temp = temp.split("```json")[1].strip().split("```")[0].strip()

    if "{{" in temp:
        temp = temp.replace("{{", "{").replace("}}", "}")

    r = temp.rfind("}")
    temp = temp[: r + 1]

    try:
        temp = json.loads(temp)
    except json.JSONDecodeError as e:
        return [], None, str(e)

    temp["goal"][1] = wp_num

    return temp, temp["task"], None


def render_context_aware_reasoning(reasoning) -> str:
    reasoning = reasoning.replace("<reasoning>:", "<reasoning>")
    sep_str = "<reasoning>"

    temp = reasoning.split(sep_str)[-1].strip()
    if "```json" in temp:
        temp = temp.split("```json")[1].strip().split("```")[0].strip()

    if "{{" in temp:
        temp = temp.replace("{{", "{").replace("}}", "}")

    r = temp.rfind("}")
    temp = temp[: r + 1]

    try:
        temp_dict = json.loads(temp)
    except json.JSONDecodeError as e:
        return None, str(e)

    return temp_dict, None


def render_gpt4_plan(plan: str, replan: bool = False) -> PlanList:
    plan = plan.replace("<task planning>:", "<task planning>").replace("**", "")
    # plan = plan.replace("<replan>:", "<replan>").replace("**", "")
    # sep_str = "<replan>" if replan else "<task planning>"
    plan = plan.replace("<replan>:", "<replan>").replace("<replan>", "<task planning>")
    sep_str = "<task planning>"

    temp = plan.split(sep_str)[-1].strip()
    if "```json" in temp:
        temp = temp.split("```json")[1].strip().split("```")[0].strip()

    if "{{" in temp:
        temp = temp.replace("{{", "{").replace("}}", "}")

    r = temp.rfind("}")
    temp = temp[: r + 1]

    try:
        temp = json.loads(temp)
    except json.JSONDecodeError as e:
        return [], str(e)

    sub_plans = [
        temp[step]
        for step in temp.keys()
        if "open" not in temp[step]["task"]
        and "place" not in temp[step]["task"]
        and "access" not in temp[step]["task"]
    ]

    for p in sub_plans:
        p["task"] = p["task"].replace("punch", "chop").replace("collect", "chop").replace("gather", "chop")

    return sub_plans, None


def render_plan(plan: str, wp_num: int = 1) -> PlanList:
    plan = plan.replace("<task planning>:", "<task planning>").replace("**", "")
    plan = plan.replace("<replan>:", "<replan>").replace("<replan>", "<task planning>")
    sep_str = "<task planning>"

    temp = plan.split(sep_str)[-1].strip()
    if "```json" in temp:
        temp = temp.split("```json")[1].strip().split("```")[0].strip()

    if "{{" in temp:
        temp = temp.replace("{{", "{").replace("}}", "}")

    r = temp.rfind("}")
    temp = temp[: r + 1]
    temp_str = copy.deepcopy(temp)

    try:
        temp = json.loads(temp)
    except json.JSONDecodeError as e:
        return [], None, str(e)

    sub_plans = [
        temp[step]
        for step in temp.keys()
        if "open" not in temp[step]["task"]
        and "place" not in temp[step]["task"]
        and "access" not in temp[step]["task"]
    ]

    for p in sub_plans:
        p["task"] = p["task"].replace("punch", "chop").replace("collect", "chop").replace("gather", "chop")
        p['goal'][1] = wp_num

    return sub_plans, temp_str, None


def render_reflection(reflection: str):
    """Environment: <Ocean>
    Situation: <Replan>
    Predicament: <In_water>
    """
    reflection = reflection.strip().replace(": ", ": <")

    matches = re.findall(r"<([^<]+)$", reflection, re.MULTILINE)
    rp = None
    if len(matches) == 3:
        rp = matches[2].split("/")[0].strip().lower()
    res = (
        matches[0].split("/")[0].replace(">", "").strip().lower().split(" ")[0].split("\n")[0],
        matches[1].split("/")[0].replace(">", "").strip().lower().split(" ")[0].split("\n")[0],
        rp,
    )
    return res


def render_recipe(recipe) -> str:
    lst = [f'"{k}": {v}' for k, v in recipe.items()]

    return "{" + ", ".join(lst) + "}"


def render_replan_example(fixed_plan: List[Dict[str, Any]]):
    res = {}
    for idx, plan in enumerate(fixed_plan):
        res[f"step {idx + 1}"] = plan
    return json.dumps(res)


def oracle_graph_to_plan(graph, subgoal_item_name, subgoal_num):
    plan_dict = dict()
    plan_idx = 1

    if "Just mine it" in graph:
        if subgoal_item_name in ['log', 'logs']:
            task_str = "chop a tree"
            need_item = "logs"
        elif subgoal_item_name in ["cobblestone", "iron_ore", "gold_ore"]:
            need_item = subgoal_item_name
            task_str = f"dig down and break down {need_item.replace('_', ' ')}"
        elif subgoal_item_name in ["diamond", "diamond_ore"]:
            need_item = subgoal_item_name.replace('_ore', ' ')
            task_str = f"dig down and mine {need_item}"
        else:
            need_item = subgoal_item_name
            task_str = f"mine {need_item}"

        dict_key = f"step {plan_idx}"
        
        plan_dict[dict_key] = {
            "task": task_str,
            "goal": [need_item, subgoal_num]
        }
        return json.dumps(plan_dict)


    lines = graph.strip().splitlines()
    for line in lines:
        if "summary:" in line:
            continue
        if "diamond_ore" in line or "diamond ore" in line:
            continue
    
        tmp = line.split(": need")
        need_item = tmp[0].split('. ')[1]
        need_num = int(tmp[1].strip())

        if need_item in ['log', 'logs']:
            task_str = "chop a tree"
            need_item = "logs"
        elif need_item in ["cobblestone", "iron_ore", "gold_ore"]:
            task_str = f"dig down and break down {need_item.replace('_', ' ')}"
        elif need_item in ["diamond"]:
            task_str = f"dig down and mine {need_item}"
        elif "ingot" in need_item:
            task_str = "smelt " + need_item.replace('_ingot', ' ore')
        else:
            task_str = "craft " + need_item.replace('_', ' ')
    
        dict_key = f"step {plan_idx}"
        plan_dict[dict_key] = {
            "task": task_str,
            "goal": [need_item, need_num]
        }
        plan_idx += 1

    return json.dumps(plan_dict)

if __name__ == "__main__":
    plan = """{
    "step 1": {"task": "punch a tree", "goal": ["logs", 3]},
    "step 2": {"task": "open inventory", "goal": ["inventory accessed", 1]},
    "step 3": {"task": "craft planks", "goal": ["planks", 12]},
    "step 4": {"task": "craft sticks", "goal": ["sticks", 4]},
    "step 5": {"task": "place crafting table", "goal": ["crafting_table placed", 1]},
    "step 6": {"task": "use crafting table", "goal": ["crafting_table used", 1]},
    "step 7": {"task": "craft wooden pickaxe", "goal": ["wooden_pickaxe", 1]}
}"""
    temp = json.loads(plan)
    print(temp["step 1"]["task"])
    sub_plans = [
        temp[step] for step in temp.keys() if "open" not in temp[step]["task"] and "place" not in temp[step]["task"]
    ]
    print(sub_plans)
    # print(render_reflection(plan))
