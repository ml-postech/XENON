from typing import Any, Dict, List
import copy
from omegaconf import DictConfig

from .mod import Mod


# class TaskCheckerModNew(Mod):
#     _cache: Dict[str, Any]

#     def __init__(self, cfg: DictConfig) -> None:
#         super().__init__(cfg)

#         self._cache = {"inventory": {}}

#     def reset(self, inventory: Dict[str, Any] | None = None):
#         if inventory is None:
#             inventory = {}
#         self._cache["inventory"] = inventory

#     def step(self, inventory, goal: tuple[str, int] | None, check_original_goal=False):
#         if goal is None:
#             return False
#         item, number = goal
#         item = self._expand_item(item)
#         need_item = [[item, number]]
#         return self._check_number(inventory, need_item)

#     # # Old checker focus on the change of items. The agent needs to gather 'number' items from the self._cache['inventory']
#     # def _check_number_old(self, inventory: Dict[str, Any], need_item: List[list]) -> bool:
#     #     # [ [["xx"], 1]] ]
#     #     total = 0
#     #     for [item_list, number] in need_item:
#     #         s = 0
#     #         p = 0
#     #         for item in item_list:
#     #             s += inventory[item] if item in inventory else 0
#     #             p += self._cache["inventory"][item] if item in self._cache["inventory"] else 0
#     #         if s >= p + number:
#     #             total += 1
#     #     return total >= len(need_item)

#     # def check_already_achieved(self, inventory, goal: tuple[str, int] | None):
#     #     if inventory is None:
#     #         return False
#     #     if goal is None:
#     #         return False
#     #     item, number = goal
#     #     item = self._expand_item(item)
#     #     need_item = [[item, number]]
#     #     return self._check_number(inventory, need_item)

#     # New checker focus on the state to be reached. The agent just needs to have 'number' items.
#     def _check_number(self, inventory: Dict[str, Any], need_item: List[list]) -> bool:
#         total = 0
#         for [item_list, number] in need_item:
#             s = 0
#             for item in item_list:
#                 s += inventory[item] if item in inventory else 0
#             if s >= number:
#                 total += 1
#         return total >= len(need_item)

#     def _expand_item(self, item: str) -> List[str]:
#         # TODO: update check item list
#         if "log" in item or "logs" in item:
#             return [
#                 "acacia_log",
#                 "birch_log",
#                 "dark_oak_log",
#                 "jungle_log",
#                 "oak_log",
#                 "spruce_log",
#             ]
#         elif "plank" in item or "planks" in item:
#             return [
#                 "acacia_planks",
#                 "birch_planks",
#                 "dark_oak_planks",
#                 "jungle_planks",
#                 "oak_planks",
#                 "dark_oak_planks",
#                 "spruce_planks",
#             ]
#         elif "redstone" in item:
#             return ["redstone"]
#         elif "stone" in item:
#             return ["cobblestone"]
#         elif "coal" in item:
#             return ["coal"]
#         else:
#             return [item]


class TaskCheckerMod(Mod):
    _cache: Dict[str, Any]

    def __init__(self, cfg: DictConfig) -> None:
        super().__init__(cfg)

        self._cache = {"inventory": {}}

    def reset(self, inventory: Dict[str, Any] | None = None):
        if inventory is None:
            inventory = {}
        self._cache["inventory"] = inventory

    def step(self, inventory, goal: tuple[str, int] | None, check_original_goal=False):
        if goal is None:
            return False
        
        goal_item, goal_number = copy.deepcopy(goal)
        item, number = goal
        item = self._expand_item(item)
        need_item = [[item, number]]
        if check_original_goal:
            return self._new_check_number(inventory, need_item) or self._new_check_number(inventory, [[[goal_item], goal_number]])
        return self._check_number(inventory, need_item)

    # def check_already_achieved(self, inventory, goal: tuple[str, int] | None):
    #     if inventory is None:
    #         return False
    #     if goal is None:
    #         return False
    #     item, number = goal
    #     item = self._expand_item(item)
    #     need_item = [[item, number]]
    #     return self._check_number(inventory, need_item)
    
    # New checker focus on the state to be reached. The agent just needs to have 'number' of items.
    # This function is only used for "check_original_goal"
    def _new_check_number(self, inventory: Dict[str, Any], need_item: List[list]) -> bool:
        print(f'In _new_check_number()')
        print(f'inventory: {inventory}')
        print(f'need_item: {need_item}')

        total = 0
        for [item_list, number] in need_item:
            s = 0
            for item in item_list:
                s += inventory[item] if item in inventory else 0
            if s >= number:
                total += 1
        return total >= len(need_item)

    # original checker focus on the change of items. The agent needs to gather 'number' items from the self._cache['inventory']
    def _check_number(self, inventory: Dict[str, Any], need_item: List[list]) -> bool:
        # [ [["xx"], 1]] ]
        total = 0
        for [item_list, number] in need_item:
            s = 0
            p = 0
            for item in item_list:
                s += inventory[item] if item in inventory else 0
                p += self._cache["inventory"][item] if item in self._cache["inventory"] else 0
            if s >= p + number:
                total += 1
        return total == len(need_item)

    def _expand_item(self, item: str) -> List[str]:
        # TODO: update check item list
        if "charcoal" in item:
            return ["charcoal"]
        elif "log" in item or "logs" in item:
            return [
                "acacia_log",
                "birch_log",
                "dark_oak_log",
                "jungle_log",
                "oak_log",
                "spruce_log",
            ]
        elif "plank" in item or "planks" in item:
            return [
                "acacia_planks",
                "birch_planks",
                "dark_oak_planks",
                "jungle_planks",
                "oak_planks",
                "dark_oak_planks",
                "spruce_planks",
            ]
        # elif "redstone" in item:
        #     return ["redstone"]
        # elif "stone" in item:
        #     return ["cobblestone"]
        elif "coal" in item:
            return ["coal"]
        else:
            return [item]
