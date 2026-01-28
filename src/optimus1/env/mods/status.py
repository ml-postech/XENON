import logging
from typing import Any, Dict

from omegaconf import DictConfig

from .mod import Mod


class StatusMod(Mod):
    inventory: Dict[str, Any]
    _inventory_change: bool = False
    _inventory_new_item: bool = False
    inventory_with_slot: Dict[int, Any]

    equipment: str = "none"
    _equipment_change: bool = False
    location_stats: Dict[str, Any]

    _cache: Dict[str, Any]

    def __init__(self, cfg: DictConfig, logger: logging.Logger):
        super().__init__(cfg)

        self.logger = logger

        self.reset()

    def reset(self):
        self.inventory = {}
        self._inventory_change = False
        self._inventory_new_item = False

        self.inventory_with_slot = {}
        self.equipment = "none"
        self._equipment_change = False
        self.location_stats = {}

        self._cache = {}

    def step(self, observation: Dict[str, Any], action: Dict[str, Any], num_steps: int):
        self.inventory = self._current_inventory(observation["inventory"])

        self.location_stats = observation["location_stats"]
        # self.location_stats.pop("biome_id")  # biome_id always 0

        self.inventory_with_slot = observation["plain_inventory"]

        self.equipment = self._current_equip(action, self.inventory_with_slot)

        if self._inventory_change:
            self.logger.info(f"[magenta]Current Inventory: {self.inventory}, position: {self.get_position()}, env num_steps: {num_steps}[/magenta]")
        if self._equipment_change:
            self.logger.info(f"[magenta]Current Equipment: {self.equipment}, env num_steps: {num_steps}[/magenta]")
        # self.logger.info(f"Current location status: {self.location_stats}")

    def _current_inventory(self, inventory):
        self._cache["last_inventory"] = self.inventory
        ci = {k: inventory[k].item() for k in inventory if inventory[k] > 0}
        self._inventory_change = ci != self.inventory
        self._inventory_new_item = len(ci) > len(self._cache["last_inventory"])

        # self.logger.info(f'ci: {ci}')
        # self.logger.info(f'self._cache["last_inventory"]: {self._cache["last_inventory"]}')
        # self.logger.info(f'self._inventory_new_item: {str(self._inventory_new_item)}\n')

        return ci

    def _current_equip(self, action, inventory) -> str:
        hotbar = -1
        for i in range(1, 10):
            if f"hotbar.{i}" in action and action[f"hotbar.{i}"] > 0:
                hotbar = i
                break
        if hotbar == -1:
            self._equipment_change = False
            return self.equipment
        self._equipment_change = inventory[hotbar - 1]["type"] != self.equipment
        return inventory[hotbar - 1]["type"]

    @property
    def inventory_change(self):
        return self._inventory_change

    @property
    def inventory_new_item(self):
        return self._inventory_new_item

    def get_position(self):
        loc = self.location_stats
        xpos, ypos, zpos = loc["xpos"].item(), loc["ypos"].item(), loc["zpos"].item()
        return xpos, ypos, zpos

    def get_height(self):
        loc = self.location_stats
        ypos = loc["ypos"].item()
        return ypos

    def inventory_change_what(self):
        diff = {}
        if self._inventory_change:
            last = self._cache["last_inventory"]
            for item, number in self.inventory.items():
                if item not in last:
                    diff[item] = number
                elif number > last[item]:
                    diff[item] = number - last[item]
        return diff

    def inventory_new_item_what(self):
        diff = {}
        if self._inventory_new_item:
            last = self._cache["last_inventory"]
            for item, number in self.inventory.items():
                if item not in last:
                    diff[item] = number
        return diff

    def get_status(self):
        return {
            "inventory": self.inventory,
            "equipment": self.equipment,
            "location_stats": self.location_stats,
            "plain_inventory": self.inventory_with_slot,
        }
