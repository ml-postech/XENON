import copy
import json
import os
import shutil
import fcntl
import time
from typing import Any, Dict

import torch
import torch.nn.functional as F
from sentence_transformers import SentenceTransformer

# from .oracle_graph import KnowledgeGraph as OracleGraph


def get_item_name(item_dict: dict) -> str:
    if "item" in item_dict:
        item = item_dict["item"][10:]
    elif "tag" in item_dict:
        item = item_dict["tag"][10:]
    if item == "stone_crafting_materials" or item == "stone_tool_materials":
        item = "cobblestone"
    return item


PICKAXE_TO_INT = {
    'wooden_pickaxe': 1,
    'stone_pickaxe': 2,
    'iron_pickaxe': 3,
}

INT_TO_PICKAXE = {
    1: 'wooden_pickaxe',
    2: 'stone_pickaxe',
    3: 'iron_pickaxe',
}


class HypothesizedRecipeGraph:
    _all_recipes: Dict[str, Any]
    item_output_number: Dict[str, int]

    item_number_graph: Dict[str, Dict[str, int]]

    def __init__(self, cfg, logger):
        # self.oracle_graph = OracleGraph() # just for checking item names
        self.cfg = cfg
        self.logger = logger
        self.device = f'cuda'
        self.exploration_uuid = cfg["exploration_uuid"]

        # List of all goal items
        self.goal_items = [
            "wooden_shovel", "wooden_pickaxe", "wooden_axe", "wooden_hoe", "stick", "crafting_table", "wooden_sword", "chest", "bowl", "ladder", "logs", "dirt",
            "stone_shovel", "stone_pickaxe", "stone_axe", "stone_hoe", "charcoal", "smoker", "stone_sword", "furnace", "torch", "cobblestone",
            "iron_shovel", "iron_pickaxe", "iron_axe", "iron_hoe", "bucket", "hopper", "rail", "iron_sword", "shears", "smithing_table", "tripwire_hook", "chain", "iron_bars", "iron_nugget", "blast_furnace", "stonecutter", "iron_ore",
            "golden_shovel", "golden_pickaxe", "golden_axe", "golden_hoe", "golden_sword", "gold_ingot", "gold_ore",
            "diamond_shovel", "diamond_pickaxe", "diamond_axe", "diamond_hoe", "diamond_sword", "diamond", "jukebox", 
            "piston", "redstone_torch", "activator_rail", "compass", "dropper", "note_block", "redstone",
            "shield", "iron_chestplate", "iron_boots", "iron_leggings", "iron_helmet", "diamond_helmet", "diamond_chestplate", "diamond_leggings", "diamond_boots", "golden_helmet", "golden_leggings", "golden_boots", "golden_chestplate",
        ]

        self.bert_encoder = SentenceTransformer('all-MiniLM-L6-v2', device=self.device)

        self.root_path = self.cfg["memory"]["path"]
        self.logger.info(f'In HypothesizedRecipeGraph. self.root_path: {self.root_path}')
        os.makedirs(self.root_path, exist_ok=True)

        self.recipe_base = os.path.join(self.root_path, self.cfg["memory"]["recipe"]["base"])
        self.logger.info(f'In HypothesizedRecipeGraph. recipe_base: {self.recipe_base}')

        self.verified_recipe_dir = os.path.join(
            self.recipe_base, self.cfg["memory"]["recipe"]["verified"]["path"]
        )
        os.makedirs(self.verified_recipe_dir, exist_ok=True)

        self.hypothesized_recipe_dir = os.path.join(
            self.recipe_base, self.cfg["memory"]["recipe"]["hypothesized"]["path"]
        )
        os.makedirs(self.hypothesized_recipe_dir, exist_ok=True)

        self.topK = int(self.cfg["memory"]["topK"])
        self.inadmissible_threshold = int(self.cfg["memory"]["inadmissible_threshold"])

        self.load_and_init_all_recipes()


    def load_and_init_all_recipes(self):
        # recipe graph. if an item is newly verified, self.graph and self.item_number_graph should be updated.
        self.graph = {}
        self.item_number_graph = {}

        self.all_item_names = []
        self.verified_item_names = []
        self.hypothesized_item_names = []
        self.frontier_item_names = []
        self.inadmissible_item_names = []
        self.crafting_resources = [] # list of verified items && used for crafting. (e.g. logs, planks, stick, cobblestone, iron_ingot)

        self.recipe_revised_items = []

        self.exploration_count_dict = {} # save exploration count for all unverified nodes
        self.knowledge_score_dict = {} # save knowledge score for all frontier nodes
        self.level_dict = {} # save level for all hypothesized nodes
        self.path_dict = {} # save path for all hypothesized nodes

        self._load_verified_recipes()
        self._load_hypothesized_recipes()
        self._prepare_item_name_embeddings()
        self.find_frontiers()

        self.all_item_names = list(set(self.all_item_names))
        self.crafting_resources = list(set(self.crafting_resources))
        self.verified_item_names = list(set(self.verified_item_names))
        self.hypothesized_item_names = list(set(self.hypothesized_item_names))
        self.frontier_item_names = list(set(self.frontier_item_names))


    def _load_verified_recipes(self):
        """
        input: paths in cfg
        output: self.graph, self.item_number_graph, self.verified_item_names, self.all_item_names, self.crafting_resources
        """

        for file_name in os.listdir(self.verified_recipe_dir):
            if not file_name.endswith(".json"):
                continue

            with open(os.path.join(self.verified_recipe_dir, file_name), "r") as fp:
                data = json.load(fp)

            recipe_data = copy.deepcopy(data)

            # update self.item_number_graph
            item_name = recipe_data["item_name"]
            for material, num_needed in recipe_data["ingredients"].items():
                if material not in self.item_number_graph:
                    self.item_number_graph[material] = {}
                self.item_number_graph[material][item_name] = num_needed

            # update self.graph
            required_pickaxe = INT_TO_PICKAXE.get(int(recipe_data["required_pickaxe"]), None)
            if required_pickaxe:
                recipe_data["ingredients"].update({required_pickaxe: 1})

            self.graph[item_name] = {
                "ingredients": recipe_data["ingredients"],
                "output_qty": recipe_data["output_qty"],
                "is_verified": True,
            }

            self.verified_item_names.append(item_name)
            self.all_item_names.append(item_name)

            if bool(recipe_data["is_crafting_resource"]):
                self.crafting_resources.append(item_name)

    def _load_hypothesized_recipes(self):
        """
        API spec
            input: paths in cfg
            output: self.graph, self.item_number_graph, self.hypothesized_item_names, self.all_item_names
        """
        for file_name in os.listdir(self.hypothesized_recipe_dir):
            if not file_name.endswith(".json"):
                continue
            if file_name.replace(".json", "") in self.verified_item_names:
                continue

            with open(os.path.join(self.hypothesized_recipe_dir, file_name), "r") as fp:
                data = json.load(fp)
            if data["item_name"] in self.verified_item_names:
                continue

            recipe_data = copy.deepcopy(data)

            # update self.item_number_graph
            item_name = recipe_data["item_name"]
            for material, num_needed in recipe_data["ingredients"].items():
                if material not in self.item_number_graph:
                    self.item_number_graph[material] = {}
                self.item_number_graph[material][item_name] = num_needed

            # update self.graph
            required_pickaxe = INT_TO_PICKAXE.get(int(recipe_data["required_pickaxe"]), None)
            if required_pickaxe:
                recipe_data["ingredients"].update({required_pickaxe: 1})

            self.graph[item_name] = {
                "ingredients": recipe_data["ingredients"],
                "output_qty": recipe_data["output_qty"],
                "is_verified": False,
            }

            self.exploration_count_dict[item_name] = recipe_data["exploration_count"]
            if recipe_data["exploration_count"] > self.inadmissible_threshold:
                self.inadmissible_item_names.append(item_name)

            self.hypothesized_item_names.append(item_name)
            self.all_item_names.append(item_name)


    def get_recipe(self, item: str):
        return self.graph.get(item)

    def _init_hypothesized_recipes(self):
        """
        API spec
            input: list of goal items, list of verified items, recipes of verified items
            output: hypothesized recipes of items

        Procedure:
            for g in goal items:
                if g in verified recipes:
                    continue
                _generate_hypothesized_recipe(g)
        """

        for g in self.goal_items:
            if g in self.verified_item_names:
                continue

            # create empty json file
            with open(os.path.join(self.hypothesized_recipe_dir, f"{g}.json"), "w") as fp:
                fp.write("")

            self._generate_hypothesized_recipe(g)


    def _generate_hypothesized_recipe(self, item_name):
        """
        API spec
            input: item_name
            output: hypothesized recipe of the item
        
        Procedure:
            find top-3 similar items, and their recipes
            generate a hypothesized recipe for g

            for material in hypothesized recipe:
                if material not in graph:
                    generate a hypothesized recipe for the material.
                    until item_name is connected to self.graph.
        """

        """
        allowed recipes
            1. dictionary, whose keys are in self.graph
            2. dictionary, which has no key. it means that the item is obtained by mining.

<item_name>: stone_pickaxe
<required_items>: {{"cobblestone": 3, "stick": 2, "crafting_table": 1}}
        """

        topK_similarity_score, topK_verified_items = self.find_top_similar_verified_items(item_name, topK=self.topK)

        topK_verified_recipes = []
        for verified_item in topK_verified_items:
            recipe_data = self.get_recipe(verified_item)
            topK_verified_recipes.append(recipe_data['ingredients'])

        # TODO: run this function with your LLM.
        # hypothesized_recipe, error_msg = __call_llm(item_name, topK_verified_items, topK_verified_recipes)
        # print("Generation")
        # print(f"item_name: {item_name}")
        # print(f'topK_verified_items: {topK_verified_items}')
        # print(hypothesized_recipe)
        # print()
        # if error_msg is not None:
        #     print("Error!!!!")
        #     print(f"item_name: {item_name}")
        #     print(hypothesized_recipe)
        #     print()
        hypothesized_recipe = {} # LLM_generate_recipe(item_name, topK_verified_items, topK_verified_recipes)

        hypothesized_recipe = self._change_material_names(item_name, hypothesized_recipe) # change material names using oracle graph and self.all_item_names

        required_pickaxe = [PICKAXE_TO_INT.get(k, 0) for k in hypothesized_recipe.keys()]
        required_pickaxe = max(required_pickaxe)

        hypothesized_recipe_json_data = {
            "item_name": item_name,
            "output_qty": 1,
            "ingredients": hypothesized_recipe,
            "required_pickaxe": required_pickaxe,
            "is_crafting_resource": False,
            "exploration_count": 1,
        }
        with open(os.path.join(self.hypothesized_recipe_dir, f"{item_name}.json"), "w") as fp:
            json.dump(hypothesized_recipe_json_data, fp, indent=2)
        
        self.hypothesized_item_names.append(item_name)
        self.all_item_names.append(item_name)

        # update self.graph and self.item_number_graph
        self.graph[item_name]  = {
            "ingredients": hypothesized_recipe,
            "output_qty": 1,
            "is_verified": False
        }
        for material, num_needed in hypothesized_recipe.items():
            if material not in self.item_number_graph:
                self.item_number_graph[material] = {}
            self.item_number_graph[material][item_name] = num_needed

        # if material not in self.graph, generate a hypothesized recipe for the material recursively.
        # This makes item_name connected in self.graph
        for material, num_needed in hypothesized_recipe.items():
            if material not in self.graph and material not in self.hypothesized_item_names:
                self._generate_hypothesized_recipe(material)


    def _change_material_names(self, item_name, hypothesized_recipe):
        # <item_name>: "stone_pickaxe"
        # <hypothesized_recipe>: {"cobblestone": 3, "stick": 2, "crafting_table": 1}
        recipe_data = {}
        for material, num_needed in hypothesized_recipe.items():
            # material name should be checked.
            # 1. if material == item_name, that material should not be included in recipe_data
            # 2. if 'wood' in material, then recipe_data['logs'] += num_needed and recipe_data['planks'] += num_needed
            # 3. if self.oracle_graph.check_valid_item_name(material), then recipe_data[material] = num_needed
            # 4. if self._check_item_in_all_item_names(material), then recipe_data[material] = num_needed. This checks material in self.all_item_names
            # else: recipe_data[material] = num_needed.
                # So, recipe_data could include materials which are not in Minecraft. e.g. "saw", "hammer", "rope"
            
            pluar_form = material + "s" if not material.endswith("s") else material
            singular_form = material[:-1] if material.endswith("s") else material

            # 1. if material == item_name, that material should not be included in recipe_data
            if (material == item_name) or (pluar_form == item_name) or (singular_form == item_name):
                continue

            # 2. if 'wood' in material, then recipe_data['logs'] += num_needed and recipe_data['planks'] += num_needed
            if "wood" in material:
                if "logs" not in recipe_data:
                    recipe_data["logs"] = 0
                recipe_data["logs"] += num_needed
                if "planks" not in recipe_data:
                    recipe_data["planks"] = 0
                recipe_data["planks"] += num_needed
                continue
            if material == "coal":
                if "coals" not in recipe_data:
                    recipe_data["coals"] = 0
                recipe_data["coals"] += num_needed
                continue

            # 3. if self.oracle_graph.check_valid_item_name(material), then recipe_data[material] = num_needed
            is_true_item_name = self.oracle_graph.check_valid_item_name(material)
            is_true_item_name_plural = self.oracle_graph.check_valid_item_name(pluar_form)
            is_true_item_name_singular = self.oracle_graph.check_valid_item_name(singular_form)
            if is_true_item_name:
                if material not in recipe_data:
                    recipe_data[material] = 0
                recipe_data[material] += num_needed
                continue
            if is_true_item_name_plural:
                if pluar_form not in recipe_data:
                    recipe_data[pluar_form] = 0
                recipe_data[pluar_form] += num_needed
                continue
            if is_true_item_name_singular:
                if singular_form not in recipe_data:
                    recipe_data[singular_form] = 0
                recipe_data[singular_form] += num_needed
                continue

            # 4. if self._check_item_in_all_item_names(material), then recipe_data[material] = num_needed. This checks material in self.all_item_names
            is_seen_item_name = self._check_item_in_all_item_names(material)
            is_seen_item_name_plural = self._check_item_in_all_item_names(pluar_form)
            is_seen_item_name_singular = self._check_item_in_all_item_names(singular_form)
            if is_seen_item_name:
                if material not in recipe_data:
                    recipe_data[material] = 0
                recipe_data[material] += num_needed
                continue
            if is_seen_item_name_plural:
                if pluar_form not in recipe_data:
                    recipe_data[pluar_form] = 0
                recipe_data[pluar_form] += num_needed
                continue
            if is_seen_item_name_singular:
                if singular_form not in recipe_data:
                    recipe_data[singular_form] = 0
                recipe_data[singular_form] += num_needed
                continue

            # else: recipe_data[material] = num_needed. Add material to self.all_item_names
            recipe_data[material] = num_needed
            self.all_item_names.append(material)

        return recipe_data

    def _check_item_in_all_item_names(self, item):
        if item in self.all_item_names:
            return True
        return False

    def _prepare_item_name_embeddings(self):
        lst_verified_dir = os.listdir(self.verified_recipe_dir)
        for file_name in lst_verified_dir:
            if not file_name.endswith(".json"):
                continue

            item_name = file_name.replace(".json", "")
            if f"{item_name}.pt" in lst_verified_dir:
                continue

            item_embed = torch.tensor(self.bert_encoder.encode(item_name)).unsqueeze(0).to(self.device)
            torch.save(item_embed, os.path.join(self.verified_recipe_dir, f"{item_name}.pt"))
        
        lst_hypothesized_dir = os.listdir(self.hypothesized_recipe_dir)
        for file_name in lst_hypothesized_dir:
            if not file_name.endswith(".json"):
                continue

            item_name = file_name.replace(".json", "")
            if f"{item_name}.pt" in lst_hypothesized_dir:
                continue

            item_embed = torch.tensor(self.bert_encoder.encode(item_name)).unsqueeze(0).to(self.device)
            torch.save(item_embed, os.path.join(self.hypothesized_recipe_dir, f"{item_name}.pt"))


    def find_top_similar_verified_items(self, item_name, topK=3):

        sorted_verified_items = sorted(self.verified_item_names)
        verified_embedding_tensors = [torch.load(os.path.join(self.verified_recipe_dir, f'{name}.pt')) for name in sorted_verified_items]
        verified_embedding_matrix = torch.cat(verified_embedding_tensors, dim=0)
        verified_embedding_matrix = verified_embedding_matrix.to(self.device)

        item_embedding = torch.tensor(self.bert_encoder.encode(item_name)).unsqueeze(0).to(self.device)

        cosine_similarities = torch.matmul(verified_embedding_matrix, item_embedding.T).squeeze()

        topK_similarities, topK_indices = torch.topk(cosine_similarities, topK)

        topK_verified_items = [sorted_verified_items[i] for i in topK_indices.tolist()]
        
        return torch.mean(topK_similarities).item(), topK_verified_items
    
    def get_verified_item_names(self):
        return self.verified_item_names

    def find_frontiers(self):
        """
        API spec
            input: list of verified items, list of hypothesized items, recipes of hypothesized items
            output: self.frontier_item_names. (hypothesized items whose materials are all from the verified items)
        """
        # condition of frontier item
        # 1. self.graph[item_name]['ingredients'].keys() are all in self.verified_item_names
        # 2. self.graph[item_name]['ingredients'].keys() are empty

        self.frontier_item_names = []

        for hypo_item_name in self.hypothesized_item_names:
            material_names = list(self.graph[hypo_item_name]['ingredients'].keys())
            if len(material_names) == 0:
                self.frontier_item_names.append(hypo_item_name)
                continue

            material_set = set(material_names)
            verified_set = set(self.verified_item_names)
            if material_set.issubset(verified_set):
                self.frontier_item_names.append(hypo_item_name)

        self.frontier_item_names = list(set(self.frontier_item_names))
        return self.frontier_item_names


    def find_frontiers_related_to_goal(self, goal_item_name):
        """
        API spec
            input: goal_item_name, self.frontier_item_names
            output: list of frontier items related to the goal item (use compile() from relative_graph.py)
        
        """


        pass


    # def calculate_knowledge_all_frontiers(self):
    #     """
    #     API spec
    #         input: self.frontier_item_names, self.verified_item_names
    #         output: self.frontier_knowledge_score_dict
    #     """
    #     self.frontier_knowledge_score_dict = {}

    #     for item_name in self.frontier_item_names:
    #         self.frontier_knowledge_score_dict[item_name] = self._calculate_knowledge(item_name)

    #     return self.frontier_knowledge_score_dict

    def calculate_knowledge_all_hypothesized(self):
        self.knowledge_score_dict = {}

        for item_name in self.hypothesized_item_names:
            self.knowledge_score_dict[item_name] = self._calculate_knowledge(item_name)

        return self.knowledge_score_dict


    def _calculate_knowledge(self, item_name):
        """
        API spec
            input: item_name
            output: knowledge score
        """
        topK_similarity_score, topK_verified_items = self.find_top_similar_verified_items(item_name, topK=self.topK)
        return float(topK_similarity_score)
    
    def get_exploration_count_all_hypothesized(self):
        hypothesized_exploration_count_dict = {}
        
        for item_name in self.hypothesized_item_names:
            cnt = self.exploration_count_dict.get(item_name, 1)
            # should be at least 1
            hypothesized_exploration_count_dict[item_name] = max(cnt, 1)

        return hypothesized_exploration_count_dict

    # def get_exploration_count_all_frontiers(self):
    #     """
    #     API spec
    #         input: self.frontier_item_names, self.exploration_count_dict
    #         output: self.frontier_exploration_count_dict
    #     """
    #     self.frontier_exploration_count_dict = {}
        
    #     for item_name in self.frontier_item_names:
    #         cnt = self.exploration_count_dict.get(item_name, 1)
    #         # should be at least 1
    #         self.frontier_exploration_count_dict[item_name] = max(cnt, 1)

    #     return self.frontier_exploration_count_dict

    # TODO: calculate_level_all_frontiers with current inventory???

    # def calculate_level_all_frontiers(self):
    #     """
    #     API spec
    #         input: list of hypothesized items, self.graph
    #         output: self.level_dict

    #     Procedure:
    #         call calculate_path_dict_all_frontiers()
    #         count the number of elements from path_dict
    #     """
    #     self.calculate_path_dict_all_frontiers()

    #     self.frontier_level_dict = {}
    #     for frontier_item_name, path in self.frontier_path_dict.items():
    #         self.frontier_level_dict[frontier_item_name] = len(path)

    #     return self.frontier_level_dict

    def calculate_level_all_hypothesized(self):
        """
        API spec
            input: list of hypothesized items, self.graph
            output: self.level_dict

        Procedure:
            call calculate_path_dict_all_hypothesized()
            count the number of elements from path_dict
        """
        self.calculate_path_dict_all_hypothesized()

        self.level_dict = {}
        for hypothesized_item_name, path in self.path_dict.items():
            self.level_dict[hypothesized_item_name] = len(path)

        return self.level_dict

    # def calculate_path_dict_all_frontiers(self):
    #     """
    #     API spec
    #         input: list of hypothesized items, self.graph
    #         output: self.path_dict

    #     Procedure:
    #         use compile() from relative_graph.py
    #     """
    #     # {"wooden_pickaxe": [["logs", 4], ["planks": 16], ["stick", 8], ["crafting_table", 1], ["wooden_pickaxe", 1]]}
    #     self.frontier_path_dict = {}
    #     for item_name in self.frontier_item_names:
    #         self.frontier_path_dict[item_name] = self._calculate_path(item_name)

    #     return self.frontier_path_dict

    def calculate_path_dict_all_hypothesized(self):
        """
        API spec
            input: list of hypothesized items, self.graph
            output: self.path_dict

        Procedure:
            use compile() from relative_graph.py
        """
        self.path_dict = {}
        for item_name in self.hypothesized_item_names:
            self.path_dict[item_name] = self._calculate_path(item_name)

        return self.path_dict

    def _calculate_path(self, item_name):
        """
        API spec
            input: item_name, self.graph
            output: path

        Procedure:
            use compile() from relative_graph.py
        """
        # {"wooden_pickaxe": [["logs", 4], ["planks": 16], ["stick", 8], ["crafting_table", 1], ["wooden_pickaxe", 1]]}
        pretty_result, ordered_text, ordered_item, ordered_item_quantity \
            = self.compile(item_name, 1)

        return ordered_item_quantity

    def update_hypothesis(self, item_name):
        """
        API spec
            input: item_name, list of verified items, recipes of verified items
            output: hypothesized recipe of the item, and updated self.graph

        Procedure:
            read all hypotehsized recipe and update exploration counts, self.exploration_count_dict

            if exploration_count_dict[item_name] <= threshold: # threshold = 2 or 3
                find top-3 similar items, and their recipes
                recipe(item_name) = aggregate(recipes of top-3 similar items)
                update self.graph
            else:
                recipe(item_name) = {k: 8 for all k in self.crafting_resources}
                update self.graph
        """
        if item_name in self.verified_item_names:
            return
        if self.exploration_count_dict[item_name] <= 1:
            return

        self.logger.info(f"Update hypothesis of {item_name}")
        self.logger.info(f"Previous recipe of {item_name}: {self.graph[item_name]}")

        new_recipe = {}

        if self.exploration_count_dict[item_name] <= self.inadmissible_threshold:
            topK_similarity_score, topK_verified_items = self.find_top_similar_verified_items(item_name)
            for verified_item in topK_verified_items:
                recipe_data = self.get_recipe(verified_item)
                assert recipe_data['is_verified'], f"In update_hypothesis(), {verified_item} is not in verified items"

                for material, num_needed in recipe_data["ingredients"].items():
                    if material in ["crafting_table", "furnace"] or "pickaxe" in material or material == "coals" or material == "coal":
                        new_recipe[material] = 1
                    elif material in self.crafting_resources:
                        new_recipe[material] = 2 * self.exploration_count_dict[item_name]
                    else:
                        new_recipe[material] = 1

            # update self.graph and self.item_number_graph
            self.graph[item_name] = {
                "ingredients": new_recipe,
                "output_qty": 1,
                "is_verified": False
            }
            for material, num_needed in new_recipe.items():
                if material not in self.item_number_graph:
                    self.item_number_graph[material] = {}
                self.item_number_graph[material][item_name] = num_needed

        else:
            # inadmissible item. recipe(item_name) = {k: 8 for all k in self.crafting_resources}
            new_recipe = {}
            for crafting_resource in self.crafting_resources:
                new_recipe[crafting_resource] = 8

            # update self.graph and self.item_number_graph
            self.graph[item_name] = {
                "ingredients": new_recipe,
                "output_qty": 1,
                "is_verified": False
            }
            for crafting_resource, num_needed in new_recipe.items():
                if crafting_resource not in self.item_number_graph:
                    self.item_number_graph[crafting_resource] = {}
                self.item_number_graph[crafting_resource][item_name] = num_needed

        self.logger.info(f"New hypothesized recipe of {item_name}: {self.graph[item_name]}\n\n")

        new_hypothesized_recipe_data = {
            "item_name": item_name,
            "output_qty": 1,
            "ingredients": new_recipe,
            "required_pickaxe": 0,
            "is_crafting_resource": False,
            "exploration_count": self.exploration_count_dict[item_name],
        }
        self.recipe_revised_items.append(item_name)
        self._save_hypothesized_recipe_data(item_name, new_hypothesized_recipe_data)


    def _save_hypothesized_recipe_data(self, item_name, _recipe_data=None):
        json_file_name = f"{item_name}.json"
        if _recipe_data is None:
            with open(os.path.join(self.hypothesized_recipe_dir, json_file_name), "r+") as fp:
                fcntl.flock(fp, fcntl.LOCK_EX)

                recipe_data = json.load(fp)
                recipe_data["exploration_count"] = self.exploration_count_dict[item_name]

                fp.seek(0)
                fp.truncate()
                json.dump(recipe_data, fp, indent=2)
                
                fcntl.flock(fp, fcntl.LOCK_UN)
        else:
            recipe_data = copy.deepcopy(_recipe_data)
            with open(os.path.join(self.hypothesized_recipe_dir, json_file_name), "r+") as fp:
                fcntl.flock(fp, fcntl.LOCK_EX)

                fp.seek(0)
                fp.truncate()
                json.dump(recipe_data, fp, indent=2)
                
                fcntl.flock(fp, fcntl.LOCK_UN)

    def remove_verified_recipe(self, item_name):
        if item_name not in self.verified_item_names:
            return
        
        json_file_name = f"{item_name}.json"
        lst_dir = os.listdir(self.verified_recipe_dir)
        if json_file_name not in lst_dir:
            return
        
        os.remove(os.path.join(self.verified_recipe_dir, json_file_name))
        self.load_and_init_all_recipes()

    def increment_count(self, item_name, prefix):
        """
        API spec
            input: item_name
            output: self.exploration_count_dict

        Procedure:
            self.exploration_count_dict[item_name] += 1
            if self.exploration_count_dict[item_name] > inadmissible_threshold:
                spread_inadmissible(item_name)
        """
        if item_name in self.verified_item_names:
            if "ours" in prefix or "feasibility" in prefix or "frontier" in prefix:
                self.remove_verified_recipe(item_name)
                self.recipe_revised_items.append(item_name)
            elif "adam" in prefix or "deckard" in prefix:
                # Not update hypothesized recipe
                return

        # Read file and update exploration count, because there are many processes run parallelly.
        self._load_hypothesized_recipes()

        if item_name not in self.exploration_count_dict:
            self.exploration_count_dict[item_name] = 1

        self.exploration_count_dict[item_name] += 1
        if self.exploration_count_dict[item_name] > self.inadmissible_threshold:
            self.inadmissible_item_names.append(item_name)

        if "adam" in prefix or "deckard" in prefix:
            # only update exploration count
            # Not update hypothesized recipe
            self._save_hypothesized_recipe_data(item_name, None)
            return

        self.recipe_revised_items = []
        self.update_hypothesis(item_name)
        if self.exploration_count_dict[item_name] > self.inadmissible_threshold:
            self.spread_inadmissible(item_name)


    def spread_inadmissible(self, inadmissible_item_name):
        """
        API spec
            input: inadmissible_item_name, self.graph
            output: self.exploration_count_dict
        
        Procedure:
            for all unverified items which have inadmissible_item_name as part of their recipe:
                increment_count(item_name)
        """
        child_of_inadmissible = []
        for result_item_name, recipe in self.graph.items():
            if recipe["is_verified"]:
                continue
            if inadmissible_item_name in list(recipe["ingredients"].keys()):
                child_of_inadmissible.append(result_item_name)
        
        for item_name in child_of_inadmissible:
            self.exploration_count_dict[item_name] += 1
            self.update_hypothesis(item_name)

    def get_recipe_revised_items(self):
        return self.recipe_revised_items
    
    def reset_recipe_revised_items(self):
        self.recipe_revised_items = []

    def select_non_conflicting_goal(self, ordered_item_lists):
        json_file_name = "current_exploring_goals.json"
        json_path = os.path.join(self.recipe_base, json_file_name)

        with open(json_path, "r+") as fp:
            fcntl.flock(fp, fcntl.LOCK_EX)
            try:
                exploring_goal_dict = json.load(fp)
            except json.JSONDecodeError:
                exploring_goal_dict = {}

            # key is self.exploration_uuid, and value is goal item name of the run
            exploring_goal_items = list(exploring_goal_dict.values())
            
            current_goal = ordered_item_lists[0]
            for candidate_goal in ordered_item_lists:
                if candidate_goal not in exploring_goal_items:
                    current_goal = candidate_goal
                    break
            
            exploring_goal_dict[self.exploration_uuid] = current_goal
            fp.seek(0)
            fp.truncate()
            json.dump(exploring_goal_dict, fp, indent=2)
            fcntl.flock(fp, fcntl.LOCK_UN)

        self.logger.info(f"In select_non_conflicting_goal()\n")
        self.logger.info(f"exploring_goal_items: {exploring_goal_items}")
        self.logger.info(f"ordered_item_lists: {ordered_item_lists}\n")

        return current_goal

    def free_exploring_goal(self):
        json_file_name = "current_exploring_goals.json"
        json_path = os.path.join(self.recipe_base, json_file_name)

        with open(json_path, "r+") as fp:
            fcntl.flock(fp, fcntl.LOCK_EX)
            exploring_goal_dict = json.load(fp)

            if self.exploration_uuid in exploring_goal_dict:
                del exploring_goal_dict[self.exploration_uuid]

            fp.seek(0)
            fp.truncate()
            json.dump(exploring_goal_dict, fp, indent=2)
            fcntl.flock(fp, fcntl.LOCK_UN)

    def increment_num_episodes_save_memory(self):
        json_file_name = "track_num_episodes.json"
        json_path = os.path.join(self.recipe_base, json_file_name)
        total_episodes = 0

        with open(json_path, "r+") as fp:
            fcntl.flock(fp, fcntl.LOCK_EX)
            data = json.load(fp)

            data["num_episodes"] += 1
            total_episodes = data["num_episodes"]
            fp.seek(0)
            fp.truncate()
            json.dump(data, fp, indent=2)
            fcntl.flock(fp, fcntl.LOCK_UN)
        
        self.logger.info(f"Total episodes: {total_episodes}")
        if total_episodes in [49, 50, 51,
                              99, 100, 101,
                              149, 150, 151,
                              199, 200, 201,
                              249, 250, 251,
                              299, 300, 301,
                              349, 350, 351,
                              399, 400, 401,
                              449, 450, 451,
                              499, 500, 501,]:
            self._save_memory_snapshot(total_episodes)

    def _save_memory_snapshot(self, total_episodes):
        self.snapshot_base = self.cfg["memory"]["snapshot_base"]
        os.makedirs(self.snapshot_base, exist_ok=True)
        destination = os.path.join(self.snapshot_base, str(total_episodes).zfill(4))

        # Check if destination exists; if so, do not copy
        if os.path.exists(destination):
            self.logger.info(f"Destination folder {destination} already exists. Skipping copy.")
            return

        shutil.copytree(self.root_path, destination)
        self.logger.info(f"Copied {self.root_path} to {destination}")


    def save_verified_recipe_data(self, recipe_data):

        if recipe_data["item_name"] in self.verified_item_names:
            self.logger.info(f"Already verified item {recipe_data['item_name']}. Skip the recipe update.")
            # Skip the item if it is already verified.
            # self._save_verified_recipe_data(recipe_data)

            if recipe_data["is_crafting"]:
                ingredient_names = list(recipe_data["ingredients"].keys())
                for ingredient in ingredient_names:
                    if ingredient in ["crafting_table", "furnace"]:
                        continue
                    if ingredient not in self.crafting_resources:
                        self._update_crafing_resources(ingredient)

            return 0

        # From now, the item is newly verified item.
        self.logger.info(f"New verified item {recipe_data['item_name']}. Save {recipe_data} into new file")
        self._save_verified_recipe_data(recipe_data)

        if recipe_data["is_crafting"]:
            ingredient_names = list(recipe_data["ingredients"].keys())
            for ingredient in ingredient_names:
                if ingredient in ["crafting_table", "furnace"]:
                    continue
                if ingredient not in self.crafting_resources:
                    self._update_crafing_resources(ingredient)

        self.load_and_init_all_recipes()
        return 1


    def _save_verified_recipe_data(self, _recipe_data):
        recipe_data = copy.deepcopy(_recipe_data)
        del recipe_data['is_crafting']
        recipe_data['is_crafting_resource'] = False

        item_name = recipe_data['item_name']

        json_file_name = f"{item_name}.json"
        lst_dir = os.listdir(self.verified_recipe_dir)

        if json_file_name not in lst_dir:
            with open(os.path.join(self.verified_recipe_dir, json_file_name), "w+") as fp:
                fcntl.flock(fp, fcntl.LOCK_EX)

                fp.seek(0)
                fp.truncate()
                json.dump(recipe_data, fp, indent=2)

                fcntl.flock(fp, fcntl.LOCK_UN)

        # There is recipe already
        else:
            with open(os.path.join(self.verified_recipe_dir, json_file_name), "r+") as fp:
                fcntl.flock(fp, fcntl.LOCK_EX)
                previous_recipe_data = json.load(fp)

                # Sometimes, craftable item is obtained by mining. (e.g. stick)
                # In this case, we should update the recipe data.
                if len(previous_recipe_data['ingredients']) == 0 and len(recipe_data['ingredients']) > 0:
                    previous_recipe_data = recipe_data

                # previous_recipe_data['required_pickaxe'] = min(previous_recipe_data['required_pickaxe'],
                #                                                 recipe_data['required_pickaxe'])
                previous_recipe_data['is_crafting_resource'] = (previous_recipe_data['is_crafting_resource'] or recipe_data['is_crafting_resource'])

                fp.seek(0)
                fp.truncate()
                json.dump(previous_recipe_data, fp, indent=2)
                fcntl.flock(fp, fcntl.LOCK_UN)


    def _update_crafing_resources(self, item_name):
        json_file_name = f"{item_name}.json"

        with open(os.path.join(self.verified_recipe_dir, json_file_name), "r+") as fp:
            fcntl.flock(fp, fcntl.LOCK_EX)
            previous_recipe_data = json.load(fp)

            previous_recipe_data['is_crafting_resource'] = True

            fp.seek(0)
            fp.truncate()
            json.dump(previous_recipe_data, fp, indent=2)
            fcntl.flock(fp, fcntl.LOCK_UN)



    def _compile_steps(
        self,
        item: str,
        quantity=1,
        steps=None,
        summary=None,
        sub_graph=None,
        in_degree=None,
        cur_inventory=dict(),
    ):
        if steps is None:
            steps = []
        if summary is None:
            summary = {}
        if sub_graph is None:
            sub_graph = {}
        if in_degree is None:
            in_degree = {}

        if item not in sub_graph:
            sub_graph[item] = {}

        recipe = self.get_recipe(item)
        if recipe is None:
            return steps

        if item not in summary:
            summary[item] = 0

        if item in ["crafting_table", "furnace"] or "pickaxe" in item or item == "coals" or item == "coal":
            if item in cur_inventory:
                return steps
            if summary[item] == 1:
                return steps
            quantity = 1
            summary[item] = 1
        else:
            summary[item] += quantity

        if item not in in_degree:
            in_degree[item] = 0

        produced_qty = recipe["output_qty"]
        runs_needed = (quantity + produced_qty - 1) // produced_qty

        for ingredient, qty_needed in recipe["ingredients"].items():
            # if 'logs' in ingredient:
            #     ingredient = 'logs'
            # If the agent has the ingredient enough already, skip this ingredient.
            needed_num = qty_needed * runs_needed
            ingredient_having_num = cur_inventory.get(ingredient, 0)
            if ingredient_having_num >= needed_num:
                cur_inventory[ingredient] -= (0 if ingredient in ["crafting_table", "furnace"] else needed_num)
                continue

            in_degree[item] += 1
            if ingredient not in sub_graph:
                sub_graph[ingredient] = {}
            sub_graph[ingredient][item] = 1

            if ingredient_having_num:
                cur_inventory[ingredient] -= ingredient_having_num
            self._compile_steps(
                ingredient,
                qty_needed * runs_needed - ingredient_having_num,
                steps,
                summary,
                sub_graph,
                in_degree,
                cur_inventory,
            )

            if self.get_recipe(ingredient) is None: # base items such as cobblestone, diamond_ore, gold_ore
                if ingredient not in summary:
                    summary[ingredient] = 0
                summary[ingredient] += qty_needed * runs_needed - ingredient_having_num

        steps.append(
            f"Craft {runs_needed} {item} to make {runs_needed * produced_qty} using {self._pretty_dict(recipe['ingredients'])}"
        )

        return steps

    def _topo_sort(self, sub_graph, in_degree):
        from queue import Queue

        q = Queue()
        for k, v in in_degree.items():
            if v == 0 and k != "":
                q.put(k)
        res = []
        while not q.empty():
            cur = q.get()
            res.append(cur)
            for k in sub_graph[cur]:
                in_degree[k] -= 1
                if in_degree[k] == 0 and k != "":
                    q.put(k)
        return res

    def _pretty_dict(self, reicpe: dict):
        return ", ".join([f"{v} {k}" for k, v in reicpe.items()])

    def _pretty_result(
        self, _summary: dict, base: dict, sub_graph: dict, in_degree: dict | None = None
    ):
        if in_degree is None:
            in_degree = {}

        summary = {}
        for k, v in _summary.items():
            if v == 0:
                del sub_graph[k]
                continue
            summary[k] = v

        for cause, result in sub_graph.items():
            if cause not in in_degree:
                in_degree[cause] = 0
            for k in result:
                if k not in in_degree:
                    in_degree[k] = 0
                in_degree[k] += 1

        order = self._topo_sort(sub_graph, in_degree)

        res = [
            f"{idx+1}. {item.lower()}: need {summary[item] if summary[item] > 0 else '??'}"
            for idx, item in enumerate(order)
        ]

        # [["logs", 4], ["planks": 16], ["stick", 8], ["crafting_table", 1], ["wooden_pickaxe", 1]],
        ordered_item_quantity = [
            [item, summary[item]]
            for idx, item in enumerate(order)
        ]

        return "\n".join(res), res, order, ordered_item_quantity

    def compile(self, item: str, number: int = 1, __cur_inventory: dict = dict()):
        self.sub_graph = {}
        self._in_degree = {}
        self.cur_inventory = {}
        summary = {}

        cur_inventory = dict()

        for k, v in __cur_inventory.items():
            if k.endswith('_planks'):
                if 'planks' not in cur_inventory:
                    cur_inventory['planks'] = 0
                cur_inventory['planks'] += v

            elif k.endswith('_log'):
                if 'logs' not in cur_inventory:
                    cur_inventory['logs'] = 0
                cur_inventory['logs'] += v
            
            elif k == 'coal':
                cur_inventory['coals'] = v
                cur_inventory['coal'] = v
            else:
                cur_inventory[k] = v

        self.cur_inventory = cur_inventory

        self._compile_steps(
            item,
            number,
            summary=summary,
            sub_graph=self.sub_graph,
            in_degree=self._in_degree,
            cur_inventory=copy.deepcopy(self.cur_inventory),
        )

        if self.get_recipe(item) is not None and len(summary) == 1:
            # all conditions are met to make the item
            print(f'All conditions are met. Just craft it!')

            pretty_result, ordered_text, ordered_item, ordered_item_quantity = self._pretty_result(summary, dict(), self.sub_graph)
            return f"craft {number} {item} summary:\n" + pretty_result + "\n", ordered_text, ordered_item, ordered_item_quantity

        # If summary is empty, it means the item is collectable without any tools. get_recipe() returns None.
        if len(summary) == 0:
            summary[item] = number

        pretty_result, ordered_text, ordered_item, ordered_item_quantity = self._pretty_result(summary, dict(), self.sub_graph)
        return f"craft {number} {item} summary:\n" + pretty_result + "\n", ordered_text, ordered_item, ordered_item_quantity


# Cycle check
# for final_target_item in hypothesized_item_names:
#     global_needed_items = []
#     global_needed_items.append(final_target_item)

#     def check_cycle(target):
#         # print(f'check_cycle. {target}')
#         ingredients = graph[target]['ingredients']
#         if len(ingredients.keys()) == 0:
#             return
    
#         for k in ingredients.keys():
#             if k == final_target_item:
#                 print(f"final_target_item: {final_target_item}")
#                 print(ingredients)
#                 print(global_needed_items)
#                 print()
    
#                 return
#             if k in global_needed_items:
#                 continue
#             global_needed_items.append(k)
#             check_cycle(k)
    
    
#     check_cycle(final_target_item)
#     # print(final_target_item)
#     # print(global_needed_items)
#     # print()
