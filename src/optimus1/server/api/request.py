from enum import Enum
from typing import Any, List, Dict

from pydantic import BaseModel


class MCRequest(BaseModel):
    rgb_images: List[Any]

    done_imgs: List[Any] | None = None
    cont_imgs: List[Any] | None = None
    replan_imgs: List[Any] | None = None

    task_or_instruction: str
    goal: str | None = None

    current_step: int = 0

    history: List[str] | None = None

    temperature: float = 0.2
    system_prompt: str | None = None

    type: str | None = None  # plan|action|replan|reflection
    error_info: str | None = None
    example: str | None = None
    graph: str | None = None
    visual_info: str | None = None

    waypoint: str | None = None
    similar_wp_sg_dict: Dict[Any, Any] | None = None
    failed_sg_list_for_wp: List[str] | None = None

    hydra_path: str | None = None
    run_uuid: str | None = None
    errorneous_planning: str | None = None

class MCResponse(BaseModel):
    response: Any = None

    status_code: int = 200
    message: str = ""

    appendix: Any | None = None