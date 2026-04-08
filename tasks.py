from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

from env import UrbanRescueEnv


@dataclass(frozen=True)
class TaskSpec:
    level: str
    description: str
    max_steps: int
    rescue_goal: int


TASK_SPECS: Dict[str, TaskSpec] = {
    "easy": TaskSpec(
        level="easy",
        description="Rescue 3 civilians in a low-intensity disaster zone.",
        max_steps=30,
        rescue_goal=3,
    ),
    "medium": TaskSpec(
        level="medium",
        description="Rescue all civilians before the mission timer expires.",
        max_steps=40,
        rescue_goal=4,
    ),
    "hard": TaskSpec(
        level="hard",
        description="Rescue at least 3 civilians while managing supplies and hazard control in a dense disaster zone.",
        max_steps=55,
        rescue_goal=4,
    ),
}


def create_task(level: str) -> UrbanRescueEnv:
    normalized = level.lower()
    if normalized not in TASK_SPECS:
        raise ValueError(f"Unsupported level: {level}")

    spec = TASK_SPECS[normalized]
    return UrbanRescueEnv(
        level=spec.level,
        max_steps=spec.max_steps,
        rescue_goal=spec.rescue_goal,
    )


def list_tasks() -> Dict[str, TaskSpec]:
    return dict(TASK_SPECS)
