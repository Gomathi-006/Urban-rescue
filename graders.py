from __future__ import annotations

from typing import Dict


def _safe_ratio(numerator: float, denominator: float) -> float:
    if denominator <= 0:
        return 0.0
    return max(0.0, min(1.0, numerator / denominator))


def _clamp_score(score: float) -> float:
    return max(0.0, min(1.0, score))


def grade_easy(final_state: Dict[str, object]) -> float:
    metrics = final_state["metrics"]
    civilians = final_state["civilians"]
    rescued_ratio = _safe_ratio(metrics["rescued"], 3)
    efficiency = 1.0 - _safe_ratio(final_state["step"], final_state["max_steps"])
    death_penalty = _safe_ratio(metrics["deaths"], max(1, len(civilians)))
    return _clamp_score((0.7 * rescued_ratio) + (0.3 * efficiency) - (0.2 * death_penalty))


def grade_medium(final_state: Dict[str, object]) -> float:
    metrics = final_state["metrics"]
    civilians = final_state["civilians"]
    total = len(civilians)
    rescued_ratio = _safe_ratio(metrics["rescued"], total)
    on_time = 1.0 if final_state["step"] <= final_state["max_steps"] else 0.0
    hazard_control = _safe_ratio(metrics["hazards_extinguished"], max(1, total))
    return _clamp_score((0.6 * rescued_ratio) + (0.2 * on_time) + (0.2 * hazard_control))


def grade_hard(final_state: Dict[str, object]) -> float:
    metrics = final_state["metrics"]
    civilians = final_state["civilians"]
    total = len(civilians)
    survival_ratio = _safe_ratio(metrics["rescued"], total)
    resource_efficiency = _safe_ratio(metrics["supplies_delivered"] + metrics["coordination_events"], total)
    time_efficiency = 1.0 - _safe_ratio(final_state["step"], final_state["max_steps"])
    loss_penalty = _safe_ratio(metrics["deaths"], total)
    return _clamp_score(
        (0.5 * survival_ratio)
        + (0.25 * resource_efficiency)
        + (0.25 * time_efficiency)
        - (0.3 * loss_penalty)
    )


def grade_level(level: str, final_state: Dict[str, object]) -> float:
    normalized = level.lower()
    if normalized == "easy":
        return grade_easy(final_state)
    if normalized == "medium":
        return grade_medium(final_state)
    if normalized == "hard":
        return grade_hard(final_state)
    raise ValueError(f"Unsupported level: {level}")
