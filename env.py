from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional, Tuple

try:
    from transformers import pipeline
except Exception:  # pragma: no cover - transformers is optional at runtime
    pipeline = None


Position = Tuple[int, int]


def _clamp(value: int, minimum: int, maximum: int) -> int:
    return max(minimum, min(value, maximum))


def _manhattan(a: Position, b: Position) -> int:
    return abs(a[0] - b[0]) + abs(a[1] - b[1])


@dataclass
class Civilian:
    civilian_id: str
    position: Position
    health: int = 100
    rescued: bool = False
    supplied: bool = False

    def as_dict(self) -> Dict[str, object]:
        return {
            "id": self.civilian_id,
            "position": self.position,
            "health": self.health,
            "rescued": self.rescued,
            "supplied": self.supplied,
        }


@dataclass
class Hazard:
    position: Position
    intensity: int = 1
    kind: str = "fire"

    def as_dict(self) -> Dict[str, object]:
        return {
            "position": self.position,
            "intensity": self.intensity,
            "kind": self.kind,
        }


@dataclass
class AgentUnit:
    name: str
    role: str
    position: Position
    inventory: int = 0
    max_inventory: int = 0
    rescued_count: int = 0

    def as_dict(self) -> Dict[str, object]:
        return {
            "name": self.name,
            "role": self.role,
            "position": self.position,
            "inventory": self.inventory,
            "max_inventory": self.max_inventory,
            "rescued_count": self.rescued_count,
        }


class HFAgentPolicy:
    """Action policy backed by a Hugging Face pipeline with heuristic fallback and loss learning."""

    ACTIONS = (
        "move_up",
        "move_down",
        "move_left",
        "move_right",
        "stay",
        "rescue",
        "extinguish",
        "pickup_supply",
        "deliver_supply",
    )

    def __init__(
        self,
        role: str,
        model_name: str = "google/flan-t5-small",
        task: str = "text2text-generation",
    ) -> None:
        self.role = role
        self.model_name = model_name
        self.task = task
        self._pipeline = None
        self._pipeline_error: Optional[str] = None
        self.action_scores: Dict[str, Dict[str, float]] = {}
        self.episode_memory: List[Dict[str, object]] = []
        self.games_played = 0
        self.wins = 0
        self.losses = 0
        self.strategy_biases = self._initial_strategy_biases()

    def _ensure_pipeline(self) -> None:
        if self._pipeline is not None or self._pipeline_error is not None:
            return
        if pipeline is None:
            self._pipeline_error = "transformers is not installed"
            return
        try:
            self._pipeline = pipeline(self.task, model=self.model_name)
        except Exception as exc:  # pragma: no cover - depends on local HF setup
            self._pipeline_error = str(exc)

    def decide_action(self, env_state: Dict[str, object]) -> str:
        self._ensure_pipeline()
        heuristic = self._heuristic_action(env_state)
        context = self._context_key(env_state)
        if self._pipeline is None:
            return self._select_action(context, heuristic, env_state)

        prompt = self._build_prompt(env_state, heuristic)
        try:
            result = self._pipeline(prompt, max_new_tokens=8)
            if not result:
                return self._select_action(context, heuristic, env_state)
            generated = result[0].get("generated_text") or result[0].get("summary_text") or ""
            action = self._parse_action(generated)
            candidate = action or heuristic
            return self._select_action(context, candidate, env_state)
        except Exception:  # pragma: no cover - depends on local HF runtime
            return self._select_action(context, heuristic, env_state)

    def _build_prompt(self, env_state: Dict[str, object], heuristic: str) -> str:
        agents = env_state["agents"]
        self_agent = agents[self.role]
        my_pos = tuple(self_agent["position"])
        civilians = [c for c in env_state["civilians"] if not c["rescued"] and c["health"] > 0]
        hazards = env_state["hazards"]
        nearest_civilian = min((_manhattan(my_pos, tuple(c["position"])) for c in civilians), default=-1)
        nearest_fire = min((_manhattan(my_pos, tuple(h["position"])) for h in hazards), default=-1)
        return (
            f"You control the {self.role} agent in an urban rescue grid world. "
            f"Choose exactly one action from {', '.join(self.ACTIONS)}. "
            f"Prefer safe and cooperative behavior. "
            f"Role={self.role}. "
            f"Nearest civilian distance={nearest_civilian}. "
            f"Nearest fire distance={nearest_fire}. "
            f"Civilians remaining={len(civilians)}. "
            f"Hazards active={len(hazards)}. "
            f"Inventory={self_agent['inventory']}. "
            f"If unsure, use {heuristic}. Action:"
        )

    def _parse_action(self, text: str) -> Optional[str]:
        lower = text.lower()
        for action in self.ACTIONS:
            if action in lower:
                return action
        return None

    def _heuristic_action(self, env_state: Dict[str, object]) -> str:
        agents = env_state["agents"]
        self_agent = agents[self.role]
        my_pos = tuple(self_agent["position"])

        civilians = [c for c in env_state["civilians"] if not c["rescued"] and c["health"] > 0]
        hazards = env_state["hazards"]
        depots = [tuple(pos) for pos in env_state["resource_depots"]]
        safe_zone = tuple(env_state["safe_zone"])
        fire_agent_pos = tuple(agents["fire_control"]["position"])

        if self.role == "rescue":
            same_cell_civilian = next((c for c in civilians if tuple(c["position"]) == my_pos), None)
            if same_cell_civilian:
                return "rescue"
            if civilians:
                target = min(
                    civilians,
                    key=lambda c: self._rescue_priority(my_pos, c, hazards, fire_agent_pos),
                )
                return self._step_toward(my_pos, tuple(target["position"]))
            return self._step_toward(my_pos, safe_zone)

        if self.role == "fire_control":
            same_cell_hazard = next((h for h in hazards if tuple(h["position"]) == my_pos), None)
            if same_cell_hazard:
                return "extinguish"
            if hazards:
                priority_civilians = self._priority_civilians(civilians, hazards)
                target = min(hazards, key=lambda h: self._fire_priority(h, priority_civilians, my_pos))
                return self._step_toward(my_pos, tuple(target["position"]))
            return "stay"

        if self.role == "supply":
            critical_needy = [c for c in civilians if not c["supplied"] and self._is_supply_priority(c, hazards)]

            if self_agent["inventory"] <= 0 and depots:
                if my_pos in depots:
                    return "pickup_supply"
                depot = min(depots, key=lambda p: _manhattan(my_pos, p))
                return self._step_toward(my_pos, depot)

            if my_pos in depots and self_agent["inventory"] < self_agent["max_inventory"] and critical_needy:
                return "pickup_supply"

            needy = [c for c in civilians if not c["supplied"] and self._is_supply_priority(c, hazards)]
            same_cell_needy = next((c for c in needy if tuple(c["position"]) == my_pos), None)
            if same_cell_needy and self_agent["inventory"] > 0:
                return "deliver_supply"
            if needy:
                target = min(
                    needy,
                    key=lambda c: self._supply_priority(my_pos, c, hazards),
                )
                return self._step_toward(my_pos, tuple(target["position"]))
            if civilians and self_agent["inventory"] > 0:
                target = min(
                    civilians,
                    key=lambda c: self._support_priority(my_pos, c, hazards),
                )
                return self._step_toward(my_pos, tuple(target["position"]))
            return "stay"

        return "stay"

    def record_decision(self, env_state: Dict[str, object], action: str) -> None:
        self.episode_memory.append(
            {
                "context": self._context_key(env_state),
                "action": action,
                "expected": self._heuristic_action(env_state),
            }
        )

    def observe_outcome(self, result: str, final_state: Dict[str, object]) -> None:
        self.games_played += 1
        if result == "won":
            self.wins += 1
            delta = 0.35
        else:
            self.losses += 1
            delta = -0.2
            self._learn_from_loss(final_state)

        role_adjustment = self._role_adjustment(final_state, result)
        for memory in self.episode_memory:
            context_scores = self.action_scores.setdefault(memory["context"], {})
            action = str(memory["action"])
            applied = self._memory_adjustment(memory, delta + role_adjustment)
            context_scores[action] = context_scores.get(action, 0.0) + applied

        self.episode_memory.clear()

    def learning_summary(self) -> Dict[str, object]:
        top_adjustments: List[str] = []
        top_adjustments.extend(self._strategy_lessons())
        for context, scores in sorted(
            self.action_scores.items(),
            key=lambda item: max((abs(value) for value in item[1].values()), default=0.0),
            reverse=True,
        ):
            if len(top_adjustments) >= 3:
                break
            if not scores:
                continue
            action, value = max(scores.items(), key=lambda item: abs(item[1]))
            direction = "prefer" if value >= 0 else "avoid"
            top_adjustments.append(f"{context}: {direction} {action}")
        return {
            "games_played": self.games_played,
            "wins": self.wins,
            "losses": self.losses,
            "top_adjustments": top_adjustments,
        }

    def _select_action(self, context: str, candidate: str, env_state: Dict[str, object]) -> str:
        scores = self.action_scores.get(context, {})
        candidate_actions = self._candidate_actions(env_state)
        best_action = candidate
        best_score = scores.get(candidate, 0.0) + self._strategy_bonus(candidate, env_state)

        for action in candidate_actions:
            action_score = scores.get(action, 0.0) + self._strategy_bonus(action, env_state)
            if action_score > best_score + 0.25:
                best_action = action
                best_score = action_score

        return best_action

    def _initial_strategy_biases(self) -> Dict[str, float]:
        if self.role == "rescue":
            return {
                "urgent_civilian": 0.0,
                "hazard_avoidance": 0.0,
                "avoid_idle": 0.0,
            }
        if self.role == "fire_control":
            return {
                "urgent_fire": 0.0,
                "protect_civilians": 0.0,
                "avoid_idle": 0.0,
            }
        return {
            "critical_support": 0.0,
            "stock_supplies": 0.0,
            "avoid_idle": 0.0,
        }

    def _learn_from_loss(self, final_state: Dict[str, object]) -> None:
        metrics = final_state["metrics"]
        civilians = final_state["civilians"]
        hazards = final_state["hazards"]
        step = int(final_state["step"])
        max_steps = int(final_state["max_steps"])
        timed_out = step >= max_steps and not self._goal_from_state(final_state)
        vulnerable_waiting = sum(
            1
            for civilian in civilians
            if not civilian["rescued"]
            and civilian["health"] > 0
            and (civilian["health"] <= 55 or self._hazard_distance(tuple(civilian["position"]), hazards) <= 2)
        )
        active_hazards = len(hazards)

        if self.role == "rescue":
            self._increase_bias("urgent_civilian", 0.25 + (0.12 * metrics["deaths"]) + (0.08 if timed_out else 0.0))
            if metrics["deaths"] > 0:
                self._increase_bias("hazard_avoidance", 0.18 + (0.05 * metrics["deaths"]))
            if timed_out or vulnerable_waiting:
                self._increase_bias("avoid_idle", 0.14)
            return

        if self.role == "fire_control":
            if active_hazards or metrics["deaths"] > 0:
                self._increase_bias("urgent_fire", 0.2 + (0.05 * active_hazards) + (0.08 * metrics["deaths"]))
            if vulnerable_waiting or metrics["deaths"] > 0:
                self._increase_bias("protect_civilians", 0.22 + (0.04 * vulnerable_waiting))
            if timed_out:
                self._increase_bias("avoid_idle", 0.12)
            return

        if vulnerable_waiting or metrics["deaths"] > 0:
            self._increase_bias("critical_support", 0.2 + (0.08 * vulnerable_waiting) + (0.05 * metrics["deaths"]))
        if metrics["supplies_delivered"] == 0 or timed_out:
            self._increase_bias("stock_supplies", 0.18)
        if timed_out or vulnerable_waiting:
            self._increase_bias("avoid_idle", 0.12)

    def _goal_from_state(self, state: Dict[str, object]) -> bool:
        return int(state["metrics"]["rescued"]) >= int(state.get("rescue_goal", self._infer_rescue_goal(state)))

    def _infer_rescue_goal(self, state: Dict[str, object]) -> int:
        civilians = state["civilians"]
        if state["level"] == "hard":
            return min(3, len(civilians))
        return len(civilians)

    def _increase_bias(self, key: str, amount: float) -> None:
        self.strategy_biases[key] = min(1.5, self.strategy_biases.get(key, 0.0) + amount)

    def _strategy_lessons(self) -> List[str]:
        lessons: List[str] = []
        if self.role == "rescue":
            if self.strategy_biases["urgent_civilian"] >= 0.2:
                lessons.append("loss learning: prioritize the most threatened civilians sooner")
            if self.strategy_biases["hazard_avoidance"] >= 0.2:
                lessons.append("loss learning: avoid rescues that leave civilians exposed to nearby fire")
        elif self.role == "fire_control":
            if self.strategy_biases["urgent_fire"] >= 0.2:
                lessons.append("loss learning: clear active fires faster after failed missions")
            if self.strategy_biases["protect_civilians"] >= 0.2:
                lessons.append("loss learning: focus fires closest to weak civilians")
        else:
            if self.strategy_biases["critical_support"] >= 0.2:
                lessons.append("loss learning: deliver supplies earlier to high-risk civilians")
            if self.strategy_biases["stock_supplies"] >= 0.2:
                lessons.append("loss learning: restock sooner when support runs short")

        if self.strategy_biases.get("avoid_idle", 0.0) >= 0.2:
            lessons.append("loss learning: avoid wasting turns when objectives remain")
        return lessons[:3]

    def _strategy_bonus(self, action: str, env_state: Dict[str, object]) -> float:
        agents = env_state["agents"]
        self_agent = agents[self.role]
        my_pos = tuple(self_agent["position"])
        civilians = [c for c in env_state["civilians"] if not c["rescued"] and c["health"] > 0]
        hazards = env_state["hazards"]
        depots = [tuple(pos) for pos in env_state["resource_depots"]]
        objectives_remaining = bool(civilians or hazards)
        bonus = 0.0

        if action == "stay" and objectives_remaining:
            bonus -= 0.45 * self.strategy_biases.get("avoid_idle", 0.0)

        if self.role == "rescue":
            same_cell_civilian = any(tuple(c["position"]) == my_pos for c in civilians)
            if same_cell_civilian and action == "rescue":
                bonus += 0.55 * self.strategy_biases["urgent_civilian"]
        elif self.role == "fire_control":
            same_cell_hazard = any(tuple(h["position"]) == my_pos for h in hazards)
            if same_cell_hazard and action == "extinguish":
                bonus += 0.6 * self.strategy_biases["urgent_fire"]
        else:
            needy = [c for c in civilians if not c["supplied"] and self._is_supply_priority(c, hazards)]
            same_cell_needy = any(tuple(c["position"]) == my_pos for c in needy)
            if same_cell_needy and action == "deliver_supply":
                bonus += 0.55 * self.strategy_biases["critical_support"]
            if my_pos in depots and action == "pickup_supply":
                bonus += 0.45 * self.strategy_biases["stock_supplies"]

        if action not in self.ACTIONS:
            return -1.0
        return bonus

    def _candidate_actions(self, env_state: Dict[str, object]) -> List[str]:
        heuristic = self._heuristic_action(env_state)
        actions = [heuristic, "stay"]
        agents = env_state["agents"]
        self_agent = agents[self.role]
        my_pos = tuple(self_agent["position"])

        civilians = [c for c in env_state["civilians"] if not c["rescued"] and c["health"] > 0]
        hazards = env_state["hazards"]
        depots = [tuple(pos) for pos in env_state["resource_depots"]]

        if self.role == "rescue":
            if any(tuple(c["position"]) == my_pos for c in civilians):
                actions.append("rescue")
            fire_agent_pos = tuple(agents["fire_control"]["position"])
            target = min(
                civilians,
                key=lambda c: self._rescue_priority(my_pos, c, hazards, fire_agent_pos),
                default=None,
            )
            if target is not None:
                actions.extend(self._movement_options_toward(my_pos, tuple(target["position"])))

        elif self.role == "fire_control":
            if any(tuple(h["position"]) == my_pos for h in hazards):
                actions.append("extinguish")
            if hazards:
                priority_civilians = self._priority_civilians(civilians, hazards)
                target = min(hazards, key=lambda h: self._fire_priority(h, priority_civilians, my_pos))
                actions.extend(self._movement_options_toward(my_pos, tuple(target["position"])))

        elif self.role == "supply":
            if my_pos in depots:
                actions.append("pickup_supply")
            needy = [c for c in civilians if not c["supplied"] and self._is_supply_priority(c, hazards)]
            if any(tuple(c["position"]) == my_pos for c in needy):
                actions.append("deliver_supply")
            target = min(
                needy or civilians,
                key=lambda c: self._support_priority(my_pos, c, hazards),
                default=None,
            )
            if target is not None:
                actions.extend(self._movement_options_toward(my_pos, tuple(target["position"])))

        deduped: List[str] = []
        for action in actions:
            if action in self.ACTIONS and action not in deduped:
                deduped.append(action)
        return deduped or [candidate for candidate in self.ACTIONS if candidate == "stay"]

    def _movement_options_toward(self, start: Position, target: Position) -> List[str]:
        options: List[str] = []
        dx = target[0] - start[0]
        dy = target[1] - start[1]
        if dx < 0:
            options.append("move_up")
        elif dx > 0:
            options.append("move_down")
        if dy < 0:
            options.append("move_left")
        elif dy > 0:
            options.append("move_right")
        return options

    def _context_key(self, env_state: Dict[str, object]) -> str:
        agents = env_state["agents"]
        self_agent = agents[self.role]
        my_pos = tuple(self_agent["position"])
        civilians = [c for c in env_state["civilians"] if not c["rescued"] and c["health"] > 0]
        hazards = env_state["hazards"]
        fire_density = min(len(hazards), 3)
        civilians_left = min(len(civilians), 5)

        on_hazard = any(tuple(h["position"]) == my_pos for h in hazards)
        nearest_hazard = min((_manhattan(my_pos, tuple(h["position"])) for h in hazards), default=9)
        nearest_civilian = min((_manhattan(my_pos, tuple(c["position"])) for c in civilians), default=9)
        low_health_waiting = sum(1 for c in civilians if c["health"] <= 40)

        if self.role == "rescue":
            same_cell_civilian = any(tuple(c["position"]) == my_pos for c in civilians)
            return (
                f"rescue|same_cell={int(same_cell_civilian)}|nearest_civ={min(nearest_civilian, 4)}"
                f"|near_fire={min(nearest_hazard, 4)}|low_hp={min(low_health_waiting, 2)}"
                f"|fire_density={fire_density}|civ_left={civilians_left}"
            )

        if self.role == "fire_control":
            hottest_fire = max((h["intensity"] for h in hazards), default=0)
            return (
                f"fire|on_fire={int(on_hazard)}|nearest_fire={min(nearest_hazard, 4)}"
                f"|hottest={hottest_fire}|low_hp={min(low_health_waiting, 2)}"
                f"|fire_density={fire_density}|civ_left={civilians_left}"
            )

        return (
            f"supply|inventory={min(int(self_agent['inventory']), 2)}|nearest_civ={min(nearest_civilian, 4)}"
            f"|nearest_fire={min(nearest_hazard, 4)}|low_hp={min(low_health_waiting, 2)}"
            f"|fire_density={fire_density}|civ_left={civilians_left}"
        )

    def _role_adjustment(self, final_state: Dict[str, object], result: str) -> float:
        metrics = final_state["metrics"]
        civilians = final_state["civilians"]

        if self.role == "rescue":
            total = max(1, len(civilians))
            return (metrics["rescued"] / total) * 0.35 if result == "won" else -(metrics["deaths"] / total) * 0.4

        if self.role == "fire_control":
            total = max(1, len(civilians))
            return (
                (metrics["hazards_extinguished"] / total) * 0.3
                if result == "won"
                else -(metrics["deaths"] / total) * 0.25
            )

        total = max(1, len(civilians))
        return (
            (metrics["supplies_delivered"] / total) * 0.25
            if result == "won"
            else -(metrics["deaths"] / total) * 0.2
        )

    def _memory_adjustment(self, memory: Dict[str, object], episode_delta: float) -> float:
        action = str(memory["action"])
        expected = str(memory["expected"])
        if action == expected:
            return episode_delta * 0.35 if episode_delta < 0 else episode_delta * 1.1
        if action == "stay" and expected != "stay":
            return episode_delta * 1.4
        return episode_delta

    def _step_toward(self, start: Position, target: Position) -> str:
        dx = target[0] - start[0]
        dy = target[1] - start[1]
        if abs(dx) > abs(dy):
            return "move_down" if dx > 0 else "move_up"
        if dy != 0:
            return "move_right" if dy > 0 else "move_left"
        if dx != 0:
            return "move_down" if dx > 0 else "move_up"
        return "stay"

    def _hazard_distance(self, position: Position, hazards: List[Dict[str, object]]) -> int:
        return min((_manhattan(position, tuple(h["position"])) for h in hazards), default=9)

    def _hazard_pressure(self, position: Position, hazards: List[Dict[str, object]]) -> int:
        on_hazard = next((h for h in hazards if tuple(h["position"]) == position), None)
        if on_hazard is not None:
            return 20 + (5 * int(on_hazard["intensity"]))
        if any(_manhattan(position, tuple(h["position"])) == 1 for h in hazards):
            return 10
        return 2

    def _priority_civilians(
        self,
        civilians: List[Dict[str, object]],
        hazards: List[Dict[str, object]],
    ) -> List[Dict[str, object]]:
        ranked = sorted(
            civilians,
            key=lambda c: (
                self._hazard_distance(tuple(c["position"]), hazards),
                int(c["health"]),
            ),
        )
        return ranked[:2] if ranked else []

    def _rescue_priority(
        self,
        my_pos: Position,
        civilian: Dict[str, object],
        hazards: List[Dict[str, object]],
        fire_agent_pos: Position,
    ) -> Tuple[int, int, int, int]:
        civ_pos = tuple(civilian["position"])
        distance = _manhattan(my_pos, civ_pos)
        hazard_pressure = self._hazard_pressure(civ_pos, hazards)
        fire_support_distance = _manhattan(fire_agent_pos, civ_pos)
        survival_margin = int(civilian["health"]) - (hazard_pressure * max(1, distance))
        urgency_bias = self.strategy_biases.get("urgent_civilian", 0.0)
        hazard_bias = self.strategy_biases.get("hazard_avoidance", 0.0)
        return (
            distance * 8
            - min(20 + int(8 * hazard_bias), hazard_pressure)
            - min(12 + int(12 * urgency_bias), max(0, 75 - int(civilian["health"]))),
            -survival_margin - int(10 * urgency_bias),
            distance + min(fire_support_distance, 2),
            self._hazard_distance(civ_pos, hazards),
        )

    def _fire_priority(
        self,
        hazard: Dict[str, object],
        civilians: List[Dict[str, object]],
        my_pos: Position,
    ) -> Tuple[int, int, int, int]:
        hazard_pos = tuple(hazard["position"])
        nearest_civilian = min((_manhattan(hazard_pos, tuple(c["position"])) for c in civilians), default=9)
        weakest_nearby = min(
            (int(c["health"]) for c in civilians if _manhattan(hazard_pos, tuple(c["position"])) <= 2),
            default=101,
        )
        urgent_fire_bias = self.strategy_biases.get("urgent_fire", 0.0)
        protect_bias = self.strategy_biases.get("protect_civilians", 0.0)
        return (
            nearest_civilian - int(2 * protect_bias),
            weakest_nearby - int(16 * protect_bias),
            _manhattan(my_pos, hazard_pos),
            -int(hazard["intensity"]) - int(2 * urgent_fire_bias) - max(0, 4 - nearest_civilian),
        )

    def _is_supply_priority(self, civilian: Dict[str, object], hazards: List[Dict[str, object]]) -> bool:
        civ_pos = tuple(civilian["position"])
        hazard_distance = self._hazard_distance(civ_pos, hazards)
        support_bias = self.strategy_biases.get("critical_support", 0.0)
        return int(civilian["health"]) <= 95 + int(8 * support_bias) or hazard_distance <= 3

    def _supply_priority(
        self,
        my_pos: Position,
        civilian: Dict[str, object],
        hazards: List[Dict[str, object]],
    ) -> Tuple[int, int, int]:
        civ_pos = tuple(civilian["position"])
        hazard_pressure = self._hazard_pressure(civ_pos, hazards)
        distance = _manhattan(my_pos, civ_pos)
        survival_margin = int(civilian["health"]) - (hazard_pressure * max(1, distance))
        support_bias = self.strategy_biases.get("critical_support", 0.0)
        return (
            distance * 7
            - min(20 + int(8 * support_bias), hazard_pressure)
            - min(20 + int(12 * support_bias), max(0, 78 - int(civilian["health"]))),
            -survival_margin - int(10 * support_bias),
            distance,
        )

    def _support_priority(
        self,
        my_pos: Position,
        civilian: Dict[str, object],
        hazards: List[Dict[str, object]],
    ) -> Tuple[int, int, int]:
        civ_pos = tuple(civilian["position"])
        return (
            _manhattan(my_pos, civ_pos),
            self._hazard_distance(civ_pos, hazards),
            int(civilian["health"]),
        )


@dataclass
class UrbanRescueEnv:
    width: int = 7
    height: int = 7
    max_steps: int = 40
    level: str = "easy"
    random_seed: Optional[int] = None
    rescue_goal: int = 3
    total_supplies: int = 6
    safe_zone: Position = (0, 0)
    resource_depots: List[Position] = field(default_factory=lambda: [(0, 6), (6, 0)])

    def __post_init__(self) -> None:
        seed = self.random_seed if self.random_seed is not None else random.SystemRandom().randrange(1, 10**9)
        self.random = random.Random(seed)
        self.game_number = 0
        self.agent_policies = {
            "rescue": HFAgentPolicy("rescue"),
            "fire_control": HFAgentPolicy("fire_control"),
            "supply": HFAgentPolicy("supply"),
        }
        self.reset()

    def reset(self) -> Dict[str, object]:
        self.game_number += 1
        self.current_step = 0
        self.done = False
        self.total_supplies = self._starting_supplies()
        self.metrics = {
            "rescued": 0,
            "deaths": 0,
            "hazards_extinguished": 0,
            "supplies_delivered": 0,
            "coordination_events": 0,
            "reward_total": 0.0,
        }
        self._randomize_layout()
        return self.state()

    def state(self) -> Dict[str, object]:
        return {
            "game_number": self.game_number,
            "level": self.level,
            "grid": {"width": self.width, "height": self.height},
            "step": self.current_step,
            "max_steps": self.max_steps,
            "safe_zone": self.safe_zone,
            "resource_depots": list(self.resource_depots),
            "agents": {role: agent.as_dict() for role, agent in self.agents.items()},
            "civilians": [civilian.as_dict() for civilian in self.civilians],
            "hazards": [hazard.as_dict() for hazard in self.hazards],
            "metrics": dict(self.metrics),
            "learning": {
                role: policy.learning_summary()
                for role, policy in self.agent_policies.items()
            },
            "result": self._mission_result(),
        }

    def step(self, actions: Dict[str, str]) -> Tuple[Dict[str, object], float, bool, Dict[str, object]]:
        if self.done:
            return self.state(), 0.0, True, {"message": "Environment already completed"}

        decision_state = self.state()
        for role, policy in self.agent_policies.items():
            policy.record_decision(decision_state, actions.get(role, "stay"))

        self.current_step += 1
        reward = 0.0
        coordination_bonus = self._apply_agent_actions(actions)
        reward += coordination_bonus
        reward += self._collision_penalty()
        reward += self._advance_hazards()
        reward += self._update_civilians()
        reward += self._efficiency_bonus()

        self.metrics["reward_total"] += reward
        self.done = self._check_done()
        final_state = self.state()
        if self.done:
            result = final_state["result"]
            for policy in self.agent_policies.values():
                policy.observe_outcome(result, final_state)
            final_state = self.state()
        info = {
            "coordination_bonus": coordination_bonus,
            "metrics": dict(self.metrics),
            "goal_reached": self._goal_reached(),
        }
        return final_state, reward, self.done, info

    def decide_actions(self) -> Dict[str, str]:
        current_state = self.state()
        return {
            role: policy.decide_action(current_state)
            for role, policy in self.agent_policies.items()
        }

    def _randomize_layout(self) -> None:
        civilian_count = self.rescue_goal
        hazard_count = {"easy": 2, "medium": 3, "hard": 4}[self.level]
        hazard_min_intensity = {"easy": 1, "medium": 1, "hard": 1}[self.level]
        hazard_max_intensity = {"easy": 1, "medium": 2, "hard": 2}[self.level]

        occupied: set[Position] = set()
        corners = [(0, 0), (0, self.width - 1), (self.height - 1, 0), (self.height - 1, self.width - 1)]

        self.safe_zone = self.random.choice(corners)
        occupied.add(self.safe_zone)

        remaining_corners = [corner for corner in corners if corner != self.safe_zone]
        self.resource_depots = self.random.sample(remaining_corners, k=min(2, len(remaining_corners)))
        occupied.update(self.resource_depots)

        rescue_start = self.safe_zone
        occupied.add(rescue_start)
        fire_start = self._sample_position(occupied, preferred_positions=remaining_corners)
        occupied.add(fire_start)
        supply_start = self._sample_position(occupied)
        occupied.add(supply_start)

        self.agents = {
            "rescue": AgentUnit("rescue_1", "rescue", rescue_start),
            "fire_control": AgentUnit("fire_1", "fire_control", fire_start),
            "supply": AgentUnit("supply_1", "supply", supply_start, inventory=1, max_inventory=2),
        }

        self.civilians = []
        for index in range(civilian_count):
            position = self._sample_position(occupied)
            occupied.add(position)
            self.civilians.append(Civilian(f"c{index + 1}", position))

        self.hazards = []
        for _ in range(hazard_count):
            position = self._sample_position(occupied)
            occupied.add(position)
            intensity = self.random.randint(hazard_min_intensity, hazard_max_intensity)
            self.hazards.append(Hazard(position, intensity))

    def _sample_position(
        self,
        occupied: set[Position],
        preferred_positions: Optional[List[Position]] = None,
    ) -> Position:
        candidates = preferred_positions or [
            (row, col)
            for row in range(self.height)
            for col in range(self.width)
        ]
        available = [position for position in candidates if position not in occupied]
        if not available:
            raise ValueError("No available positions left to sample")
        return self.random.choice(available)

    def _apply_agent_actions(self, actions: Dict[str, str]) -> float:
        reward = 0.0
        acted_on_hazard_positions: List[Position] = []

        for role in ("fire_control", "supply", "rescue"):
            action = actions.get(role, "stay")
            agent = self.agents[role]

            if action.startswith("move_"):
                agent.position = self._move(agent.position, action)
                continue

            if action == "extinguish" and role == "fire_control":
                target = self._hazard_at(agent.position)
                if target is not None:
                    target.intensity -= 1
                    acted_on_hazard_positions.append(target.position)
                    if target.intensity <= 0:
                        self.hazards.remove(target)
                        self.metrics["hazards_extinguished"] += 1
                        reward += 1.5
                else:
                    reward -= 0.1
                continue

            if action == "pickup_supply" and role == "supply":
                if agent.position in self.resource_depots and agent.inventory < agent.max_inventory and self.total_supplies > 0:
                    agent.inventory += 1
                    self.total_supplies -= 1
                    reward += 0.2
                else:
                    reward -= 0.05
                continue

            if action == "deliver_supply" and role == "supply":
                civilian = self._civilian_at(agent.position)
                if civilian and not civilian.rescued and civilian.health > 0 and agent.inventory > 0:
                    civilian.health = min(100, civilian.health + 20)
                    civilian.supplied = True
                    agent.inventory -= 1
                    self.metrics["supplies_delivered"] += 1
                    reward += 1.0
                else:
                    reward -= 0.1
                continue

            if action == "rescue" and role == "rescue":
                civilian = self._civilian_at(agent.position)
                if civilian and not civilian.rescued and civilian.health > 0:
                    # Rescue reward is intentionally shaped by recent fire control and supply support.
                    rescue_reward = 5.0
                    if civilian.position != self.safe_zone and self._hazard_near(civilian.position):
                        rescue_reward *= 0.5
                    if civilian.supplied:
                        rescue_reward += 1.0
                    else:
                        rescue_reward -= 0.5
                    civilian.rescued = True
                    civilian.position = self.safe_zone
                    agent.rescued_count += 1
                    self.metrics["rescued"] += 1
                    reward += rescue_reward
                    if self._was_recently_secured(agent.position, acted_on_hazard_positions):
                        self.metrics["coordination_events"] += 1
                        # Reward tighter rescue/fire sequencing so agents benefit from cooperation.
                        reward += 1.0
                else:
                    reward -= 0.1
                continue

        return reward

    def _advance_hazards(self) -> float:
        reward = 0.0
        new_hazards: List[Hazard] = []

        for hazard in list(self.hazards):
            if self.random.random() < 0.25 and hazard.intensity < 3:
                hazard.intensity += 1

            spread_probability = self._hazard_spread_probability(hazard)
            if self.random.random() < spread_probability:
                spread_target = self._random_neighbor(hazard.position)
                if spread_target and not self._hazard_at(spread_target):
                    # Fire growth is harsher on denser boards and hard mode to force fire-control urgency.
                    chain_intensity = 2 if self._has_adjacent_hazard(spread_target) else 1
                    new_hazards.append(Hazard(spread_target, chain_intensity))
                    reward -= 0.3

        self.hazards.extend(new_hazards)
        if len(self.hazards) > self._hazard_pressure_threshold():
            reward -= 1.0
        return reward

    def _update_civilians(self) -> float:
        reward = 0.0
        for civilian in self.civilians:
            if civilian.rescued or civilian.health <= 0:
                continue

            if self._hazard_at(civilian.position):
                civilian.health -= 18 if self.level == "hard" else 20
            elif any(_manhattan(civilian.position, hazard.position) == 1 for hazard in self.hazards):
                civilian.health -= 8 if self.level == "hard" else 10
            else:
                civilian.health -= 1 if self.level == "hard" else 2

            if civilian.health <= 0:
                civilian.health = 0
                self.metrics["deaths"] += 1
                reward -= 4.0

        return reward

    def _efficiency_bonus(self) -> float:
        active_civilians = len([c for c in self.civilians if not c.rescued and c.health > 0])
        if active_civilians == 0:
            return 2.0
        if self.current_step <= self.max_steps // 2:
            return 0.05
        return -0.02

    def _starting_supplies(self) -> int:
        return {"easy": 7, "medium": 6, "hard": 5}[self.level]

    def _collision_penalty(self) -> float:
        positions = [agent.position for agent in self.agents.values()]
        return -0.1 if len(set(positions)) < len(positions) else 0.0

    def _hazard_spread_probability(self, hazard: Hazard) -> float:
        if self.level == "easy":
            base = 0.05
        elif self.level == "hard":
            base = 0.25
        else:
            base = 0.20
        if hazard.intensity > 2:
            base = max(base, 0.32 if self.level == "hard" else 0.40)
        if self._has_adjacent_hazard(hazard.position):
            if self.level == "easy":
                base = max(base, 0.10)
            elif self.level == "hard":
                base = max(base, 0.32)
            else:
                base = max(base, 0.40)
        return min(base, 0.55 if self.level == "hard" else 0.65)

    def _hazard_pressure_threshold(self) -> int:
        return {"easy": 5, "medium": 4, "hard": 4}[self.level]

    def _goal_reached(self) -> bool:
        if self.level == "hard":
            return self._hard_victory_met()
        return self.metrics["rescued"] >= self.rescue_goal

    def _hard_victory_met(self) -> bool:
        return (
            self.metrics["rescued"] >= 3
            and self.metrics["hazards_extinguished"] >= 2
            and self.metrics["deaths"] <= 1
        )

    def _check_done(self) -> bool:
        if self.current_step >= self.max_steps:
            return True
        alive_unrescued = [c for c in self.civilians if not c.rescued and c.health > 0]
        if self.level == "easy":
            return self.metrics["rescued"] >= 3 or not alive_unrescued
        return not alive_unrescued or self.metrics["rescued"] == len(self.civilians)

    def _mission_result(self) -> Optional[str]:
        if not self.done:
            return None
        return "won" if self._goal_reached() else "lost"

    def _move(self, position: Position, action: str) -> Position:
        x, y = position
        if action == "move_up":
            x -= 1
        elif action == "move_down":
            x += 1
        elif action == "move_left":
            y -= 1
        elif action == "move_right":
            y += 1
        return (_clamp(x, 0, self.height - 1), _clamp(y, 0, self.width - 1))

    def _random_neighbor(self, position: Position) -> Optional[Position]:
        choices = []
        x, y = position
        for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            nx = x + dx
            ny = y + dy
            if 0 <= nx < self.height and 0 <= ny < self.width:
                choices.append((nx, ny))
        if not choices:
            return None
        return self.random.choice(choices)

    def _hazard_at(self, position: Position) -> Optional[Hazard]:
        return next((hazard for hazard in self.hazards if hazard.position == position), None)

    def _hazard_near(self, position: Position) -> bool:
        return any(_manhattan(position, hazard.position) <= 1 for hazard in self.hazards)

    def _has_adjacent_hazard(self, position: Position) -> bool:
        return any(_manhattan(position, hazard.position) == 1 for hazard in self.hazards)

    def _civilian_at(self, position: Position) -> Optional[Civilian]:
        return next(
            (
                civilian
                for civilian in self.civilians
                if civilian.position == position and not civilian.rescued and civilian.health > 0
            ),
            None,
        )

    def _was_recently_secured(self, position: Position, hazard_positions: Iterable[Position]) -> bool:
        return any(_manhattan(position, hazard_position) <= 1 for hazard_position in hazard_positions)
