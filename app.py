from __future__ import annotations
from typing import Dict, Tuple

try:
    import gradio as gr
except Exception:  # pragma: no cover - optional UI dependency
    gr = None

from graders import grade_level
from tasks import TASK_SPECS, create_task


ACTION_CHOICES = [
    "auto",
    "stay",
    "move_up",
    "move_down",
    "move_left",
    "move_right",
    "rescue",
    "extinguish",
    "pickup_supply",
    "deliver_supply",
]


def _result_label(result: str | None) -> str:
    if result == "won":
        return "You Won"
    if result == "lost":
        return "You Lost"
    return "In Progress"


def _render_result_banner(state: Dict[str, object]) -> str:
    result = state["result"]
    if result == "won":
        background = "#dcfce7"
        border = "#16a34a"
        text = "#166534"
    elif result == "lost":
        background = "#fee2e2"
        border = "#dc2626"
        text = "#991b1b"
    else:
        background = "#dbeafe"
        border = "#2563eb"
        text = "#1d4ed8"

    return (
        f"<div style='margin:12px 0; padding:14px 18px; border-left:6px solid {border}; "
        f"background:{background}; color:{text}; border-radius:10px; font-weight:700; font-size:20px;'>"
        f"Mission Status: {_result_label(result)}"
        "</div>"
    )


def _format_position(position: Tuple[int, int]) -> str:
    return f"({position[0]}, {position[1]})"


def _new_env(level: str):
    env = create_task(level)
    state = env.state()
    return env, state


def _render_grid(state: Dict[str, object]) -> str:
    width = state["grid"]["width"]
    height = state["grid"]["height"]
    grid: list[list[list[str]]] = [[[] for _ in range(width)] for _ in range(height)]

    safe_x, safe_y = state["safe_zone"]
    grid[safe_x][safe_y].append("Safe zone")

    for depot_x, depot_y in state["resource_depots"]:
        grid[depot_x][depot_y].append("Supply depot")

    for hazard in state["hazards"]:
        x, y = hazard["position"]
        grid[x][y].append(f"Fire level {hazard['intensity']}")

    for civilian in state["civilians"]:
        if civilian["rescued"] or civilian["health"] <= 0:
            continue
        x, y = civilian["position"]
        grid[x][y].append(f"Civilian {civilian['id']} (HP {max(0, civilian['health'])})")

    label_map = {
        "rescue": "Rescue team",
        "fire_control": "Fire team",
        "supply": "Supply team",
    }
    for role, agent in state["agents"].items():
        x, y = agent["position"]
        grid[x][y].append(label_map[role])

    header = "".join(f"<th>Col {column}</th>" for column in range(width))
    rows = []
    for row_index, row in enumerate(grid):
        cells = []
        for cell in row:
            if cell:
                content = "<br>".join(cell)
            else:
                content = '<span style="color:#6b7280;">Empty</span>'
            cells.append(
                "<td style='vertical-align:top; min-width:150px; padding:10px; border:1px solid #d1d5db;'>"
                f"{content}</td>"
            )
        rows.append(
            "<tr>"
            f"<th style='padding:10px; border:1px solid #d1d5db; background:#f8fafc;'>Row {row_index}</th>"
            + "".join(cells)
            + "</tr>"
        )

    return (
        "### Mission Map\n"
        "<div style='overflow-x:auto;'>"
        "<table style='border-collapse:collapse; width:100%;'>"
        "<thead><tr>"
        "<th style='padding:10px; border:1px solid #d1d5db; background:#f1f5f9;'>Grid</th>"
        f"{header}"
        "</tr></thead>"
        "<tbody>"
        + "".join(rows)
        + "</tbody></table></div>"
    )


def _render_summary(state: Dict[str, object], reward: float = 0.0, score: float | None = None) -> str:
    metrics = state["metrics"]
    learning = state["learning"]
    goal = TASK_SPECS[state["level"]].description
    result_label = _result_label(state["result"])
    summary = [
        f"Game: {state['game_number']}",
        f"Mission: {state['level'].title()}",
        f"Mission status: {result_label}",
        f"Goal: {goal}",
        f"Turn: {state['step']} of {state['max_steps']}",
        f"Last step reward: {reward:.2f}",
        f"Civilians rescued: {metrics['rescued']}",
        f"Civilian deaths: {metrics['deaths']}",
        f"Fires extinguished: {metrics['hazards_extinguished']}",
        f"Supplies delivered: {metrics['supplies_delivered']}",
        f"Team coordination bonuses: {metrics['coordination_events']}",
        f"Total mission reward: {metrics['reward_total']:.2f}",
        (
            "Agent record: "
            f"rescue {learning['rescue']['wins']}W-{learning['rescue']['losses']}L, "
            f"fire {learning['fire_control']['wins']}W-{learning['fire_control']['losses']}L, "
            f"supply {learning['supply']['wins']}W-{learning['supply']['losses']}L"
        ),
    ]
    if score is not None:
        summary.append(f"Grade: {score:.3f}")
    return "\n".join(summary)


def _render_status(state: Dict[str, object], actions: Dict[str, str] | None = None, done: bool = False) -> str:
    team_names = {
        "rescue": "Rescue team",
        "fire_control": "Fire team",
        "supply": "Supply team",
    }
    civilians = []
    for civilian in state["civilians"]:
        status = "rescued" if civilian["rescued"] else "waiting"
        if civilian["health"] <= 0:
            status = "dead"
        civilians.append(
            f"- {civilian['id']}: position {_format_position(tuple(civilian['position']))}, "
            f"health {civilian['health']}, status {status}, supplied {civilian['supplied']}"
        )

    hazards = [
        f"- Fire at {_format_position(tuple(h['position']))}: level {h['intensity']}"
        for h in state["hazards"]
    ]
    agents = []
    for role, agent in state["agents"].items():
        learning = state["learning"][role]
        adaptations = learning["top_adjustments"] or ["No learned adjustments yet"]
        agents.append(
            f"- {team_names[role]}: position {_format_position(tuple(agent['position']))}, "
            f"inventory {agent['inventory']}, rescued civilians {agent['rescued_count']}, "
            f"record {learning['wins']}W-{learning['losses']}L, "
            f"learning {'; '.join(adaptations)}"
        )

    parts = [
        "TEAMS",
        "\n".join(agents),
        "",
        "CIVILIANS",
        "\n".join(civilians) if civilians else "None",
        "",
        "FIRES",
        "\n".join(hazards) if hazards else "None",
    ]

    if actions is not None:
        action_lines = [
            f"- {team_names[role]} action: {action}"
            for role, action in actions.items()
        ]
        parts.extend(["", "LAST ACTIONS", "\n".join(action_lines)])
    if done:
        parts.extend(["", "MISSION STATUS", f"Game Over: {_result_label(state['result'])}."])
    else:
        parts.extend(["", "MISSION STATUS", "Game status: In Progress."])
    return "\n".join(parts)


def _render_how_to_play() -> str:
    return "\n".join(
        [
            "## How To Read The Game",
            "- The map shows full words, not short codes.",
            "- `Safe zone`: where rescued civilians are taken.",
            "- `Supply depot`: where the supply team picks up supplies.",
            "- `Fire level N`: a fire hazard. Higher number means more dangerous.",
            "- `Civilian cX (HP N)`: a person waiting to be saved. Lower HP means more urgent.",
            "- `Rescue team`: moves to civilians and rescues them.",
            "- `Fire team`: moves to fires and extinguishes them.",
            "- `Supply team`: picks up supplies and delivers them to weak civilians.",
            "",
            "## Controls",
            "- Leave a team on `auto` to let the Hugging Face-backed policy choose an action.",
            "- Use `Run One Step` to advance one turn and inspect what changed.",
            "- Use `Autoplay` to let the teams play several turns automatically.",
        ]
    )


def _resolve_actions(
    env,
    rescue_action: str,
    fire_action: str,
    supply_action: str,
) -> Dict[str, str]:
    auto_actions = env.decide_actions()
    return {
        "rescue": auto_actions["rescue"] if rescue_action == "auto" else rescue_action,
        "fire_control": auto_actions["fire_control"] if fire_action == "auto" else fire_action,
        "supply": auto_actions["supply"] if supply_action == "auto" else supply_action,
    }


def reset_ui(level: str) -> Tuple[object, str, str, str]:
    env, state = _new_env(level)
    return (
        env,
        _render_result_banner(state),
        _render_grid(state),
        _render_summary(state),
        _render_status(state),
    )


def step_ui(env, level: str, rescue_action: str, fire_action: str, supply_action: str):
    if env is None or getattr(env, "level", None) != level:
        env, _ = _new_env(level)

    actions = _resolve_actions(env, rescue_action, fire_action, supply_action)
    state, reward, done, _info = env.step(actions)
    score = grade_level(level, state) if done else None
    return (
        env,
        _render_result_banner(state),
        _render_grid(state),
        _render_summary(state, reward=reward, score=score),
        _render_status(state, actions=actions, done=done),
    )


def autoplay_ui(env, level: str, max_rollout_steps: int):
    if env is None or getattr(env, "level", None) != level:
        env, _ = _new_env(level)

    reward = 0.0
    steps_run = 0
    actions = None
    done = False

    for _ in range(max_rollout_steps):
        actions = env.decide_actions()
        state, reward, done, _info = env.step(actions)
        steps_run += 1
        if done:
            break
    else:
        state = env.state()

    score = grade_level(level, state) if done else None
    status = _render_status(state, actions=actions, done=done)
    status += f"\n\nAutoplay steps executed: {steps_run}"
    return (
        env,
        _render_result_banner(state),
        _render_grid(state),
        _render_summary(state, reward=reward, score=score),
        status,
    )


def build_app() -> gr.Blocks:
    if gr is None:
        raise ModuleNotFoundError(
            "gradio is not installed. Install it with `pip install gradio` to launch the UI."
        )
    with gr.Blocks(title="UrbanRescueEnv") as demo:
        env_state = gr.State(value=None)

        gr.Markdown(
            "# UrbanRescueEnv\n"
            "Interactive disaster response game with three teams: rescue, fire control, and supply."
        )

        with gr.Row():
            level = gr.Dropdown(
                choices=["easy", "medium", "hard"],
                value="easy",
                label="Mission Level",
            )
            rollout_steps = gr.Slider(
                minimum=1,
                maximum=50,
                value=50,
                step=1,
                label="Autoplay Steps",
            )

        gr.Markdown(
            "\n".join(
                ["## Mission Goals"]
                + [f"- `{name}`: {spec.description}" for name, spec in TASK_SPECS.items()]
            )
        )
        gr.Markdown(_render_how_to_play())

        with gr.Row():
            rescue_action = gr.Dropdown(ACTION_CHOICES, value="auto", label="Rescue Team Action")
            fire_action = gr.Dropdown(ACTION_CHOICES, value="auto", label="Fire Team Action")
            supply_action = gr.Dropdown(ACTION_CHOICES, value="auto", label="Supply Team Action")

        with gr.Row():
            reset_button = gr.Button("Reset Mission", variant="secondary")
            step_button = gr.Button("Run One Step", variant="primary")
            autoplay_button = gr.Button("Autoplay", variant="secondary")

        result_banner = gr.Markdown(label="Mission Result")
        grid = gr.Markdown(label="Mission Map")
        summary = gr.Textbox(label="Mission Summary", lines=10)
        status = gr.Textbox(label="Detailed Status", lines=18)

        demo.load(
            fn=reset_ui,
            inputs=[level],
            outputs=[env_state, result_banner, grid, summary, status],
        )
        level.change(
            fn=reset_ui,
            inputs=[level],
            outputs=[env_state, result_banner, grid, summary, status],
        )
        reset_button.click(
            fn=reset_ui,
            inputs=[level],
            outputs=[env_state, result_banner, grid, summary, status],
        )
        step_button.click(
            fn=step_ui,
            inputs=[env_state, level, rescue_action, fire_action, supply_action],
            outputs=[env_state, result_banner, grid, summary, status],
        )
        autoplay_button.click(
            fn=autoplay_ui,
            inputs=[env_state, level, rollout_steps],
            outputs=[env_state, result_banner, grid, summary, status],
        )

    return demo


if __name__ == "__main__":
    build_app().launch()
