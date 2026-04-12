import json
import os
import re

from openai import OpenAI

from tasks import create_task


API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
MODEL_NAME = os.getenv("MODEL_NAME", "meta-llama/Llama-3.1-8B-Instruct:cerebras")
HF_TOKEN = os.getenv("HF_TOKEN")
ACTION_NAMES = {
    "move_up",
    "move_down",
    "move_left",
    "move_right",
    "stay",
    "rescue",
    "extinguish",
    "pickup_supply",
    "deliver_supply",
}

if not HF_TOKEN:
    raise ValueError("Set HF_TOKEN before running inference.py")

client = OpenAI(
    base_url=API_BASE_URL,
    api_key=HF_TOKEN,
)


def build_prompt(state):
    civilians = [c for c in state["civilians"] if not c["rescued"] and c["health"] > 0]
    hazards = state["hazards"]
    agents = state["agents"]

    civilian_lines = [
        f"- {c['id']}: pos={tuple(c['position'])} health={c['health']} supplied={c['supplied']}"
        for c in civilians
    ] or ["- none"]
    hazard_lines = [
        f"- pos={tuple(h['position'])} intensity={h['intensity']} kind={h['kind']}"
        for h in hazards
    ] or ["- none"]
    agent_lines = [
        (
            f"- {role}: pos={tuple(agent['position'])} inventory={agent['inventory']} "
            f"rescued_count={agent['rescued_count']}"
        )
        for role, agent in agents.items()
    ]

    return f"""
You are controlling 3 agents in a disaster rescue simulation:

- rescue: saves civilians
- fire_control: extinguishes fires
- supply: delivers supplies

State:
Step: {state["step"]}
Civilians remaining: {len(civilians)}
Hazards: {len(hazards)}
Safe zone: {tuple(state["safe_zone"])}
Resource depots: {', '.join(str(tuple(pos)) for pos in state["resource_depots"])}

Agents:
{chr(10).join(agent_lines)}

Unrescued civilians:
{chr(10).join(civilian_lines)}

Hazards:
{chr(10).join(hazard_lines)}

Return JSON only in this format:
{{"rescue":"<action>","fire_control":"<action>","supply":"<action>"}}

Valid actions:
move_up, move_down, move_left, move_right, stay,
rescue, extinguish, pickup_supply, deliver_supply

Choose actions that move agents toward their role goals immediately.
""".strip()


def _extract_action(value):
    if not isinstance(value, str):
        return None
    candidate = value.strip().lower()
    return candidate if candidate in ACTION_NAMES else None


def parse_action(text, default_actions):
    actions = dict(default_actions)

    try:
        parsed = json.loads(text)
        if isinstance(parsed, dict):
            for role in actions:
                candidate = _extract_action(parsed.get(role))
                if candidate is not None:
                    actions[role] = candidate
            return actions
    except Exception:
        pass

    for role in actions:
        match = re.search(rf"{role}\s*[:=]\s*([a-z_]+)", text.lower())
        if not match:
            continue
        candidate = _extract_action(match.group(1))
        if candidate is not None:
            actions[role] = candidate

    return actions


def fallback_action(env):
    return env.decide_actions()


def run_episode():
    env = create_task("medium")
    state = env.reset()

    print(
        f"[START] task=urban_rescue env=openenv provider=openai_compatible "
        f"model={MODEL_NAME} base_url={API_BASE_URL}"
    )

    step_count = 0
    rewards = []
    success = False

    try:
        while True:
            prompt = build_prompt(state)
            error = None

            try:
                response = client.chat.completions.create(
                    model=MODEL_NAME,
                    messages=[{"role": "user", "content": prompt}],
                    response_format={"type": "json_object"},
                )
                text = response.choices[0].message.content or ""
                action = parse_action(text, env.decide_actions())
            except Exception as exc:
                text = "fallback"
                error = "null"
                action = fallback_action(env)

            state, reward, done, info = env.step(action)

            rewards.append(f"{reward:.2f}")
            step_count += 1

            print(
                f"[STEP] step={step_count} action={action} "
                f"reward={reward:.2f} done={str(done).lower()} "
                f"error={error or 'null'}"
            )

            if done:
                success = state["result"] == "won"
                break
    except Exception as exc:
        print(
            f"[STEP] step={step_count} action=error "
            f"reward=0.00 done=true error={str(exc)}"
        )
        success = False

    print(
        f"[END] success={str(success).lower()} "
        f"steps={step_count} rewards={','.join(rewards)}"
    )


if __name__ == "__main__":
    run_episode()
