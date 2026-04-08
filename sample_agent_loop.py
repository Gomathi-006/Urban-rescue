from __future__ import annotations

from pprint import pprint

from graders import grade_level
from tasks import TASK_SPECS, create_task


def run_level(level: str, episodes: int = 3) -> None:
    env = create_task(level)
    for episode in range(episodes):
        state = env.state() if episode == 0 else env.reset()

        print(f"\n=== {level.upper()} | GAME {state['game_number']} ===")
        print(TASK_SPECS[level].description)

        while True:
            actions = env.decide_actions()
            state, reward, done, info = env.step(actions)
            print(f"step={state['step']:02d} actions={actions} reward={reward:.2f}")
            if done:
                break

        score = grade_level(level, state)
        print(f"result={state['result']}")
        print(f"final_score={score:.3f}")
        pprint(info["metrics"])
        print("learning_record=")
        pprint(state["learning"])


if __name__ == "__main__":
    for level_name in ("easy", "medium", "hard"):
        run_level(level_name)
