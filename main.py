from fastapi import FastAPI
from pydantic import BaseModel
from typing import Dict

from tasks import create_task

app = FastAPI()

env = create_task("medium")


# ✅ Request models (fixes your error)
class ResetRequest(BaseModel):
    level: str = "medium"


class StepRequest(BaseModel):
    actions: Dict[str, str]


@app.post("/reset")
def reset(req: ResetRequest):
    global env
    env = create_task(req.level)
    state = env.reset()
    return state


@app.post("/step")
def step(req: StepRequest):
    state, reward, done, info = env.step(req.actions)
    return {
        "state": state,
        "reward": reward,
        "done": done,
        "info": info,
    }


@app.get("/state")
def state():
    return env.state()