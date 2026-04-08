from fastapi import FastAPI
from pydantic import BaseModel
from typing import Dict, Optional

from tasks import create_task

app = FastAPI()

env = create_task("medium")


# ✅ Make body OPTIONAL
class ResetRequest(BaseModel):
    level: Optional[str] = "medium"


class StepRequest(BaseModel):
    actions: Dict[str, str]


@app.post("/reset")
def reset(req: ResetRequest = None):   # ✅ THIS IS THE FIX
    global env

    level = "medium"
    if req and req.level:
        level = req.level

    env = create_task(level)
    return env.reset()


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
