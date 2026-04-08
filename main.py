from fastapi import FastAPI, Body
from pydantic import BaseModel
from typing import Dict, Optional

from tasks import create_task

app = FastAPI()

env = create_task("medium")


class ResetRequest(BaseModel):
    level: Optional[str] = "medium"


class StepRequest(BaseModel):
    actions: Dict[str, str]


# ✅ CRITICAL FIX
@app.post("/reset")
def reset(req: Optional[ResetRequest] = Body(default=None)):
    global env

    level = "medium"
    if req is not None and req.level is not None:
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
