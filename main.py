from fastapi import FastAPI
from pydantic import BaseModel
from typing import Dict

try:
    import gradio as gr
except Exception:  # pragma: no cover - optional UI dependency
    gr = None

from app import build_app
from tasks import create_task


api = FastAPI(title="UrbanRescueEnv")
env = create_task("medium")


class ResetRequest(BaseModel):
    level: str = "medium"


class StepRequest(BaseModel):
    actions: Dict[str, str]


@api.get("/health")
def health():
    return {"status": "ok"}


@api.post("/reset")
def reset(req: ResetRequest = ResetRequest()):
    global env
    env = create_task(req.level)
    return env.reset()


@api.post("/step")
def step(req: StepRequest):
    state, reward, done, info = env.step(req.actions)
    return {
        "state": state,
        "reward": reward,
        "done": done,
        "info": info,
    }


@api.get("/state")
def state():
    return env.state()


if gr is not None:
    app = gr.mount_gradio_app(api, build_app(), path="/")
else:  # pragma: no cover - optional UI dependency
    @api.get("/")
    def root():
        return {
            "message": "UrbanRescueEnv API is running. Install gradio to serve the UI at /."
        }

    app = api
