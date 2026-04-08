"""
server/app.py

FastAPI server for the Manufacturing Defect Triage RL environment.
Exposes the environment over HTTP on port 7860 (Hugging Face Spaces standard).

Endpoints:
    GET  /health  — liveness probe
    POST /reset   — start a new episode
    POST /step    — execute one agent step
    GET  /state   — inspect current environment state
"""

from __future__ import annotations

import os
from contextlib import asynccontextmanager
from typing import Any, Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from env.environment import ManufacturingDefectEnv
from env.models import Action, Observation, Reward

# ---------------------------------------------------------------------------
# Global environment instance (single-session; for concurrent use add sessions)
# ---------------------------------------------------------------------------

_env: ManufacturingDefectEnv | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialise a default environment on startup."""
    global _env
    _env = ManufacturingDefectEnv(task_id="task_1_classify", seed=42)
    yield


app = FastAPI(
    title="Manufacturing Defect Triage — OpenEnv",
    description=(
        "An RL environment where an agent triages manufacturing defects on a "
        "production line. Classifies defect types, prioritises repair queues, "
        "and diagnoses root causes."
    ),
    version="1.0.0",
    lifespan=lifespan,
)


# ---------------------------------------------------------------------------
# Request / response schemas
# ---------------------------------------------------------------------------

class ResetRequest(BaseModel):
    task_id: str = "task_1_classify"
    seed: int = 42

    model_config = {"extra": "ignore"}  # silently ignore unknown fields


class StepResponse(BaseModel):
    observation: Observation
    reward: Reward
    done: bool
    info: dict[str, Any]


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.get("/health", tags=["infra"])
def health() -> dict[str, str]:
    """Liveness probe — returns 200 OK if the server is running."""
    return {"status": "ok"}


@app.post("/reset", response_model=Observation, tags=["env"])
def reset(request: Optional[ResetRequest] = None) -> Observation:
    """
    Start a new episode.

    Accepts: {"task_id": str, "seed": int}
    Returns: Initial Observation JSON.
    """
    global _env
    try:
        if request is None:
            request = ResetRequest()
        _env = ManufacturingDefectEnv(task_id=request.task_id, seed=request.seed)
        observation = _env.reset()
        return observation
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.post("/step", response_model=StepResponse, tags=["env"])
def step(action: Action) -> StepResponse:
    """
    Execute one agent step.

    Accepts: Action JSON body.
    Returns: {observation, reward, done, info}
    """
    global _env
    if _env is None:
        raise HTTPException(status_code=400, detail="Call /reset before /step.")
    try:
        obs, reward, done, info = _env.step(action)
        return StepResponse(observation=obs, reward=reward, done=done, info=info)
    except RuntimeError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc


@app.get("/state", tags=["env"])
def state() -> dict[str, Any]:
    """Return the current internal environment state."""
    global _env
    if _env is None:
        raise HTTPException(status_code=400, detail="Call /reset to initialise the environment.")
    return _env.state()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    """Entry point for the OpenEnv server."""
    import uvicorn
    port = int(os.environ.get("PORT", 7860))
    uvicorn.run("server.app:app", host="0.0.0.0", port=port, reload=False)


if __name__ == "__main__":
    main()
