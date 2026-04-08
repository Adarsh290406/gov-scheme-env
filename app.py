import inference
"""
app.py — FastAPI Server
=======================
Wraps the GovSchemeEnvironment in a REST API.
Any agent can interact with the environment
by sending HTTP requests to this server.

Endpoints:
  GET  /health          -> Check if server is running
  GET  /info            -> Environment metadata
  POST /reset           -> Start a new episode
  POST /step            -> Take an action
  GET  /state           -> Get current full state
  GET  /tasks           -> List all available tasks
  POST /tasks/{name}    -> Start a specific task (easy/medium/hard)
  POST /grade           -> Grade the agent's last recommendation
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional
import uuid

from models import (
    Action, Observation, Difficulty,
    StepResult, State
)
from environment import GovSchemeEnvironment
from tasks.easy import (
    run_task_with_fixed_citizen as easy_task,
    grade as easy_grade,
    TASK_DESCRIPTION as EASY_DESC
)
from tasks.medium import (
    run_task_with_fixed_citizen as medium_task,
    grade as medium_grade,
    TASK_DESCRIPTION as MEDIUM_DESC
)
from tasks.hard import (
    run_task_with_fixed_citizen as hard_task,
    grade as hard_grade,
    TASK_DESCRIPTION as HARD_DESC
)


# -----------------------------------------
# APP SETUP
# -----------------------------------------

app = FastAPI(
    title="Gov Scheme Finder — OpenEnv",
    description=(
        "A real-world RL environment where agents learn to match "
        "Indian citizens to the correct government schemes by asking "
        "smart questions. Implements the OpenEnv step/reset/state API."
    ),
    version="1.0.0",
)

# Allow all origins so agents can connect from anywhere
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# -----------------------------------------
# SESSION STORE
# Stores one environment per session_id
# so multiple agents can run simultaneously
# -----------------------------------------

sessions: dict[str, GovSchemeEnvironment] = {}
session_tasks: dict[str, str] = {}          # session_id -> task name
session_recommendations: dict[str, str] = {} # session_id -> last recommended scheme


# -----------------------------------------
# REQUEST / RESPONSE MODELS
# -----------------------------------------

class ResetRequest(BaseModel):
    difficulty: Optional[Difficulty] = Difficulty.EASY
    session_id: Optional[str] = None


class StepRequest(BaseModel):
    session_id: str
    action: Action


class GradeRequest(BaseModel):
    session_id: str
    task: str   # "easy", "medium", or "hard"


class ResetResponse(BaseModel):
    session_id: str
    observation: Observation
    message: str


# -----------------------------------------
# HELPER — Get environment by session
# -----------------------------------------

def get_env(session_id: str) -> GovSchemeEnvironment:
    if session_id not in sessions:
        raise HTTPException(
            status_code=404,
            detail=f"Session '{session_id}' not found. Call /reset first."
        )
    return sessions[session_id]


# -----------------------------------------
# ENDPOINTS
# -----------------------------------------

@app.get("/")
def read_root():
    """Root endpoint so the Hugging Face space doesn't show Not Found."""
    return {
        "status": "ok",
        "message": "Gov Scheme Finder API is running. Check /info for details."
    }


@app.get("/health")
def health():
    """Check if the server is running"""
    return {
        "status": "ok",
        "message": "Gov Scheme Finder environment is running!",
        "active_sessions": len(sessions)
    }


@app.get("/info")
def info():
    """Return environment metadata"""
    return {
        "name": "gov-scheme-finder",
        "version": "1.0.0",
        "description": "RL environment for matching Indian citizens to government schemes",
        "action_space": [
            "ask_age", "ask_income", "ask_gender", "ask_caste",
            "ask_location", "ask_occupation", "ask_disability",
            "ask_bpl", "ask_education", "ask_bank_account",
            "ask_ration_card", "ask_marital_status",
            "ask_land_ownership", "ask_state", "recommend_scheme"
        ],
        "observation_space": [
            "age", "income", "income_context", "gender", "caste",
            "location", "occupation", "has_disability", "is_bpl",
            "education", "has_bank_account", "has_ration_card",
            "marital_status", "land_ownership", "state",
            "step_count", "max_steps", "last_action_result",
            "available_schemes", "done"
        ],
        "difficulties": ["easy", "medium", "hard"],
        "tasks": {
            "easy":   {"schemes": 5,  "max_steps": 10, "pass_threshold": 0.5},
            "medium": {"schemes": 10, "max_steps": 8,  "pass_threshold": 0.4},
            "hard":   {"schemes": 12, "max_steps": 6,  "pass_threshold": 0.3},
        }
    }


@app.get("/tasks")
def list_tasks():
    """List all available tasks with descriptions"""
    return {
        "tasks": {
            "easy":   EASY_DESC.strip(),
            "medium": MEDIUM_DESC.strip(),
            "hard":   HARD_DESC.strip(),
        }
    }


@app.post("/reset", response_model=ResetResponse)
def reset(request: ResetRequest = None):
    if request is None:
        request = ResetRequest()
    """
    Start a new episode.
    Returns a session_id — use this in all future requests.
    """
    session_id = request.session_id or str(uuid.uuid4())

    env = GovSchemeEnvironment(difficulty=request.difficulty)
    obs = env.reset()

    sessions[session_id] = env
    session_tasks[session_id] = request.difficulty.value
    session_recommendations.pop(session_id, None)

    return ResetResponse(
        session_id=session_id,
        observation=obs,
        message=(
            f"New episode started! Difficulty: {request.difficulty.value}. "
            f"Use session_id '{session_id}' for all future requests."
        )
    )


@app.post("/tasks/{task_name}", response_model=ResetResponse)
def start_task(task_name: str, session_id: Optional[str] = None):
    """
    Start a specific task with a fixed citizen profile.
    Use this for deterministic grading.
    """
    task_name = task_name.lower()
    if task_name not in ["easy", "medium", "hard"]:
        raise HTTPException(
            status_code=400,
            detail="Task must be one of: easy, medium, hard"
        )

    session_id = session_id or str(uuid.uuid4())

    if task_name == "easy":
        env = easy_task()
    elif task_name == "medium":
        env = medium_task()
    else:
        env = hard_task()

    obs = env.state.observation
    sessions[session_id] = env
    session_tasks[session_id] = task_name
    session_recommendations.pop(session_id, None)

    return ResetResponse(
        session_id=session_id,
        observation=obs,
        message=(
            f"Task '{task_name}' started with fixed citizen profile. "
            f"Use session_id '{session_id}' for all future requests."
        )
    )


@app.post("/step")
def step(request: StepRequest):
    """
    Take one action in the environment.
    Returns observation, reward, done, info.
    """
    env = get_env(request.session_id)

    if env.state and env.state.is_done:
        raise HTTPException(
            status_code=400,
            detail="Episode is already done. Call /reset or /tasks/{name} to start a new one."
        )

    result = env.step(request.action)

    # Track last recommendation for grading
    if request.action.action_type.value == "recommend_scheme":
        session_recommendations[request.session_id] = request.action.scheme_name

    return {
        "session_id": request.session_id,
        "observation": result.observation,
        "reward": result.reward,
        "done": result.done,
        "info": result.info
    }


@app.get("/state/{session_id}")
def get_state(session_id: str):
    """
    Get the full internal state of the environment.
    Note: This reveals the hidden citizen profile.
    Use only for debugging — agents should not call this.
    """
    env = get_env(session_id)
    state = env.get_state()
    return {
        "session_id": session_id,
        "state": state
    }


@app.post("/grade")
def grade(request: GradeRequest):
    """
    Grade the agent's performance on a task.
    Call this after the episode is done.
    """
    env = get_env(request.session_id)
    state = env.get_state()

    if not state.is_done:
        raise HTTPException(
            status_code=400,
            detail="Episode is not done yet. Complete the episode before grading."
        )

    recommended = session_recommendations.get(request.session_id, "")
    task = request.task.lower()

    if task == "easy":
        result = easy_grade(
            recommended_scheme=recommended,
            questions_asked=state.questions_asked,
            steps_taken=state.step_count,
            total_reward=state.total_reward
        )
    elif task == "medium":
        result = medium_grade(
            recommended_scheme=recommended,
            questions_asked=state.questions_asked,
            steps_taken=state.step_count,
            total_reward=state.total_reward
        )
    elif task == "hard":
        result = hard_grade(
            recommended_scheme=recommended,
            questions_asked=state.questions_asked,
            steps_taken=state.step_count,
            total_reward=state.total_reward
        )
    else:
        raise HTTPException(
            status_code=400,
            detail="Task must be one of: easy, medium, hard"
        )

    return {
        "session_id": request.session_id,
        "task": task,
        "grade": result
    }


@app.delete("/session/{session_id}")
def delete_session(session_id: str):
    """Clean up a session when done"""
    sessions.pop(session_id, None)
    session_tasks.pop(session_id, None)
    session_recommendations.pop(session_id, None)
    return {"message": f"Session '{session_id}' deleted."}


# -----------------------------------------
# RUN SERVER
# -----------------------------------------

def main():
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=7860)

if __name__ == "__main__":
    main()