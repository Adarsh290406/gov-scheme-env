"""
TASK 1 — EASY
=============
Citizen Profile: Obvious, single correct scheme
Schemes available: 10
Max steps: 10
Expected score for a smart agent: 0.7 to 1.0

Scenario:
  A rural BPL woman who needs LPG connection.
  The correct scheme is obviously PM Ujjwala Yojana (Ujjwala Scheme).
  Agent should figure this out in 3-4 questions.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models import (
    CitizenProfile, Gender, CasteCategory,
    Location, Occupation, Difficulty
)
from environment import GovSchemeEnvironment, ALL_SCHEMES, check_scheme_conditions

# -----------------------------------------
# FIXED CITIZEN PROFILE
# Same every time — makes grader deterministic
# -----------------------------------------

EASY_CITIZEN = CitizenProfile(
    age=32,
    income=85000.0,          # Household income (she is daily wage worker)
    gender=Gender.FEMALE,
    caste=CasteCategory.OBC,
    location=Location.RURAL,
    occupation=Occupation.DAILY_WAGE,
    has_disability=False,
    is_bpl=True,
    education="8th",
    has_bank_account=True,
    has_ration_card=True,
    marital_status="married",
    land_ownership="none",
    state="Uttar Pradesh",
    correct_schemes=[]
)

# Compute correct schemes dynamically for this citizen from ALL_SCHEMES
for name, scheme in ALL_SCHEMES.items():
    is_match, _ = check_scheme_conditions(EASY_CITIZEN, scheme)
    if is_match:
        EASY_CITIZEN.correct_schemes.append(name)

TASK_DESCRIPTION = """
TASK 1 (EASY) — Find the Right Scheme
--------------------------------------
A 32-year-old married rural woman comes to you for help.
She is a daily wage worker living in a rural area.
Her household income is Rs.85,000 per year.
She belongs to OBC category, has an 8th pass education,
and is registered as BPL with a bank account and ration card.

Your job: Ask the right questions and recommend
the most relevant government scheme for her.

Hint: Think about what a rural BPL woman needs most.
"""

# -----------------------------------------
# GRADER — Scores agent 0.0 to 1.0
# -----------------------------------------

def grade(
    recommended_scheme: str,
    questions_asked: list,
    steps_taken: int,
    total_reward: float,
    max_steps: int = 10
) -> dict:
    score = 0.0
    feedback = []

    # 1. Scheme correctness (0.0 to 0.5)
    if recommended_scheme in EASY_CITIZEN.correct_schemes:
        if recommended_scheme in ["Ujjwala Scheme", "Pradhan Mantri Ujjwala Yojana"]:
            scheme_score = 0.5
            feedback.append("Perfect scheme choice — PM Ujjwala Yojana is the top priority.")
        else:
            scheme_score = 0.35
            feedback.append(f"Correct scheme but not the most relevant. BPL woman -> Ujjwala Scheme.")
    else:
        scheme_score = 0.0
        feedback.append(f"Wrong scheme. Valid schemes: {EASY_CITIZEN.correct_schemes[:5]}")

    score += scheme_score

    # 2. Efficiency score (0.0 to 0.3)
    if steps_taken <= 4:
        efficiency_score = 0.3
        feedback.append("Excellent efficiency — solved in 4 steps or fewer!")
    elif steps_taken <= 6:
        efficiency_score = 0.2
        feedback.append("Good efficiency.")
    elif steps_taken <= 8:
        efficiency_score = 0.1
        feedback.append("Average efficiency.")
    else:
        efficiency_score = 0.0
        feedback.append("Poor efficiency — too many steps taken.")

    score += efficiency_score

    # 3. Question quality (0.0 to 0.2)
    quality_score = 0.0
    key_questions = ["ask_occupation", "ask_bpl", "ask_gender", "ask_location"]
    asked_key = [q for q in questions_asked if q in key_questions]

    if "ask_occupation" in questions_asked and questions_asked.index("ask_occupation") == 0:
        quality_score += 0.1
        feedback.append("Smart: asked occupation first.")

    if len(asked_key) >= 3:
        quality_score += 0.1
        feedback.append("Asked the right questions.")
    elif len(asked_key) >= 2:
        quality_score += 0.05
        feedback.append("Missed some key questions.")

    score += quality_score
    final_score = round(min(1.0, max(0.0, score)), 3)

    return {
        "task": "easy",
        "score": final_score,
        "scheme_score": scheme_score,
        "efficiency_score": efficiency_score,
        "quality_score": quality_score,
        "recommended_scheme": recommended_scheme,
        "correct_schemes": EASY_CITIZEN.correct_schemes,
        "steps_taken": steps_taken,
        "questions_asked": questions_asked,
        "feedback": feedback,
        "passed": final_score >= 0.5
    }


def create_task_env() -> GovSchemeEnvironment:
    return GovSchemeEnvironment(difficulty=Difficulty.EASY)

def run_task_with_fixed_citizen() -> GovSchemeEnvironment:
    env = create_task_env()
    obs = env.reset()
    env.state.citizen_profile = EASY_CITIZEN
    
    # Ensure Ujjwala is in the available schemes pool for the agent
    if "Ujjwala Scheme" not in env.available_schemes:
        if len(env.available_schemes) >= env.max_steps:
            env.available_schemes.pop()
        env.available_schemes.append("Ujjwala Scheme")
        
    obs.available_schemes = env.available_schemes
    return env

if __name__ == "__main__":
    from models import Action, ActionType
    env = run_task_with_fixed_citizen()
    steps = [
        Action(action_type=ActionType.ASK_OCCUPATION),
        Action(action_type=ActionType.ASK_GENDER),
        Action(action_type=ActionType.ASK_BPL),
        Action(action_type=ActionType.RECOMMEND_SCHEME, scheme_name="Ujjwala Scheme"),
    ]
    for action in steps:
        env.step(action)
    state = env.get_state()
    res = grade("Ujjwala Scheme", state.questions_asked, state.step_count, state.total_reward)
    print(res)