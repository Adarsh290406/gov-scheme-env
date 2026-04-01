"""
TASK 2 — MEDIUM
===============
Citizen Profile: Ambiguous — qualifies for multiple schemes
Schemes available: 25
Max steps: 8
Expected score for a smart agent: 0.5 to 0.8

Scenario:
  A rural SC category farmer with low income.
  Qualifies for multiple schemes — agent must
  identify the MOST relevant one efficiently (PM Kisan Samman Nidhi).
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
# -----------------------------------------

MEDIUM_CITIZEN = CitizenProfile(
    age=45,
    income=95000.0,          # Annual crop income
    gender=Gender.MALE,
    caste=CasteCategory.SC,
    location=Location.RURAL,
    occupation=Occupation.FARMER,
    has_disability=False,
    is_bpl=True,
    education="none",
    has_bank_account=True,
    has_ration_card=True,
    marital_status="married",
    land_ownership="owner",
    state="Maharashtra",
    correct_schemes=[]
)

# Compute correct schemes dynamically for this citizen from ALL_SCHEMES
for name, scheme in ALL_SCHEMES.items():
    is_match, _ = check_scheme_conditions(MEDIUM_CITIZEN, scheme)
    if is_match:
        MEDIUM_CITIZEN.correct_schemes.append(name)

TASK_DESCRIPTION = """
TASK 2 (MEDIUM) — Ambiguous Profile
--------------------------------------
A 45-year-old married man comes for help.
He is a farmer in a rural area who owns his land.
His annual crop income is Rs.95,000.
He belongs to SC category, has no formal education,
and is registered as BPL with a bank account.

Your job: Ask smart questions and recommend
the MOST relevant scheme for his situation.

Hint: A farmer with land ownership qualifies for
direct agricultural income support schemes.
"""

# Priority ranking — most relevant scheme first
PRIORITY_SCHEMES = [
    "PM Kisan Samman Nidhi",       # Direct income support for farmer
    "Kisan Credit Card",           # Subsidized agricultural credit
    "Ayushman Bharat",             # Health insurance for BPL
    "MGNREGA",                     # Rural employment
    "Swachh Bharat Mission Gramin" # Rural sanitation
]

def grade(
    recommended_scheme: str,
    questions_asked: list,
    steps_taken: int,
    total_reward: float,
    max_steps: int = 8
) -> dict:
    score = 0.0
    feedback = []

    # 1. Scheme correctness (0.0 to 0.4)
    if recommended_scheme in MEDIUM_CITIZEN.correct_schemes:
        scheme_score = 0.4
        feedback.append(f"Correct! '{recommended_scheme}' is valid for this citizen.")
    else:
        scheme_score = 0.0
        feedback.append(f"Wrong scheme. Valid options: {MEDIUM_CITIZEN.correct_schemes[:5]}")

    score += scheme_score

    # 2. Priority score (0.0 to 0.2)
    if recommended_scheme in PRIORITY_SCHEMES:
        priority_index = PRIORITY_SCHEMES.index(recommended_scheme)
        if priority_index == 0:
            priority_score = 0.2
            feedback.append("Best possible scheme chosen!")
        elif priority_index == 1:
            priority_score = 0.15
            feedback.append("Great choice — second most relevant scheme.")
        elif priority_index == 2:
            priority_score = 0.1
            feedback.append("Good choice but not the most relevant.")
        else:
            priority_score = 0.05
            feedback.append("Valid but not the priority scheme for this citizen.")
    else:
        priority_score = 0.0

    score += priority_score

    # 3. Efficiency (0.0 to 0.2)
    if steps_taken <= 5:
        efficiency_score = 0.2
        feedback.append("Excellent efficiency!")
    elif steps_taken <= 7:
        efficiency_score = 0.1
        feedback.append("Good efficiency.")
    else:
        efficiency_score = 0.0
        feedback.append("Too many steps — needs to be more decisive.")

    score += efficiency_score

    # 4. Question quality (0.0 to 0.2)
    quality_score = 0.0
    key_questions = ["ask_occupation", "ask_land_ownership", "ask_income", "ask_location"]
    asked_key = [q for q in questions_asked if q in key_questions]

    if "ask_occupation" in questions_asked and questions_asked.index("ask_occupation") == 0:
        quality_score += 0.1
        feedback.append("Smart: asked occupation first.")

    if len(asked_key) >= 3:
        quality_score += 0.1
        feedback.append("Asked all critical questions including land ownership.")
    elif len(asked_key) == 2:
        quality_score += 0.05

    score += quality_score
    final_score = round(min(1.0, max(0.0, score)), 3)

    return {
        "task": "medium",
        "score": final_score,
        "scheme_score": scheme_score,
        "priority_score": priority_score,
        "efficiency_score": efficiency_score,
        "quality_score": quality_score,
        "recommended_scheme": recommended_scheme,
        "correct_schemes": MEDIUM_CITIZEN.correct_schemes,
        "priority_schemes": PRIORITY_SCHEMES,
        "steps_taken": steps_taken,
        "questions_asked": questions_asked,
        "feedback": feedback,
        "passed": final_score >= 0.4
    }

def create_task_env() -> GovSchemeEnvironment:
    return GovSchemeEnvironment(difficulty=Difficulty.MEDIUM)

def run_task_with_fixed_citizen() -> GovSchemeEnvironment:
    env = create_task_env()
    obs = env.reset()
    env.state.citizen_profile = MEDIUM_CITIZEN
    
    # Ensure PM Kisan is available
    if "PM Kisan Samman Nidhi" not in env.available_schemes:
        if len(env.available_schemes) >= env.max_steps:
            env.available_schemes.pop()
        env.available_schemes.append("PM Kisan Samman Nidhi")
        
    obs.available_schemes = env.available_schemes
    return env

if __name__ == "__main__":
    from models import Action, ActionType
    env = run_task_with_fixed_citizen()
    steps = [
        Action(action_type=ActionType.ASK_OCCUPATION),
        Action(action_type=ActionType.ASK_LAND_OWNERSHIP),
        Action(action_type=ActionType.ASK_INCOME),
        Action(action_type=ActionType.RECOMMEND_SCHEME, scheme_name="PM Kisan Samman Nidhi"),
    ]
    for action in steps:
        env.step(action)
    state = env.get_state()
    res = grade("PM Kisan Samman Nidhi", state.questions_asked, state.step_count, state.total_reward)
    print(res)