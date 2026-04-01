"""
TASK 2 — MEDIUM
===============
Citizen Profile: Ambiguous — qualifies for multiple schemes
Schemes available: 10
Max steps: 8
Expected score for a smart agent: 0.5 to 0.8

Scenario:
  A rural SC category farmer with low income.
  Qualifies for multiple schemes — agent must
  identify the MOST relevant one efficiently.
  Harder because more schemes, fewer steps.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


from models import (
    CitizenProfile, Gender, CasteCategory,
    Location, Occupation, Difficulty
)
from environment import GovSchemeEnvironment


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
    correct_schemes=[
        "PM Kisan Samman Nidhi",
        "Ayushman Bharat",
        "MGNREGA",
        "PM Awas Yojana (Gramin)",
        "Fasal Bima Yojana"
    ]
)

TASK_DESCRIPTION = """
TASK 2 (MEDIUM) — Ambiguous Profile
--------------------------------------
A 45-year-old man comes for help.
He is a farmer in a rural area.
His annual crop income is Rs.95,000.
He belongs to SC category and is registered as BPL.

Your job: Ask smart questions and recommend
the MOST relevant scheme for his situation.

Available schemes: 10 schemes across farming,
housing, health, and employment categories.

Hint: A farmer with BPL status qualifies for
multiple schemes — find the most impactful one.
"""


# -----------------------------------------
# GRADER
# -----------------------------------------

# Priority ranking — most relevant scheme first
PRIORITY_SCHEMES = [
    "PM Kisan Samman Nidhi",       # Direct income support for farmer
    "Fasal Bima Yojana",           # Crop insurance — very relevant
    "Ayushman Bharat",             # Health insurance for BPL
    "PM Awas Yojana (Gramin)",     # Housing for rural BPL
    "MGNREGA",                     # Employment guarantee
]


def grade(
    recommended_scheme: str,
    questions_asked: list,
    steps_taken: int,
    total_reward: float,
    max_steps: int = 8
) -> dict:
    """
    Grade the agent's performance on the medium task.

    Scoring breakdown:
      - Correct scheme found        : 0.0 to 0.4
      - Priority of scheme chosen   : 0.0 to 0.2
      - Efficiency                  : 0.0 to 0.2
      - Question quality            : 0.0 to 0.2
    """

    score = 0.0
    feedback = []

    # ── 1. Scheme correctness (0.0 to 0.4) ──
    if recommended_scheme in MEDIUM_CITIZEN.correct_schemes:
        scheme_score = 0.4
        feedback.append(f"Correct! '{recommended_scheme}' is valid for this citizen.")
    else:
        scheme_score = 0.0
        feedback.append(f"Wrong scheme. Valid options: {MEDIUM_CITIZEN.correct_schemes}")

    score += scheme_score

    # ── 2. Priority score (0.0 to 0.2) ──
    # Rewards picking the MOST relevant scheme
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

    # ── 3. Efficiency (0.0 to 0.2) ──
    # Should solve in 5 steps or fewer with 8 max
    if steps_taken <= 4:
        efficiency_score = 0.2
        feedback.append("Excellent efficiency!")
    elif steps_taken <= 6:
        efficiency_score = 0.1
        feedback.append("Good efficiency.")
    else:
        efficiency_score = 0.0
        feedback.append("Too many steps — needs to be more decisive.")

    score += efficiency_score

    # ── 4. Question quality (0.0 to 0.2) ──
    quality_score = 0.0
    key_questions = ["ask_occupation", "ask_bpl", "ask_income", "ask_location"]
    asked_key = [q for q in questions_asked if q in key_questions]

    if "ask_occupation" in questions_asked and questions_asked.index("ask_occupation") == 0:
        quality_score += 0.1
        feedback.append("Smart: asked occupation first.")

    if len(asked_key) >= 3:
        quality_score += 0.1
        feedback.append("Asked all critical questions.")
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


# -----------------------------------------
# TASK RUNNER
# -----------------------------------------

def create_task_env() -> GovSchemeEnvironment:
    env = GovSchemeEnvironment(difficulty=Difficulty.MEDIUM)
    return env


def run_task_with_fixed_citizen() -> GovSchemeEnvironment:
    env = create_task_env()
    obs = env.reset()
    env.state.citizen_profile = MEDIUM_CITIZEN
    obs.available_schemes = env.available_schemes
    return env


# -----------------------------------------
# QUICK TEST
# -----------------------------------------

if __name__ == "__main__":
    print("=" * 50)
    print("TASK 2 — MEDIUM")
    print("=" * 50)
    print(TASK_DESCRIPTION)

    env = run_task_with_fixed_citizen()

    from models import Action, ActionType

    steps = [
        Action(action_type=ActionType.ASK_OCCUPATION),
        Action(action_type=ActionType.ASK_INCOME),
        Action(action_type=ActionType.ASK_BPL),
        Action(action_type=ActionType.ASK_LOCATION),
        Action(action_type=ActionType.RECOMMEND_SCHEME, scheme_name="PM Kisan Samman Nidhi"),
    ]

    print("Simulating smart agent...\n")
    for i, action in enumerate(steps):
        result = env.step(action)
        print(f"Step {i+1}: {action.action_type.value}")
        print(f"  Reward : {result.reward.value}")
        print(f"  Reason : {result.reward.reason[:80]}")
        if result.done:
            break

    state = env.get_state()
    grade_result = grade(
        recommended_scheme="PM Kisan Samman Nidhi",
        questions_asked=state.questions_asked,
        steps_taken=state.step_count,
        total_reward=state.total_reward
    )

    print(f"\n--- GRADER RESULT ---")
    print(f"Final Score : {grade_result['score']}")
    print(f"Passed      : {grade_result['passed']}")
    print(f"Feedback    : {grade_result['feedback']}")
    print("=" * 50)