"""
TASK 3 — HARD
=============
Citizen Profile: Complex — qualifies for many schemes,
                 agent must pick the highest priority scheme.
Schemes available: All
Max steps: 6
Expected score for a smart agent: 0.3 to 0.6

Scenario:
  A disabled SC student from a rural BPL family.
  Qualifies for 15+ schemes simultaneously.
  Agent must recommend the single HIGHEST PRIORITY
  scheme (Divyangjan Scholarship / Disability Pension) within just 6 steps.
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

HARD_CITIZEN = CitizenProfile(
    age=19,
    income=120000.0,         # Parent's annual income (student)
    gender=Gender.FEMALE,
    caste=CasteCategory.SC,
    location=Location.RURAL,
    occupation=Occupation.STUDENT,
    has_disability=True,
    is_bpl=True,
    education="12th",
    has_bank_account=True,
    has_ration_card=True,
    marital_status="single",
    land_ownership="none",
    state="Bihar",
    correct_schemes=[]
)

# Compute correct schemes dynamically for this citizen from ALL_SCHEMES
for name, scheme in ALL_SCHEMES.items():
    is_match, _ = check_scheme_conditions(HARD_CITIZEN, scheme)
    if is_match:
        HARD_CITIZEN.correct_schemes.append(name)

TASK_DESCRIPTION = """
TASK 3 (HARD) — Complex Overlapping Eligibility
-------------------------------------------------
A 19-year-old single woman approaches for help.
She is a student from a rural SC family, 12th pass.
Her parents earn Rs.1,20,000 per year combined.
The family is registered as BPL with Bank/Ration cards.
She has a physical disability.

Your job: With only 6 steps available, ask the
most critical questions and recommend the SINGLE
HIGHEST PRIORITY scheme for her situation.

Hint: With a disability + BPL + SC + student status,
find the scheme that specifically targets these groups.
"""

PRIORITY_SCHEMES = [
    "Post Matric Scholarship for SC Students",
    "National Overseas Scholarship for SC Students",
    "Beti Bachao Beti Padhao",
    "National Scheme of Incentive to Girls for Education",
    "Swachh Bharat Mission Gramin"
]

def grade(
    recommended_scheme: str,
    questions_asked: list,
    steps_taken: int,
    total_reward: float,
    max_steps: int = 6
) -> dict:
    score = 0.0
    feedback = []

    # 1. Scheme correctness (0.0 to 0.3)
    if recommended_scheme in HARD_CITIZEN.correct_schemes:
        scheme_score = 0.3
        feedback.append(f"Correct — '{recommended_scheme}' applies to this citizen.")
    else:
        scheme_score = 0.0
        feedback.append(f"Wrong scheme. Valid schemes: {HARD_CITIZEN.correct_schemes[:5]}")

    score += scheme_score

    # 2. Priority score (0.0 to 0.3)
    if recommended_scheme in PRIORITY_SCHEMES:
        priority_index = PRIORITY_SCHEMES.index(recommended_scheme)
        if priority_index == 0:
            priority_score = 0.3
            feedback.append("Perfect priority choice — most targeted scheme for her situation!")
        elif priority_index == 1:
            priority_score = 0.25
            feedback.append("Excellent — second highest priority scheme.")
        elif priority_index == 2:
            priority_score = 0.2
            feedback.append("Great choice — Beti Bachao is highly relevant.")
        elif priority_index <= 4:
            priority_score = 0.15
            feedback.append("Good choice but not the highest priority.")
        else:
            priority_score = 0.05
            feedback.append("Valid but lower priority scheme chosen.")
    else:
        priority_score = 0.0

    score += priority_score

    # 3. Efficiency (0.0 to 0.2)
    if steps_taken <= 4:
        efficiency_score = 0.2
        feedback.append("Outstanding efficiency — solved in 4 steps or fewer!")
    elif steps_taken <= 5:
        efficiency_score = 0.1
        feedback.append("Decent efficiency.")
    else:
        efficiency_score = 0.05
        feedback.append("Used all steps — needs better decision making.")

    score += efficiency_score

    # 4. Question quality (0.0 to 0.2)
    quality_score = 0.0
    critical_questions = ["ask_occupation", "ask_disability", "ask_caste", "ask_education", "ask_gender"]
    asked_critical = [q for q in questions_asked if q in critical_questions]

    if "ask_occupation" in questions_asked and questions_asked.index("ask_occupation") == 0:
        quality_score += 0.1
        feedback.append("Smart: asked occupation first even under time pressure.")

    if len(asked_critical) >= 3:
        quality_score += 0.1
        feedback.append("Asked crucial targeted questions.")
    elif len(asked_critical) >= 2:
        quality_score += 0.05

    score += quality_score
    final_score = round(min(1.0, max(0.0, score)), 3)

    return {
        "task": "hard",
        "score": final_score,
        "scheme_score": scheme_score,
        "priority_score": priority_score,
        "efficiency_score": efficiency_score,
        "quality_score": quality_score,
        "recommended_scheme": recommended_scheme,
        "correct_schemes": HARD_CITIZEN.correct_schemes,
        "priority_schemes": PRIORITY_SCHEMES[:3],
        "steps_taken": steps_taken,
        "questions_asked": questions_asked,
        "feedback": feedback,
        "passed": final_score >= 0.3
    }


def create_task_env() -> GovSchemeEnvironment:
    return GovSchemeEnvironment(difficulty=Difficulty.HARD)

def run_task_with_fixed_citizen() -> GovSchemeEnvironment:
    env = create_task_env()
    obs = env.reset()
    env.state.citizen_profile = HARD_CITIZEN
    obs.available_schemes = env.available_schemes
    return env

if __name__ == "__main__":
    from models import Action, ActionType
    env = run_task_with_fixed_citizen()
    steps = [
        Action(action_type=ActionType.ASK_OCCUPATION),
        Action(action_type=ActionType.ASK_CASTE),
        Action(action_type=ActionType.ASK_EDUCATION),
        Action(action_type=ActionType.RECOMMEND_SCHEME, scheme_name="Post Matric Scholarship for SC Students"),
    ]
    for action in steps:
        env.step(action)
    state = env.get_state()
    res = grade("Post Matric Scholarship for SC Students", state.questions_asked, state.step_count, state.total_reward)
    print(res)