"""
TASK 2 — MEDIUM
===============
Citizen: Rural SC farmer, land owner, BPL
Schemes: 25 | Max steps: 8
Expected score: 0.5 to 0.8
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models import CitizenProfile, Gender, CasteCategory, Location, Occupation, Difficulty
from environment import GovSchemeEnvironment, ALL_SCHEMES, check_scheme_conditions

MEDIUM_CITIZEN = CitizenProfile(
    age=45, income=95000.0,
    gender=Gender.MALE, caste=CasteCategory.SC,
    location=Location.RURAL, occupation=Occupation.FARMER,
    has_disability=False, is_bpl=True,
    education="none", has_bank_account=True, has_ration_card=True,
    marital_status="married", land_ownership="owner",
    state="Maharashtra", correct_schemes=[]
)

for name, scheme in ALL_SCHEMES.items():
    is_match, _ = check_scheme_conditions(MEDIUM_CITIZEN, scheme)
    if is_match:
        MEDIUM_CITIZEN.correct_schemes.append(name)

TASK_DESCRIPTION = """
TASK 2 (MEDIUM) — Ambiguous Profile
--------------------------------------
A 45-year-old married male farmer, rural, SC category.
Annual crop income Rs.95,000. No formal education.
BPL, owns land, has bank account. State: Maharashtra.
Recommend the MOST relevant scheme for his situation.
Hint: A farmer with land ownership needs direct income support.
"""

TOP_PRIORITY = [
    "PM Kisan Samman Nidhi",
    "Kisan Credit Card",
    "Fasal Bima Yojana",
    "Pradhan Mantri Fasal Bima Yojana (PMFBY)",
    "Soil Health Card Scheme",
    "National Agriculture Market (e-NAM)",
    "Pradhan Mantri Krishi Sinchayee Yojana (PMKSY)",
    "Rashtriya Krishi Vikas Yojana (RKVY)",
    "Ayushman Bharat",
    "MGNREGA",
]


def grade(recommended_scheme, questions_asked, steps_taken, total_reward, max_steps=8):
    score = 0.01
    feedback = []

    is_correct = recommended_scheme in MEDIUM_CITIZEN.correct_schemes or recommended_scheme in TOP_PRIORITY

    if is_correct:
        if recommended_scheme in ["PM Kisan Samman Nidhi", "Kisan Credit Card"]:
            scheme_score = 0.39
            feedback.append(f"Best choice! '{recommended_scheme}' is the top priority for this farmer.")
        elif recommended_scheme in ["Fasal Bima Yojana", "Pradhan Mantri Fasal Bima Yojana (PMFBY)",
                                     "Soil Health Card Scheme", "Pradhan Mantri Krishi Sinchayee Yojana (PMKSY)"]:
            scheme_score = 0.34
            feedback.append(f"Great agricultural scheme choice — '{recommended_scheme}'.")
        else:
            scheme_score = 0.24
            feedback.append(f"Correct — '{recommended_scheme}' applies but not the top priority.")
    else:
        scheme_score = 0.01
        feedback.append("Wrong scheme. Top options: PM Kisan Samman Nidhi, Kisan Credit Card, Fasal Bima Yojana")

    score += scheme_score

    if recommended_scheme in TOP_PRIORITY[:2]:
        pri = 0.19; feedback.append("Best possible scheme chosen!")
    elif recommended_scheme in TOP_PRIORITY[2:5]:
        pri = 0.14; feedback.append("Great agricultural scheme.")
    elif is_correct:
        pri = 0.04; feedback.append("Valid but not the highest priority.")
    else:
        pri = 0.01
    score += pri

    if steps_taken <= 4:
        eff = 0.19; feedback.append("Excellent efficiency!")
    elif steps_taken <= 6:
        eff = 0.09; feedback.append("Good efficiency.")
    else:
        eff = 0.01; feedback.append("Too many steps.")
    score += eff

    qual = 0.01
    key = ["ask_occupation", "ask_land_ownership", "ask_income", "ask_location"]
    if questions_asked and questions_asked[0] == "ask_occupation":
        qual += 0.09; feedback.append("Smart: asked occupation first.")
    if len([q for q in questions_asked if q in key]) >= 3:
        qual += 0.09; feedback.append("Asked all critical questions including land ownership.")
    elif len([q for q in questions_asked if q in key]) >= 2:
        qual += 0.04
    score += qual

    # Change the very last line of your grade function
    final = float(f"{max(0.05, min(0.95, float(score))):.2f}")

    return {
        "task": "medium", "score": final,
        "scheme_score": scheme_score, "priority_score": pri,
        "efficiency_score": eff, "quality_score": qual,
        "recommended_scheme": recommended_scheme,
        "correct_schemes": MEDIUM_CITIZEN.correct_schemes[:5],
        "steps_taken": steps_taken, "questions_asked": questions_asked,
        "feedback": feedback, "passed": final >= 0.4
    }


def create_task_env():
    return GovSchemeEnvironment(difficulty=Difficulty.MEDIUM)


def run_task_with_fixed_citizen():
    env = create_task_env()
    obs = env.reset()
    env.state.citizen_profile = MEDIUM_CITIZEN
    for scheme in ["PM Kisan Samman Nidhi", "Kisan Credit Card", "Fasal Bima Yojana"]:
        if scheme not in env.available_schemes and scheme in ALL_SCHEMES:
            env.available_schemes.append(scheme)
    obs.available_schemes = env.available_schemes
    return env


if __name__ == "__main__":
    from models import Action, ActionType
    env = run_task_with_fixed_citizen()
    print(f"Correct schemes (top 5): {MEDIUM_CITIZEN.correct_schemes[:5]}")
    steps = [
        Action(action_type=ActionType.ASK_OCCUPATION),
        Action(action_type=ActionType.ASK_LAND_OWNERSHIP),
        Action(action_type=ActionType.ASK_INCOME),
        Action(action_type=ActionType.RECOMMEND_SCHEME, scheme_name="PM Kisan Samman Nidhi"),
    ]
    for a in steps:
        r = env.step(a)
        print(f"{a.action_type.value} -> {r.reward.value}")
    state = env.get_state()
    res = grade("PM Kisan Samman Nidhi", state.questions_asked, state.step_count, state.total_reward)
    print(f"Score: {res['score']} | Passed: {res['passed']} | Feedback: {res['feedback']}")