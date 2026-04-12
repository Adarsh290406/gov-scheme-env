"""
TASK 1 — EASY
=============
Citizen: Rural BPL woman, daily wage worker
Schemes: 10 | Max steps: 10
Expected score: 0.7 to 1.0
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models import CitizenProfile, Gender, CasteCategory, Location, Occupation, Difficulty
from environment import GovSchemeEnvironment, ALL_SCHEMES, check_scheme_conditions

EASY_CITIZEN = CitizenProfile(
    age=32, income=85000.0,
    gender=Gender.FEMALE, caste=CasteCategory.OBC,
    location=Location.RURAL, occupation=Occupation.DAILY_WAGE,
    has_disability=False, is_bpl=True,
    education="8th", has_bank_account=True, has_ration_card=True,
    marital_status="married", land_ownership="none",
    state="Uttar Pradesh", correct_schemes=[]
)

for name, scheme in ALL_SCHEMES.items():
    is_match, _ = check_scheme_conditions(EASY_CITIZEN, scheme)
    if is_match:
        EASY_CITIZEN.correct_schemes.append(name)

TASK_DESCRIPTION = """
TASK 1 (EASY) — Find the Right Scheme
--------------------------------------
A 32-year-old married rural woman, daily wage worker.
Household income Rs.85,000/year. OBC, BPL, 8th pass.
Has bank account and ration card. State: Uttar Pradesh.
Recommend the most relevant government scheme for her.
Hint: Think about what a rural BPL woman needs most.
"""

TOP_PRIORITY = ["PM Ujjwala Yojana", "Ujjwala Scheme", "Ayushman Bharat",
                "MGNREGA", "PM Awas Yojana Gramin", "Pradhan Mantri Matru Vandana Yojana",
                "Beti Bachao Beti Padhao", "Pradhan Mantri Jan-Dhan Yojana",
                "Pradhan Mantri Mahila Shakti Kendra", "Pradhan Mantri Kaushal Vikas Yojana"]


def grade(recommended_scheme, questions_asked, steps_taken, total_reward, max_steps=10):
    score = 0.01
    feedback = []

    is_correct = recommended_scheme in EASY_CITIZEN.correct_schemes or recommended_scheme in TOP_PRIORITY

    if is_correct:
        if recommended_scheme in ["PM Ujjwala Yojana", "Ujjwala Scheme"]:
            scheme_score = 0.49
            feedback.append("Perfect! PM Ujjwala Yojana is the top priority for a rural BPL woman.")
        elif recommended_scheme in ["Ayushman Bharat", "MGNREGA", "PM Awas Yojana Gramin"]:
            scheme_score = 0.39
            feedback.append(f"Great choice! '{recommended_scheme}' is highly relevant.")
        else:
            scheme_score = 0.29
            feedback.append(f"Correct — '{recommended_scheme}' applies to this citizen.")
    else:
        scheme_score = 0.01
        feedback.append("Wrong scheme. Top options: PM Ujjwala Yojana, Ayushman Bharat, MGNREGA")

    score += scheme_score

    if steps_taken <= 4:
        eff = 0.29; feedback.append("Excellent efficiency!")
    elif steps_taken <= 6:
        eff = 0.19; feedback.append("Good efficiency.")
    elif steps_taken <= 8:
        eff = 0.09; feedback.append("Average efficiency.")
    else:
        eff = 0.01; feedback.append("Poor efficiency.")
    score += eff

    qual = 0.01
    key = ["ask_occupation", "ask_bpl", "ask_gender", "ask_location"]
    if questions_asked and questions_asked[0] == "ask_occupation":
        qual += 0.09; feedback.append("Smart: asked occupation first.")
    if len([q for q in questions_asked if q in key]) >= 3:
        qual += 0.09; feedback.append("Asked the right questions.")
    elif len([q for q in questions_asked if q in key]) >= 2:
        qual += 0.04
    score += qual

    # Change the very last line of your grade function
    final = round(max(0.05, min(0.95, score)), 2)

    return {
        "task": "easy", "score": final,
        "scheme_score": scheme_score, "efficiency_score": eff, "quality_score": qual,
        "recommended_scheme": recommended_scheme,
        "correct_schemes": EASY_CITIZEN.correct_schemes[:5],
        "steps_taken": steps_taken, "questions_asked": questions_asked,
        "feedback": feedback, "passed": final >= 0.5
    }


def create_task_env():
    return GovSchemeEnvironment(difficulty=Difficulty.EASY)


def run_task_with_fixed_citizen():
    env = create_task_env()
    obs = env.reset()
    env.state.citizen_profile = EASY_CITIZEN
    for scheme in ["PM Ujjwala Yojana", "Ayushman Bharat", "MGNREGA"]:
        if scheme not in env.available_schemes and scheme in ALL_SCHEMES:
            env.available_schemes.append(scheme)
    obs.available_schemes = env.available_schemes
    return env


if __name__ == "__main__":
    from models import Action, ActionType
    env = run_task_with_fixed_citizen()
    print(f"Correct schemes (top 5): {EASY_CITIZEN.correct_schemes[:5]}")
    steps = [
        Action(action_type=ActionType.ASK_OCCUPATION),
        Action(action_type=ActionType.ASK_GENDER),
        Action(action_type=ActionType.ASK_BPL),
        Action(action_type=ActionType.RECOMMEND_SCHEME, scheme_name="PM Ujjwala Yojana"),
    ]
    for a in steps:
        r = env.step(a)
        print(f"{a.action_type.value} -> {r.reward.value}")
    state = env.get_state()
    res = grade("PM Ujjwala Yojana", state.questions_asked, state.step_count, state.total_reward)
    print(f"Score: {res['score']} | Passed: {res['passed']} | Feedback: {res['feedback']}")