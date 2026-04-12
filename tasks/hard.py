"""
TASK 3 — HARD
=============
Citizen: Disabled SC female student, rural BPL
Schemes: All 67 | Max steps: 6
Expected score: 0.3 to 0.6
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models import CitizenProfile, Gender, CasteCategory, Location, Occupation, Difficulty
from environment import GovSchemeEnvironment, ALL_SCHEMES, check_scheme_conditions

HARD_CITIZEN = CitizenProfile(
    age=19, income=120000.0,
    gender=Gender.FEMALE, caste=CasteCategory.SC,
    location=Location.RURAL, occupation=Occupation.STUDENT,
    has_disability=True, is_bpl=True,
    education="12th", has_bank_account=True, has_ration_card=True,
    marital_status="single", land_ownership="none",
    state="Bihar", correct_schemes=[]
)

for name, scheme in ALL_SCHEMES.items():
    is_match, _ = check_scheme_conditions(HARD_CITIZEN, scheme)
    if is_match:
        HARD_CITIZEN.correct_schemes.append(name)

TASK_DESCRIPTION = """
TASK 3 (HARD) — Complex Overlapping Eligibility
-------------------------------------------------
A 19-year-old single female student, SC category, rural Bihar.
Parents earn Rs.1,20,000/year. 12th pass, BPL.
Has physical disability, bank account and ration card.
With only 6 steps — find the HIGHEST PRIORITY scheme.
Hint: Disability + SC + student = very targeted schemes exist.
"""

# Priority based on actual scheme names in schemes.json
TOP_PRIORITY = [
    "Divyangjan Scholarship",
    "Indira Gandhi Disability Pension",
    "Post Matric Scholarship for SC Students",
    "Post-Matric Scholarship Scheme for SC Students",
    "National Overseas Scholarship for SC Students",
    "National Overseas Scholarship Scheme for SC Students",
    "National Fellowship for Scheduled Caste Students",
    "SC ST Scholarship",
    "PM Scholarship Scheme",
    "National Scholarship Portal",
    "Beti Bachao Beti Padhao",
    "National Scheme of Incentive to Girls for Education",
    "Pradhan Mantri Matru Vandana Yojana",
    "Ayushman Bharat",
]


def grade(recommended_scheme, questions_asked, steps_taken, total_reward, max_steps=6):
    score = 0.0
    feedback = []

    is_correct = recommended_scheme in HARD_CITIZEN.correct_schemes or recommended_scheme in TOP_PRIORITY

    if is_correct:
        scheme_score = 0.29
        feedback.append(f"Correct — '{recommended_scheme}' applies to this citizen.")
    else:
        scheme_score = 0.01
        feedback.append(f"Wrong scheme. Top options: Divyangjan Scholarship, Post Matric Scholarship for SC Students")
    score += scheme_score

    if recommended_scheme in TOP_PRIORITY[:2]:
        pri = 0.29; feedback.append("Perfect priority — most targeted scheme for disability + SC!")
    elif recommended_scheme in TOP_PRIORITY[2:6]:
        pri = 0.19; feedback.append("Excellent — highly targeted SC scholarship chosen.")
    elif recommended_scheme in TOP_PRIORITY[6:10]:
        pri = 0.14; feedback.append("Good scholarship choice.")
    elif is_correct:
        pri = 0.04; feedback.append("Valid but not the highest priority.")
    else:
        pri = 0.01
    score += pri

    if steps_taken <= 3:
        eff = 0.19; feedback.append("Outstanding efficiency!")
    elif steps_taken <= 4:
        eff = 0.14; feedback.append("Very efficient.")
    elif steps_taken <= 5:
        eff = 0.09; feedback.append("Decent efficiency.")
    else:
        eff = 0.01; feedback.append("Used all steps — needs better decision making.")
    score += eff

    qual = 0.01
    critical = ["ask_occupation", "ask_disability", "ask_caste", "ask_education", "ask_gender"]
    if questions_asked and questions_asked[0] == "ask_occupation":
        qual += 0.09; feedback.append("Smart: asked occupation first under time pressure.")
    if len([q for q in questions_asked if q in critical]) >= 3:
        qual += 0.09; feedback.append("Asked crucial targeted questions.")
    elif len([q for q in questions_asked if q in critical]) >= 2:
        qual += 0.04
    score += qual

    # Change the very last line of your grade function
    final = round(max(0.05, min(0.95, score)), 2)

    return {
        "task": "hard", "score": final,
        "scheme_score": scheme_score, "priority_score": pri,
        "efficiency_score": eff, "quality_score": qual,
        "recommended_scheme": recommended_scheme,
        "correct_schemes": HARD_CITIZEN.correct_schemes[:5],
        "top_priority": TOP_PRIORITY[:3],
        "steps_taken": steps_taken, "questions_asked": questions_asked,
        "feedback": feedback, "passed": final >= 0.3
    }


def create_task_env():
    return GovSchemeEnvironment(difficulty=Difficulty.HARD)


def run_task_with_fixed_citizen():
    env = create_task_env()
    obs = env.reset()
    env.state.citizen_profile = HARD_CITIZEN
    obs.available_schemes = env.available_schemes
    return env


if __name__ == "__main__":
    from models import Action, ActionType
    env = run_task_with_fixed_citizen()
    print(f"Correct schemes (top 5): {HARD_CITIZEN.correct_schemes[:5]}")
    steps = [
        Action(action_type=ActionType.ASK_OCCUPATION),
        Action(action_type=ActionType.ASK_DISABILITY),
        Action(action_type=ActionType.ASK_CASTE),
        Action(action_type=ActionType.RECOMMEND_SCHEME, scheme_name="Divyangjan Scholarship"),
    ]
    for a in steps:
        r = env.step(a)
        print(f"{a.action_type.value} -> {r.reward.value}")
    state = env.get_state()
    res = grade("Divyangjan Scholarship", state.questions_asked, state.step_count, state.total_reward)
    print(f"Score: {res['score']} | Passed: {res['passed']} | Feedback: {res['feedback']}")