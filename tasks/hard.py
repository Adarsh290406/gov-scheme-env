"""
TASK 3 — HARD
=============
Citizen Profile: Complex — qualifies for many schemes,
                 agent must find ALL of them, not just one
Schemes available: 12 (all schemes)
Max steps: 6 (very tight!)
Expected score for a smart agent: 0.3 to 0.6

Scenario:
  A disabled SC student from a rural BPL family.
  Qualifies for 5+ schemes simultaneously.
  Agent must recommend the single HIGHEST PRIORITY
  scheme within just 6 steps.
  Hardest because: all 12 schemes, only 6 steps,
  complex overlapping eligibility.
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

HARD_CITIZEN = CitizenProfile(
    age=19,
    income=120000.0,         # Parent's annual income (student)
    gender=Gender.FEMALE,
    caste=CasteCategory.SC,
    location=Location.RURAL,
    occupation=Occupation.STUDENT,
    has_disability=True,
    is_bpl=True,
    correct_schemes=[
        "Ayushman Bharat",
        "PM Scholarship Scheme",
        "SC/ST Scholarship",
        "Divyangjan Scholarship",
        "MGNREGA",
        "PM Awas Yojana (Gramin)",
        "Indira Gandhi National Disability Pension",
        "PM Ujjwala Yojana",
    ]
)

TASK_DESCRIPTION = """
TASK 3 (HARD) — Complex Overlapping Eligibility
-------------------------------------------------
A 19-year-old woman approaches for help.
She is a student from a rural SC family.
Her parents earn Rs.1,20,000 per year combined.
The family is registered as BPL.
She has a physical disability.

Your job: With only 6 steps available, ask the
most critical questions and recommend the SINGLE
HIGHEST PRIORITY scheme for her situation.

All 12 schemes are available. Multiple schemes
apply — but you must pick the most impactful one.

Hint: With a disability + BPL + student status,
which single scheme would change her life the most?
"""


# -----------------------------------------
# PRIORITY RANKING FOR HARD TASK
# -----------------------------------------

PRIORITY_SCHEMES = [
    "Divyangjan Scholarship",                   # Disability-specific, most targeted
    "Indira Gandhi National Disability Pension", # BPL + disability = direct income
    "SC/ST Scholarship",                         # SC student — targeted support
    "PM Scholarship Scheme",                     # General student support
    "Ayushman Bharat",                           # Health insurance for BPL
    "PM Ujjwala Yojana",                         # BPL woman — LPG
    "PM Awas Yojana (Gramin)",                   # Rural BPL housing
    "MGNREGA",                                   # Rural employment
]


def grade(
    recommended_scheme: str,
    questions_asked: list,
    steps_taken: int,
    total_reward: float,
    max_steps: int = 6
) -> dict:
    """
    Grade the agent's performance on the hard task.

    Scoring breakdown:
      - Scheme in correct list      : 0.0 to 0.3
      - Priority of scheme chosen   : 0.0 to 0.3
      - Efficiency (6 step limit!)  : 0.0 to 0.2
      - Question quality            : 0.0 to 0.2
    """

    score = 0.0
    feedback = []

    # ── 1. Scheme correctness (0.0 to 0.3) ──
    if recommended_scheme in HARD_CITIZEN.correct_schemes:
        scheme_score = 0.3
        feedback.append(f"Correct — '{recommended_scheme}' applies to this citizen.")
    else:
        scheme_score = 0.0
        feedback.append(f"Wrong scheme. This citizen qualifies for {len(HARD_CITIZEN.correct_schemes)} schemes.")

    score += scheme_score

    # ── 2. Priority score (0.0 to 0.3) ──
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
            feedback.append("Great choice — SC/ST scholarship is very relevant.")
        elif priority_index <= 4:
            priority_score = 0.15
            feedback.append("Good choice but not the highest priority.")
        else:
            priority_score = 0.05
            feedback.append("Valid but lower priority scheme chosen.")
    else:
        priority_score = 0.0

    score += priority_score

    # ── 3. Efficiency (0.0 to 0.2) ──
    # Only 6 steps — must be very decisive
    if steps_taken <= 3:
        efficiency_score = 0.2
        feedback.append("Outstanding efficiency — solved in 3 steps or fewer!")
    elif steps_taken <= 4:
        efficiency_score = 0.15
        feedback.append("Very efficient.")
    elif steps_taken <= 5:
        efficiency_score = 0.1
        feedback.append("Decent efficiency.")
    else:
        efficiency_score = 0.05
        feedback.append("Used all steps — needs better decision making.")

    score += efficiency_score

    # ── 4. Question quality (0.0 to 0.2) ──
    # On hard, asking disability + occupation + caste is critical
    quality_score = 0.0
    critical_questions = ["ask_occupation", "ask_disability", "ask_caste", "ask_bpl"]
    asked_critical = [q for q in questions_asked if q in critical_questions]

    if "ask_occupation" in questions_asked and questions_asked.index("ask_occupation") == 0:
        quality_score += 0.1
        feedback.append("Smart: asked occupation first even under time pressure.")

    if "ask_disability" in questions_asked:
        quality_score += 0.05
        feedback.append("Good: checked for disability status.")

    if len(asked_critical) >= 3:
        quality_score += 0.05
        feedback.append("Asked the right critical questions.")

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


# -----------------------------------------
# TASK RUNNER
# -----------------------------------------

def create_task_env() -> GovSchemeEnvironment:
    env = GovSchemeEnvironment(difficulty=Difficulty.HARD)
    return env


def run_task_with_fixed_citizen() -> GovSchemeEnvironment:
    env = create_task_env()
    obs = env.reset()
    env.state.citizen_profile = HARD_CITIZEN
    obs.available_schemes = env.available_schemes
    return env


# -----------------------------------------
# QUICK TEST
# -----------------------------------------

if __name__ == "__main__":
    print("=" * 55)
    print("TASK 3 — HARD")
    print("=" * 55)
    print(TASK_DESCRIPTION)

    env = run_task_with_fixed_citizen()

    from models import Action, ActionType

    # Smart agent — asks only the most critical questions
    steps = [
        Action(action_type=ActionType.ASK_OCCUPATION),
        Action(action_type=ActionType.ASK_DISABILITY),
        Action(action_type=ActionType.ASK_CASTE),
        Action(action_type=ActionType.RECOMMEND_SCHEME, scheme_name="Divyangjan Scholarship"),
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
        recommended_scheme="Divyangjan Scholarship",
        questions_asked=state.questions_asked,
        steps_taken=state.step_count,
        total_reward=state.total_reward
    )

    print(f"\n--- GRADER RESULT ---")
    print(f"Final Score : {grade_result['score']}")
    print(f"Passed      : {grade_result['passed']}")
    print(f"Feedback    : {grade_result['feedback']}")
    print("=" * 55)