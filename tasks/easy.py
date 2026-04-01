"""
TASK 1 — EASY
=============
Citizen Profile: Obvious, single correct scheme
Schemes available: 5
Max steps: 10
Expected score for a smart agent: 0.7 to 1.0

Scenario:
  A rural BPL woman who needs LPG connection.
  The correct scheme is obviously PM Ujjwala Yojana.
  Agent should figure this out in 3-4 questions.
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
    correct_schemes=["PM Ujjwala Yojana", "Ayushman Bharat", "MGNREGA"]
)

TASK_DESCRIPTION = """
TASK 1 (EASY) — Find the Right Scheme
--------------------------------------
A 32-year-old rural woman comes to you for help.
She is a daily wage worker living in a rural area.
Her household income is Rs.85,000 per year.
She belongs to OBC category and is registered as BPL.

Your job: Ask the right questions and recommend
the most relevant government scheme for her.

Available schemes: PM Ujjwala Yojana, PM Kisan Samman Nidhi,
Ayushman Bharat, PM Scholarship Scheme, MGNREGA

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
    """
    Grade the agent's performance on the easy task.

    Scoring breakdown:
      - Correct scheme found        : 0.0 to 0.5
      - Efficiency (fewer steps)    : 0.0 to 0.3
      - Question quality            : 0.0 to 0.2
    """

    score = 0.0
    feedback = []

    # ── 1. Scheme correctness (0.0 to 0.5) ──
    if recommended_scheme in EASY_CITIZEN.correct_schemes:
        if recommended_scheme == "PM Ujjwala Yojana":
            scheme_score = 0.5      # Best match for this citizen
            feedback.append("Perfect scheme choice — PM Ujjwala Yojana is the top priority.")
        else:
            scheme_score = 0.35     # Correct but not the most relevant
            feedback.append(f"Correct scheme but not the most relevant. Best: PM Ujjwala Yojana.")
    else:
        scheme_score = 0.0
        feedback.append(f"Wrong scheme. Correct options: {EASY_CITIZEN.correct_schemes}")

    score += scheme_score

    # ── 2. Efficiency score (0.0 to 0.3) ──
    # Agent should solve this in 4 steps or fewer
    if steps_taken <= 3:
        efficiency_score = 0.3
        feedback.append("Excellent efficiency — solved in 3 steps or fewer!")
    elif steps_taken <= 5:
        efficiency_score = 0.2
        feedback.append("Good efficiency.")
    elif steps_taken <= 7:
        efficiency_score = 0.1
        feedback.append("Average efficiency — could ask fewer questions.")
    else:
        efficiency_score = 0.0
        feedback.append("Poor efficiency — too many steps taken.")

    score += efficiency_score

    # ── 3. Question quality (0.0 to 0.2) ──
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

    # ── Clamp final score ──
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


# -----------------------------------------
# TASK RUNNER — Creates env with fixed citizen
# -----------------------------------------

def create_task_env() -> GovSchemeEnvironment:
    """Create an easy difficulty environment"""
    env = GovSchemeEnvironment(difficulty=Difficulty.EASY)
    return env


def run_task_with_fixed_citizen() -> GovSchemeEnvironment:
    """
    Creates env and injects the fixed citizen profile.
    Use this for deterministic grading.
    """
    env = create_task_env()
    obs = env.reset()

    # Override with fixed citizen
    env.state.citizen_profile = EASY_CITIZEN
    obs.available_schemes = env.available_schemes

    return env


# -----------------------------------------
# QUICK TEST
# -----------------------------------------

if __name__ == "__main__":
    print("=" * 50)
    print("TASK 1 — EASY")
    print("=" * 50)
    print(TASK_DESCRIPTION)

    # Simulate a smart agent solving the easy task
    env = run_task_with_fixed_citizen()

    from models import Action, ActionType

    steps = [
        Action(action_type=ActionType.ASK_OCCUPATION),
        Action(action_type=ActionType.ASK_GENDER),
        Action(action_type=ActionType.ASK_BPL),
        Action(action_type=ActionType.RECOMMEND_SCHEME, scheme_name="PM Ujjwala Yojana"),
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
        recommended_scheme="PM Ujjwala Yojana",
        questions_asked=state.questions_asked,
        steps_taken=state.step_count,
        total_reward=state.total_reward
    )

    print(f"\n--- GRADER RESULT ---")
    print(f"Final Score : {grade_result['score']}")
    print(f"Passed      : {grade_result['passed']}")
    print(f"Feedback    : {grade_result['feedback']}")
    print("=" * 50)