"""
baseline.py — Baseline Inference Script
========================================
Runs an LLM agent against all 3 tasks and
produces reproducible baseline scores.

Uses the OpenAI client library pointed at Groq's
free API (fully OpenAI-compatible).

Usage:
  python baseline.py

Requirements:
  - OPENAI_API_KEY set in .env file (your Groq key)
  - pip install openai python-dotenv
"""

import os
import sys
import json
from dotenv import load_dotenv
from openai import OpenAI

# Load .env file
load_dotenv()

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from models import Action, ActionType, Difficulty
from tasks.easy import run_task_with_fixed_citizen as easy_task, grade as easy_grade
from tasks.medium import run_task_with_fixed_citizen as medium_task, grade as medium_grade
from tasks.hard import run_task_with_fixed_citizen as hard_task, grade as hard_grade

# -----------------------------------------
# SETUP OPENAI CLIENT → POINTING TO GROQ
# -----------------------------------------

api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    raise ValueError(
        "OPENAI_API_KEY not found! "
        "Make sure you have a .env file with your Groq key."
    )

client = OpenAI(
    api_key=api_key,
    base_url="https://api.groq.com/openai/v1"  # Free Groq API
)

MODEL = "llama-3.1-8b-instant"   # Free Groq model

# -----------------------------------------
# SYSTEM PROMPT
# Tells the LLM how to behave as an agent
# -----------------------------------------

SYSTEM_PROMPT = """
You are an AI agent helping Indian citizens find the right government schemes.
You interact with an environment using a strict JSON format.

AVAILABLE ACTIONS:
- ask_occupation   : Ask citizen's occupation (ask this FIRST always)
- ask_income       : Ask citizen's income
- ask_bpl          : Ask if citizen is Below Poverty Line
- ask_location     : Ask if citizen is rural or urban
- ask_gender       : Ask citizen's gender
- ask_caste        : Ask citizen's caste category
- ask_disability   : Ask if citizen has a disability
- ask_age          : Ask citizen's age
- recommend_scheme : Recommend a government scheme (ends episode)

RULES:
1. Always ask occupation FIRST
2. Never repeat a question
3. Recommend a scheme as soon as you have enough information
4. Be efficient — you have limited steps

RESPONSE FORMAT:
You must ALWAYS respond with valid JSON only. No extra text.

For asking a question:
{"action_type": "ask_occupation"}

For recommending a scheme:
{"action_type": "recommend_scheme", "scheme_name": "PM Ujjwala Yojana"}
"""

# -----------------------------------------
# AGENT RUNNER
# Runs the LLM agent against one task
# -----------------------------------------

def run_agent(env, task_name: str, available_schemes: list) -> dict:
    """
    Runs the LLM agent against the environment
    until the episode is done or step limit reached.
    Returns the final state for grading.
    """

    print(f"\n  Running agent on {task_name} task...")

    # Build conversation history
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {
            "role": "user",
            "content": (
                f"New citizen has arrived. Help them find the right government scheme.\n"
                f"Available schemes: {json.dumps(available_schemes)}\n"
                f"Start by asking the most important question first."
            )
        }
    ]

    last_recommendation = ""
    step = 0

    while True:
        # Get LLM response
        try:
            response = client.chat.completions.create(
                model=MODEL,
                messages=messages,
                temperature=0.1,   # Low temperature = more deterministic
                max_tokens=100,
            )
            raw = response.choices[0].message.content.strip()

        except Exception as e:
            print(f"  API error: {e}")
            break

        # Parse JSON response
        try:
            # Clean up response in case model adds extra text
            if "{" in raw and "}" in raw:
                raw = raw[raw.index("{"):raw.rindex("}")+1]
            action_data = json.loads(raw)
        except json.JSONDecodeError:
            print(f"  Could not parse response: {raw}")
            break

        # Build action
        try:
            action = Action(**action_data)
        except Exception as e:
            print(f"  Invalid action: {e}")
            break

        # Track recommendation
        if action.action_type == ActionType.RECOMMEND_SCHEME:
            last_recommendation = action.scheme_name or ""

        # Take step in environment
        result = env.step(action)
        step += 1

        print(f"  Step {step}: {action.action_type.value}"
              + (f" → {action.scheme_name}" if action.scheme_name else "")
              + f" | Reward: {result.reward.value}")

        # Add result to conversation so LLM knows what happened
        messages.append({"role": "assistant", "content": raw})
        messages.append({
            "role": "user",
            "content": (
                f"Result: {result.reward.reason}\n"
                f"Current info: {result.observation.model_dump_json()}\n"
                + ("Episode done." if result.done else "Continue.")
            )
        })

        if result.done:
            break

    return {
        "last_recommendation": last_recommendation,
        "state": env.get_state()
    }


# -----------------------------------------
# MAIN — Run all 3 tasks
# -----------------------------------------

def main():
    print("=" * 60)
    print("Gov Scheme Finder — Baseline Inference Script")
    print(f"Model: {MODEL} (via Groq)")
    print("=" * 60)

    results = {}

    # ── TASK 1: EASY ──
    print("\n[TASK 1] EASY")
    print("-" * 40)
    env = easy_task()
    available = env.available_schemes
    run_result = run_agent(env, "easy", available)
    state = run_result["state"]
    grade_result = easy_grade(
        recommended_scheme=run_result["last_recommendation"],
        questions_asked=state.questions_asked,
        steps_taken=state.step_count,
        total_reward=state.total_reward
    )
    results["easy"] = grade_result
    print(f"\n  Score    : {grade_result['score']}")
    print(f"  Passed   : {grade_result['passed']}")
    print(f"  Feedback : {grade_result['feedback']}")

    # ── TASK 2: MEDIUM ──
    print("\n[TASK 2] MEDIUM")
    print("-" * 40)
    env = medium_task()
    available = env.available_schemes
    run_result = run_agent(env, "medium", available)
    state = run_result["state"]
    grade_result = medium_grade(
        recommended_scheme=run_result["last_recommendation"],
        questions_asked=state.questions_asked,
        steps_taken=state.step_count,
        total_reward=state.total_reward
    )
    results["medium"] = grade_result
    print(f"\n  Score    : {grade_result['score']}")
    print(f"  Passed   : {grade_result['passed']}")
    print(f"  Feedback : {grade_result['feedback']}")

    # ── TASK 3: HARD ──
    print("\n[TASK 3] HARD")
    print("-" * 40)
    env = hard_task()
    available = env.available_schemes
    run_result = run_agent(env, "hard", available)
    state = run_result["state"]
    grade_result = hard_grade(
        recommended_scheme=run_result["last_recommendation"],
        questions_asked=state.questions_asked,
        steps_taken=state.step_count,
        total_reward=state.total_reward
    )
    results["hard"] = grade_result
    print(f"\n  Score    : {grade_result['score']}")
    print(f"  Passed   : {grade_result['passed']}")
    print(f"  Feedback : {grade_result['feedback']}")

    # ── FINAL SUMMARY ──
    avg_score = round(
        sum(r["score"] for r in results.values()) / len(results), 3
    )

    print("\n" + "=" * 60)
    print("BASELINE RESULTS SUMMARY")
    print("=" * 60)
    print(f"  Task 1 (Easy)   : {results['easy']['score']}")
    print(f"  Task 2 (Medium) : {results['medium']['score']}")
    print(f"  Task 3 (Hard)   : {results['hard']['score']}")
    print(f"  Average Score   : {avg_score}")
    print("=" * 60)
    print("\nCopy these scores into your README.md baseline section!")


if __name__ == "__main__":
    main()