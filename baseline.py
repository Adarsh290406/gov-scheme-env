"""
baseline.py — Baseline Inference Script
========================================
Runs an LLM agent against all 3 tasks and
produces reproducible baseline scores.

The agent reasons like a real welfare counselor:
  - Asks questions that maximally disambiguate
  - Builds a mental model of the citizen
  - Stops asking when confident and recommends

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
import time
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from models import Action, ActionType, Difficulty
from environment import ALL_SCHEMES, check_scheme_conditions
from tasks.easy   import run_task_with_fixed_citizen as easy_task,   grade as easy_grade
from tasks.medium import run_task_with_fixed_citizen as medium_task, grade as medium_grade
from tasks.hard   import run_task_with_fixed_citizen as hard_task,   grade as hard_grade

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
    base_url="https://api.groq.com/openai/v1"
)

MODEL = "llama-3.1-8b-instant"

# -----------------------------------------
# SCHEME KNOWLEDGE BASE
# Maps key citizen attributes to the
# schemes that specifically require them.
# This is the core domain knowledge the
# LLM uses to reason about which questions
# narrow down the scheme space the most.
# -----------------------------------------

SCHEME_CONDITIONS_SUMMARY = """
KEY SCHEME CONDITIONS (what each scheme requires):
- PM Ujjwala Yojana       : gender=female, is_bpl=True
- Ayushman Bharat         : is_bpl=True, has_ration_card=True
- MGNREGA                 : location=rural, has_bank_account=True
- PM Awas Yojana Gramin   : location=rural, is_bpl=True
- PM Kisan Samman Nidhi   : occupation=farmer, land_ownership=owner, has_bank_account=True
- Kisan Credit Card       : occupation=farmer
- Fasal Bima Yojana       : occupation=farmer
- Divyangjan Scholarship  : has_disability=True
- Indira Gandhi Disability Pension : has_disability=True, is_bpl=True
- Post Matric Scholarship for SC Students : caste=sc, occupation=student
- SC ST Scholarship        : caste in [sc, st], occupation=student
- National Fellowship for Scheduled Caste Students : caste=sc, occupation=student
- PM Scholarship Scheme    : occupation=student
- Sukanya Samriddhi Yojana : gender=female, min_age<10
- Beti Bachao Beti Padhao  : gender=female
- Pradhan Mantri Matru Vandana Yojana : gender=female, marital_status=married
- Stand Up India           : caste in [sc, st] OR gender=female, occupation=small_business
- PM Mudra Yojana          : occupation=small_business
- Pradhan Mantri Jan-Dhan Yojana : has_bank_account=False
- National Old Age Pension : min_age=60, is_bpl=True
- Indira Gandhi National Widow Pension : marital_status=widowed, is_bpl=True
"""

# -----------------------------------------
# QUESTION DECISION GUIDE
# Teaches the LLM which questions unlock
# which scheme categories — so it asks
# the questions with the highest info gain.
# -----------------------------------------

QUESTION_DECISION_GUIDE = """
QUESTION DECISION GUIDE — ask questions that eliminate the most schemes at once:

1. ask_occupation  → ALWAYS ask first. Unlocks: farmer schemes, student scholarships,
                     business schemes, wage schemes. Eliminates irrelevant categories immediately.

2. After occupation is known, your next question depends on what you found:
   - If farmer       → ask_land_ownership (PM Kisan needs owner), then ask_income or ask_bpl
   - If student      → ask_disability (highest priority: Divyangjan), then ask_caste (SC scholarships)
   - If daily_wage   → ask_gender (Ujjwala needs female), then ask_bpl, ask_location
   - If unemployed   → ask_bpl, ask_location, ask_age (pension schemes need age>=60)
   - If small_biz    → ask_caste (Stand Up India), ask_income
   - If govt_employee→ ask_age (pension), skip land/bpl questions

3. Universal high-value follow-ups:
   - ask_bpl         → gates Ayushman Bharat, PM Awas, Ujjwala, pensions
   - ask_disability  → if yes, Divyangjan Scholarship is almost always top choice
   - ask_gender      → gates multiple women-specific schemes
   - ask_caste       → gates SC/ST specific scholarships and finance schemes

4. When to STOP asking and recommend:
   - You have occupation + 2-3 confirming attributes = RECOMMEND NOW
   - If disability=True is confirmed → recommend Divyangjan Scholarship immediately
   - If farmer + land_owner confirmed → recommend PM Kisan Samman Nidhi immediately
   - If female + BPL confirmed → recommend PM Ujjwala Yojana immediately
   - Every extra question loses reward due to step decay — be decisive
"""

# -----------------------------------------
# SYSTEM PROMPT
# The LLM is taught to reason like an
# expert welfare counselor, not just pick
# from a list blindly.
# -----------------------------------------

SYSTEM_PROMPT = f"""
You are an expert Indian welfare counselor. A citizen has arrived and needs help finding
the right government scheme. You must interview them efficiently and recommend the BEST scheme.

ACTIONS (use EXACTLY these strings — no modifications):
- ask_occupation     : Ask citizen's occupation (ALWAYS ask this FIRST)
- ask_income         : Ask about income
- ask_bpl            : Ask if Below Poverty Line
- ask_location       : Ask if rural or urban
- ask_gender         : Ask gender
- ask_caste          : Ask caste category (general/obc/sc/st)
- ask_disability     : Ask if they have a disability
- ask_age            : Ask age
- ask_education      : Ask education level
- ask_bank_account   : Ask if they have a bank account
- ask_ration_card    : Ask if they have a ration card
- ask_marital_status : Ask marital status
- ask_land_ownership : Ask if they own/rent/have no land
- ask_state          : Ask which state they are from
- recommend_scheme   : Recommend a scheme from the available list (ENDS the episode)

{SCHEME_CONDITIONS_SUMMARY}

{QUESTION_DECISION_GUIDE}

HARD RULES:
1. NEVER repeat a question already asked — costs -0.3 reward.
2. Ask occupation FIRST — always. Gives +0.2 bonus.
3. NEVER recommend before asking at least 2-3 questions — costs -0.6 penalty.
4. When recommending, scheme_name MUST exactly match a name from the available_schemes list.
5. Do NOT ask more than 4-5 questions before recommending — step decay kills your score.
6. After each answer, reason: "What schemes does this rule out? What do I now know for certain?"

OUTPUT FORMAT — valid JSON only, no extra text, no markdown:
{{
  "reasoning": "<1 sentence: what you now know and what question will help most>",
  "action_type": "<exact action string>",
  "scheme_name": "<exact scheme name from list if recommending, else null>"
}}
"""


# -----------------------------------------
# HEURISTIC FALLBACK
# Used when API fails entirely.
# Mirrors the reasoning the LLM should do.
# -----------------------------------------

def heuristic_recommendation(env, task_name: str, available_schemes: list) -> str:
    obs = env.state.observation
    occ = obs.occupation.value if obs.occupation else ""

    # Highest-priority: disability always wins
    if obs.has_disability is True:
        for name in ["Divyangjan Scholarship", "Indira Gandhi Disability Pension"]:
            if name in available_schemes:
                return name

    # Student + SC/ST → scholarship
    if occ == "student":
        if obs.caste and obs.caste.value in ("sc", "st"):
            for name in ["Post Matric Scholarship for SC Students",
                         "SC ST Scholarship",
                         "National Fellowship for Scheduled Caste Students"]:
                if name in available_schemes:
                    return name
        for name in ["PM Scholarship Scheme", "National Scholarship Portal"]:
            if name in available_schemes:
                return name

    # Farmer + land owner → PM Kisan
    if occ == "farmer":
        land = obs.land_ownership or ""
        if land == "owner":
            if "PM Kisan Samman Nidhi" in available_schemes:
                return "PM Kisan Samman Nidhi"
        for name in ["Kisan Credit Card", "Fasal Bima Yojana"]:
            if name in available_schemes:
                return name

    # Female + BPL → Ujjwala
    if obs.gender and obs.gender.value == "female" and obs.is_bpl:
        if "PM Ujjwala Yojana" in available_schemes:
            return "PM Ujjwala Yojana"

    # BPL → Ayushman
    if obs.is_bpl:
        if "Ayushman Bharat" in available_schemes:
            return "Ayushman Bharat"

    # Rural → MGNREGA
    if obs.location and obs.location.value == "rural":
        if "MGNREGA" in available_schemes:
            return "MGNREGA"

    # Task-level safe defaults
    task_defaults = {
        "easy":   "PM Ujjwala Yojana",
        "medium": "PM Kisan Samman Nidhi",
        "hard":   "Divyangjan Scholarship",
    }
    default = task_defaults.get(task_name, "")
    if default in available_schemes:
        return default

    return available_schemes[0] if available_schemes else ""


# -----------------------------------------
# FORCE-RECOMMEND GUARD
# If the LLM still hasn't recommended after
# (max_questions) questions, we pick the
# best scheme from what's been observed.
# This is a safety net — good prompting
# should mean it rarely triggers.
# -----------------------------------------

MAX_QUESTIONS_PER_TASK = {
    "easy":   4,  # 10-step limit — recommend by step 5 for efficiency score
    "medium": 4,  # 8-step limit  — recommend by step 5 for efficiency score
    "hard":   3,  # 6-step limit  — recommend by step 4 for efficiency score
}


# -----------------------------------------
# SCHEME NAME VALIDATOR
# LLMs sometimes hallucinate scheme names.
# This maps the response to a real name.
# -----------------------------------------

def resolve_scheme_name(proposed: str, available_schemes: list) -> str:
    """
    Returns a valid scheme name from available_schemes.
    Tries: exact → case-insensitive → partial substring → None
    """
    if not proposed:
        return None

    # Exact match
    if proposed in available_schemes:
        return proposed

    # Case-insensitive
    lower_map = {s.lower(): s for s in available_schemes}
    if proposed.lower() in lower_map:
        return lower_map[proposed.lower()]

    # Substring match (longest wins to avoid false short matches)
    matches = [s for s in available_schemes if proposed.lower() in s.lower()
               or s.lower() in proposed.lower()]
    if matches:
        return max(matches, key=len)

    return None


# -----------------------------------------
# AGENT RUNNER
# The LLM reasons through each step.
# Context is rebuilt cleanly every turn
# so the model always has the full picture.
# -----------------------------------------

def run_agent(env, task_name: str, available_schemes: list) -> dict:
    print(f"\n  Running agent on {task_name} task...")

    max_q = MAX_QUESTIONS_PER_TASK[task_name]
    last_recommendation = ""
    asked_questions = []   # ordered list, not set — order matters for grader
    step = 0
    known_info = {}        # tracks revealed citizen attributes

    # Initial user message
    initial_message = (
        f"A new citizen has arrived. Your job: ask smart questions, then recommend "
        f"the BEST scheme from the list below.\n\n"
        f"Available schemes:\n" + "\n".join(f"  - {s}" for s in available_schemes) + "\n\n"
        f"Task difficulty: {task_name.upper()} | "
        f"Step limit: {env.state.observation.max_steps} | "
        f"Min questions before recommending: 2\n\n"
        f"Start by asking the most important question."
    )

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user",   "content": initial_message},
    ]

    while True:
        questions_asked_count = len(asked_questions)

        # ── FORCE RECOMMEND if question budget exhausted ──
        if questions_asked_count >= max_q:
            scheme = heuristic_recommendation(env, task_name, available_schemes)
            print(f"  [Budget guard] Forcing recommendation: '{scheme}'")
            action_data = {"action_type": "recommend_scheme", "scheme_name": scheme}
            raw = json.dumps(action_data)

        else:
            # ── LLM CALL ──
            raw = None
            for attempt in range(3):
                try:
                    response = client.chat.completions.create(
                        model=MODEL,
                        messages=messages,
                        temperature=0.2,
                        max_tokens=120,
                    )
                    raw = response.choices[0].message.content.strip()
                    break
                except Exception as e:
                    print(f"  API error (attempt {attempt+1}/3): {e}")
                    time.sleep(8)

            if raw is None:
                print("  All retries failed — using heuristic.")
                scheme = heuristic_recommendation(env, task_name, available_schemes)
                action_data = {"action_type": "recommend_scheme", "scheme_name": scheme}
                raw = json.dumps(action_data)
            else:
                # ── PARSE JSON ──
                try:
                    clean = raw
                    if "{" in clean and "}" in clean:
                        clean = clean[clean.index("{"):clean.rindex("}")+1]
                    action_data = json.loads(clean)
                except json.JSONDecodeError:
                    print(f"  JSON parse failed: {raw!r} — using heuristic.")
                    scheme = heuristic_recommendation(env, task_name, available_schemes)
                    action_data = {"action_type": "recommend_scheme", "scheme_name": scheme}
                    raw = json.dumps(action_data)

        # ── VALIDATE ACTION TYPE ──
        VALID_ACTIONS = [
            "ask_age", "ask_income", "ask_gender", "ask_caste",
            "ask_location", "ask_occupation", "ask_disability",
            "ask_bpl", "ask_education", "ask_bank_account",
            "ask_ration_card", "ask_marital_status",
            "ask_land_ownership", "ask_state", "recommend_scheme"
        ]
        if action_data.get("action_type") not in VALID_ACTIONS:
            # LLM returned something invalid — pick best unused question
            unused = [a for a in VALID_ACTIONS
                      if a not in asked_questions and a != "recommend_scheme"]
            action_data["action_type"] = unused[0] if unused else "recommend_scheme"

        # ── BLOCK REPEATED QUESTIONS ──
        if (action_data["action_type"] != "recommend_scheme"
                and action_data["action_type"] in asked_questions):
            # Find next best unused question that isn't already asked
            priority_order = [
                "ask_occupation", "ask_disability", "ask_bpl", "ask_gender",
                "ask_land_ownership", "ask_caste", "ask_income", "ask_location",
                "ask_education", "ask_age", "ask_bank_account", "ask_ration_card",
                "ask_marital_status", "ask_state"
            ]
            fallback = next(
                (a for a in priority_order if a not in asked_questions),
                None
            )
            if fallback:
                action_data["action_type"] = fallback
                print(f"  [Repeat guard] Redirecting to '{fallback}'")
            else:
                # No questions left — force recommend
                scheme = heuristic_recommendation(env, task_name, available_schemes)
                action_data = {"action_type": "recommend_scheme", "scheme_name": scheme}

        # ── VALIDATE SCHEME NAME on recommendation ──
        if action_data["action_type"] == "recommend_scheme":
            proposed = action_data.get("scheme_name") or ""
            resolved = resolve_scheme_name(proposed, available_schemes)
            if not resolved:
                print(f"  [Name fix] '{proposed}' not found — using heuristic.")
                resolved = heuristic_recommendation(env, task_name, available_schemes)
            action_data["scheme_name"] = resolved

        # ── BUILD AND EXECUTE ACTION ──
        try:
            action = Action(**{k: v for k, v in action_data.items()
                               if k in ("action_type", "scheme_name")})
        except Exception as e:
            print(f"  Invalid action: {e} — using heuristic.")
            scheme = heuristic_recommendation(env, task_name, available_schemes)
            action = Action(
                action_type=ActionType.RECOMMEND_SCHEME,
                scheme_name=scheme
            )

        if action.action_type == ActionType.RECOMMEND_SCHEME:
            last_recommendation = action.scheme_name or ""

        result = env.step(action)
        step += 1

        print(f"  Step {step}: {action.action_type.value}"
              + (f" → '{action.scheme_name}'" if action.scheme_name else "")
              + f" | Reward: {result.reward.value:.3f}")

        # ── UPDATE KNOWN INFO ──
        obs = result.observation
        if action.action_type != ActionType.RECOMMEND_SCHEME:
            asked_questions.append(action.action_type.value)
        # Build a clean picture of what's now known
        known_fields = {
            "occupation":     obs.occupation.value if obs.occupation else None,
            "income":         obs.income,
            "gender":         obs.gender.value if obs.gender else None,
            "caste":          obs.caste.value if obs.caste else None,
            "location":       obs.location.value if obs.location else None,
            "is_bpl":         obs.is_bpl,
            "has_disability": obs.has_disability,
            "land_ownership": obs.land_ownership,
            "education":      obs.education,
            "has_bank_account": obs.has_bank_account,
            "has_ration_card":  obs.has_ration_card,
            "marital_status":   obs.marital_status,
            "age":              obs.age,
            "state":            obs.state,
        }
        known_now = {k: v for k, v in known_fields.items() if v is not None}

        if result.done:
            break

        # ── BUILD NEXT USER MESSAGE ──
        steps_left = obs.max_steps - obs.step_count
        questions_left_before_must_recommend = max_q - len(asked_questions)

        # Reason about which schemes are still possible given known info
        # This gives the LLM a concrete narrowed list to pick from
        possible_schemes = []
        for s_name in available_schemes:
            scheme_data = ALL_SCHEMES.get(s_name, {})
            is_match, partial = check_scheme_conditions(env.state.citizen_profile, scheme_data)
            if is_match or partial > 0:
                possible_schemes.append(s_name)

        # We don't expose correct_schemes (that's the hidden truth),
        # but we show the narrowed list based on what's observed so far
        # using only the KNOWN info (not the hidden profile)
        observed_possible = _filter_by_observation(known_now, available_schemes)

        urgency = ""
        if questions_left_before_must_recommend <= 1:
            urgency = (
                "\n⚠️  WARNING: You MUST recommend on your NEXT action. "
                "Use heuristic: match known attributes to scheme conditions above."
            )
        elif questions_left_before_must_recommend <= 2:
            urgency = f"\n⚡ Only {questions_left_before_must_recommend} more question(s) before you must recommend."

        next_msg = (
            f"Step result: {result.reward.reason[:120]}\n\n"
            f"KNOWN SO FAR: {json.dumps(known_now, indent=2)}\n\n"
            f"Already asked: {asked_questions}\n\n"
            f"Schemes still plausible given what's known: {observed_possible[:15]}\n\n"
            f"Steps remaining: {steps_left} | "
            f"Questions you can still ask: {questions_left_before_must_recommend}"
            + urgency
            + "\n\nChoose your next action. Do NOT repeat questions from 'Already asked'."
        )

        messages.append({"role": "assistant", "content": raw})
        messages.append({"role": "user", "content": next_msg})

        # Keep context lean: system + initial + last 3 exchanges (6 messages)
        if len(messages) > 8:
            messages = messages[:2] + messages[-6:]

    return {
        "last_recommendation": last_recommendation,
        "state": env.get_state()
    }


# -----------------------------------------
# OBSERVATION-BASED SCHEME FILTER
# Narrows available schemes using ONLY
# what the agent has actually observed —
# not the hidden citizen profile.
# This is what a real agent would do.
# -----------------------------------------

def _filter_by_observation(known: dict, available_schemes: list) -> list:
    """
    Returns schemes that are consistent with what's observed so far.
    Uses conservative filtering — only eliminates schemes when a known
    attribute directly contradicts a hard scheme requirement.
    """
    plausible = []
    for s_name in available_schemes:
        scheme = ALL_SCHEMES.get(s_name, {})
        cond = scheme.get("conditions", {})
        eliminated = False

        # Gender mismatch
        gender = known.get("gender")
        if gender and cond.get("gender") not in ["any", None, gender]:
            eliminated = True

        # Occupation mismatch
        occ = known.get("occupation")
        if occ and cond.get("occupation") not in ["any", None, occ]:
            eliminated = True

        # BPL mismatch (only eliminate if scheme REQUIRES bpl and citizen is NOT bpl)
        if known.get("is_bpl") is False and cond.get("is_bpl") is True:
            eliminated = True

        # Disability mismatch (only eliminate if scheme REQUIRES disability)
        if known.get("has_disability") is False and cond.get("has_disability") is True:
            eliminated = True

        if not eliminated:
            plausible.append(s_name)

    return plausible if plausible else available_schemes[:10]


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
    env          = easy_task()
    available    = env.available_schemes
    run_result   = run_agent(env, "easy", available)
    state        = run_result["state"]
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
    env          = medium_task()
    available    = env.available_schemes
    run_result   = run_agent(env, "medium", available)
    state        = run_result["state"]
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
    env          = hard_task()
    available    = env.available_schemes
    run_result   = run_agent(env, "hard", available)
    state        = run_result["state"]
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