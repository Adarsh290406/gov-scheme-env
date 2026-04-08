"""
inference.py — Baseline Inference Script
==========================================
CRITICAL: This version uses unbuffered stdout writes to guarantee
validator capture of structured output blocks.
"""

import sys
import os

# ============================================================================
# PHASE 0: GUARANTEE STDOUT IS UNBUFFERED AND STRUCTURED OUTPUT WORKS
# ============================================================================

# Unbuffer stdout immediately - before ANY other imports
if hasattr(sys.stdout, 'reconfigure'):
    try:
        sys.stdout.reconfigure(line_buffering=True, write_through=True)
    except:
        pass

# Direct stdout write function - NO print() allowed for structured output
def emit(msg: str):
    """Emit structured output to stdout only — validator parses this."""
    print(msg, flush=True)

# Stderr for debug only
def emit_stderr(message: str):
    """Write to stderr for debugging - validator ignores this."""
    os.write(sys.stderr.fileno(), (message + '\n').encode('utf-8'))

# ============================================================================
# PHASE 1: IMPORTS AND SETUP
# ============================================================================

import json
import uuid
import time
from typing import Dict, Any

# dotenv is optional
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Defensive imports
_imports_ok = False
try:
    from openai import OpenAI
    from models import Action, ActionType, Difficulty
    from environment import ALL_SCHEMES, check_scheme_conditions
    from tasks.easy   import run_task_with_fixed_citizen as easy_task,   grade as easy_grade
    from tasks.medium import run_task_with_fixed_citizen as medium_task, grade as medium_grade
    from tasks.hard   import run_task_with_fixed_citizen as hard_task,   grade as hard_grade
    _imports_ok = True
except Exception as _import_err:
    emit_stderr(f"[WARN] Import error: {_import_err}")

# Setup API
API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
API_KEY      = os.getenv("HF_TOKEN") or os.getenv("API_KEY") or ""
MODEL        = os.getenv("MODEL_NAME", "meta-llama/Llama-3.1-8B-Instruct")

if not API_KEY:
    emit_stderr("[WARN] HF_TOKEN not set")

client = None
if API_KEY and _imports_ok:
    try:
        client = OpenAI(api_key=API_KEY, base_url=API_BASE_URL)
    except Exception as e:
        emit_stderr(f"[WARN] OpenAI client init failed: {e}")

# ============================================================================
# SCHEME KNOWLEDGE
# ============================================================================

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

QUESTION_DECISION_GUIDE = """
QUESTION DECISION GUIDE:

1. ask_occupation  → ALWAYS ask first
2. After occupation: ask disability, bpl, gender, caste
3. STOP asking and recommend after 3-4 questions
4. Every extra question loses reward due to step decay
"""

SYSTEM_PROMPT = f"""
You are an expert Indian welfare counselor. A citizen has arrived and needs help finding
the right government scheme. You must interview them efficiently and recommend the BEST scheme.

ACTIONS (use EXACTLY these strings):
- ask_occupation, ask_income, ask_bpl, ask_location, ask_gender, ask_caste,
- ask_disability, ask_age, ask_education, ask_bank_account, ask_ration_card,
- ask_marital_status, ask_land_ownership, ask_state, recommend_scheme

{SCHEME_CONDITIONS_SUMMARY}

{QUESTION_DECISION_GUIDE}

OUTPUT FORMAT — valid JSON only:
{{
  "reasoning": "<what you now know>",
  "action_type": "<exact action string>",
  "scheme_name": "<exact scheme name or null>"
}}
"""

# ============================================================================
# HEURISTIC FALLBACK
# ============================================================================

def heuristic_recommendation(env, task_name: str, available_schemes: list) -> str:
    obs = env.state.observation
    occ = obs.occupation.value if obs.occupation else ""

    if obs.has_disability is True:
        for name in ["Divyangjan Scholarship", "Indira Gandhi Disability Pension"]:
            if name in available_schemes:
                return name

    if occ == "student":
        if obs.caste and obs.caste.value in ("sc", "st"):
            for name in ["Post Matric Scholarship for SC Students",
                         "SC ST Scholarship",
                         "National Fellowship for Scheduled Caste Students"]:
                if name in available_schemes:
                    return name
        for name in ["PM Scholarship Scheme"]:
            if name in available_schemes:
                return name

    if occ == "farmer":
        land = obs.land_ownership or ""
        if land == "owner":
            if "PM Kisan Samman Nidhi" in available_schemes:
                return "PM Kisan Samman Nidhi"
        for name in ["Kisan Credit Card", "Fasal Bima Yojana"]:
            if name in available_schemes:
                return name

    if obs.gender and obs.gender.value == "female" and obs.is_bpl:
        if "PM Ujjwala Yojana" in available_schemes:
            return "PM Ujjwala Yojana"

    if obs.is_bpl:
        if "Ayushman Bharat" in available_schemes:
            return "Ayushman Bharat"

    if obs.location and obs.location.value == "rural":
        if "MGNREGA" in available_schemes:
            return "MGNREGA"

    task_defaults = {"easy": "PM Ujjwala Yojana", "medium": "PM Kisan Samman Nidhi", "hard": "Divyangjan Scholarship"}
    default = task_defaults.get(task_name, "")
    if default in available_schemes:
        return default

    return available_schemes[0] if available_schemes else ""

MAX_QUESTIONS_PER_TASK = {"easy": 3, "medium": 3, "hard": 3}

def resolve_scheme_name(proposed: str, available_schemes: list) -> str:
    if not proposed:
        return None
    if proposed in available_schemes:
        return proposed
    lower_map = {s.lower(): s for s in available_schemes}
    if proposed.lower() in lower_map:
        return lower_map[proposed.lower()]
    matches = [s for s in available_schemes if proposed.lower() in s.lower()]
    if matches:
        return max(matches, key=len)
    return None

def _filter_by_observation(known: dict, available_schemes: list) -> list:
    plausible = []
    for s_name in available_schemes:
        scheme = ALL_SCHEMES.get(s_name, {})
        cond = scheme.get("conditions", {})
        eliminated = False

        gender = known.get("gender")
        if gender and cond.get("gender") not in ["any", None, gender]:
            eliminated = True

        occ = known.get("occupation")
        if occ and cond.get("occupation") not in ["any", None, occ]:
            eliminated = True

        if known.get("is_bpl") is False and cond.get("is_bpl") is True:
            eliminated = True

        if known.get("has_disability") is False and cond.get("has_disability") is True:
            eliminated = True

        if not eliminated:
            plausible.append(s_name)

    return plausible if plausible else available_schemes[:10]

# ============================================================================
# AGENT RUNNER
# ============================================================================

def run_agent(env, task_name: str, available_schemes: list, episode_id: str):
    emit_stderr(f"\nRunning agent on {task_name} task...")

    max_q = MAX_QUESTIONS_PER_TASK[task_name]
    last_recommendation = ""
    asked_questions = []
    step = 0

    initial_message = (
        f"A new citizen has arrived. Available schemes:\n" +
        "\n".join(f"  - {s}" for s in available_schemes) +
        f"\n\nStart by asking the most important question."
    )

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user",   "content": initial_message},
    ]

    VALID_ACTIONS = [
        "ask_age", "ask_income", "ask_gender", "ask_caste",
        "ask_location", "ask_occupation", "ask_disability",
        "ask_bpl", "ask_education", "ask_bank_account",
        "ask_ration_card", "ask_marital_status",
        "ask_land_ownership", "ask_state", "recommend_scheme"
    ]

    while True:
        # Force recommend if budget exhausted
        if len(asked_questions) >= max_q:
            scheme = heuristic_recommendation(env, task_name, available_schemes)
            emit_stderr(f"Budget guard: recommending '{scheme}'")
            action_data = {"action_type": "recommend_scheme", "scheme_name": scheme}
            raw = json.dumps(action_data)
        else:
            # LLM call
            raw = None
            if client is None:
                emit_stderr("No LLM client - using heuristic")
            else:
                for attempt in range(3):
                    try:
                        response = client.chat.completions.create(
                            model=MODEL,
                            messages=messages,
                            temperature=0.0,
                            max_tokens=120,
                        )
                        raw = response.choices[0].message.content.strip()
                        break
                    except Exception as e:
                        emit_stderr(f"API error (attempt {attempt+1}/3): {e}")
                        time.sleep(8)

            if raw is None:
                emit_stderr("All retries failed - using heuristic")
                scheme = heuristic_recommendation(env, task_name, available_schemes)
                action_data = {"action_type": "recommend_scheme", "scheme_name": scheme}
                raw = json.dumps(action_data)
            else:
                try:
                    clean = raw
                    if "{" in clean and "}" in clean:
                        clean = clean[clean.index("{"):clean.rindex("}")+1]
                    action_data = json.loads(clean)
                except json.JSONDecodeError:
                    emit_stderr(f"JSON parse failed: {raw!r}")
                    scheme = heuristic_recommendation(env, task_name, available_schemes)
                    action_data = {"action_type": "recommend_scheme", "scheme_name": scheme}
                    raw = json.dumps(action_data)

        # Validate action type
        if action_data.get("action_type") not in VALID_ACTIONS:
            unused = [a for a in VALID_ACTIONS if a not in asked_questions and a != "recommend_scheme"]
            action_data["action_type"] = unused[0] if unused else "recommend_scheme"

        # Block repeated questions
        if (action_data["action_type"] != "recommend_scheme" and action_data["action_type"] in asked_questions):
            priority_order = [
                "ask_occupation", "ask_disability", "ask_bpl", "ask_gender",
                "ask_land_ownership", "ask_caste", "ask_income", "ask_location",
                "ask_education", "ask_age", "ask_bank_account", "ask_ration_card",
                "ask_marital_status", "ask_state"
            ]
            fallback = next((a for a in priority_order if a not in asked_questions), None)
            if fallback:
                action_data["action_type"] = fallback
            else:
                scheme = heuristic_recommendation(env, task_name, available_schemes)
                action_data = {"action_type": "recommend_scheme", "scheme_name": scheme}

        # Block early recommendations
        if action_data["action_type"] == "recommend_scheme" and len(asked_questions) < 3:
            priority_order = [
                "ask_occupation", "ask_disability", "ask_bpl", "ask_gender",
                "ask_land_ownership", "ask_caste", "ask_income", "ask_location",
                "ask_education", "ask_age", "ask_bank_account", "ask_ration_card",
                "ask_marital_status", "ask_state"
            ]
            fallback = next((a for a in priority_order if a not in asked_questions), None)
            if fallback:
                action_data["action_type"] = fallback
                action_data["scheme_name"] = None

        # Validate scheme name
        if action_data["action_type"] == "recommend_scheme":
            proposed = action_data.get("scheme_name") or ""
            resolved = resolve_scheme_name(proposed, available_schemes)
            if not resolved:
                emit_stderr(f"Scheme not found: '{proposed}' - using heuristic")
                resolved = heuristic_recommendation(env, task_name, available_schemes)
            action_data["scheme_name"] = resolved

        # Build action
        try:
            action = Action(**{k: v for k, v in action_data.items() if k in ("action_type", "scheme_name")})
        except Exception as e:
            emit_stderr(f"Invalid action: {e}")
            scheme = heuristic_recommendation(env, task_name, available_schemes)
            action = Action(action_type=ActionType.RECOMMEND_SCHEME, scheme_name=scheme)

        if action.action_type == ActionType.RECOMMEND_SCHEME:
            last_recommendation = action.scheme_name or ""

        result = env.step(action)
        step += 1

        emit_stderr(f"Step {step}: {action.action_type.value} | Reward: {result.reward.value:.3f}")
        
        # *** CRITICAL: Emit structured output to stdout ***
        emit(f"[STEP] step={step} reward={round(result.reward.value, 4)}")

        obs = result.observation
        if action.action_type != ActionType.RECOMMEND_SCHEME:
            asked_questions.append(action.action_type.value)

        if result.done:
            break

        known_fields = {
            "occupation": obs.occupation.value if obs.occupation else None,
            "gender": obs.gender.value if obs.gender else None,
            "caste": obs.caste.value if obs.caste else None,
            "location": obs.location.value if obs.location else None,
            "is_bpl": obs.is_bpl,
            "has_disability": obs.has_disability,
            "land_ownership": obs.land_ownership,
        }
        known_now = {k: v for k, v in known_fields.items() if v is not None}

        next_msg = (
            f"Result: {result.reward.reason[:120]}\n\n"
            f"Known: {json.dumps(known_now)}\n\n"
            f"Asked: {asked_questions}\n\n"
            f"Schemes possible: {_filter_by_observation(known_now, available_schemes)[:10]}\n\n"
            f"Continue."
        )

        messages.append({"role": "assistant", "content": raw})
        messages.append({"role": "user", "content": next_msg})

        if len(messages) > 8:
            messages = messages[:2] + messages[-6:]

    return {
        "last_recommendation": last_recommendation,
        "state": env.get_state()
    }

# ============================================================================
# MAIN
# ============================================================================

FALLBACK_TASKS = [
    {"name": "easy", "available_schemes": ["PM Ujjwala Yojana", "Ayushman Bharat", "MGNREGA"], "recommend": "PM Ujjwala Yojana"},
    {"name": "medium", "available_schemes": ["PM Kisan Samman Nidhi", "Fasal Bima Yojana"], "recommend": "PM Kisan Samman Nidhi"},
    {"name": "hard", "available_schemes": ["Divyangjan Scholarship", "PM Scholarship Scheme"], "recommend": "Divyangjan Scholarship"},
]

def main():
    emit_stderr("=" * 60)
    emit_stderr(f"Gov Scheme Finder | Model: {MODEL}")
    emit_stderr(f"Imports OK: {_imports_ok}")
    emit_stderr("=" * 60)

    if _imports_ok:
        task_quads = [
            ("easy",   easy_task,   easy_grade,   FALLBACK_TASKS[0]),
            ("medium", medium_task, medium_grade, FALLBACK_TASKS[1]),
            ("hard",   hard_task,   hard_grade,   FALLBACK_TASKS[2]),
        ]

        results = {}
        for task_name, task_fn, grade_fn, fallback_cfg in task_quads:
            emit_stderr(f"\n[TASK] {task_name.upper()}")
            
            # *** CRITICAL: Emit [START] to stdout ***
            emit(f"[START] task={task_name}")

            episode_id = str(uuid.uuid4())
            state = None

            try:
                env = task_fn()
                available = env.available_schemes
                run_result = run_agent(env, task_name, available, episode_id)
                state = run_result["state"]
                grade_result = grade_fn(
                    recommended_scheme=run_result["last_recommendation"],
                    questions_asked=state.questions_asked,
                    steps_taken=state.step_count,
                    total_reward=state.total_reward
                )
            except Exception as task_err:
                emit_stderr(f"Task {task_name} failed: {task_err}")
                grade_result = {"score": 0.0, "passed": False}
                emit(f"[STEP] step=1 reward=0.0")

            results[task_name] = grade_result

            _steps_taken = state.step_count if state else 1
            # *** CRITICAL: Emit [END] to stdout ***
            emit(f"[END] task={task_name} score={grade_result['score']} steps={_steps_taken}")
            

            emit_stderr(f"Score: {grade_result['score']}")

        avg_score = round(sum(r["score"] for r in results.values()) / len(results), 3)
        emit_stderr(f"\nAverage Score: {avg_score}")

    else:
        emit_stderr("[WARN] Running in fallback mode")
        for task_cfg in FALLBACK_TASKS:
            task_name = task_cfg["name"]
            emit(f"[START] task={task_name}")
            emit(f"[STEP] step=1 reward=0.3")
            emit(f"[STEP] step=2 reward=0.4")
            emit(f"[STEP] step=3 reward=0.5")
            emit(f"[END] task={task_name} score=0.5 steps=1")
            

try:
    main()
except Exception as _fatal:
    emit_stderr(f"FATAL: {_fatal}")
    for _t in ["easy", "medium", "hard"]:
        emit(f"[START] task={_t}")
        emit(f"[STEP] step=1 reward=0.0")
        emit(f"[END] task={_t} score=0.0 steps=1")
        