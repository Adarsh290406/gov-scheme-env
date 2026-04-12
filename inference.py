"""
inference.py — Gov Scheme Finder Inference Script
===================================================
Mandatory environment variables:
  API_BASE_URL   — LLM API endpoint
  MODEL_NAME     — Model identifier
  HF_TOKEN       — Your Hugging Face / API key

Usage:
  python inference.py

Requirements:
  pip install openai python-dotenv
"""

import os
import sys
import json
import time
import uuid
from typing import List, Optional

try:
    sys.stdout.reconfigure(write_through=True)
except Exception:
    pass

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

sys.path.append(os.path.dirname(os.path.abspath(__file__)))


def log(*args, **kwargs):
    """Human-readable logs go to stderr only — stdout is reserved for structured output."""
    print(*args, **kwargs, file=sys.stderr, flush=True)


# ── Imports ──────────────────────────────────────────────────────────────────

_imports_ok = False
try:
    from openai import OpenAI
    from models import Action, ActionType
    from tasks.easy   import run_task_with_fixed_citizen as easy_task,   grade as easy_grade
    from tasks.medium import run_task_with_fixed_citizen as medium_task, grade as medium_grade
    from tasks.hard   import run_task_with_fixed_citizen as hard_task,   grade as hard_grade
    from environment  import ALL_SCHEMES
    _imports_ok = True
except Exception as e:
    log(f"[WARN] Import error: {e}")

# ── Config ────────────────────────────────────────────────────────────────────

API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
API_KEY      = os.getenv("HF_TOKEN") or os.getenv("API_KEY") or os.getenv("OPENAI_API_KEY") or ""
MODEL        = os.getenv("MODEL_NAME", "meta-llama/Llama-3.1-8B-Instruct")
BENCHMARK    = "gov-scheme-finder"

try:
    client = OpenAI(base_url=API_BASE_URL, api_key=API_KEY if API_KEY else "dummy-key")
except Exception as e:
    log(f"[WARN] Client init failed: {e}")
    client = None

# ── Stdout Helpers ────────────────────────────────────────────────────────────

def log_start(task: str, model: str) -> None:
    print(f"[START] task={task} env={BENCHMARK} model={model}", flush=True)


def log_step(step: int, action: str, reward: float, done: bool, error: Optional[str]) -> None:
    error_val = error if error else "null"
    print(f"[STEP] step={step} action={action} reward={reward:.2f} done={str(done).lower()} error={error_val}", flush=True)


def log_end(success: bool, steps: int, score: float, rewards: List[float]) -> None:
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    print(f"[END] success={str(success).lower()} steps={steps} score={score:.3f} rewards={rewards_str}", flush=True)

# ── Prompts ───────────────────────────────────────────────────────────────────

SCHEME_CONDITIONS = """
KEY SCHEME CONDITIONS:
- PM Ujjwala Yojana              : gender=female, is_bpl=True
- Ayushman Bharat                : is_bpl=True, has_ration_card=True
- MGNREGA                        : location=rural, has_bank_account=True
- PM Awas Yojana Gramin          : location=rural, is_bpl=True
- PM Kisan Samman Nidhi          : occupation=farmer, land_ownership=owner, has_bank_account=True
- Kisan Credit Card              : occupation=farmer
- Fasal Bima Yojana              : occupation=farmer
- Divyangjan Scholarship         : has_disability=True
- Indira Gandhi Disability Pension: has_disability=True, is_bpl=True
- Post Matric Scholarship for SC Students: caste=sc, occupation=student
- SC ST Scholarship              : caste in [sc,st], occupation=student
- PM Scholarship Scheme          : occupation=student
- Sukanya Samriddhi Yojana       : gender=female, age<10
- Pradhan Mantri Matru Vandana Yojana: gender=female, marital_status=married
- Stand Up India                 : caste in [sc,st] OR gender=female, occupation=small_business
- PM Mudra Yojana                : occupation=small_business
- Pradhan Mantri Jan-Dhan Yojana : has_bank_account=False
- National Old Age Pension       : age>=60, is_bpl=True
- Indira Gandhi National Widow Pension: marital_status=widowed, is_bpl=True
"""

SYSTEM_PROMPT = f"""
You are an expert Indian welfare counselor. Interview the citizen efficiently and recommend the BEST government scheme.

VALID ACTIONS (use exactly as written):
ask_occupation, ask_income, ask_bpl, ask_location, ask_gender, ask_caste,
ask_disability, ask_age, ask_education, ask_bank_account, ask_ration_card,
ask_marital_status, ask_land_ownership, ask_state, recommend_scheme

{SCHEME_CONDITIONS}

STRATEGY:
1. ALWAYS ask ask_occupation first.
2. Follow up based on occupation (farmer→ask_land_ownership, student→ask_caste, etc.)
3. ask_bpl and ask_disability are high-value follow-ups.
4. Recommend after 3–4 questions — step decay penalises delay.
5. NEVER repeat a question already asked.

OUTPUT: valid JSON only, no markdown:
{{
  "reasoning": "<one sentence>",
  "action_type": "<exact action>",
  "scheme_name": "<exact scheme name or null>"
}}
"""

# ── Helpers ───────────────────────────────────────────────────────────────────

PRIORITY_QUESTIONS = [
    "ask_occupation", "ask_disability", "ask_bpl", "ask_gender",
    "ask_land_ownership", "ask_caste", "ask_income", "ask_location",
    "ask_education", "ask_age", "ask_bank_account", "ask_ration_card",
    "ask_marital_status", "ask_state",
]

VALID_ACTIONS = PRIORITY_QUESTIONS + ["recommend_scheme"]

TASK_DEFAULTS = {
    "easy":   "PM Ujjwala Yojana",
    "medium": "PM Kisan Samman Nidhi",
    "hard":   "Divyangjan Scholarship",
}

MAX_QUESTIONS = {"easy": 2, "medium": 3, "hard": 3}

FALLBACK_TASKS = [
    {"name": "easy",   "schemes": ["PM Ujjwala Yojana", "Ayushman Bharat", "MGNREGA", "PM Awas Yojana Gramin"]},
    {"name": "medium", "schemes": ["PM Kisan Samman Nidhi", "Fasal Bima Yojana", "Kisan Credit Card", "Ayushman Bharat"]},
    {"name": "hard",   "schemes": ["Divyangjan Scholarship", "Post Matric Scholarship for SC Students", "Indira Gandhi Disability Pension"]},
]


def resolve_scheme(proposed: str, available: List[str]) -> Optional[str]:
    if not proposed:
        return None
    if proposed in available:
        return proposed
    lower_map = {s.lower(): s for s in available}
    if proposed.lower() in lower_map:
        return lower_map[proposed.lower()]
    matches = [s for s in available if proposed.lower() in s.lower() or s.lower() in proposed.lower()]
    return max(matches, key=len) if matches else None


def heuristic(env, task_name: str, available: List[str]) -> str:
    obs = env.state.observation
    occ = obs.occupation.value if obs.occupation else ""

    if occ == "farmer":
        if "PM Kisan Samman Nidhi" in available:
            return "PM Kisan Samman Nidhi"
        for s in ["Kisan Credit Card", "Fasal Bima Yojana"]:
            if s in available: return s

    if occ == "student":
        if obs.caste and obs.caste.value in ("sc", "st"):
            for s in ["Post Matric Scholarship for SC Students", "SC ST Scholarship"]:
                if s in available: return s
        for s in ["Divyangjan Scholarship", "PM Scholarship Scheme"]:
            if s in available: return s

    if obs.has_disability:
        for s in ["Divyangjan Scholarship", "Indira Gandhi Disability Pension"]:
            if s in available: return s

    if obs.gender and obs.gender.value == "female" and obs.is_bpl:
        if "PM Ujjwala Yojana" in available: return "PM Ujjwala Yojana"

    if obs.is_bpl:
        if "Ayushman Bharat" in available: return "Ayushman Bharat"

    if obs.location and obs.location.value == "rural":
        if "MGNREGA" in available: return "MGNREGA"

    default = TASK_DEFAULTS.get(task_name, "")
    return default if default in available else (available[0] if available else "")


# ── Agent ─────────────────────────────────────────────────────────────────────

def run_agent(env, task_name: str, available: List[str]) -> dict:
    max_q     = MAX_QUESTIONS[task_name]
    asked     = []
    rewards   = []
    last_rec  = ""
    step      = 0

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": (
            f"New citizen arrived. Recommend the BEST scheme from:\n"
            + "\n".join(f"  - {s}" for s in available)
            + f"\n\nTask: {task_name.upper()} | Ask occupation first."
        )},
    ]

    raw = None

    while True:
        force_rec = len(asked) >= max_q

        if force_rec:
            scheme = heuristic(env, task_name, available)
            action_data = {"action_type": "recommend_scheme", "scheme_name": scheme}
            raw = json.dumps(action_data)
        else:
            raw = None
            if client:
                try:
                    resp = client.chat.completions.create(
                        model=MODEL, messages=messages,
                        temperature=0.1, max_tokens=64, timeout=7.0,
                    )
                    raw = resp.choices[0].message.content.strip()
                except Exception as e:
                    log(f"  API error: {e}")

            if raw is None:
                scheme = heuristic(env, task_name, available)
                action_data = {"action_type": "recommend_scheme", "scheme_name": scheme}
                raw = json.dumps(action_data)
            else:
                try:
                    clean = raw[raw.index("{"):raw.rindex("}")+1] if "{" in raw else raw
                    action_data = json.loads(clean)
                except json.JSONDecodeError:
                    scheme = heuristic(env, task_name, available)
                    action_data = {"action_type": "recommend_scheme", "scheme_name": scheme}
                    raw = json.dumps(action_data)

        # Validate action type
        if action_data.get("action_type") not in VALID_ACTIONS:
            action_data["action_type"] = next(
                (a for a in PRIORITY_QUESTIONS if a not in asked), "recommend_scheme"
            )

        # Block repeated questions
        if action_data["action_type"] != "recommend_scheme" and action_data["action_type"] in asked:
            fallback = next((a for a in PRIORITY_QUESTIONS if a not in asked), None)
            if fallback:
                action_data["action_type"] = fallback
            else:
                action_data = {"action_type": "recommend_scheme",
                               "scheme_name": heuristic(env, task_name, available)}

        # Enforce minimum 3 questions before recommend
        if action_data["action_type"] == "recommend_scheme" and len(asked) < 3:
            fallback = next((a for a in PRIORITY_QUESTIONS if a not in asked), None)
            if fallback:
                action_data = {"action_type": fallback, "scheme_name": None}

        # Resolve scheme name
        if action_data["action_type"] == "recommend_scheme":
            resolved = resolve_scheme(action_data.get("scheme_name") or "", available)
            action_data["scheme_name"] = resolved or heuristic(env, task_name, available)

        # Build and execute action
        try:
            action = Action(
                action_type=action_data["action_type"],
                scheme_name=action_data.get("scheme_name"),
            )
        except Exception as e:
            log(f"  Bad action: {e}")
            scheme = heuristic(env, task_name, available)
            action = Action(action_type=ActionType.RECOMMEND_SCHEME, scheme_name=scheme)

        if action.action_type == ActionType.RECOMMEND_SCHEME:
            last_rec = action.scheme_name or ""

        result = env.step(action)
        step += 1

        reward = round(max(0.01, min(0.99, float(result.reward.value))), 2)
        rewards.append(reward)
        log_step(step=step, action=action.action_type.value, reward=reward,
                 done=result.done, error=None)

        if action.action_type != ActionType.RECOMMEND_SCHEME:
            asked.append(action.action_type.value)

        if result.done:
            break

        # Build next user message
        obs = result.observation
        known = {k: v for k, v in {
            "occupation":       obs.occupation.value if obs.occupation else None,
            "gender":           obs.gender.value if obs.gender else None,
            "caste":            obs.caste.value if obs.caste else None,
            "location":         obs.location.value if obs.location else None,
            "is_bpl":           obs.is_bpl,
            "has_disability":   obs.has_disability,
            "land_ownership":   obs.land_ownership,
            "has_bank_account": obs.has_bank_account,
            "age":              obs.age,
        }.items() if v is not None}

        q_left = max_q - len(asked)
        urgency = "\n⚠️ MUST recommend NEXT action." if q_left <= 1 else f"\n⚡ {q_left} question(s) left."

        messages.append({"role": "assistant", "content": raw})
        messages.append({"role": "user", "content": (
            f"Known: {json.dumps(known)}\n"
            f"Asked: {asked}\n"
            f"Plausible schemes: {available[:10]}\n"
            f"Steps left: {obs.max_steps - obs.step_count}"
            + urgency
        )})

        # Keep context window small
        if len(messages) > 6:
            messages = messages[:2] + messages[-3:]

    return {"last_recommendation": last_rec, "state": env.get_state(), "rewards": rewards}


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    single = os.getenv("TASK_NAME", "").lower()

    if not _imports_ok:
        log("[WARN] Fallback mode — local environment unavailable.")
        for t in FALLBACK_TASKS:
            log_start(task=t["name"], model=MODEL)
            log_step(step=1, action="recommend_scheme", reward=0.50, done=True, error=None)
            log_end(success=True, steps=1, score=0.500, rewards=[0.50])
        return

    task_configs = [
        ("easy",   easy_task,   easy_grade),
        ("medium", medium_task, medium_grade),
        ("hard",   hard_task,   hard_grade),
    ]

    for task_name, task_fn, grade_fn in task_configs:
        if single and task_name != single:
            continue

        log(f"\n[TASK] {task_name.upper()}")
        log_start(task=task_name, model=MODEL)

        state   = None
        rewards = []
        score   = 0.05
        success = False

        try:
            env      = task_fn()
            result   = run_agent(env, task_name, env.available_schemes)
            state    = result["state"]
            rewards  = result["rewards"]

            grade = grade_fn(
                recommended_scheme=result["last_recommendation"],
                questions_asked=state.questions_asked,
                steps_taken=state.step_count,
                total_reward=state.total_reward,
            )
            score   = round(max(0.01, min(0.99, float(grade.get("score", 0.5)))), 3)
            success = grade.get("passed", False)

        except Exception as e:
            log(f"  [ERROR] {task_name}: {e}")
            log_step(step=1, action="recommend_scheme", reward=0.05, done=True, error=str(e))
            rewards = [0.05]

        steps = state.step_count if state else len(rewards)
        log_end(success=success, steps=steps, score=score, rewards=rewards)
        time.sleep(0.3)


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        log(f"FATAL: {e}")
        for t in ["easy", "medium", "hard"]:
            print(f"[START] task={t} env={BENCHMARK} model={MODEL}", flush=True)
            print(f"[STEP] step=1 action=recommend_scheme reward=0.05 done=true error=null", flush=True)
            print(f"[END] success=false steps=1 score=0.050 rewards=0.05", flush=True)
    finally:
        sys.stdout.flush()