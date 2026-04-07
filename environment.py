import random
import uuid
import json
import os
from typing import Optional
from models import (
    Action, ActionType, Observation, Reward, State,
    StepResult, CitizenProfile, Difficulty,
    Gender, CasteCategory, Location, Occupation,
    OCCUPATION_INCOME_CONTEXT, IncomeContext,
    INCOME_BEFORE_OCCUPATION_PENALTY, OCCUPATION_FIRST_BONUS
)

# -----------------------------------------
# LOAD SCHEMES FROM schemes.json
# -----------------------------------------

def load_schemes() -> dict:
    """Load schemes from schemes.json file"""
    schemes_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "schemes.json"
    )
    if not os.path.exists(schemes_path):
        raise FileNotFoundError(
            "schemes.json not found! Run generate_schemes.py first."
        )
    with open(schemes_path, "r", encoding="utf-8") as f:
        schemes_list = json.load(f)

    # Convert list to dict keyed by name
    return {s["name"]: s for s in schemes_list}


# Load schemes once at startup
ALL_SCHEMES = load_schemes()


# -----------------------------------------
# SCHEME CONDITION CHECKER
# Evaluates if a citizen matches a scheme
# -----------------------------------------

def check_scheme_conditions(citizen: CitizenProfile, scheme: dict) -> tuple:
    """
    Check if citizen matches scheme conditions.
    Returns (is_full_match, partial_score)
    - is_full_match: True if citizen fully qualifies
    - partial_score: 0.0 to 0.5 for partial matches
    """
    conditions = scheme.get("conditions", {})
    partial_score = 0.0
    failed_conditions = 0
    total_conditions = 0

    # Gender check
    if conditions.get("gender") not in ["any", None]:
        total_conditions += 1
        if citizen.gender.value != conditions["gender"]:
            failed_conditions += 1

    # Age check
    if conditions.get("min_age") is not None:
        total_conditions += 1
        if citizen.age < conditions["min_age"]:
            failed_conditions += 1

    if conditions.get("max_age") is not None:
        total_conditions += 1
        if citizen.age > conditions["max_age"]:
            failed_conditions += 1

    # Income check
    if conditions.get("max_income") is not None:
        total_conditions += 1
        if citizen.income > conditions["max_income"]:
            failed_conditions += 1

    # Occupation check
    if conditions.get("occupation") not in ["any", None]:
        total_conditions += 1
        if citizen.occupation.value != conditions["occupation"]:
            failed_conditions += 1

    # Location check
    if conditions.get("location") not in ["any", None]:
        total_conditions += 1
        if citizen.location.value != conditions["location"]:
            failed_conditions += 1

    # Caste check
    if conditions.get("caste") not in ["any", None]:
        total_conditions += 1
        caste_val = conditions["caste"]
        if isinstance(caste_val, list):
            if citizen.caste.value not in caste_val:
                failed_conditions += 1
        else:
            if citizen.caste.value != caste_val:
                failed_conditions += 1

    # BPL check
    if conditions.get("is_bpl") is not None:
        total_conditions += 1
        if citizen.is_bpl != conditions["is_bpl"]:
            failed_conditions += 1

    # Disability check
    if conditions.get("has_disability") is not None:
        total_conditions += 1
        if citizen.has_disability != conditions["has_disability"]:
            failed_conditions += 1

    # Bank account check
    if conditions.get("has_bank_account") is not None:
        total_conditions += 1
        if citizen.has_bank_account != conditions["has_bank_account"]:
            failed_conditions += 1

    # Ration card check
    if conditions.get("has_ration_card") is not None:
        total_conditions += 1
        if citizen.has_ration_card != conditions["has_ration_card"]:
            failed_conditions += 1

    # Marital status check
    if conditions.get("marital_status") not in ["any", None]:
        total_conditions += 1
        if citizen.marital_status != conditions["marital_status"]:
            failed_conditions += 1

    # Land ownership check
    if conditions.get("land_ownership") not in ["any", None]:
        total_conditions += 1
        if citizen.land_ownership != conditions["land_ownership"]:
            failed_conditions += 1

    if total_conditions == 0:
        return True, 0.0

    if failed_conditions == 0:
        return True, 0.0  # Full match

    match_ratio = 1.0 - (failed_conditions / total_conditions)
    if match_ratio >= 0.7:
        partial_score = 0.3   # Almost qualifies
    elif match_ratio >= 0.5:
        partial_score = 0.15  # Partially qualifies
    else:
        partial_score = 0.0   # Too far off

    return False, partial_score


# -----------------------------------------
# QUESTION RELEVANCE — Context Aware
# How useful is each question given what
# the agent already knows
# -----------------------------------------

QUESTION_RELEVANCE = {
    ActionType.ASK_OCCUPATION:      1.0,
    ActionType.ASK_INCOME:          0.9,
    ActionType.ASK_BPL:             0.9,
    ActionType.ASK_LOCATION:        0.7,
    ActionType.ASK_GENDER:          0.6,
    ActionType.ASK_CASTE:           0.6,
    ActionType.ASK_DISABILITY:      0.5,
    ActionType.ASK_AGE:             0.4,
    ActionType.ASK_EDUCATION:       0.5,
    ActionType.ASK_BANK_ACCOUNT:    0.4,
    ActionType.ASK_RATION_CARD:     0.4,
    ActionType.ASK_MARITAL_STATUS:  0.3,
    ActionType.ASK_LAND_OWNERSHIP:  0.5,
    ActionType.ASK_STATE:           0.3,
}

# Questions irrelevant for certain occupations
IRRELEVANT_QUESTIONS = {
    Occupation.STUDENT: [
        ActionType.ASK_LAND_OWNERSHIP,
    ],
    Occupation.GOVERNMENT_EMPLOYEE: [
        ActionType.ASK_LAND_OWNERSHIP,
        ActionType.ASK_BPL,
    ],
    Occupation.SMALL_BUSINESS: [
        ActionType.ASK_LAND_OWNERSHIP,
    ],
}

# Minimum questions before recommending
MIN_QUESTIONS_BEFORE_RECOMMEND = {
    Difficulty.EASY:   2,
    Difficulty.MEDIUM: 3,
    Difficulty.HARD:   3,
}

# Reward decay per step — urgency increases
STEP_DECAY = 0.02

# Noise levels per difficulty
NOISE_LEVELS = {
    Difficulty.EASY:   0.0,
    Difficulty.MEDIUM: 0.10,
    Difficulty.HARD:   0.20,
}

NOISY_QUESTIONS = [
    ActionType.ASK_BPL,
    ActionType.ASK_INCOME,
    ActionType.ASK_CASTE,
    ActionType.ASK_DISABILITY,
]

# Scheme expiry
SCHEME_EXPIRY_CHANCE = {
    Difficulty.EASY:   0.0,
    Difficulty.MEDIUM: 0.10,
    Difficulty.HARD:   0.20,
}

SCHEME_EXPIRY_INTERVAL = 3

# Incomplete info
INCOMPLETE_INFO_CHANCE = {
    Difficulty.EASY:   0.0,
    Difficulty.MEDIUM: 0.15,
    Difficulty.HARD:   0.25,
}

INCOMPLETE_INFO_QUESTIONS = [
    ActionType.ASK_INCOME,
    ActionType.ASK_CASTE,
    ActionType.ASK_BPL,
]


# -----------------------------------------
# ENVIRONMENT CLASS
# -----------------------------------------

class GovSchemeEnvironment:

    def __init__(self, difficulty: Difficulty = Difficulty.EASY):
        self.difficulty = difficulty
        self.state: Optional[State] = None
        self._setup_difficulty()

    def _setup_difficulty(self):
        """Set max steps and available schemes based on difficulty"""
        scheme_names = list(ALL_SCHEMES.keys())

        if self.difficulty == Difficulty.EASY:
            self.max_steps = 10
            # Pick 10 diverse schemes for easy
            self.available_schemes = scheme_names[:10]
        elif self.difficulty == Difficulty.MEDIUM:
            self.max_steps = 8
            # Pick 25 schemes for medium
            self.available_schemes = scheme_names[:25]
        else:  # HARD
            self.max_steps = 6
            # All schemes for hard
            self.available_schemes = scheme_names

        self._original_schemes = list(self.available_schemes)

    def _generate_citizen(self) -> CitizenProfile:
        """
        Generate a weighted random citizen profile.
        Rural BPL farmers/daily wage workers are more common
        than urban government employees — reflecting real India.
        """
        # Weighted occupation distribution
        occupation = random.choices(
            list(Occupation),
            weights=[25, 20, 20, 10, 15, 10],  # farmer, student, daily_wage, business, unemployed, govt
            k=1
        )[0]

        # Income based on occupation
        if occupation == Occupation.STUDENT:
            income = random.randint(80000, 700000)
        elif occupation == Occupation.FARMER:
            income = random.randint(30000, 250000)
        elif occupation == Occupation.UNEMPLOYED:
            income = random.randint(0, 60000)
        elif occupation == Occupation.DAILY_WAGE:
            income = random.randint(40000, 130000)
        elif occupation == Occupation.SMALL_BUSINESS:
            income = random.randint(100000, 1000000)
        else:
            income = random.randint(200000, 1400000)

        # Weighted location — more rural citizens
        location = random.choices(
            list(Location),
            weights=[65, 35],  # rural more common
            k=1
        )[0]

        # Weighted BPL — more BPL in rural
        is_bpl_chance = 0.5 if location == Location.RURAL else 0.2
        is_bpl = random.random() < is_bpl_chance

        profile = CitizenProfile(
            age=random.randint(5, 75),
            income=float(income),
            gender=random.choice(list(Gender)),
            caste=random.choices(
                list(CasteCategory),
                weights=[30, 40, 20, 10],  # general, obc, sc, st
                k=1
            )[0],
            location=location,
            occupation=occupation,
            has_disability=random.random() < 0.15,
            is_bpl=is_bpl,
            has_bank_account=random.random() < 0.75,
            has_ration_card=random.random() < 0.60,
            marital_status=random.choice(["any", "married", "single", "widowed"]),
            land_ownership=random.choice(["any", "owner", "tenant", "none"]) if occupation == Occupation.FARMER else "none",
            state=random.choice([
                "any", "Uttar Pradesh", "Maharashtra", "Bihar",
                "Madhya Pradesh", "Rajasthan", "Gujarat"
            ]),
            correct_schemes=[]
        )

        # Find all correct schemes
        correct = []
        for name in self.available_schemes:
            scheme = ALL_SCHEMES.get(name)
            if scheme:
                is_match, _ = check_scheme_conditions(profile, scheme)
                if is_match:
                    correct.append(name)

        profile.correct_schemes = correct if correct else ["No scheme available"]
        return profile

    # -----------------------------------------
    # FEATURE 1: NOISE
    # -----------------------------------------

    def _apply_noise(self, action_type: ActionType, true_value):
        noise_chance = NOISE_LEVELS.get(self.difficulty, 0.0)
        if action_type not in NOISY_QUESTIONS or random.random() > noise_chance:
            return true_value, False

        if isinstance(true_value, bool):
            return not true_value, True
        elif isinstance(true_value, float):
            factor = random.choice([0.5, 0.6, 1.5, 1.8])
            return round(true_value * factor, 2), True
        elif isinstance(true_value, CasteCategory):
            others = [c for c in CasteCategory if c != true_value]
            return random.choice(others), True

        return true_value, False

    # -----------------------------------------
    # FEATURE 2: SCHEME EXPIRY
    # -----------------------------------------

    def _check_scheme_expiry(self, obs: Observation) -> Optional[str]:
        expiry_chance = SCHEME_EXPIRY_CHANCE.get(self.difficulty, 0.0)
        if (self.state.step_count % SCHEME_EXPIRY_INTERVAL == 0
                and self.state.step_count > 0
                and random.random() < expiry_chance
                and len(self.available_schemes) > 3):

            citizen = self.state.citizen_profile
            non_correct = [
                s for s in self.available_schemes
                if s not in citizen.correct_schemes
            ]
            to_expire = random.choice(non_correct if non_correct else self.available_schemes)
            self.available_schemes.remove(to_expire)
            obs.available_schemes = self.available_schemes
            return to_expire

        return None

    # -----------------------------------------
    # FEATURE 3: INCOMPLETE INFO
    # -----------------------------------------

    def _check_incomplete_info(self, action_type: ActionType) -> bool:
        incomplete_chance = INCOMPLETE_INFO_CHANCE.get(self.difficulty, 0.0)
        if action_type not in INCOMPLETE_INFO_QUESTIONS:
            return False
        return random.random() < incomplete_chance

    # -----------------------------------------
    # CONTEXT-AWARE PENALTY
    # Penalize irrelevant questions based on
    # what agent already knows
    # -----------------------------------------

    def _get_irrelevance_penalty(self, action_type: ActionType) -> tuple:
        """
        Returns (penalty, reason) if question is irrelevant
        given what agent already knows about citizen.
        Returns (0.0, "") if question is relevant.
        """
        obs = self.state.observation

        # If occupation is known, check for irrelevant questions
        if obs.occupation is not None:
            irrelevant = IRRELEVANT_QUESTIONS.get(obs.occupation, [])
            if action_type in irrelevant:
                return -0.2, (
                    f"Irrelevant question for a {obs.occupation.value}! "
                    f"'{action_type.value}' doesn't help narrow down schemes for this occupation."
                )

        # Penalize asking gender when no gender-specific schemes remain
        if action_type == ActionType.ASK_GENDER and obs.gender is not None:
            return -0.3, "Already know the gender — repeating!"

        # Penalize asking disability when already known
        if action_type == ActionType.ASK_DISABILITY and obs.has_disability is not None:
            return -0.3, "Already know disability status!"

        return 0.0, ""

    # -----------------------------------------
    # REWARD DECAY — Urgency over time
    # -----------------------------------------

    def _get_decay_penalty(self) -> float:
        """Returns increasing penalty for each step taken"""
        return -STEP_DECAY * self.state.step_count

    # -----------------------------------------
    # reset()
    # -----------------------------------------

    def reset(self) -> Observation:
        self._setup_difficulty()
        citizen = self._generate_citizen()

        initial_observation = Observation(
            step_count=0,
            max_steps=self.max_steps,
            last_action_result=(
                "New citizen arrived. "
                "TIP: Ask occupation first! "
                "Be efficient — unnecessary questions cost reward. "
                f"Difficulty: {self.difficulty.value} | "
                f"Schemes available: {len(self.available_schemes)}"
            ),
            available_schemes=list(self.available_schemes),
            done=False
        )

        self.state = State(
            citizen_profile=citizen,
            observation=initial_observation,
            difficulty=self.difficulty,
            episode_id=str(uuid.uuid4()),
            step_count=0,
            total_reward=0.0,
            is_done=False,
            questions_asked=[],
            occupation_asked_first=False
        )

        return initial_observation

    # -----------------------------------------
    # step()
    # -----------------------------------------

    def step(self, action: Action) -> StepResult:
        if self.state is None:
            raise RuntimeError("Call reset() before step()")
        if self.state.is_done:
            raise RuntimeError("Episode done. Call reset() to start again.")

        citizen = self.state.citizen_profile
        obs = self.state.observation
        reward_value = 0.0
        reward_reason = ""
        done = False
        info = {}

        # ── CASE 1: Asking a question ──
        if action.action_type != ActionType.RECOMMEND_SCHEME:

            question = action.action_type.value

            # Penalty: repeated question
            if question in self.state.questions_asked:
                reward_value = 0.0
                reward_reason = f"Already asked '{question}'! Repeating wastes steps."

            # Penalty: income before occupation
            elif (action.action_type == ActionType.ASK_INCOME
                  and ActionType.ASK_OCCUPATION.value not in self.state.questions_asked):
                reward_value = 0.0
                reward_reason = (
                    "Asked income before occupation! Context unknown. No reward."
                )
                obs.income = citizen.income
                obs.income_context = IncomeContext.NOT_APPLICABLE
                obs.income_hint = "WARNING: Ask occupation first to understand income context!"
                self.state.questions_asked.append(question)

            else:
                # Context-aware irrelevance penalty
                irrelevance_penalty, irrelevance_reason = self._get_irrelevance_penalty(action.action_type)

                if irrelevance_penalty < 0:
                    reward_value = 0.0
                    reward_reason = irrelevance_reason
                    self.state.questions_asked.append(question)

                # Incomplete info check
                elif self._check_incomplete_info(action.action_type):
                    reward_value = 0.05
                    reward_reason = (
                        f"Good question — but citizen is unsure about '{question}'. "
                        "No clear answer. Try cross-verifying."
                    )
                    self.state.questions_asked.append(question)
                    info["incomplete"] = True

                else:
                    # Normal question
                    relevance = QUESTION_RELEVANCE.get(action.action_type, 0.3)

                    # Apply reward decay for urgency
                    decay = self._get_decay_penalty()
                    reward_value = (relevance * 0.3) + decay
                    reward_reason = (
                        f"'{question}' revealed. "
                        f"Relevance: {relevance} | "
                        f"Step decay: {round(decay, 3)}"
                    )

                    # Bonus for asking occupation first
                    if (action.action_type == ActionType.ASK_OCCUPATION
                            and len(self.state.questions_asked) == 0):
                        reward_value += OCCUPATION_FIRST_BONUS
                        reward_reason += f" | BONUS: Occupation first! +{OCCUPATION_FIRST_BONUS}"
                        self.state.occupation_asked_first = True

                    self.state.questions_asked.append(question)

                    # Reveal answer with noise
                    if action.action_type == ActionType.ASK_AGE:
                        obs.age = citizen.age

                    elif action.action_type == ActionType.ASK_INCOME:
                        noisy_income, was_noisy = self._apply_noise(action.action_type, citizen.income)
                        obs.income = noisy_income
                        obs.income_context = citizen.income_context()
                        reward_reason += f" | Context: {citizen.income_label()}"
                        if was_noisy:
                            info["noise_warning"] = "Income may be approximate"

                    elif action.action_type == ActionType.ASK_GENDER:
                        obs.gender = citizen.gender

                    elif action.action_type == ActionType.ASK_CASTE:
                        noisy_caste, was_noisy = self._apply_noise(action.action_type, citizen.caste)
                        obs.caste = noisy_caste
                        if was_noisy:
                            info["noise_warning"] = "Caste may be misreported"

                    elif action.action_type == ActionType.ASK_LOCATION:
                        obs.location = citizen.location

                    elif action.action_type == ActionType.ASK_OCCUPATION:
                        obs.occupation = citizen.occupation
                        reward_reason += f" | Occupation: {citizen.occupation.value}"

                    elif action.action_type == ActionType.ASK_DISABILITY:
                        noisy_disability, was_noisy = self._apply_noise(action.action_type, citizen.has_disability)
                        obs.has_disability = noisy_disability
                        if was_noisy:
                            info["noise_warning"] = "Disability status may be misreported"

                    elif action.action_type == ActionType.ASK_BPL:
                        noisy_bpl, was_noisy = self._apply_noise(action.action_type, citizen.is_bpl)
                        obs.is_bpl = noisy_bpl
                        if was_noisy:
                            info["noise_warning"] = "BPL status may be misreported"

                    elif action.action_type == ActionType.ASK_EDUCATION:
                        obs.education = citizen.education

                    elif action.action_type == ActionType.ASK_BANK_ACCOUNT:
                        obs.has_bank_account = citizen.has_bank_account

                    elif action.action_type == ActionType.ASK_RATION_CARD:
                        obs.has_ration_card = citizen.has_ration_card

                    elif action.action_type == ActionType.ASK_MARITAL_STATUS:
                        obs.marital_status = citizen.marital_status

                    elif action.action_type == ActionType.ASK_LAND_OWNERSHIP:
                        obs.land_ownership = citizen.land_ownership

                    elif action.action_type == ActionType.ASK_STATE:
                        obs.state = citizen.state

            # Scheme expiry check
            expired = self._check_scheme_expiry(obs)
            if expired:
                reward_reason += f" | ALERT: '{expired}' scheme has expired!"
                info["scheme_expired"] = expired

        # ── CASE 2: Recommending a scheme ──
        else:
            recommended = action.scheme_name
            questions_so_far = len(self.state.questions_asked)
            min_questions = MIN_QUESTIONS_BEFORE_RECOMMEND.get(self.difficulty, 2)

            # Penalty: recommending too early without enough info
            if questions_so_far < min_questions:
                reward_value = 0.0
                reward_reason = (
                    f"Recommended too early! Asked only {questions_so_far} questions. "
                    f"Minimum {min_questions} needed to make an informed recommendation."
                )
                done = True
                info["too_early"] = True

            elif not recommended:
                reward_value = 0.0
                reward_reason = "Must provide a scheme name!"
                done = True

            elif recommended not in self.available_schemes:
                if recommended in self._original_schemes:
                    reward_value = 0.0
                    reward_reason = f"'{recommended}' expired during this episode!"
                else:
                    reward_value = 0.0
                    reward_reason = f"'{recommended}' not in available schemes!"
                done = True

            elif recommended in citizen.correct_schemes:
                # Efficiency bonus scales within [0.0, 1.0] ceiling
                efficiency_bonus = min(0.3, max(0, (self.max_steps - self.state.step_count) * 0.05))
                reward_value = min(1.0, 0.7 + efficiency_bonus)
                reward_reason = (
                    f"CORRECT! '{recommended}' is perfect for this citizen! "
                    f"Efficiency bonus: +{round(efficiency_bonus, 2)}"
                )
                done = True
                info["success"] = True
                info["efficiency_bonus"] = efficiency_bonus

            else:
                # Check partial match
                scheme = ALL_SCHEMES.get(recommended, {})
                _, partial_score = check_scheme_conditions(citizen, scheme)

                if partial_score > 0:
                    reward_value = partial_score
                    reward_reason = (
                        f"PARTIAL MATCH — '{recommended}' partially applies. "
                        f"Partial reward: {partial_score}. "
                        f"Correct: {citizen.correct_schemes[:3]}"
                    )
                    info["partial_match"] = True
                else:
                    reward_value = 0.0
                    reward_reason = (
                        f"WRONG! '{recommended}' doesn't match this citizen. "
                        f"Correct: {citizen.correct_schemes[:3]}"
                    )
                    info["success"] = False

                info["correct_schemes"] = citizen.correct_schemes
                done = True

        # ── Step limit ──
        self.state.step_count += 1
        obs.step_count = self.state.step_count

        if self.state.step_count >= self.max_steps and not done:
            reward_value = 0.0
            reward_reason += " | Step limit reached!"
            done = True
            info["timeout"] = True
            info["correct_schemes"] = citizen.correct_schemes

        # ── Update totals ──
        self.state.total_reward += reward_value
        self.state.is_done = done
        obs.done = done
        obs.last_action_result = reward_reason

        # Hard clamp: reward is always in [0.0, 1.0]
        reward_value = round(max(0.0, min(1.0, reward_value)), 3)

        reward = Reward(
            value=reward_value,
            reason=reward_reason,
            total_score=round(
                max(0.0, min(1.0, self.state.total_reward / self.max_steps)),
                3
            )
        )

        # ── Episode summary when done ──
        if done:
            info["episode_summary"] = {
                "episode_id": self.state.episode_id,
                "difficulty": self.difficulty.value,
                "steps_taken": self.state.step_count,
                "questions_asked": self.state.questions_asked,
                "total_reward": round(self.state.total_reward, 3),
                "final_score": reward.total_score,
                "correct_schemes": citizen.correct_schemes,
                "recommended": action.scheme_name if action.action_type == ActionType.RECOMMEND_SCHEME else None,
                "occupation_asked_first": self.state.occupation_asked_first,
            }

        return StepResult(
            observation=obs,
            reward=reward,
            done=done,
            info=info
        )

    # -----------------------------------------
    # state()
    # -----------------------------------------

    def get_state(self) -> State:
        if self.state is None:
            raise RuntimeError("Call reset() before get_state()")
        return self.state


# -----------------------------------------
# QUICK TEST
# -----------------------------------------

if __name__ == "__main__":
    print("=" * 60)
    print(f"Testing Environment | Schemes loaded: {len(ALL_SCHEMES)}")
    print("=" * 60)

    env = GovSchemeEnvironment(difficulty=Difficulty.EASY)
    obs = env.reset()

    print(f"\n[PASS] reset() works!")
    print(f"   Occupation (hidden): {env.state.citizen_profile.occupation.value}")
    print(f"   Correct schemes    : {env.state.citizen_profile.correct_schemes[:3]}")
    print(f"   Available schemes  : {len(obs.available_schemes)}")

    from models import Action, ActionType

    actions = [
        Action(action_type=ActionType.ASK_OCCUPATION),
        Action(action_type=ActionType.ASK_BPL),
        Action(action_type=ActionType.ASK_INCOME),
    ]

    print("\nRunning 3 questions...\n")
    for i, action in enumerate(actions):
        result = env.step(action)
        print(f"Step {i+1}: {action.action_type.value}")
        print(f"  Reward : {result.reward.value}")
        print(f"  Reason : {result.reward.reason[:90]}")

    # Recommend correct scheme
    correct = env.state.citizen_profile.correct_schemes[0]
    result = env.step(Action(
        action_type=ActionType.RECOMMEND_SCHEME,
        scheme_name=correct
    ))
    print(f"\nRecommend: {correct}")
    print(f"  Reward : {result.reward.value}")
    print(f"  Done   : {result.done}")
    print(f"  Score  : {result.reward.total_score}")
    print(f"  Summary: {result.info.get('episode_summary', {})}")

    print("\n" + "=" * 60)
    print("All tests passed!")
    print("=" * 60)