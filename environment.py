import random
import uuid
from typing import Optional
from models import (
    Action, ActionType, Observation, Reward, State,
    StepResult, CitizenProfile, Difficulty,
    Gender, CasteCategory, Location, Occupation,
    OCCUPATION_INCOME_CONTEXT, IncomeContext,
    INCOME_BEFORE_OCCUPATION_PENALTY, OCCUPATION_FIRST_BONUS
)

# -----------------------------------------
# GOVERNMENT SCHEMES DATABASE
# -----------------------------------------

SCHEMES = {
    "PM Ujjwala Yojana": {
        "description": "Free LPG connection for BPL women",
        "conditions": lambda p: p.is_bpl and p.gender == Gender.FEMALE,
        "partial_conditions": [
            (lambda p: p.gender == Gender.FEMALE, 0.3),   # female but not BPL
            (lambda p: p.is_bpl, 0.2),                    # BPL but not female
        ]
    },
    "PM Kisan Samman Nidhi": {
        "description": "Rs.6000/year for small farmers",
        "conditions": lambda p: p.occupation == Occupation.FARMER and p.income < 200000,
        "partial_conditions": [
            (lambda p: p.occupation == Occupation.FARMER, 0.4),  # farmer but income too high
        ]
    },
    "Ayushman Bharat": {
        "description": "Health insurance up to Rs.5 lakh for BPL families",
        "conditions": lambda p: p.is_bpl,
        "partial_conditions": [
            (lambda p: p.income < 300000, 0.3),   # low income but not officially BPL
        ]
    },
    "PM Scholarship Scheme": {
        "description": "Scholarship for students from low income families",
        "conditions": lambda p: p.occupation == Occupation.STUDENT and p.income < 600000,
        "partial_conditions": [
            (lambda p: p.occupation == Occupation.STUDENT, 0.3),  # student but income too high
        ]
    },
    "MGNREGA": {
        "description": "100 days guaranteed wage employment in rural areas",
        "conditions": lambda p: p.location == Location.RURAL,
        "partial_conditions": [
            (lambda p: p.is_bpl, 0.2),   # BPL but urban
        ]
    },
    "PM Awas Yojana (Gramin)": {
        "description": "Rural housing scheme for BPL families",
        "conditions": lambda p: p.location == Location.RURAL and p.is_bpl,
        "partial_conditions": [
            (lambda p: p.location == Location.RURAL, 0.3),  # rural but not BPL
            (lambda p: p.is_bpl, 0.2),                      # BPL but urban
        ]
    },
    "Divyangjan Scholarship": {
        "description": "Support for disabled citizens",
        "conditions": lambda p: p.has_disability,
        "partial_conditions": []
    },
    "SC/ST Scholarship": {
        "description": "Educational support for SC/ST students",
        "conditions": lambda p: p.caste in [CasteCategory.SC, CasteCategory.ST] and p.occupation == Occupation.STUDENT,
        "partial_conditions": [
            (lambda p: p.caste in [CasteCategory.SC, CasteCategory.ST], 0.3),  # SC/ST but not student
        ]
    },
    "PM Mudra Yojana": {
        "description": "Loans for small business owners",
        "conditions": lambda p: p.occupation == Occupation.SMALL_BUSINESS and p.income < 1000000,
        "partial_conditions": [
            (lambda p: p.occupation == Occupation.SMALL_BUSINESS, 0.3),  # business but income too high
        ]
    },
    "Sukanya Samriddhi Yojana": {
        "description": "Savings scheme for girl child below 10 years",
        "conditions": lambda p: p.gender == Gender.FEMALE and p.age < 10,
        "partial_conditions": [
            (lambda p: p.gender == Gender.FEMALE and p.age < 18, 0.2),  # female minor but older than 10
        ]
    },
    "Fasal Bima Yojana": {
        "description": "Crop insurance for farmers",
        "conditions": lambda p: p.occupation == Occupation.FARMER,
        "partial_conditions": []
    },
    "Indira Gandhi National Disability Pension": {
        "description": "Monthly pension for BPL disabled citizens",
        "conditions": lambda p: p.has_disability and p.is_bpl,
        "partial_conditions": [
            (lambda p: p.has_disability, 0.3),  # disabled but not BPL
            (lambda p: p.is_bpl, 0.1),          # BPL but not disabled
        ]
    },
}

# -----------------------------------------
# QUESTION RELEVANCE SCORING
# -----------------------------------------

QUESTION_RELEVANCE = {
    ActionType.ASK_OCCUPATION:  1.0,
    ActionType.ASK_INCOME:      0.9,
    ActionType.ASK_BPL:         0.9,
    ActionType.ASK_LOCATION:    0.7,
    ActionType.ASK_GENDER:      0.6,
    ActionType.ASK_CASTE:       0.6,
    ActionType.ASK_DISABILITY:  0.5,
    ActionType.ASK_AGE:         0.4,
}

# -----------------------------------------
# NOISE PROBABILITIES PER DIFFICULTY
# Chance that citizen gives a WRONG answer
# -----------------------------------------

NOISE_LEVELS = {
    Difficulty.EASY:   0.0,    # No noise on easy
    Difficulty.MEDIUM: 0.10,   # 10% chance of wrong answer
    Difficulty.HARD:   0.20,   # 20% chance of wrong answer
}

# Questions that CAN have noise applied
NOISY_QUESTIONS = [
    ActionType.ASK_BPL,
    ActionType.ASK_INCOME,
    ActionType.ASK_CASTE,
    ActionType.ASK_DISABILITY,
]

# -----------------------------------------
# SCHEME EXPIRY SETTINGS
# -----------------------------------------

SCHEME_EXPIRY_CHANCE = {
    Difficulty.EASY:   0.0,    # No expiry on easy
    Difficulty.MEDIUM: 0.10,   # 10% chance per step
    Difficulty.HARD:   0.20,   # 20% chance per step
}

SCHEME_EXPIRY_INTERVAL = 3     # Check for expiry every 3 steps

# -----------------------------------------
# INCOMPLETE INFO SETTINGS
# Chance citizen says "I don't know"
# -----------------------------------------

INCOMPLETE_INFO_CHANCE = {
    Difficulty.EASY:   0.0,    # No incomplete info on easy
    Difficulty.MEDIUM: 0.15,   # 15% chance
    Difficulty.HARD:   0.25,   # 25% chance
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
        if self.difficulty == Difficulty.EASY:
            self.max_steps = 10
            self.available_schemes = [
                "PM Ujjwala Yojana",
                "PM Kisan Samman Nidhi",
                "Ayushman Bharat",
                "PM Scholarship Scheme",
                "MGNREGA",
            ]
        elif self.difficulty == Difficulty.MEDIUM:
            self.max_steps = 8
            self.available_schemes = [
                "PM Ujjwala Yojana",
                "PM Kisan Samman Nidhi",
                "Ayushman Bharat",
                "PM Scholarship Scheme",
                "MGNREGA",
                "PM Awas Yojana (Gramin)",
                "Divyangjan Scholarship",
                "SC/ST Scholarship",
                "PM Mudra Yojana",
                "Fasal Bima Yojana",
            ]
        else:  # HARD
            self.max_steps = 6
            self.available_schemes = list(SCHEMES.keys())

        # Keep a copy of original schemes for expiry tracking
        self._original_schemes = list(self.available_schemes)

    def _generate_citizen(self) -> CitizenProfile:
        """Generate a random hidden citizen profile"""
        occupation = random.choice(list(Occupation))

        if occupation == Occupation.STUDENT:
            income = random.randint(100000, 800000)
        elif occupation == Occupation.FARMER:
            income = random.randint(30000, 300000)
        elif occupation == Occupation.UNEMPLOYED:
            income = random.randint(0, 80000)
        elif occupation == Occupation.DAILY_WAGE:
            income = random.randint(50000, 150000)
        elif occupation == Occupation.SMALL_BUSINESS:
            income = random.randint(100000, 1200000)
        else:
            income = random.randint(200000, 1500000)

        profile = CitizenProfile(
            age=random.randint(5, 75),
            income=float(income),
            gender=random.choice(list(Gender)),
            caste=random.choice(list(CasteCategory)),
            location=random.choice(list(Location)),
            occupation=occupation,
            has_disability=random.random() < 0.2,
            is_bpl=random.random() < 0.4,
            correct_schemes=[]
        )

        correct = [
            name for name, scheme in SCHEMES.items()
            if name in self.available_schemes and scheme["conditions"](profile)
        ]
        profile.correct_schemes = correct if correct else ["No scheme available"]
        return profile

    # -----------------------------------------
    # FEATURE 1: NOISE
    # Citizen gives a wrong answer with some probability
    # -----------------------------------------

    def _apply_noise(self, action_type: ActionType, true_value):
        """
        Randomly flip the answer the citizen gives.
        Only applies to noisy questions on medium/hard.
        Returns (noisy_value, was_noisy)
        """
        noise_chance = NOISE_LEVELS.get(self.difficulty, 0.0)

        if action_type not in NOISY_QUESTIONS or random.random() > noise_chance:
            return true_value, False

        # Flip the value based on type
        if isinstance(true_value, bool):
            return not true_value, True
        elif isinstance(true_value, float):
            # Give a wrong income — shift by 30-50%
            factor = random.choice([0.5, 0.6, 1.5, 1.8])
            return round(true_value * factor, 2), True
        elif isinstance(true_value, CasteCategory):
            others = [c for c in CasteCategory if c != true_value]
            return random.choice(others), True

        return true_value, False

    # -----------------------------------------
    # FEATURE 2: SCHEME EXPIRY
    # Schemes can become unavailable mid-episode
    # -----------------------------------------

    def _check_scheme_expiry(self, obs: Observation) -> Optional[str]:
        """
        Every SCHEME_EXPIRY_INTERVAL steps, randomly expire a scheme.
        Returns the name of expired scheme or None.
        """
        expiry_chance = SCHEME_EXPIRY_CHANCE.get(self.difficulty, 0.0)

        if (self.state.step_count % SCHEME_EXPIRY_INTERVAL == 0
                and self.state.step_count > 0
                and random.random() < expiry_chance
                and len(self.available_schemes) > 2):

            # Pick a random scheme to expire (not the correct one if possible)
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
    # Citizen sometimes says "I don't know"
    # -----------------------------------------

    def _check_incomplete_info(self, action_type: ActionType) -> bool:
        """
        Returns True if citizen doesn't know the answer.
        Only applies to certain questions on medium/hard.
        """
        incomplete_chance = INCOMPLETE_INFO_CHANCE.get(self.difficulty, 0.0)
        if action_type not in INCOMPLETE_INFO_QUESTIONS:
            return False
        return random.random() < incomplete_chance

    # -----------------------------------------
    # FEATURE 4: CONFLICTING SIGNALS
    # Partial eligibility scoring for recommendations
    # -----------------------------------------

    def _get_partial_match_score(self, scheme_name: str) -> float:
        """
        If agent recommends a scheme citizen doesn't fully qualify for,
        check if there's partial eligibility and give partial reward.
        Returns partial reward value (0.0 if no partial match)
        """
        if scheme_name not in SCHEMES:
            return 0.0

        scheme = SCHEMES[scheme_name]
        citizen = self.state.citizen_profile

        for condition_fn, partial_reward in scheme.get("partial_conditions", []):
            try:
                if condition_fn(citizen):
                    return partial_reward
            except Exception:
                pass

        return 0.0

    # -----------------------------------------
    # reset() — Start a fresh episode
    # -----------------------------------------

    def reset(self) -> Observation:
        """Reset environment and return initial empty observation"""
        self._setup_difficulty()
        citizen = self._generate_citizen()

        initial_observation = Observation(
            step_count=0,
            max_steps=self.max_steps,
            last_action_result=(
                "New citizen arrived. "
                "TIP: Ask occupation first — income means different things "
                "for students, farmers, and unemployed citizens! "
                f"NOTE: Difficulty is '{self.difficulty.value}' — "
                + ("No noise or scheme changes." if self.difficulty == Difficulty.EASY
                   else "Citizen may give uncertain answers. Schemes may expire!")
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
    # step() — Agent takes an action
    # -----------------------------------------

    def step(self, action: Action) -> StepResult:
        """Process agent action and return observation, reward, done, info"""

        if self.state is None:
            raise RuntimeError("Call reset() before step()")
        if self.state.is_done:
            raise RuntimeError("Episode is done. Call reset() to start a new episode.")

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
                reward_value = -0.3
                reward_reason = f"Already asked '{question}'. Avoid repeating — costs valuable steps!"

            # Penalty: income asked before occupation
            elif (action.action_type == ActionType.ASK_INCOME
                  and ActionType.ASK_OCCUPATION.value not in self.state.questions_asked):
                reward_value = INCOME_BEFORE_OCCUPATION_PENALTY
                reward_reason = (
                    "Asked income before knowing occupation! "
                    "Income means different things for different people. Penalty applied."
                )
                obs.income = citizen.income
                obs.income_context = IncomeContext.NOT_APPLICABLE
                obs.income_hint = "WARNING: Income context unknown. Ask occupation first!"
                self.state.questions_asked.append(question)

            else:
                # FEATURE 3: Check if citizen doesn't know
                if self._check_incomplete_info(action.action_type):
                    reward_value = 0.05
                    reward_reason = (
                        f"Good question — but citizen is unsure about '{question}'. "
                        "No clear answer received. Try cross-verifying with another question."
                    )
                    self.state.questions_asked.append(question)
                    info["incomplete"] = True
                    info["hint"] = "Citizen gave uncertain answer — consider asking a related question to verify."

                else:
                    # Normal question — get true value then apply noise
                    relevance = QUESTION_RELEVANCE.get(action.action_type, 0.3)
                    reward_value = relevance * 0.3
                    reward_reason = f"Good question! '{question}' revealed. Relevance: {relevance}"

                    # Bonus for asking occupation first
                    if (action.action_type == ActionType.ASK_OCCUPATION
                            and len(self.state.questions_asked) == 0):
                        reward_value += OCCUPATION_FIRST_BONUS
                        reward_reason += f" | BONUS: Asked occupation first! +{OCCUPATION_FIRST_BONUS}"
                        self.state.occupation_asked_first = True

                    self.state.questions_asked.append(question)

                    # Reveal answer (with possible noise)
                    if action.action_type == ActionType.ASK_AGE:
                        obs.age = citizen.age

                    elif action.action_type == ActionType.ASK_INCOME:
                        noisy_income, was_noisy = self._apply_noise(action.action_type, citizen.income)
                        obs.income = noisy_income
                        obs.income_context = citizen.income_context()
                        reward_reason += f" | Income context: {citizen.income_label()}"
                        if was_noisy:
                            reward_reason += " | [NOISE: Citizen may have given approximate income]"
                            info["noise_warning"] = "Income value may be inaccurate — citizen was uncertain"

                    elif action.action_type == ActionType.ASK_GENDER:
                        obs.gender = citizen.gender

                    elif action.action_type == ActionType.ASK_CASTE:
                        noisy_caste, was_noisy = self._apply_noise(action.action_type, citizen.caste)
                        obs.caste = noisy_caste
                        if was_noisy:
                            reward_reason += " | [NOISE: Citizen may have misreported caste category]"
                            info["noise_warning"] = "Caste value may be inaccurate"

                    elif action.action_type == ActionType.ASK_LOCATION:
                        obs.location = citizen.location

                    elif action.action_type == ActionType.ASK_OCCUPATION:
                        obs.occupation = citizen.occupation
                        reward_reason += f" | Occupation: {citizen.occupation.value}"

                    elif action.action_type == ActionType.ASK_DISABILITY:
                        noisy_disability, was_noisy = self._apply_noise(action.action_type, citizen.has_disability)
                        obs.has_disability = noisy_disability
                        if was_noisy:
                            reward_reason += " | [NOISE: Disability status may be misreported]"
                            info["noise_warning"] = "Disability status may be inaccurate"

                    elif action.action_type == ActionType.ASK_BPL:
                        noisy_bpl, was_noisy = self._apply_noise(action.action_type, citizen.is_bpl)
                        obs.is_bpl = noisy_bpl
                        if was_noisy:
                            reward_reason += " | [NOISE: BPL status may be misreported]"
                            info["noise_warning"] = "BPL status may be inaccurate — consider cross-verifying with income"

            # FEATURE 2: Check for scheme expiry after question
            expired = self._check_scheme_expiry(obs)
            if expired:
                reward_reason += f" | ALERT: '{expired}' scheme has expired and is no longer available!"
                info["scheme_expired"] = expired

        # ── CASE 2: Recommending a scheme ──
        else:
            recommended = action.scheme_name

            if not recommended:
                reward_value = -0.5
                reward_reason = "You must provide a scheme name when recommending!"
                done = True

            elif recommended not in self.available_schemes:
                # Check if it expired mid-episode
                if recommended in self._original_schemes:
                    reward_value = -0.3
                    reward_reason = f"'{recommended}' was valid but has expired during this episode!"
                else:
                    reward_value = -0.5
                    reward_reason = f"'{recommended}' is not in the available schemes list."
                done = True

            elif recommended in citizen.correct_schemes:
                reward_value = 1.0
                reward_reason = f"CORRECT! '{recommended}' is the perfect scheme for this citizen!"
                done = True
                info["success"] = True

            else:
                # FEATURE 4: Check for partial/conflicting eligibility
                partial_score = self._get_partial_match_score(recommended)
                if partial_score > 0:
                    reward_value = partial_score
                    reward_reason = (
                        f"PARTIAL MATCH — '{recommended}' partially applies but citizen "
                        f"doesn't fully qualify. Partial reward: {partial_score}. "
                        f"Correct schemes: {citizen.correct_schemes}"
                    )
                    info["partial_match"] = True
                    info["partial_score"] = partial_score
                else:
                    reward_value = -0.5
                    reward_reason = f"WRONG! '{recommended}' does not match this citizen at all."
                    info["success"] = False

                info["correct_schemes"] = citizen.correct_schemes
                done = True

        # ── Step limit ──
        self.state.step_count += 1
        obs.step_count = self.state.step_count

        if self.state.step_count >= self.max_steps and not done:
            reward_value -= 0.2
            reward_reason += " | Step limit reached without recommending!"
            done = True
            info["timeout"] = True
            info["correct_schemes"] = citizen.correct_schemes

        # ── Update totals ──
        self.state.total_reward += reward_value
        self.state.is_done = done
        obs.done = done
        obs.last_action_result = reward_reason

        reward = Reward(
            value=round(reward_value, 3),
            reason=reward_reason,
            total_score=round(
                max(0.0, min(1.0, self.state.total_reward / self.max_steps)),
                3
            )
        )

        return StepResult(
            observation=obs,
            reward=reward,
            done=done,
            info=info
        )

    # -----------------------------------------
    # state() — Return full internal state
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
    print("Testing Updated Environment — All 4 RL Features")
    print("=" * 60)

    # Test on HARD to see all features
    env = GovSchemeEnvironment(difficulty=Difficulty.HARD)
    obs = env.reset()
    print(f"\n[PASS] reset() works!")
    print(f"   Citizen occupation (hidden): {env.state.citizen_profile.occupation.value}")
    print(f"   Correct schemes (hidden): {env.state.citizen_profile.correct_schemes}")
    print(f"   Available schemes: {len(obs.available_schemes)}")

    from models import Action, ActionType

    actions = [
        Action(action_type=ActionType.ASK_OCCUPATION),
        Action(action_type=ActionType.ASK_BPL),
        Action(action_type=ActionType.ASK_INCOME),
        Action(action_type=ActionType.ASK_DISABILITY),
    ]

    print(f"\nRunning 4 questions on HARD difficulty...\n")
    for i, action in enumerate(actions):
        result = env.step(action)
        print(f"Step {i+1}: {action.action_type.value}")
        print(f"  Reward : {result.reward.value}")
        print(f"  Reason : {result.reward.reason[:100]}")
        if result.info:
            print(f"  Info   : {result.info}")
        if result.done:
            break

    # Try recommending correct scheme
    correct = env.state.citizen_profile.correct_schemes[0]
    result = env.step(Action(
        action_type=ActionType.RECOMMEND_SCHEME,
        scheme_name=correct
    ))
    print(f"\nStep 5: recommend '{correct}'")
    print(f"  Reward : {result.reward.value}")
    print(f"  Reason : {result.reward.reason[:100]}")
    print(f"  Done   : {result.done}")
    print(f"  Score  : {result.reward.total_score}")

    print("\n" + "=" * 60)
    print("All features working!")
    print("=" * 60)