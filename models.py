from pydantic import BaseModel, Field
from typing import Optional, List
from enum import Enum


# -----------------------------------------
# ENUMS — Fixed choices for the environment
# -----------------------------------------

class ActionType(str, Enum):
    ASK_AGE = "ask_age"
    ASK_INCOME = "ask_income"           # Context-aware — means different things based on occupation
    ASK_GENDER = "ask_gender"
    ASK_CASTE = "ask_caste"
    ASK_LOCATION = "ask_location"
    ASK_OCCUPATION = "ask_occupation"   # Should always be asked FIRST
    ASK_DISABILITY = "ask_disability"
    ASK_BPL = "ask_bpl"
    RECOMMEND_SCHEME = "recommend_scheme"


class IncomeContext(str, Enum):
    """
    Describes WHOSE income is being reported.
    This changes automatically based on occupation:
      - Student     -> parent's income
      - Farmer      -> annual crop income
      - Unemployed  -> household income
      - Daily wage  -> monthly wage x 12
      - Business    -> business income
      - Govt job    -> personal salary
    """
    PERSONAL = "personal"
    PARENTAL = "parental"
    CROP = "crop"
    HOUSEHOLD = "household"
    NOT_APPLICABLE = "not_applicable"


class Gender(str, Enum):
    MALE = "male"
    FEMALE = "female"
    OTHER = "other"


class CasteCategory(str, Enum):
    GENERAL = "general"
    OBC = "obc"
    SC = "sc"
    ST = "st"


class Location(str, Enum):
    RURAL = "rural"
    URBAN = "urban"


class Occupation(str, Enum):
    FARMER = "farmer"
    STUDENT = "student"
    DAILY_WAGE = "daily_wage"
    SMALL_BUSINESS = "small_business"
    UNEMPLOYED = "unemployed"
    GOVERNMENT_EMPLOYEE = "government_employee"


class Difficulty(str, Enum):
    EASY = "easy"
    MEDIUM = "medium"
    HARD = "hard"


# -----------------------------------------
# HELPER — Map occupation to income context
# -----------------------------------------

OCCUPATION_INCOME_CONTEXT = {
    Occupation.STUDENT:             IncomeContext.PARENTAL,
    Occupation.FARMER:              IncomeContext.CROP,
    Occupation.DAILY_WAGE:          IncomeContext.HOUSEHOLD,
    Occupation.UNEMPLOYED:          IncomeContext.HOUSEHOLD,
    Occupation.SMALL_BUSINESS:      IncomeContext.PERSONAL,
    Occupation.GOVERNMENT_EMPLOYEE: IncomeContext.PERSONAL,
}

# Penalty for asking income BEFORE occupation
INCOME_BEFORE_OCCUPATION_PENALTY = -0.4

# Bonus for asking occupation FIRST (smart maze navigation)
OCCUPATION_FIRST_BONUS = 0.2


# -----------------------------------------
# ACTION — What the agent can do each step
# -----------------------------------------

class Action(BaseModel):
    action_type: ActionType = Field(
        ...,
        description="The type of action the agent wants to take"
    )
    scheme_name: Optional[str] = Field(
        default=None,
        description="Name of scheme to recommend. Only needed for RECOMMEND_SCHEME"
    )


# -----------------------------------------
# OBSERVATION — What the agent sees
# -----------------------------------------

class Observation(BaseModel):
    # Revealed one by one as agent asks questions
    age: Optional[int] = Field(default=None, description="Citizen's age")
    income: Optional[float] = Field(default=None, description="Income in rupees (context depends on occupation)")
    income_context: Optional[IncomeContext] = Field(default=None, description="Whose income this is — revealed with income")
    gender: Optional[Gender] = Field(default=None)
    caste: Optional[CasteCategory] = Field(default=None)
    location: Optional[Location] = Field(default=None)
    occupation: Optional[Occupation] = Field(default=None)
    has_disability: Optional[bool] = Field(default=None)
    is_bpl: Optional[bool] = Field(default=None)

    # Hint shown to agent when income asked before occupation
    income_hint: Optional[str] = Field(
        default=None,
        description="Hint shown to agent when income is asked before occupation"
    )

    # Episode progress
    step_count: int = Field(default=0)
    max_steps: int = Field(default=10)
    last_action_result: Optional[str] = Field(default=None)
    available_schemes: List[str] = Field(default_factory=list)
    done: bool = Field(default=False)


# -----------------------------------------
# REWARD — Score after each step
# -----------------------------------------

class Reward(BaseModel):
    value: float = Field(..., description="Reward for this step")
    reason: str = Field(..., description="Why this reward was given")
    total_score: float = Field(default=0.0, description="Running score 0.0 to 1.0")


# -----------------------------------------
# CITIZEN PROFILE — Hidden from agent
# -----------------------------------------

class CitizenProfile(BaseModel):
    """True citizen profile — agent cannot see this directly"""
    age: int
    income: float
    gender: Gender
    caste: CasteCategory
    location: Location
    occupation: Occupation
    has_disability: bool
    is_bpl: bool
    correct_schemes: List[str] = Field(description="Correct schemes for this citizen")

    def income_context(self) -> IncomeContext:
        """Returns the correct income context based on occupation"""
        return OCCUPATION_INCOME_CONTEXT.get(self.occupation, IncomeContext.PERSONAL)

    def income_label(self) -> str:
        """Returns a human readable label for what income means for this citizen"""
        context = self.income_context()
        labels = {
            IncomeContext.PARENTAL:         "Parent's annual income",
            IncomeContext.CROP:             "Annual crop income",
            IncomeContext.HOUSEHOLD:        "Household annual income",
            IncomeContext.PERSONAL:         "Personal annual income",
            IncomeContext.NOT_APPLICABLE:   "No income applicable",
        }
        return labels.get(context, "Annual income")


# -----------------------------------------
# STATE — Full internal environment state
# -----------------------------------------

class State(BaseModel):
    citizen_profile: CitizenProfile
    observation: Observation
    difficulty: Difficulty
    episode_id: str
    step_count: int = Field(default=0)
    total_reward: float = Field(default=0.0)
    is_done: bool = Field(default=False)
    questions_asked: List[str] = Field(default_factory=list)
    occupation_asked_first: bool = Field(
        default=False,
        description="Tracks if agent asked occupation before income"
    )


# -----------------------------------------
# STEP RESULT — What step() returns
# -----------------------------------------

class StepResult(BaseModel):
    observation: Observation
    reward: Reward
    done: bool
    info: dict = Field(default_factory=dict)