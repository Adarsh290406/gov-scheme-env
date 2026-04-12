from pydantic import BaseModel, Field
from typing import Optional, List
from enum import Enum


# -----------------------------------------
# ENUMS
# -----------------------------------------

class ActionType(str, Enum):
    # Original questions
    ASK_AGE             = "ask_age"
    ASK_INCOME          = "ask_income"
    ASK_GENDER          = "ask_gender"
    ASK_CASTE           = "ask_caste"
    ASK_LOCATION        = "ask_location"
    ASK_OCCUPATION      = "ask_occupation"      # Always ask FIRST
    ASK_DISABILITY      = "ask_disability"
    ASK_BPL             = "ask_bpl"
    # New questions
    ASK_EDUCATION       = "ask_education"       # Education level
    ASK_BANK_ACCOUNT    = "ask_bank_account"    # Has Jan Dhan / bank account?
    ASK_RATION_CARD     = "ask_ration_card"     # Has ration card?
    ASK_MARITAL_STATUS  = "ask_marital_status"  # Married/single/widowed?
    ASK_LAND_OWNERSHIP  = "ask_land_ownership"  # Owns/rents/no land?
    ASK_STATE           = "ask_state"           # Which state?
    # Recommend
    RECOMMEND_SCHEME    = "recommend_scheme"


class IncomeContext(str, Enum):
    PERSONAL        = "personal"
    PARENTAL        = "parental"
    CROP            = "crop"
    HOUSEHOLD       = "household"
    NOT_APPLICABLE  = "not_applicable"


class Gender(str, Enum):
    MALE    = "male"
    FEMALE  = "female"
    OTHER   = "other"


class CasteCategory(str, Enum):
    GENERAL = "general"
    OBC     = "obc"
    SC      = "sc"
    ST      = "st"


class Location(str, Enum):
    RURAL = "rural"
    URBAN = "urban"


class Occupation(str, Enum):
    FARMER              = "farmer"
    STUDENT             = "student"
    DAILY_WAGE          = "daily_wage"
    SMALL_BUSINESS      = "small_business"
    UNEMPLOYED          = "unemployed"
    GOVERNMENT_EMPLOYEE = "government_employee"


class Difficulty(str, Enum):
    EASY   = "easy"
    MEDIUM = "medium"
    HARD   = "hard"


# -----------------------------------------
# HELPERS
# -----------------------------------------

OCCUPATION_INCOME_CONTEXT = {
    Occupation.STUDENT:             IncomeContext.PARENTAL,
    Occupation.FARMER:              IncomeContext.CROP,
    Occupation.DAILY_WAGE:          IncomeContext.HOUSEHOLD,
    Occupation.UNEMPLOYED:          IncomeContext.HOUSEHOLD,
    Occupation.SMALL_BUSINESS:      IncomeContext.PERSONAL,
    Occupation.GOVERNMENT_EMPLOYEE: IncomeContext.PERSONAL,
}

INCOME_BEFORE_OCCUPATION_PENALTY = -0.4
OCCUPATION_FIRST_BONUS = 0.05


# -----------------------------------------
# ACTION
# -----------------------------------------

class Action(BaseModel):
    action_type: ActionType = Field(
        ...,
        description="The type of action the agent wants to take"
    )
    scheme_name: Optional[str] = Field(
        default=None,
        description="Scheme to recommend. Only needed for RECOMMEND_SCHEME"
    )


# -----------------------------------------
# OBSERVATION — What agent sees
# -----------------------------------------

class Observation(BaseModel):
    # Original fields
    age: Optional[int]                      = Field(default=None)
    income: Optional[float]                 = Field(default=None)
    income_context: Optional[IncomeContext] = Field(default=None)
    gender: Optional[Gender]                = Field(default=None)
    caste: Optional[CasteCategory]          = Field(default=None)
    location: Optional[Location]            = Field(default=None)
    occupation: Optional[Occupation]        = Field(default=None)
    has_disability: Optional[bool]          = Field(default=None)
    is_bpl: Optional[bool]                  = Field(default=None)
    income_hint: Optional[str]              = Field(default=None)

    # New fields
    education: Optional[str]               = Field(default=None, description="Education level")
    has_bank_account: Optional[bool]        = Field(default=None)
    has_ration_card: Optional[bool]         = Field(default=None)
    marital_status: Optional[str]           = Field(default=None)
    land_ownership: Optional[str]           = Field(default=None)
    state: Optional[str]                    = Field(default=None, description="State of residence")

    # Episode progress
    step_count: int                         = Field(default=0)
    max_steps: int                          = Field(default=10)
    last_action_result: Optional[str]       = Field(default=None)
    available_schemes: List[str]            = Field(default_factory=list)
    done: bool                              = Field(default=False)


# -----------------------------------------
# REWARD
# -----------------------------------------

class Reward(BaseModel):
    value: float        = Field(..., description="Reward for this step")
    reason: str         = Field(..., description="Why this reward was given")
    total_score: float  = Field(default=0.01, description="Running score 0.01 to 0.99")


# -----------------------------------------
# CITIZEN PROFILE — Hidden from agent
# -----------------------------------------

class CitizenProfile(BaseModel):
    age: int
    income: float
    gender: Gender
    caste: CasteCategory
    location: Location
    occupation: Occupation
    has_disability: bool
    is_bpl: bool
    # New fields
    has_bank_account: bool      = Field(default=True)
    has_ration_card: bool       = Field(default=False)
    marital_status: str         = Field(default="any")
    land_ownership: str         = Field(default="any")
    state: str                  = Field(default="any")
    education: str              = Field(default="any")
    correct_schemes: List[str]  = Field(default_factory=list)

    def income_context(self) -> IncomeContext:
        return OCCUPATION_INCOME_CONTEXT.get(self.occupation, IncomeContext.PERSONAL)

    def income_label(self) -> str:
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
# STATE
# -----------------------------------------

class State(BaseModel):
    citizen_profile: CitizenProfile
    observation: Observation
    difficulty: Difficulty
    episode_id: str
    step_count: int             = Field(default=0)
    total_reward: float         = Field(default=0.01)
    is_done: bool               = Field(default=False)
    questions_asked: List[str]  = Field(default_factory=list)
    occupation_asked_first: bool = Field(default=False)


# -----------------------------------------
# STEP RESULT
# -----------------------------------------

class StepResult(BaseModel):
    observation: Observation
    reward: Reward
    done: bool
    info: dict = Field(default_factory=dict)