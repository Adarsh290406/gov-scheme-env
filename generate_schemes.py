"""
generate_schemes.py — Auto-generate Schemes Database
=====================================================
Runs ONCE to generate a comprehensive database of
Indian government schemes using Groq API.

Usage:
  python generate_schemes.py
"""

import os
import json
import re
import sys
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

client = OpenAI(
    api_key=os.getenv("OPENAI_API_KEY"),
    base_url="https://api.groq.com/openai/v1"
)

MODEL = "llama-3.1-8b-instant"

# -----------------------------------------
# PROMPTS — One per category to keep it
# focused and avoid JSON errors
# -----------------------------------------

CATEGORY_PROMPTS = [
    {
        "category": "Farmer Schemes",
        "prompt": """List 10 real Indian government schemes specifically for FARMERS.
Return ONLY a JSON array. No markdown, no backticks, no explanation.
Each item must have exactly these fields:
[
  {
    "name": "scheme name",
    "description": "one line description",
    "ministry": "ministry name",
    "benefit": "what citizen gets",
    "priority": 7,
    "conditions": {
      "gender": "any",
      "min_age": 18,
      "max_age": null,
      "max_income": 200000,
      "occupation": "farmer",
      "location": "any",
      "caste": "any",
      "is_bpl": null,
      "has_disability": null,
      "min_education": "any",
      "has_bank_account": true,
      "has_ration_card": null,
      "marital_status": "any",
      "land_ownership": "owner",
      "state": "any"
    }
  }
]"""
    },
    {
        "category": "Student Schemes",
        "prompt": """List 10 real Indian government schemes specifically for STUDENTS.
Return ONLY a JSON array. No markdown, no backticks, no explanation.
Each item must have exactly these fields:
[
  {
    "name": "scheme name",
    "description": "one line description",
    "ministry": "ministry name",
    "benefit": "what citizen gets",
    "priority": 7,
    "conditions": {
      "gender": "any",
      "min_age": 15,
      "max_age": 25,
      "max_income": 600000,
      "occupation": "student",
      "location": "any",
      "caste": "any",
      "is_bpl": null,
      "has_disability": null,
      "min_education": "10th",
      "has_bank_account": null,
      "has_ration_card": null,
      "marital_status": "any",
      "land_ownership": "any",
      "state": "any"
    }
  }
]"""
    },
    {
        "category": "Women Schemes",
        "prompt": """List 10 real Indian government schemes specifically for WOMEN.
Return ONLY a JSON array. No markdown, no backticks, no explanation.
Each item must have exactly these fields:
[
  {
    "name": "scheme name",
    "description": "one line description",
    "ministry": "ministry name",
    "benefit": "what citizen gets",
    "priority": 7,
    "conditions": {
      "gender": "female",
      "min_age": 18,
      "max_age": null,
      "max_income": null,
      "occupation": "any",
      "location": "any",
      "caste": "any",
      "is_bpl": null,
      "has_disability": null,
      "min_education": "any",
      "has_bank_account": null,
      "has_ration_card": null,
      "marital_status": "any",
      "land_ownership": "any",
      "state": "any"
    }
  }
]"""
    },
    {
        "category": "BPL and Disability Schemes",
        "prompt": """List 10 real Indian government schemes for BPL families and disabled citizens.
Return ONLY a JSON array. No markdown, no backticks, no explanation.
Each item must have exactly these fields:
[
  {
    "name": "scheme name",
    "description": "one line description",
    "ministry": "ministry name",
    "benefit": "what citizen gets",
    "priority": 8,
    "conditions": {
      "gender": "any",
      "min_age": null,
      "max_age": null,
      "max_income": null,
      "occupation": "any",
      "location": "any",
      "caste": "any",
      "is_bpl": true,
      "has_disability": null,
      "min_education": "any",
      "has_bank_account": null,
      "has_ration_card": null,
      "marital_status": "any",
      "land_ownership": "any",
      "state": "any"
    }
  }
]"""
    },
    {
        "category": "SC ST OBC Schemes",
        "prompt": """List 10 real Indian government schemes specifically for SC, ST, and OBC communities.
Return ONLY a JSON array. No markdown, no backticks, no explanation.
Each item must have exactly these fields:
[
  {
    "name": "scheme name",
    "description": "one line description",
    "ministry": "ministry name",
    "benefit": "what citizen gets",
    "priority": 7,
    "conditions": {
      "gender": "any",
      "min_age": null,
      "max_age": null,
      "max_income": null,
      "occupation": "any",
      "location": "any",
      "caste": "sc",
      "is_bpl": null,
      "has_disability": null,
      "min_education": "any",
      "has_bank_account": null,
      "has_ration_card": null,
      "marital_status": "any",
      "land_ownership": "any",
      "state": "any"
    }
  }
]"""
    },
]


# -----------------------------------------
# JSON CLEANER
# -----------------------------------------

def extract_json(raw: str) -> str:
    """Extract JSON array from messy LLM response"""

    # Remove markdown code blocks
    raw = re.sub(r'```json', '', raw)
    raw = re.sub(r'```', '', raw)
    raw = raw.strip()

    # Find the JSON array
    start = raw.find('[')
    end = raw.rfind(']')

    if start == -1 or end == -1:
        return ""

    return raw[start:end+1]


# -----------------------------------------
# GENERATE PER CATEGORY
# -----------------------------------------

def generate_category(category_info: dict) -> list:
    """Generate schemes for one category"""
    print(f"  Generating: {category_info['category']}...")

    try:
        response = client.chat.completions.create(
            model=MODEL,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are an expert on Indian government schemes. "
                        "Always respond with a valid JSON array only. "
                        "Never use markdown formatting or backticks. "
                        "Never add any text before or after the JSON array."
                    )
                },
                {
                    "role": "user",
                    "content": category_info["prompt"]
                }
            ],
            temperature=0.1,
            max_tokens=3000,
        )

        raw = response.choices[0].message.content.strip()

        # Debug — print first 100 chars
        print(f"    Response preview: {raw[:100]}")

        cleaned = extract_json(raw)
        if not cleaned:
            print(f"    Could not find JSON array in response")
            return []

        schemes = json.loads(cleaned)
        print(f"    Got {len(schemes)} schemes")
        return schemes

    except json.JSONDecodeError as e:
        print(f"    JSON error: {e}")
        return []
    except Exception as e:
        print(f"    API error: {e}")
        return []


def validate_scheme(scheme: dict) -> bool:
    """Check if scheme has minimum required fields"""
    if not isinstance(scheme, dict):
        return False
    if "name" not in scheme or not scheme["name"]:
        return False
    if "conditions" not in scheme:
        return False
    return True


def clean_scheme(scheme: dict) -> dict:
    """Fill in missing fields with safe defaults"""
    conditions = scheme.get("conditions", {})

    defaults = {
        "gender": "any",
        "min_age": None,
        "max_age": None,
        "max_income": None,
        "occupation": "any",
        "location": "any",
        "caste": "any",
        "is_bpl": None,
        "has_disability": None,
        "min_education": "any",
        "has_bank_account": None,
        "has_ration_card": None,
        "marital_status": "any",
        "land_ownership": "any",
        "state": "any"
    }

    for key, default in defaults.items():
        if key not in conditions:
            conditions[key] = default

    scheme["conditions"] = conditions
    scheme["ministry"] = scheme.get("ministry", "Government of India")
    scheme["priority"] = int(scheme.get("priority", 5))
    scheme["benefit"] = scheme.get("benefit", "Government benefit")
    scheme["description"] = scheme.get("description", "Government scheme")

    return scheme


# -----------------------------------------
# ORIGINAL SCHEMES — Always included
# -----------------------------------------

ORIGINAL_SCHEMES = [
    {
        "name": "PM Ujjwala Yojana",
        "description": "Free LPG connection for BPL women",
        "ministry": "Ministry of Petroleum",
        "benefit": "Free LPG connection + first refill",
        "priority": 8,
        "conditions": {
            "gender": "female", "min_age": 18, "max_age": None,
            "max_income": None, "occupation": "any", "location": "any",
            "caste": "any", "is_bpl": True, "has_disability": None,
            "min_education": "any", "has_bank_account": None,
            "has_ration_card": None, "marital_status": "any",
            "land_ownership": "any", "state": "any"
        }
    },
    {
        "name": "PM Kisan Samman Nidhi",
        "description": "Rs.6000/year direct income support for farmers",
        "ministry": "Ministry of Agriculture",
        "benefit": "Rs.6000 per year in 3 installments",
        "priority": 9,
        "conditions": {
            "gender": "any", "min_age": 18, "max_age": None,
            "max_income": 200000, "occupation": "farmer", "location": "any",
            "caste": "any", "is_bpl": None, "has_disability": None,
            "min_education": "any", "has_bank_account": True,
            "has_ration_card": None, "marital_status": "any",
            "land_ownership": "owner", "state": "any"
        }
    },
    {
        "name": "Ayushman Bharat",
        "description": "Health insurance up to Rs.5 lakh for BPL families",
        "ministry": "Ministry of Health",
        "benefit": "Health coverage up to Rs.5 lakh per year",
        "priority": 10,
        "conditions": {
            "gender": "any", "min_age": None, "max_age": None,
            "max_income": None, "occupation": "any", "location": "any",
            "caste": "any", "is_bpl": True, "has_disability": None,
            "min_education": "any", "has_bank_account": None,
            "has_ration_card": True, "marital_status": "any",
            "land_ownership": "any", "state": "any"
        }
    },
    {
        "name": "MGNREGA",
        "description": "100 days guaranteed wage employment in rural areas",
        "ministry": "Ministry of Rural Development",
        "benefit": "100 days guaranteed employment at minimum wage",
        "priority": 7,
        "conditions": {
            "gender": "any", "min_age": 18, "max_age": None,
            "max_income": None, "occupation": "any", "location": "rural",
            "caste": "any", "is_bpl": None, "has_disability": None,
            "min_education": "any", "has_bank_account": True,
            "has_ration_card": None, "marital_status": "any",
            "land_ownership": "any", "state": "any"
        }
    },
    {
        "name": "Fasal Bima Yojana",
        "description": "Crop insurance for farmers",
        "ministry": "Ministry of Agriculture",
        "benefit": "Crop insurance coverage against natural disasters",
        "priority": 8,
        "conditions": {
            "gender": "any", "min_age": 18, "max_age": None,
            "max_income": None, "occupation": "farmer", "location": "any",
            "caste": "any", "is_bpl": None, "has_disability": None,
            "min_education": "any", "has_bank_account": True,
            "has_ration_card": None, "marital_status": "any",
            "land_ownership": "any", "state": "any"
        }
    },
    {
        "name": "PM Awas Yojana Gramin",
        "description": "Rural housing scheme for BPL families",
        "ministry": "Ministry of Rural Development",
        "benefit": "Financial assistance for house construction",
        "priority": 8,
        "conditions": {
            "gender": "any", "min_age": 18, "max_age": None,
            "max_income": None, "occupation": "any", "location": "rural",
            "caste": "any", "is_bpl": True, "has_disability": None,
            "min_education": "any", "has_bank_account": None,
            "has_ration_card": None, "marital_status": "any",
            "land_ownership": "any", "state": "any"
        }
    },
    {
        "name": "Divyangjan Scholarship",
        "description": "Scholarship and support for disabled citizens",
        "ministry": "Ministry of Social Justice",
        "benefit": "Monthly scholarship + assistive devices",
        "priority": 8,
        "conditions": {
            "gender": "any", "min_age": None, "max_age": None,
            "max_income": None, "occupation": "any", "location": "any",
            "caste": "any", "is_bpl": None, "has_disability": True,
            "min_education": "any", "has_bank_account": None,
            "has_ration_card": None, "marital_status": "any",
            "land_ownership": "any", "state": "any"
        }
    },
    {
        "name": "SC ST Scholarship",
        "description": "Educational support for SC/ST students",
        "ministry": "Ministry of Tribal Affairs",
        "benefit": "Annual scholarship for education",
        "priority": 7,
        "conditions": {
            "gender": "any", "min_age": None, "max_age": 25,
            "max_income": 250000, "occupation": "student", "location": "any",
            "caste": "sc", "is_bpl": None, "has_disability": None,
            "min_education": "any", "has_bank_account": None,
            "has_ration_card": None, "marital_status": "any",
            "land_ownership": "any", "state": "any"
        }
    },
    {
        "name": "PM Mudra Yojana",
        "description": "Loans for small business owners",
        "ministry": "Ministry of Finance",
        "benefit": "Collateral free loans up to Rs.10 lakh",
        "priority": 7,
        "conditions": {
            "gender": "any", "min_age": 18, "max_age": None,
            "max_income": 1000000, "occupation": "small_business",
            "location": "any", "caste": "any", "is_bpl": None,
            "has_disability": None, "min_education": "any",
            "has_bank_account": True, "has_ration_card": None,
            "marital_status": "any", "land_ownership": "any", "state": "any"
        }
    },
    {
        "name": "Indira Gandhi Disability Pension",
        "description": "Monthly pension for BPL disabled citizens",
        "ministry": "Ministry of Rural Development",
        "benefit": "Rs.300-500 per month pension",
        "priority": 9,
        "conditions": {
            "gender": "any", "min_age": 18, "max_age": None,
            "max_income": None, "occupation": "any", "location": "any",
            "caste": "any", "is_bpl": True, "has_disability": True,
            "min_education": "any", "has_bank_account": None,
            "has_ration_card": None, "marital_status": "any",
            "land_ownership": "any", "state": "any"
        }
    },
    {
        "name": "PM Scholarship Scheme",
        "description": "Scholarship for students from low income families",
        "ministry": "Ministry of Education",
        "benefit": "Annual scholarship up to Rs.25000",
        "priority": 7,
        "conditions": {
            "gender": "any", "min_age": 15, "max_age": 25,
            "max_income": 600000, "occupation": "student", "location": "any",
            "caste": "any", "is_bpl": None, "has_disability": None,
            "min_education": "10th", "has_bank_account": None,
            "has_ration_card": None, "marital_status": "any",
            "land_ownership": "any", "state": "any"
        }
    },
    {
        "name": "Sukanya Samriddhi Yojana",
        "description": "Savings scheme for girl child below 10 years",
        "ministry": "Ministry of Finance",
        "benefit": "High interest savings account for girl child",
        "priority": 7,
        "conditions": {
            "gender": "female", "min_age": None, "max_age": 10,
            "max_income": None, "occupation": "any", "location": "any",
            "caste": "any", "is_bpl": None, "has_disability": None,
            "min_education": "any", "has_bank_account": None,
            "has_ration_card": None, "marital_status": "any",
            "land_ownership": "any", "state": "any"
        }
    },
]


# -----------------------------------------
# MAIN
# -----------------------------------------

def main():
    print("=" * 60)
    print("Generating Indian Government Schemes Database")
    print("=" * 60)

    all_schemes = []
    seen_names = set()

    # Add original schemes first
    for scheme in ORIGINAL_SCHEMES:
        all_schemes.append(scheme)
        seen_names.add(scheme["name"])

    print(f"\nOriginal schemes loaded: {len(all_schemes)}")
    print("\nGenerating additional schemes via Groq API...")

    # Generate per category
    for category_info in CATEGORY_PROMPTS:
        batch = generate_category(category_info)

        for scheme in batch:
            if not validate_scheme(scheme):
                continue
            if scheme["name"] in seen_names:
                continue

            cleaned = clean_scheme(scheme)
            all_schemes.append(cleaned)
            seen_names.add(scheme["name"])

    # Sort by priority
    all_schemes.sort(key=lambda x: x.get("priority", 5), reverse=True)

    print(f"\nTotal unique schemes: {len(all_schemes)}")

    # Save to schemes.json
    output_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "schemes.json"
    )
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(all_schemes, f, indent=2, ensure_ascii=False)

    # Print summary
    print("\nSchemes breakdown:")
    categories = {
        "Farmer":     [s for s in all_schemes if s["conditions"]["occupation"] == "farmer"],
        "Student":    [s for s in all_schemes if s["conditions"]["occupation"] == "student"],
        "Women only": [s for s in all_schemes if s["conditions"]["gender"] == "female"],
        "BPL":        [s for s in all_schemes if s["conditions"]["is_bpl"] == True],
        "Disability": [s for s in all_schemes if s["conditions"]["has_disability"] == True],
        "Rural":      [s for s in all_schemes if s["conditions"]["location"] == "rural"],
        "Any citizen":[s for s in all_schemes if s["conditions"]["occupation"] == "any"],
    }

    for category, schemes in categories.items():
        print(f"  {category:15} : {len(schemes)} schemes")

    print("\n" + "=" * 60)
    print(f"Done! {len(all_schemes)} schemes saved to schemes.json")
    print("=" * 60)


if __name__ == "__main__":
    main()