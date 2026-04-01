# Gov Scheme Finder

An **OpenEnv** reinforcement learning environment where AI agents act as government scheme advisors for Indian citizens. Agents must navigate 67 real-world schemes by asking the right questions under constraints (step limits, incomplete info, and dynamic matching).

## Features
- **67 Real Schemes:** Extracted dynamically using LLMs.
- **14 Actions:** The agent determines state/occupation/income/caste/education dynamically.
- **Context-Aware Penalties:** Agents are penalized for asking a student if they own land, or asking income before occupation context is established.
- **OpenEnv API Support:** Full support for `GET /info`, `POST /reset`, `POST /step`, `POST /tasks/{name}`, `POST /grade` etc.

## Setup & Running

1. **Install Dependencies**
   ```bash
   pip install -r requirements.txt
   ```

2. **Set up Environment**
   Create a `.env` file with your Groq or OpenAI key:
   ```
   OPENAI_API_KEY=your_groq_api_key_here
   ```

3. **Run the API Server**
   ```bash
   uvicorn app:app --host 0.0.0.0 --port 8000
   ```

4. **Docker Support**
   ```bash
   docker build -t gov-scheme-finder .
   docker run -p 8000:8000 --env-file .env gov-scheme-finder
   ```

## Action Space
- `ask_occupation`, `ask_income`, `ask_bpl`, `ask_location`, `ask_gender`, `ask_caste`, `ask_disability`, `ask_age`, `ask_education`, `ask_bank_account`, `ask_ration_card`, `ask_marital_status`, `ask_land_ownership`, `ask_state`
- `recommend_scheme` (ends episode)

## Observation Space
- Citizen profile clues extracted so far.
- List of `available_schemes` (which might expire dynamically mid-episode).
- Step counts and textual hints (`last_action_result`).

## Baseline LLM Agent Scores
Running `python baseline.py` evaluates `llama-3.1-8b-instant` directly against the Easy, Medium, and Hard tasks.

- **Task 1 (Easy):** `0.3`
- **Task 2 (Medium):** `0.2`
- **Task 3 (Hard):** `0.2`
- **Average:** `0.233`

*(Note: The llama-3.1-8b-instant model via Groq achieves around ~0.23 due to the tight step limits and the sheer number of overlapping schemes now natively supported in the dataset — a great benchmark for future RL agents to beat!)*
