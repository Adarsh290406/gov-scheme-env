---
title: Gov Scheme Finder
emoji: 🏛️
colorFrom: blue
colorTo: green
sdk: docker
pinned: false
---
# 🏛️ Gov Scheme Finder — OpenEnv Environment

A real-world Reinforcement Learning environment where AI agents learn to match Indian citizens to the correct government schemes by asking smart questions.

Built for the **OpenEnv Hackathon** — implements the full OpenEnv `step()` / `reset()` / `state()` API.

---

## 🌍 Motivation

India has **500+ government welfare schemes** but most citizens don't know which ones they qualify for. A rural farmer may miss out on crop insurance. A disabled student may never hear about disability scholarships. A BPL family may not know they qualify for free health insurance.

This environment trains AI agents to solve that problem — by learning to ask the right questions and recommend the most relevant scheme for any citizen profile.

---

## 🎮 How It Works

```
reset()  →  A new random citizen arrives (profile hidden from agent)
step()   →  Agent asks a question OR recommends a scheme
state()  →  Returns current known info about the citizen
```

The agent starts blind. It must ask smart questions to reveal information, then recommend the correct government scheme. Every wrong move costs reward.

---

## 🗂️ Action Space (14 actions)

|       Action         |                  Description                     |
|----------------------|--------------------------------------------------|
| `ask_occupation`     | Always ask this FIRST                            |
| `ask_income`         | Context-aware (parental/crop/household/personal) |
| `ask_bpl`            | Below Poverty Line status                        |
| `ask_location`       | Rural or urban                                   |
| `ask_gender`         | Gender                                           |
| `ask_caste`          | General/OBC/SC/ST                                |
| `ask_disability`     | Has a disability?                                |
| `ask_age`            | Age                                              |
| `ask_education`      | Education level                                  |
| `ask_bank_account`   | Has Jan Dhan / bank account?                     |
| `ask_ration_card`    | Has ration card?                                 |
| `ask_marital_status` | Married/single/widowed?                          |
| `ask_land_ownership` | Owns/rents/no land?                              |
| `ask_state`          | Which state?                                     |
| `recommend_scheme`   | Recommend a scheme (ends episode)                |
|----------------------|--------------------------------------------------|
---

## 🏆 Reward System 

|            Action            |          Reward         |
|------------------------------|-------------------------|
| Ask occupation first         | +0.5 bonus              |
| Ask relevant question        | +0.1 to +0.3            |
| Ask irrelevant question      | -0.2                    |   
| Ask income before occupation | -0.4                    |
| Repeat a question            | -0.3                    |
| Recommend too early          | -0.6                    |
| Correct scheme               | +1.0 + efficiency bonus |
| Partial match                | +0.15 to +0.3           |
| Wrong scheme                 | -0.5                    |
| Step limit reached           | -0.2                    |
|------------------------------|-------------------------|
**Reward decay per step** — forces agent to be decisive.

---

## 🎮 RL Features

- **Noise** — Citizens sometimes give wrong answers (10-20% on medium/hard)
- **Scheme Expiry** — Schemes can expire mid-episode
- **Incomplete Info** — Citizens sometimes say "I don't know"
- **Partial Eligibility** — Partial match rewards
- **Context-aware penalties** — Irrelevant questions penalized
- **Weighted citizens** — Reflects real India demographics

---

## 📋 Tasks

| Task   | Citizen                                      | Schemes | Steps | Pass |
|--------|----------------------------------------------|---------|-------|------|
| Easy   | Rural OBC BPL woman, daily wage, UP          | 10      | 10    | 0.5  | 
| Medium | Rural SC BPL farmer, land owner, Maharashtra | 25      | 8     | 0.4  |
| Hard   | Rural SC disabled female student, BPL, Bihar | 67      | 6     | 0.3  |
|--------|----------------------------------------------|---------|-------|------|


## 📁 Project Structure

```
gov-scheme-env/
├── models.py           # Pydantic models
├── environment.py      # Core logic — reset(), step(), state()
├── app.py              # FastAPI server
├── inference.py        # Hackathon inference script (API_BASE_URL, HF_TOKEN, MODEL_NAME)
├── baseline.py         # Local baseline script (Groq)
├── generate_schemes.py # Auto-generates schemes.json
├── schemes.json        # 67 real Indian government schemes
├── Dockerfile          # HF Spaces deployment
├── openenv.yaml        # Environment metadata
├── requirements.txt    # Dependencies
└── tasks/
    ├── easy.py
    ├── medium.py
    └── hard.py
```

---

## 🚀 Setup

```bash
git clone https://github.com/Adarsh290406/gov-scheme-env.git
cd gov-scheme-env
pip install -r requirements.txt
```

Run server:
```bash
python app.py
```

**Run baseline (local, uses Groq):**

Create `.env`:
```
OPENAI_API_KEY=your_groq_key_here
```
```bash
python baseline.py
```

**Run inference script (hackathon, uses HF router):**

Set environment variables:
```
API_BASE_URL=https://router.huggingface.co/v1
MODEL_NAME=meta-llama/Llama-3.1-8B-Instruct
HF_TOKEN=your_hf_token_here
```
```bash
python inference.py
```

Docker:
```bash
docker build -t gov-scheme-finder .
docker run -p 7860:7860 --env-file .env gov-scheme-finder
```

---

## 📊 Baseline Scores

Model: `llama-3.1-8b-instant` via Groq (OpenAI-compatible)

| Task        | Score    | Passed  |
|-------------|----------|---------|
| Easy        | 1.0      | ✅      |
| Medium      | 0.9      | ✅      |
| Hard        | 0.95     | ✅      |
| **Average** | **0.95** | ✅      |
|-------------|----------|---------|

The baseline agent uses a reasoning-driven approach — it asks questions with the highest information gain per step (occupation → disability/gender/land ownership → confirm BPL), narrows the scheme space after each answer, and recommends confidently once enough attributes are confirmed. This makes it an effective RL training baseline: the agent genuinely reasons rather than guessing, so RL can learn from the reward signal in a meaningful way.

---

## 🛠️ Built With

Python 3.10+ | FastAPI | Pydantic | Uvicorn | OpenEnv | Groq

## 📜 License

MIT License