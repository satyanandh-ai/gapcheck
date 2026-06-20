# GapCheck

An honest gap check between your skills and the jobs that actually exist right now.

Built solo for Anakin Blitz 2026 in a single 6-hour sprint.

## The problem

Career advice for students and early-career developers is almost always generic.
Job postings contain real, current signal about what companies actually want, but nobody connects that signal back to the individual. GapCheck closes that loop.

## What it does

1. You type your current skills and a target role.
2. Wire searches live Indeed job postings for that exact role, right now.
3. An LLM (Groq / LLaMA 3.3) compares the real postings against what you know.
4. You get a match score, your strengths, the gaps that matter, a 3-step action plan, and real postings worth applying to.

## Why Wire is core, not a bolt-on

Every job posting shown and analyzed comes from a live in_search_jobs call through the Anakin Wire API against Indeed. There is no scraping code and no static dataset. If Wire returns nothing, the app has nothing to analyze.

## Tech stack

- Anakin Wire API - live job search (in_search_jobs, Indeed)
- Groq (LLaMA 3.3 70B) - gap analysis reasoning
- Streamlit + custom CSS - frontend
- Python - backend, Wire client, polling

## Running locally

```
git clone https://github.com/satyanandh-ai/gapcheck.git
cd gapcheck
pip install -r requirements.txt
```

Create a .env file with WIRE_API_KEY and GROQ_API_KEY, then run:

```
streamlit run app.py
```

## Author

Ch. Satyanand, B.Tech AIML, ALIET Vijayawada
Built live, solo, in 6 hours for Anakin Blitz 2026.
