"""
analyzer.py
Compares a user's stated skills against REAL live job postings
(fetched via Wire/Indeed) and produces a structured gap analysis
using Groq (LLaMA 3.3).
"""

import os
import json
import re
from groq import Groq
from dotenv import load_dotenv

load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")


def _strip_html(text):
    """Remove HTML tags and collapse whitespace from a snippet string."""
    if not text:
        return ""
    clean = re.sub(r"<[^>]+>", " ", text)
    clean = re.sub(r"\s+", " ", clean).strip()
    return clean



SYSTEM_PROMPT = """You are a blunt, practical technical career coach.

You will be given:
1. A user's self-reported current skills and experience level
2. A target job role
3. A list of REAL, currently-live job postings scraped from Indeed for that role

Your job: compare what these real postings actually ask for (read the
snippets carefully) against what the user says they know, and produce a
gap analysis grounded in the actual postings, not generic advice.

Respond ONLY with valid JSON, no markdown fences, no preamble, in this exact shape:
{
  "match_score": <integer 0-100>,
  "summary": "<one or two blunt sentences on overall readiness, referencing real postings>",
  "strengths": ["<skill the user has that these postings actually want>", ...],
  "gaps": [
    {"skill": "<missing or weak skill, drawn from real postings>", "why_it_matters": "<short reason, cite a company/pattern from the postings>", "how_to_fix": "<one concrete, doable action>"}
  ],
  "top_3_actions": ["<action 1>", "<action 2>", "<action 3>"],
  "matching_jobs": [
    {"title": "<job title>", "company": "<company>", "url": "<job url>"}
  ]
}

Keep gaps to the 3-5 that matter most. For matching_jobs, pick up to 3 postings
that best fit the user's current level. Be specific — reference real company
names and real requirements from the postings provided, not generic advice."""


def analyze_gap(user_skills: str, target_role: str, job_postings: list) -> dict:
    """Calls Groq LLM to compare user skills vs real job postings."""
    if not GROQ_API_KEY:
        raise RuntimeError("GROQ_API_KEY not set - add it to your .env file")

    client = Groq(api_key=GROQ_API_KEY)

    # trim snippets to keep payload reasonable
    trimmed_postings = []
    for job in job_postings[:15]:
        trimmed_postings.append({
            "title": job.get("title"),
            "company": job.get("company"),
            "location": job.get("location"),
            "snippet": _strip_html(job.get("snippet"))[:500],
            "url": job.get("url"),
        })

    user_content = json.dumps({
        "user_skills": user_skills,
        "target_role": target_role,
        "job_postings": trimmed_postings,
    })

    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_content},
        ],
        temperature=0.4,
    )

    raw = response.choices[0].message.content.strip()
    if raw.startswith("```"):
        raw = raw.strip("`")
        if raw.startswith("json"):
            raw = raw[4:]
        raw = raw.strip()

    return json.loads(raw)
