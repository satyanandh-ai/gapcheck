"""
resume_parser.py
Turns free-text skills input and/or an uploaded resume (PDF/DOCX/TXT)
into a structured candidate profile - skills, projects,
certifications, and an estimated experience level - instead of one
flat blob of text.

This feeds retrieval.py and analyzer.py a much richer query than a
plain skills list: "built a Flask job board with Postgres" is a
concrete signal that "Flask" alone isn't. It also gives the UI
something to show the user before analysis runs, so parsing stays
transparent instead of a silent black box - consistent with the rest
of the app's approach to explainability.
"""

import io
import json
import os

from dotenv import load_dotenv
from groq import Groq

load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GROQ_MODEL = os.getenv("GROQ_MODEL", "openai/gpt-oss-120b")

EMPTY_PROFILE = {
    "skills": [],
    "projects": [],
    "certifications": [],
    "years_experience": "not stated",
    "summary": "",
}


def extract_text_from_upload(uploaded_file) -> str:
    """
    Extracts raw text from an uploaded resume file (PDF, DOCX, or
    plain text). Returns "" on any failure rather than raising - a
    broken file shouldn't crash the app, since the free-text skills
    box is always available as a fallback input.
    """
    if uploaded_file is None:
        return ""
    name = (getattr(uploaded_file, "name", "") or "").lower()
    try:
        raw = uploaded_file.read()
        if name.endswith(".pdf"):
            from pypdf import PdfReader
            reader = PdfReader(io.BytesIO(raw))
            return "\n".join((page.extract_text() or "") for page in reader.pages)
        if name.endswith(".docx"):
            import docx
            doc = docx.Document(io.BytesIO(raw))
            return "\n".join(p.text for p in doc.paragraphs)
        # .txt or unrecognized extension - best-effort decode
        return raw.decode("utf-8", errors="ignore")
    except Exception as e:
        print(f"[resume_parser] failed to extract text from '{name}': {e}")
        return ""


PARSE_SYSTEM_PROMPT = """You extract a structured candidate profile from free-text
resume content and/or self-reported skills. The input may be a full resume,
a short skills list, or both concatenated together.

Extract ONLY what's actually present. Never invent, infer, or pad -
if something isn't stated or clearly implied, leave it out or say "not stated".
Do not guess at years of experience from job titles alone unless dates or
explicit experience are given - if genuinely unclear, use "not stated".

Respond ONLY with valid JSON, no markdown fences, no preamble, in this exact shape:
{
  "skills": ["<skill>", ...],
  "projects": [{"name": "<short project name>", "description": "<one clause on what it does/used>"}],
  "certifications": ["<certification name>", ...],
  "years_experience": "<best supported estimate, e.g. '0-1 years', '2-3 years', or 'not stated'>",
  "summary": "<one plain-language sentence describing this candidate's current level>"
}"""


def parse_profile(raw_text: str) -> dict:
    """
    Calls Groq to extract a structured profile from raw resume/skills
    text. Returns EMPTY_PROFILE (never raises) if there's no input, or
    if the LLM's response isn't valid JSON - degrading to "nothing
    structured was extracted" is safer than breaking the whole flow,
    since the caller can still fall back to the raw text.
    """
    if not raw_text or not raw_text.strip():
        return dict(EMPTY_PROFILE)
    if not GROQ_API_KEY:
        raise RuntimeError("GROQ_API_KEY not set - add it to your .env file")

    client = Groq(api_key=GROQ_API_KEY)

    # Same defenses as analyzer.py's Groq call - see that file's comment
    # for why: gpt-oss-120b is a reasoning model with known cases of
    # reasoning tokens leaking into / eating into the output budget,
    # which can truncate JSON mid-response.
    last_error = None
    for attempt in range(2):
        response = client.chat.completions.create(
            model=GROQ_MODEL,
            messages=[
                {"role": "system", "content": PARSE_SYSTEM_PROMPT},
                {"role": "user", "content": raw_text[:8000]},
            ],
            temperature=0.2,
            max_tokens=3000,
            reasoning_effort="low",
            response_format={"type": "json_object"},
        )
        raw = response.choices[0].message.content.strip()
        if raw.startswith("```"):
            raw = raw.strip("`")
            if raw.startswith("json"):
                raw = raw[4:]
            raw = raw.strip()
        try:
            parsed = json.loads(raw)
            break
        except json.JSONDecodeError as e:
            last_error = e
            print(f"[resume_parser] JSON parse failed on attempt {attempt + 1}/2: {e}")
    else:
        print(f"[resume_parser] LLM returned non-JSON twice in a row, degrading to empty profile: {last_error}")
        return dict(EMPTY_PROFILE)

    # Fill in any missing keys rather than trust the LLM followed the
    # schema exactly - downstream code shouldn't need defensive .get() everywhere.
    profile = dict(EMPTY_PROFILE)
    profile.update(parsed)
    return _clean_profile(profile)


def _clean_profile(profile: dict) -> dict:
    """
    Strips out blank/empty entries the LLM sometimes includes despite
    the "never invent, leave it out if absent" instruction - e.g. a
    projects list with 3 entries that all have empty name/description.
    Left uncleaned, these render as blank bullet points in the UI and
    add nothing (or noise) to the retrieval query text. This runs once,
    here, so every consumer (the UI preview and profile_to_query_text)
    sees already-clean data instead of each needing its own filtering.
    """
    skills = [s.strip() for s in (profile.get("skills") or []) if isinstance(s, str) and s.strip()]

    projects = []
    for p in profile.get("projects") or []:
        if isinstance(p, dict):
            name = (p.get("name") or "").strip()
            desc = (p.get("description") or "").strip()
            if name or desc:
                projects.append({"name": name, "description": desc})
        elif isinstance(p, str) and p.strip():
            projects.append({"name": p.strip(), "description": ""})

    certifications = [c.strip() for c in (profile.get("certifications") or []) if isinstance(c, str) and c.strip()]

    years_experience = (profile.get("years_experience") or "").strip() or "not stated"
    summary = (profile.get("summary") or "").strip()

    return {
        "skills": skills,
        "projects": projects,
        "certifications": certifications,
        "years_experience": years_experience,
        "summary": summary,
    }


def profile_to_query_text(profile: dict) -> str:
    """
    Flattens a structured profile back into a single text string for
    retrieval + analysis - more organized than raw free text since
    skills/projects/certifications/experience are explicitly labeled,
    which helps both TF-IDF and embedding-based retrieval match the
    right postings.
    """
    parts = []
    if profile.get("skills"):
        parts.append("Skills: " + ", ".join(profile["skills"]))
    if profile.get("projects"):
        proj_strs = []
        for p in profile["projects"]:
            if isinstance(p, dict):
                text = f"{p.get('name', '')} - {p.get('description', '')}".strip(" -")
                if text:
                    proj_strs.append(text)
            elif p:
                proj_strs.append(str(p))
        if proj_strs:
            parts.append("Projects: " + "; ".join(proj_strs))
    if profile.get("certifications"):
        parts.append("Certifications: " + ", ".join(profile["certifications"]))
    years = profile.get("years_experience")
    if years and years != "not stated":
        parts.append("Experience: " + years)
    return ". ".join(parts)