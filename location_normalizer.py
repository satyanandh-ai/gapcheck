"""
location_normalizer.py
Normalizes free-text location input (any city, any country) into the
"City, State/Country" format Indeed/Wire resolves reliably, and also
extracts a best-guess ISO-3166 alpha-2 country code so wire_client can
route the search to the correct country-specific Wire action.

Primary path: ask the LLM (Groq) to normalize it — this is what lets
GapCheck support "blr", "nyc", "Vijayawada", "London UK", etc. without
a hardcoded city list.

Fallback path: if GROQ_API_KEY is missing or the call fails/times out,
fall back to a simple regex clean-up + no country guess, so the app
degrades gracefully instead of crashing.
"""

import os
import re
import json
from functools import lru_cache

from dotenv import load_dotenv

load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")

NORMALIZE_SYSTEM_PROMPT = """You convert a user-typed location into a
standardized job-search location.

Respond ONLY with valid JSON, no markdown fences, no preamble, in this
exact shape:
{"location": "<City, State/Region>", "country_code": "<ISO 3166-1 alpha-2 lowercase>"}

Rules:
- "location" should be "City, State/Region" (omit region if the city is globally unambiguous, e.g. "London").
- "country_code" is a two-letter lowercase ISO code (e.g. "in", "us", "gb", "de", "ca").
- Expand abbreviations (blr -> Bengaluru, nyc -> New York, hyd -> Hyderabad).
- Fix casing and spelling ("bangalore" -> "Bengaluru", "bombay" -> "Mumbai").
- If input is "remote" or similar, respond: {"location": "Remote", "country_code": ""}
- If input is empty, gibberish, or not a real place, respond: {"location": "UNKNOWN", "country_code": ""}
- Never invent a country if genuinely ambiguous — pick the most populous/well-known match.

Examples:
"blr" -> {"location": "Bengaluru, Karnataka", "country_code": "in"}
"nyc" -> {"location": "New York, New York", "country_code": "us"}
"london uk" -> {"location": "London", "country_code": "gb"}
"vijayawada" -> {"location": "Vijayawada, Andhra Pradesh", "country_code": "in"}
"berlin" -> {"location": "Berlin", "country_code": "de"}
"toronto" -> {"location": "Toronto, Ontario", "country_code": "ca"}
"wfh" -> {"location": "Remote", "country_code": ""}"""


def _regex_clean(location: str) -> dict:
    """Minimal fallback when the LLM path is unavailable."""
    if not location:
        return {"location": "Remote", "country_code": ""}
    cleaned = re.sub(r"\s+", " ", location.strip())
    return {"location": cleaned, "country_code": ""}


@lru_cache(maxsize=256)
def _llm_normalize(location_key: str) -> str:
    """Cached LLM call — same input string only hits Groq once per process."""
    from groq import Groq

    client = Groq(api_key=GROQ_API_KEY)
    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[
            {"role": "system", "content": NORMALIZE_SYSTEM_PROMPT},
            {"role": "user", "content": location_key},
        ],
        temperature=0,
        max_tokens=60,
    )
    return response.choices[0].message.content.strip()


def normalize_location_full(location: str) -> dict:
    """
    Returns {"location": "<City, Region>", "country_code": "<iso2 or ''>"}.
    Uses the LLM when available; falls back to basic cleanup (no country
    guess) on any failure so the app degrades gracefully instead of crashing.
    """
    if not location or not location.strip():
        return {"location": "Remote", "country_code": ""}

    key = location.strip().lower()

    if not GROQ_API_KEY:
        return _regex_clean(location)

    try:
        raw = _llm_normalize(key)
        if raw.startswith("```"):
            raw = raw.strip("`")
            if raw.startswith("json"):
                raw = raw[4:]
            raw = raw.strip()
        parsed = json.loads(raw)
        loc = parsed.get("location", "").strip()
        country = parsed.get("country_code", "").strip().lower()
        if not loc or loc.upper() == "UNKNOWN":
            return _regex_clean(location)
        return {"location": loc, "country_code": country}
    except Exception:
        # Any Groq/network/parsing failure -> degrade gracefully
        return _regex_clean(location)


def normalize_location(location: str) -> str:
    """Backwards-compatible helper: just the normalized location string."""
    return normalize_location_full(location)["location"]


def simplified_fallback(location: str) -> str:
    """
    Looser variant to retry with if the normalized location returns
    zero results — e.g. drop state/region, keep just the city.
    """
    if "," in location:
        return location.split(",")[0].strip()
    return location.strip()


def suggestions() -> list:
    """A short list of known-good locations to show the user on failure."""
    return [
        "Bengaluru, Karnataka",
        "Hyderabad, Telangana",
        "New York, USA",
        "London, UK",
        "Remote",
    ]