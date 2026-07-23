"""
location_normalizer.py
Normalizes free-text location input (any city, any country) into the
"City, State/Country" format Indeed/Wire resolves reliably.

Primary path: ask the LLM (Groq) to normalize it — this is what lets
GapCheck support "blr", "nyc", "Vijayawada", "London UK", etc. without
a hardcoded city list.

Fallback path: if GROQ_API_KEY is missing or the call fails/times out,
fall back to a simple regex clean-up so the app degrades gracefully
instead of crashing.
"""

import os
import re
from functools import lru_cache

from dotenv import load_dotenv

load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")

NORMALIZE_SYSTEM_PROMPT = """You convert a user-typed location into a
standardized job-search location string.

Rules:
- Output ONLY the normalized location, nothing else. No explanation, no punctuation beyond what's needed.
- Format: "City, State/Region, Country" when you can confidently determine all three.
- If it's a well-known global city, country can be omitted if unambiguous (e.g. "London, UK" not "London, England, United Kingdom").
- Expand abbreviations (blr -> Bengaluru, nyc -> New York, hyd -> Hyderabad).
- Fix casing and spelling ("bangalore" -> "Bengaluru", "bombay" -> "Mumbai").
- If input is "remote" or similar, output exactly: Remote
- If input is empty, gibberish, or not a real place, output exactly: UNKNOWN
- Never invent a country if the city is genuinely ambiguous across multiple countries without other context — pick the most populous/well-known match.

Examples:
blr -> Bengaluru, Karnataka
nyc -> New York, USA
bangalore -> Bengaluru, Karnataka
london uk -> London, UK
vijayawada -> Vijayawada, Andhra Pradesh
berlin -> Berlin, Germany
toronto -> Toronto, Canada
wfh -> Remote
asdkjashd -> UNKNOWN"""


def _regex_clean(location: str) -> str:
    """Minimal fallback: trim/collapse whitespace."""
    if not location:
        return "Remote"
    cleaned = re.sub(r"\s+", " ", location.strip())
    return cleaned


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
        max_tokens=30,
    )
    result = response.choices[0].message.content.strip()
    return result


def normalize_location(location: str) -> str:
    """
    Normalize any free-text location into a Wire/Indeed-friendly string.
    Uses the LLM when available; falls back to basic cleanup on any
    failure (missing key, network error, rate limit, malformed output).
    """
    if not location or not location.strip():
        return "Remote"

    key = location.strip().lower()

    if not GROQ_API_KEY:
        return _regex_clean(location)

    try:
        result = _llm_normalize(key)
        if not result or result.upper() == "UNKNOWN":
            return _regex_clean(location)
        return result
    except Exception:
        # Any Groq/network failure -> degrade gracefully, don't crash the search
        return _regex_clean(location)


def simplified_fallback(location: str) -> str:
    """
    Looser variant to retry with if the normalized location returns
    zero results — e.g. drop state/country, keep just the city.
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