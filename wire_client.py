"""
wire_client.py
Thin wrapper around the Anakin Wire API.

CONFIRMED LIVE on 2026-06-20 against the real Indeed catalog:
  - in_search_jobs runs ASYNC: POST /task returns {"job_id": ...},
    then GET /jobs/{job_id} until status == "completed".
  - Final job list lives at: response["data"]["data"]["jobs"]

NOTE ON GLOBAL SUPPORT:
The confirmed action is named "in_search_jobs" — the "in_" prefix
strongly suggests this is an India-scoped Indeed action, not a generic
global one. Rather than assume other countries work, this client
queries Wire's own catalog at runtime to discover which country-scoped
*_search_jobs (and matching *_job_details) actions actually exist, and
routes each search to the matching action for the resolved country.
If no matching action exists for a location's country, it fails fast
with a clear message instead of hanging until timeout.
"""

import os
import sys
import time
import json
import re
import requests
from functools import lru_cache
from dotenv import load_dotenv

from location_normalizer import normalize_location_full, simplified_fallback

load_dotenv()

WIRE_API_KEY = os.getenv("WIRE_API_KEY")
BASE_URL = "https://api.anakin.io/v1/wire"

# Known confirmed-working action as of 2026-06-20. Used as the fallback
# when catalog discovery fails or returns nothing usable.
DEFAULT_SEARCH_ACTION = "in_search_jobs"
DEFAULT_COUNTRY = "in"


class WireError(Exception):
    pass


def _headers():
    if not WIRE_API_KEY:
        raise WireError("WIRE_API_KEY not set — add it to your .env file")
    return {"X-API-Key": WIRE_API_KEY}


def _extract_failure_reason(polled: dict) -> str:
    """
    Wire doesn't consistently put the failure reason in the same spot.
    Check the likely locations in order, and fall back to dumping the
    raw payload rather than silently returning None.
    """
    data = polled.get("data") if isinstance(polled.get("data"), dict) else {}
    inner = data.get("data") if isinstance(data.get("data"), dict) else {}
    candidates = [
        data.get("error"),
        data.get("message"),
        inner.get("error"),
        polled.get("error"),
        polled.get("message"),
        polled.get("reason"),
    ]
    for c in candidates:
        if c:
            return str(c)
    return f"no reason provided by Wire (raw response: {json.dumps(polled)[:500]})"


def get_catalog(service: str):
    resp = requests.get(f"{BASE_URL}/catalog/{service}", headers=_headers(), timeout=20)
    resp.raise_for_status()
    return resp.json()


@lru_cache(maxsize=1)
def get_available_search_actions() -> dict:
    """
    Discover which country-scoped job-search actions Wire actually
    exposes, by querying the catalog. Returns {country_code: action_id},
    e.g. {"in": "in_search_jobs", "us": "us_search_jobs"}.

    Best-effort: catalog response shape isn't fully documented, so this
    tries a few plausible structures. Falls back to just the known
    working default if discovery fails or finds nothing.
    """
    try:
        catalog = get_catalog("indeed")
    except Exception as e:
        print(f"[wire] catalog discovery failed, using default action only: {e}")
        return {DEFAULT_COUNTRY: DEFAULT_SEARCH_ACTION}

    # Try a few likely shapes for where action IDs live in the catalog payload
    action_ids = []
    if isinstance(catalog, dict):
        for key in ("actions", "data"):
            val = catalog.get(key)
            if isinstance(val, list):
                for item in val:
                    if isinstance(item, dict):
                        # Wire's catalog uses "action_id" for the callable
                        # name (e.g. "in_search_jobs"); "id" is an internal UUID.
                        aid = item.get("action_id") or item.get("id")
                        if aid:
                            action_ids.append(aid)
                    elif isinstance(item, str):
                        action_ids.append(item)
            elif isinstance(val, dict):
                action_ids.extend(val.keys())

    found = {}
    for action_id in action_ids:
        m = re.match(r"^([a-z]{2})_search_jobs$", action_id)
        if m:
            found[m.group(1)] = action_id

    if not found:
        print(f"[wire] catalog discovery found no *_search_jobs actions, using default. Raw catalog: {json.dumps(catalog)[:300]}")
        return {DEFAULT_COUNTRY: DEFAULT_SEARCH_ACTION}

    print(f"[wire] discovered search actions for countries: {sorted(found.keys())}")
    return found


def run_action(action_id: str, params: dict, max_wait_seconds: int = 60, poll_interval: float = 2.0, debug: bool = True):
    """
    Executes a Wire action and waits for the result, handling both:
      - sync actions that return data immediately
      - async actions that return {"job_id": ...} and need polling
    Returns the full parsed JSON response once status == "completed".
    """
    resp = requests.post(
        f"{BASE_URL}/task",
        headers=_headers(),
        json={"action_id": action_id, "params": params},
        timeout=30,
    )
    resp.raise_for_status()
    result = resp.json()

    if result.get("status") == "processing" and "job_id" in result:
        job_id = result["job_id"]
        waited = 0.0
        last_status = None
        while waited < max_wait_seconds:
            time.sleep(poll_interval)
            waited += poll_interval
            poll_resp = requests.get(f"{BASE_URL}/jobs/{job_id}", headers=_headers(), timeout=20)
            poll_resp.raise_for_status()
            polled = poll_resp.json()
            status = polled.get("status")
            if debug and status != last_status:
                print(f"[wire] job {job_id} action={action_id} status={status} at {waited:.1f}s params={params}")
                last_status = status
            if status == "completed":
                return polled
            if status == "failed":
                raise WireError(f"Wire job {job_id} failed: {_extract_failure_reason(polled)}")
        raise WireError(
            f"Wire job {job_id} (action={action_id}) timed out after {max_wait_seconds}s "
            f"(last status: {last_status}, params: {params})"
        )

    return result


# --- Indeed actions ------------------------------------------

def search_jobs(query: str, location: str = "Bengaluru, Karnataka", country_domain: str = None):
    """
    Search live Indeed job postings via Wire.

    Normalizes the location (any city, any country) via LLM, resolves
    the country code, and routes to the matching country-scoped Wire
    action (e.g. "us_search_jobs" for a US location) if one exists.

    If country_domain is explicitly passed, it overrides auto-detection.

    Raises WireError immediately (no hang/timeout) if the resolved
    country has no matching Wire action available.

    Returns (jobs_list, used_live: bool, resolved_location: str).
    """
    resolved = normalize_location_full(location)
    location_str = resolved["location"]
    detected_country = country_domain or resolved["country_code"] or DEFAULT_COUNTRY

    if location_str == "Remote":
        detected_country = country_domain or DEFAULT_COUNTRY

    available = get_available_search_actions()
    action_id = available.get(detected_country)

    if not action_id:
        supported = ", ".join(sorted(available.keys())) or "none"
        raise WireError(
            f"No Wire job-search action available for country '{detected_country}' "
            f"(resolved from '{location}' -> '{location_str}'). "
            f"Currently supported countries: {supported}. "
            f"Try a location in one of those countries, or 'Remote'."
        )

    raw = run_action(action_id, {
        "query": query,
        "location": location_str,
        "country_domain": detected_country,
    })
    jobs = raw.get("data", {}).get("data", {}).get("jobs", [])

    if not jobs:
        fallback_location = simplified_fallback(location_str)
        if fallback_location != location_str:
            raw = run_action(action_id, {
                "query": query,
                "location": fallback_location,
                "country_domain": detected_country,
            })
            jobs = raw.get("data", {}).get("data", {}).get("jobs", [])
            if jobs:
                location_str = fallback_location

    return jobs, True, location_str


def get_job_details(job_key: str = None, job_url: str = None, country_domain: str = "in"):
    """Get full details for one Indeed job (in_job_details, async)."""
    params = {"country_domain": country_domain}
    if job_key:
        params["job_key"] = job_key
    if job_url:
        params["job_url"] = job_url
    raw = run_action("in_job_details", params)
    return raw.get("data", {}).get("data", {}), True


def get_salary_estimate(title: str, location: str = "Bengaluru", country: str = "IN"):
    """Get salary estimate (in_salary_search, async)."""
    raw = run_action("in_salary_search", {
        "title": title,
        "location": location,
        "country": country,
    })
    return raw.get("data", {}).get("data", {}), True


if __name__ == "__main__":
    if len(sys.argv) >= 3 and sys.argv[1] == "catalog":
        print(json.dumps(get_catalog(sys.argv[2]), indent=2))
    elif len(sys.argv) >= 2 and sys.argv[1] == "actions":
        print(json.dumps(get_available_search_actions(), indent=2))
    elif len(sys.argv) >= 2 and sys.argv[1] == "test":
        loc = sys.argv[2] if len(sys.argv) > 2 else "Bangalore"
        jobs, ok, resolved = search_jobs("backend developer", loc)
        print(f"Resolved location: {resolved}")
        print(f"Got {len(jobs)} jobs. First one:")
        print(json.dumps(jobs[0], indent=2) if jobs else "none")
    else:
        print("Usage:")
        print("  python wire_client.py catalog <service_name>")
        print("  python wire_client.py actions")
        print("  python wire_client.py test [location]")