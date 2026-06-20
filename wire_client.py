"""
wire_client.py
Thin wrapper around the Anakin Wire API.

CONFIRMED LIVE on 2026-06-20 against the real Indeed catalog:
  - in_search_jobs runs ASYNC: POST /task returns {"job_id": ...},
    then GET /jobs/{job_id} until status == "completed".
  - Final job list lives at: response["data"]["data"]["jobs"]
"""

import os
import sys
import time
import json
import requests
from dotenv import load_dotenv

load_dotenv()

WIRE_API_KEY = os.getenv("WIRE_API_KEY")
BASE_URL = "https://api.anakin.io/v1/wire"


class WireError(Exception):
    pass


def _headers():
    if not WIRE_API_KEY:
        raise WireError("WIRE_API_KEY not set — add it to your .env file")
    return {"X-API-Key": WIRE_API_KEY}


def get_catalog(service: str):
    resp = requests.get(f"{BASE_URL}/catalog/{service}", headers=_headers(), timeout=20)
    resp.raise_for_status()
    return resp.json()


def run_action(action_id: str, params: dict, max_wait_seconds: int = 30, poll_interval: float = 1.5):
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

    # Async pattern: {"status": "processing", "job_id": ..., "poll_url": ...}
    if result.get("status") == "processing" and "job_id" in result:
        job_id = result["job_id"]
        waited = 0.0
        while waited < max_wait_seconds:
            time.sleep(poll_interval)
            waited += poll_interval
            poll_resp = requests.get(f"{BASE_URL}/jobs/{job_id}", headers=_headers(), timeout=20)
            poll_resp.raise_for_status()
            polled = poll_resp.json()
            if polled.get("status") == "completed":
                return polled
            if polled.get("status") == "failed":
                raise WireError(f"Wire job {job_id} failed: {polled.get('data', {}).get('error')}")
        raise WireError(f"Wire job {job_id} timed out after {max_wait_seconds}s")

    return result


# --- CONFIRMED Indeed actions ------------------------------------------

def search_jobs(query: str, location: str = "Bengaluru, Karnataka", country_domain: str = "in"):
    """
    Search live Indeed job postings via Wire (in_search_jobs, async).
    Returns (jobs_list, used_live: bool).
    """
    raw = run_action("in_search_jobs", {
        "query": query,
        "location": location,
        "country_domain": country_domain,
    })
    jobs = raw.get("data", {}).get("data", {}).get("jobs", [])
    return jobs, True


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
    elif len(sys.argv) >= 2 and sys.argv[1] == "test":
        jobs, ok = search_jobs("backend developer")
        print(f"Got {len(jobs)} jobs. First one:")
        print(json.dumps(jobs[0], indent=2) if jobs else "none")
    else:
        print("Usage:")
        print("  python wire_client.py catalog <service_name>")
        print("  python wire_client.py test")
