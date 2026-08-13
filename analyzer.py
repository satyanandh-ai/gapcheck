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
from retrieval import retrieve_relevant_jobs
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
3. A list of REAL, currently-live job postings scraped from Indeed for
   that role, pre-filtered and ranked by a retrieval step against this
   user's specific skills/role (see each posting's relevance_to_candidate
   score, 0-1 — higher means it's a closer match to this candidate). Each
   posting has a "job_index" - use this to cite postings, never restate
   or invent a company name/URL yourself. Each posting's "description"
   field is the FULL job description when "description_is_full" is
   true, or just a short search-result preview when it's false - treat
   full descriptions as more authoritative on specifics like required
   years of experience, since a short preview may not mention them at all.
Your job: compare what these real postings actually ask for (read the
descriptions carefully) against what the user says they know, and produce a
gap analysis grounded in the actual postings, not generic advice. Weigh
higher-relevance postings more heavily when deciding what's a "real" gap
versus a one-off requirement from a single unusual posting.

SCORING - this is a points-allocation exercise, not a number you invent:
Produce a score_breakdown: the 4-6 most important skill/competency areas
these postings actually ask for. For each area, assign "max_points" (how
much this area matters to these postings, as a share of 100 - the
max_points across ALL rows in score_breakdown MUST sum to exactly 100),
and "achieved_points" (0 to max_points, how well the user's stated
skills/experience support this specific area). The final match_score is
NOT something you state directly - it is calculated afterward as the sum
of achieved_points, so your job is only to allocate max_points sensibly
across the areas that matter and score achieved_points honestly against
each one.

Keep areas distinct and non-overlapping - e.g. "MLOps" (experiment
tracking, model registry, monitoring) and "Containerization" (Docker,
Kubernetes) are different things even if related; don't create two rows
that would both be satisfied or both be missing by the same evidence.

BE STRICT ABOUT SOFT/INTERPERSONAL SKILLS (communication, teamwork,
leadership, stakeholder management, collaboration): only award
achieved_points > 0 if the user's stated profile EXPLICITLY mentions
something like leading a team, presenting to stakeholders, mentoring,
open-source contribution, or similar. Having built individual technical
projects is NOT evidence of collaboration or communication ability -
don't infer soft skills from the mere existence of projects. If there's
no explicit evidence, achieved_points must be 0 and the note should say
so plainly (e.g. "No direct evidence of team collaboration in your
profile - projects alone don't demonstrate this").

CITATIONS - this is critical for trust: every score_breakdown row and
every gap MUST include a "cited_job_indices" field - a list of the
job_index values (from the postings you were given) that actually
contain or imply that requirement. This field is required on every
row, with no exceptions: if you genuinely cannot point to a specific
posting for something, use an empty list "cited_job_indices": [] -
never omit the field, and never pad it with postings that don't
actually support the claim.

Respond ONLY with valid JSON, no markdown fences, no preamble, in this exact shape:
{
  "summary": "<one or two blunt sentences on overall readiness, referencing real postings>",
  "score_breakdown": [
    {"area": "<skill/competency area drawn from real postings>", "max_points": <int, all rows sum to 100>, "achieved_points": <int, 0 to max_points>, "note": "<one short clause explaining the rating>", "cited_job_indices": [<job_index>, ...] (use [] if none apply)}
  ],
  "strengths": ["<skill the user has that these postings actually want>", ...],
  "gaps": [
    {"skill": "<missing or weak skill, drawn from real postings>", "why_it_matters": "<short reason>", "how_to_fix": "<one concrete, doable action>", "time_estimate": "<rough time to close this gap, e.g. '1-2 weeks'>", "cited_job_indices": [<job_index>, ...] (use [] if none apply)}
  ],
  "top_3_actions": [
    {"action": "<action>", "time_estimate": "<e.g. '2 weeks'>"}
  ],
  "matching_job_indices": [<job_index>, <job_index>, <job_index>]
}
Keep gaps to the 3-5 that matter most, and don't create multiple overlapping
gaps for essentially the same underlying skill (e.g. "MLOps", "model
deployment", and "containerization" as three separate gaps when they're
really one gap with several parts - consolidate). Keep score_breakdown to
4-6 areas - these should be the areas that most influenced the score, not
an exhaustive list, and their max_points must sum to exactly 100. For
matching_job_indices, pick up to 3 job_index values for postings that best
fit the user's current level. Be specific — reference real requirements
from the postings provided, not generic advice. Time estimates should be
realistic for someone learning part-time alongside other commitments, not
idealized."""
def analyze_gap(user_skills: str, target_role: str, job_postings: list) -> dict:
    """Calls Groq LLM to compare user skills vs real job postings."""
    if not GROQ_API_KEY:
        raise RuntimeError("GROQ_API_KEY not set - add it to your .env file")
    client = Groq(api_key=GROQ_API_KEY)

    # RAG retrieval step: rank all postings Wire returned by relevance
    # to this user's skills + target role, and only pass the strongest
    # matches into the LLM's context instead of the first N in
    # whatever order Wire's API returned them. If the caller already
    # ran retrieval (job_postings already carry "_relevance"), don't
    # redo it - just use what was passed in.
    if job_postings and "_relevance" in job_postings[0]:
        relevant_postings = job_postings[:15]
    else:
        relevant_postings = retrieve_relevant_jobs(user_skills, target_role, job_postings, top_k=15)

    trimmed_postings = []
    index_to_job = {}
    for i, job in enumerate(relevant_postings):
        index_to_job[i] = job
        # Prefer the full description (fetched by retrieval.py's
        # enrichment stage) over the short search-result snippet when
        # available - it's what the LLM should actually read the
        # requirements from. Larger char budget since it's much richer
        # text than a 2-line preview.
        has_full = bool(job.get("full_description"))
        source_text = job.get("full_description") or job.get("snippet")
        char_budget = 3000 if has_full else 500
        trimmed_postings.append({
            "job_index": i,
            "title": job.get("title"),
            "company": job.get("company"),
            "location": job.get("location"),
            "description": _strip_html(source_text)[:char_budget],
            "description_is_full": has_full,
            "url": job.get("url"),
            "relevance_to_candidate": job.get("_relevance"),
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
    parsed = json.loads(raw)
    return _resolve_citations(parsed, index_to_job)


def _evidence_tier(coverage_pct: float, cited_count: int) -> str:
    """
    Buckets citation coverage into a human-readable tier - i.e. how
    much of the retrieved posting set backs a claim, NOT the
    candidate's skill level (that's `status`, derived separately from
    points). Deliberately uses different words than status's
    strong/partial/missing (well_supported/some_support/limited_
    support/unsupported) - both scales used "strong"/"moderate"
    independently, which read as two disagreeing judgments about the
    same thing when shown together, even though they measure
    different things (citation coverage vs. candidate skill level).
    """
    if cited_count == 0:
        return "unsupported"
    if coverage_pct >= 50:
        return "well_supported"
    if coverage_pct >= 20:
        return "some_support"
    return "limited_support"


def _status_from_points(achieved: int, max_points: int) -> str:
    """
    Derives the strong/partial/missing badge from the achieved/max
    points ratio, rather than asking the LLM to separately state both
    a score AND a status label for the same row - two independently
    asserted numbers can disagree with each other (e.g. "7/10 points"
    next to a "missing" badge), which is exactly the kind of
    inconsistency this whole citation/scoring system exists to avoid.
    Deriving status from the same numbers that produce the score
    guarantees they can never contradict each other.
    """
    if max_points <= 0:
        return "missing"
    ratio = achieved / max_points
    if ratio >= 0.7:
        return "strong"
    if ratio >= 0.35:
        return "partial"
    return "missing"


def _normalize_score_breakdown(score_breakdown: list) -> list:
    """
    Validates and cleans the LLM's max_points/achieved_points per row:
    coerces to int, clips achieved_points to [0, max_points] (an LLM
    claiming more achieved than max is a contradiction we shouldn't
    display), and drops rows with a non-positive max_points (can't
    compute a meaningful ratio or weight from those). Never raises -
    malformed rows are dropped rather than crashing the whole analysis.
    """
    cleaned = []
    for row in score_breakdown or []:
        try:
            max_points = int(row.get("max_points", 0))
            achieved_points = int(row.get("achieved_points", 0))
        except (TypeError, ValueError):
            continue
        if max_points <= 0:
            continue
        achieved_points = max(0, min(achieved_points, max_points))
        row["max_points"] = max_points
        row["achieved_points"] = achieved_points
        cleaned.append(row)
    return cleaned


def _compute_match_score(score_breakdown: list) -> int:
    """
    The match_score is calculated here, in code, as a percentage of
    points actually achieved - it is never taken from a number the LLM
    states directly. The prompt asks the LLM to make max_points across
    rows sum to 100, but this doesn't hard-require that: it computes
    achieved/max as a genuine percentage regardless of what the row
    totals actually sum to, so a model that doesn't follow the "sum to
    100" instruction exactly still produces a mathematically honest
    score instead of a silently wrong one.
    """
    total_max = sum(row["max_points"] for row in score_breakdown)
    total_achieved = sum(row["achieved_points"] for row in score_breakdown)
    if total_max <= 0:
        return 0
    return round(100 * total_achieved / total_max)


def _citation_summary(indices: list, index_to_job: dict) -> dict:
    """
    Turns a list of job_index values the LLM cited into trusted,
    display-ready citation data pulled from our own retrieved postings
    - never from anything the LLM generated itself. Invalid or
    out-of-range indices are silently dropped rather than breaking the
    response, since a hallucinated index shouldn't take down the whole
    gap-check.

    Also surfaces the average retrieval relevance (`_relevance`, set by
    retrieval.py) of the cited postings. This doesn't change the
    evidence tier - mixing a 0-100% coverage figure with a much
    smaller-scale similarity score needs real calibration to not
    misbehave (2 highly-relevant citations shouldn't silently
    outrank 5 moderately-relevant ones without that being deliberate).
    It's shown as supporting context instead: the data was already
    computed during retrieval, so there's no reason to hide it.
    """
    seen = []
    citations = []
    relevance_scores = []
    for idx in indices or []:
        if not isinstance(idx, int) or idx in seen or idx not in index_to_job:
            continue
        seen.append(idx)
        job = index_to_job[idx]
        citations.append({
            "title": job.get("title"),
            "company": job.get("company"),
            "url": job.get("url"),
        })
        if job.get("_relevance") is not None:
            relevance_scores.append(job["_relevance"])
    total = len(index_to_job)
    cited_count = len(citations)
    coverage_pct = round(100 * cited_count / total) if total else 0
    avg_relevance = round(sum(relevance_scores) / len(relevance_scores), 3) if relevance_scores else None
    return {
        "citations": citations,
        "avg_relevance": avg_relevance,
        "cited_count": cited_count,
        "coverage_pct": coverage_pct,
        "evidence_tier": _evidence_tier(coverage_pct, cited_count),
    }


def _resolve_citations(parsed: dict, index_to_job: dict) -> dict:
    """
    Post-processes the LLM's raw JSON:
      - normalizes score_breakdown's max_points/achieved_points and
        computes match_score from them in code (see _compute_match_score) -
        the LLM never states match_score directly, so there's no way
        for a headline number to disagree with its own breakdown
      - derives each row's status (strong/partial/missing) from that
        same achieved/max ratio instead of trusting a separately
        LLM-asserted label
      - replaces job_index citation lists with real posting data
        (title/company/url) sourced from our own retrieval results,
        and rebuilds matching_jobs from matching_job_indices the same
        way - so no company name or URL in the final response was
        ever generated by the LLM itself
    """
    total = len(index_to_job)

    parsed["score_breakdown"] = _normalize_score_breakdown(parsed.get("score_breakdown", []))
    parsed["match_score"] = _compute_match_score(parsed["score_breakdown"])

    for row in parsed["score_breakdown"]:
        row["status"] = _status_from_points(row["achieved_points"], row["max_points"])
        summary = _citation_summary(row.pop("cited_job_indices", []), index_to_job)
        row["citations"] = summary["citations"]
        row["cited_count"] = summary["cited_count"]
        row["cited_total"] = total
        row["coverage_pct"] = summary["coverage_pct"]
        row["evidence_tier"] = summary["evidence_tier"]
        row["avg_relevance"] = summary["avg_relevance"]

    for gap in parsed.get("gaps", []) or []:
        summary = _citation_summary(gap.pop("cited_job_indices", []), index_to_job)
        gap["citations"] = summary["citations"]
        gap["cited_count"] = summary["cited_count"]
        gap["cited_total"] = total
        gap["coverage_pct"] = summary["coverage_pct"]
        gap["evidence_tier"] = summary["evidence_tier"]
        gap["avg_relevance"] = summary["avg_relevance"]

    matching_indices = parsed.pop("matching_job_indices", []) or []
    matching_jobs = []
    seen = []
    for idx in matching_indices:
        if not isinstance(idx, int) or idx in seen or idx not in index_to_job:
            continue
        seen.append(idx)
        job = index_to_job[idx]
        matching_jobs.append({
            "title": job.get("title"),
            "company": job.get("company"),
            "url": job.get("url"),
        })
    parsed["matching_jobs"] = matching_jobs

    return parsed