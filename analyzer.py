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

Also produce a score_breakdown: a short list of the 4-6 most important
skill/competency areas these postings actually ask for, each rated
against what the user has. This is what makes match_score explainable
instead of an unexplained number — a reader should be able to look at
the breakdown and see exactly why the score is what it is.

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
  "match_score": <integer 0-100>,
  "summary": "<one or two blunt sentences on overall readiness, referencing real postings>",
  "score_breakdown": [
    {"area": "<skill/competency area drawn from real postings>", "status": "<strong|partial|missing>", "note": "<one short clause explaining the rating>", "cited_job_indices": [<job_index>, ...] (use [] if none apply)}
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
Keep gaps to the 3-5 that matter most. Keep score_breakdown to 4-6 areas —
these should be the areas that most influenced the match_score, not an
exhaustive list. For matching_job_indices, pick up to 3 job_index values
for postings that best fit the user's current level. Be specific —
reference real requirements from the postings provided, not generic
advice. Time estimates should be realistic for someone learning
part-time alongside other commitments, not idealized."""
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
    Buckets citation coverage into a human-readable tier. Deliberately
    called "evidence strength", not "confidence" - this reflects how
    much of the retrieved posting set backs the claim, not the LLM's
    certainty, which isn't something we can actually measure.
    """
    if cited_count == 0:
        return "unsupported"
    if coverage_pct >= 50:
        return "strong"
    if coverage_pct >= 20:
        return "moderate"
    return "limited"


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
    Post-processes the LLM's raw JSON: replaces job_index citation
    lists with real posting data (title/company/url) sourced from our
    own retrieval results, and rebuilds matching_jobs from
    matching_job_indices the same way - so no company name or URL in
    the final response was ever generated by the LLM itself.
    """
    total = len(index_to_job)

    for row in parsed.get("score_breakdown", []) or []:
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