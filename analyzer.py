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
# Groq deprecates and shuts down models on a rolling basis (they've done
# this roughly monthly through 2026 per their changelog) - keeping this
# as an env var means the next deprecation is a config change, not a
# code push. llama-3.3-70b-versatile was shut down 08/16/26; this
# default is Groq's own recommended replacement as of that deprecation.
GROQ_MODEL = os.getenv("GROQ_MODEL", "openai/gpt-oss-120b")
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

NEVER reference a posting by its job_index, or say things like "postings
0, 1, and 2" or "job posting 3", anywhere in free text (summary, note,
why_it_matters, how_to_fix, action). job_index is an internal identifier
for the cited_job_indices field only - readers should never see it. If
you want to reference postings in prose, describe them qualitatively
instead (e.g. "several retrieved postings emphasize cloud deployment").

SCORING: produce a score_breakdown - the 4-6 most important skill/
competency areas these postings actually ask for. For each area, judge
ONLY how strongly the user's stated skills/experience support it, as
"evidence_pct" (0-100). Do NOT judge how important or common this area
is in the job market yourself - that's calculated separately from which
postings you cite, using the SAME cited_job_indices you provide for that
row, not something you need to weigh in evidence_pct. Keeping these
separate matters: a skill you rate highly just because the user is good
at it, but that few of the retrieved postings actually ask for, should
not end up dominating the final score - and it won't, as long as you
score evidence_pct purely on evidence strength and let citations do the
market-relevance work.

Keep areas distinct and non-overlapping - e.g. "MLOps" (experiment
tracking, model registry, monitoring) and "Containerization" (Docker,
Kubernetes) are different things even if related; don't create two rows
that would both be satisfied or both be missing by the same evidence.

BE STRICT ABOUT SOFT/INTERPERSONAL SKILLS (communication, teamwork,
leadership, stakeholder management, collaboration): only award
evidence_pct > 0 if the user's stated profile EXPLICITLY mentions
something like leading a team, presenting to stakeholders, mentoring,
open-source contribution, or similar. Having built individual technical
projects is NOT evidence of collaboration or communication ability -
don't infer soft skills from the mere existence of projects. If there's
no explicit evidence, evidence_pct must be 0 and the note should say so
plainly (e.g. "No direct evidence of team collaboration in your profile
- projects alone don't demonstrate this").

CITATIONS - this is critical for trust: every score_breakdown row and
every gap MUST include a "cited_job_indices" field - a list of the
job_index values (from the postings you were given) that actually
contain or imply that requirement. This field is required on every
row, with no exceptions: if you genuinely cannot point to a specific
posting for something, use an empty list "cited_job_indices": [] -
never omit the field, and never pad it with postings that don't
actually support the claim. For score_breakdown rows specifically,
these citations also determine how much that area counts toward the
final score - an accurate, non-padded citation list matters as much
here as it does for trust.

Respond ONLY with valid JSON, no markdown fences, no preamble, in this exact shape:
{
  "summary": "<one or two blunt sentences on overall readiness, referencing real postings qualitatively - never by job_index or number>",
  "score_breakdown": [
    {"area": "<skill/competency area drawn from real postings>", "evidence_pct": <int 0-100, evidence strength ONLY>, "note": "<one short clause explaining the rating>", "cited_job_indices": [<job_index>, ...] (use [] if none apply)}
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
an exhaustive list. For matching_job_indices, pick up to 3 job_index values
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

    # gpt-oss-120b is a reasoning model - Groq has open, live bug reports
    # of its internal reasoning tokens occasionally leaking into or
    # eating into the same output budget as the actual answer, which can
    # truncate a long JSON response mid-string. Three defenses:
    #   - response_format=json_object: Groq enforces syntactically valid
    #     JSON server-side, the strongest guard against a cut-off response
    #   - reasoning_effort="low": this task is a direct extraction/scoring
    #     job, not something needing deep reasoning chains, so keep more
    #     of the token budget for the actual answer
    #   - generous max_tokens: this schema (up to 15 postings' worth of
    #     citations, 4-6 breakdown rows, several gaps) is verbose; give
    #     it real headroom instead of hitting a default limit
    # Even with all three, an occasional malformed response is still
    # possible (LLM output isn't 100% deterministic) - retry once on a
    # JSON parse failure before giving up, since a re-sample often
    # succeeds where the first attempt didn't.
    last_error = None
    for attempt in range(2):
        response = client.chat.completions.create(
            model=GROQ_MODEL,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_content},
            ],
            temperature=0.4,
            max_tokens=8000,
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
            return _resolve_citations(parsed, index_to_job)
        except json.JSONDecodeError as e:
            last_error = e
            print(f"[analyzer] JSON parse failed on attempt {attempt + 1}/2: {e}")

    raise RuntimeError(
        "The analysis came back malformed twice in a row - this is usually a temporary "
        "issue with the AI model's response, not something wrong with your search. "
        "Please try running the gap check again."
    ) from last_error


def _market_demand_tier(market_demand_pct: float, cited_count: int) -> str:
    """
    Buckets citation coverage - i.e. how much of the retrieved posting
    set actually asks for this - into a human-readable tier. This is
    ONLY about the market (computed from citations), never about the
    candidate's own skill level (that's `status`, derived separately
    from evidence_pct) - keeping the words distinct (well_supported/
    some_support/limited_support/unsupported vs. strong/good/partial/
    weak/not_demonstrated) avoids the two independent measurements
    reading as if they're contradicting each other when shown together.
    """
    if cited_count == 0:
        return "unsupported"
    if market_demand_pct >= 50:
        return "well_supported"
    if market_demand_pct >= 20:
        return "some_support"
    return "limited_support"


def _priority_from_market_demand(market_demand_pct: float) -> str:
    """
    Derives a gap's priority (high/medium/low) from how many retrieved
    postings actually cite it - not from how severe or generic the gap
    sounds. A gap mentioned in most retrieved postings deserves more
    attention than one that showed up in a single unusual listing,
    even if both are framed similarly in prose.
    """
    if market_demand_pct >= 50:
        return "high"
    if market_demand_pct >= 20:
        return "medium"
    return "low"


def _status_from_evidence(evidence_pct: int) -> str:
    """
    Buckets evidence_pct (the LLM's evidence-strength judgment, 0-100)
    into a 5-tier label. This measures candidate evidence only - how
    strongly the profile supports this area - not market demand, which
    is a separate, code-computed number (see market_demand_pct below).

    The exact cutoffs (80/60/30/1) are a judgment call, not a derived
    fact - there's no ground truth that makes 80% "the" boundary for
    Strong vs. 75% or 85%. Adopted as specified rather than re-derived.
    """
    if evidence_pct >= 80:
        return "strong"
    if evidence_pct >= 60:
        return "good"
    if evidence_pct >= 30:
        return "partial"
    if evidence_pct >= 1:
        return "weak"
    return "not_demonstrated"


def _normalize_score_breakdown(score_breakdown: list) -> list:
    """
    Validates and cleans the LLM's evidence_pct per row: coerces to
    int, clips to [0, 100] (defends against an out-of-range or
    malformed value), and drops rows that are missing an area name
    entirely. Never raises - malformed rows are dropped rather than
    crashing the whole analysis.
    """
    cleaned = []
    for row in score_breakdown or []:
        if not row.get("area"):
            continue
        try:
            evidence_pct = int(row.get("evidence_pct", 0))
        except (TypeError, ValueError):
            evidence_pct = 0
        row["evidence_pct"] = max(0, min(evidence_pct, 100))
        cleaned.append(row)
    return cleaned


def _compute_match_score(score_breakdown: list) -> int:
    """
    match_score is calculated here, in code, as a market-demand-
    weighted average of evidence_pct - never a number the LLM states
    directly, and never a simple unweighted average either.

    Each row's weight is its market_demand_pct (how many of the
    retrieved postings actually cite it - computed from the SAME
    cited_job_indices the LLM already provides, not a separate
    LLM-assigned "importance"). This is the fix for a real problem:
    previously the LLM could call a skill "20/20 important" while the
    citation data showed only 5 of 15 postings actually required it -
    two independently-asserted numbers with no relationship to each
    other. Now importance isn't asserted at all; it's measured from
    the same evidence used everywhere else in the app. A skill the
    candidate is strong in but that almost nothing in the retrieved
    market actually asks for can no longer dominate the score.

    Falls back to an unweighted average if every row's market_demand_pct
    is 0 (e.g. citation resolution failed across the board) - otherwise
    a total-weight of zero would make the score meaningless rather than
    just imperfectly weighted.
    """
    if not score_breakdown:
        return 0
    total_weight = sum(row["market_demand_pct"] for row in score_breakdown)
    if total_weight <= 0:
        return round(sum(row["evidence_pct"] for row in score_breakdown) / len(score_breakdown))
    weighted_sum = sum(row["market_demand_pct"] * row["evidence_pct"] for row in score_breakdown)
    return round(weighted_sum / total_weight)


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
    market_demand_pct = round(100 * cited_count / total) if total else 0
    avg_relevance = round(sum(relevance_scores) / len(relevance_scores), 3) if relevance_scores else None
    return {
        "citations": citations,
        "avg_relevance": avg_relevance,
        "cited_count": cited_count,
        "market_demand_pct": market_demand_pct,
        "market_demand_tier": _market_demand_tier(market_demand_pct, cited_count),
    }


def _resolve_citations(parsed: dict, index_to_job: dict) -> dict:
    """
    Post-processes the LLM's raw JSON:
      - resolves each score_breakdown row's citations FIRST (needed to
        compute market_demand_pct before the score can be calculated)
      - computes match_score in code as a market-demand-weighted
        average of evidence_pct (see _compute_match_score) - market
        weight comes from citations, not an LLM-asserted importance
      - derives each row's status (strong/good/partial/weak/not_
        demonstrated) from evidence_pct alone, so it can never
        contradict the number driving the score
      - derives each gap's priority (high/medium/low) from how many
        retrieved postings actually cite it, not from how the gap is
        phrased
      - replaces job_index citation lists with real posting data
        (title/company/url) sourced from our own retrieval results,
        and rebuilds matching_jobs from matching_job_indices the same
        way - so no company name or URL in the final response was
        ever generated by the LLM itself
    """
    total = len(index_to_job)

    parsed["score_breakdown"] = _normalize_score_breakdown(parsed.get("score_breakdown", []))

    for row in parsed["score_breakdown"]:
        summary = _citation_summary(row.pop("cited_job_indices", []), index_to_job)
        row["citations"] = summary["citations"]
        row["cited_count"] = summary["cited_count"]
        row["cited_total"] = total
        row["market_demand_pct"] = summary["market_demand_pct"]
        row["market_demand_tier"] = summary["market_demand_tier"]
        row["avg_relevance"] = summary["avg_relevance"]
        row["status"] = _status_from_evidence(row["evidence_pct"])

    parsed["match_score"] = _compute_match_score(parsed["score_breakdown"])

    for gap in parsed.get("gaps", []) or []:
        summary = _citation_summary(gap.pop("cited_job_indices", []), index_to_job)
        gap["citations"] = summary["citations"]
        gap["cited_count"] = summary["cited_count"]
        gap["cited_total"] = total
        gap["market_demand_pct"] = summary["market_demand_pct"]
        gap["market_demand_tier"] = summary["market_demand_tier"]
        gap["avg_relevance"] = summary["avg_relevance"]
        gap["priority"] = _priority_from_market_demand(summary["market_demand_pct"])

    # Gaps are shown in priority order - the reader's attention should
    # go to what the retrieved market actually demands most, not
    # whatever order the LLM happened to list them in.
    priority_rank = {"high": 0, "medium": 1, "low": 2}
    parsed["gaps"] = sorted(parsed.get("gaps", []) or [], key=lambda g: priority_rank.get(g["priority"], 3))

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