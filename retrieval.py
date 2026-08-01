"""
retrieval.py
RAG retrieval layer for GapCheck.

Wire returns live Indeed postings, but not all of them are equally
relevant to a specific user's skills + target role — a broad query like
"developer" can return postings for wildly different seniority levels,
stacks, or sub-roles. Previously analyzer.py just took the first N
postings in whatever order Wire's API happened to return them.

This module turns that into real retrieval, in two stages:

STAGE 1 - cheap first pass: every posting Wire returned is scored
against the user's profile using only the short search-result snippet
(title, company, ~2-line preview) - no extra network calls.

STAGE 2 - full-description refinement: the strongest candidates from
stage 1 have their FULL job description fetched via Wire's
in_job_details action, and are re-ranked using that much richer text.
A two-line snippet can easily undersell (or oversell) a posting -
"5+ years" might only appear in the full requirements section, not the
preview. This is deliberately bounded, not applied to every posting:
fetching full details is itself a live, async Wire call per posting, so
enriching all 50+ postings a broad search can return would make every
gap-check painfully slow. Only the current top candidates get enriched,
fetches run concurrently with a bounded worker pool, and any individual
failure or slow response just leaves that one posting on its snippet -
degrading gracefully rather than blocking the whole search.

Two ranking strategies, in priority order, used at both stages:

1. Semantic (OpenAI text-embedding-3-small), if OPENAI_API_KEY is set.
   Understands "React" ~= "frontend framework", "GCP" ~= "Google Cloud"
   even without literal keyword overlap - TF-IDF can't do that.
2. TF-IDF + cosine similarity, as a zero-dependency, zero-API-key
   fallback. Used automatically if no embedding key is configured, or
   if the embedding API call fails for any reason (network issue, rate
   limit, bad key) - a broken *upgrade* should never break the basic
   gap-check.

All of this is query-time retrieval over a small, per-request corpus
(the postings Wire just returned for this one search) - no persistent
vector store is needed since each search's corpus is used once and
then discarded.
"""

import concurrent.futures
import os
import re

import numpy as np
import requests
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from wire_client import get_job_details, WireError

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")
EMBEDDINGS_URL = "https://api.openai.com/v1/embeddings"

# --- Stage 2 (full-description enrichment) tuning ------------------
# These bound the worst-case latency/cost of enrichment: at most
# ENRICH_LIMIT postings get a full-detail fetch, at most
# ENRICH_WORKERS run concurrently, each individual fetch gets at most
# ENRICH_PER_JOB_MAX_WAIT seconds (shorter than a one-off lookup would
# use - this is a refinement step, not the critical path), and the
# whole enrichment stage gives up after ENRICH_OVERALL_TIMEOUT seconds
# regardless of how many postings are still in flight.
ENRICH_LIMIT = 10
ENRICH_WORKERS = 5
ENRICH_PER_JOB_MAX_WAIT = 15
ENRICH_PER_JOB_MAX_RETRIES = 1
ENRICH_OVERALL_TIMEOUT = 45

# Fields Wire's in_job_details might use for the full description text.
# UNCONFIRMED - checked defensively in priority order since this hasn't
# been verified against a live response.
DESCRIPTION_FIELD_CANDIDATES = ("description", "job_description", "full_description", "snippet")


def _strip_html(text):
    """Remove HTML tags and collapse whitespace from a snippet string."""
    if not text:
        return ""
    clean = re.sub(r"<[^>]+>", " ", text)
    clean = re.sub(r"\s+", " ", clean).strip()
    return clean


def _job_to_document(job: dict) -> str:
    """
    Flattens one job posting into a single text chunk for retrieval.
    Prefers the full description (set by enrich_with_full_descriptions)
    over the short search-result snippet when available, since it's a
    far more complete picture of what the posting actually requires.
    """
    text = job.get("full_description") or job.get("snippet")
    max_chars = 4000 if job.get("full_description") else 800
    parts = [
        job.get("title") or "",
        job.get("company") or "",
        _strip_html(text)[:max_chars],
    ]
    return " ".join(p for p in parts if p)


def _embed_texts(texts: list):
    """
    Calls OpenAI's embeddings API for a batch of texts and returns a
    list of vectors in the same order as `texts`.

    Returns None (never raises) on any failure - missing key, network
    error, bad response, rate limit - so the caller can fall back to
    TF-IDF instead of the embeddings upgrade taking down retrieval
    entirely.
    """
    if not OPENAI_API_KEY:
        return None
    try:
        resp = requests.post(
            EMBEDDINGS_URL,
            headers={
                "Authorization": f"Bearer {OPENAI_API_KEY}",
                "Content-Type": "application/json",
            },
            json={"model": EMBEDDING_MODEL, "input": texts},
            timeout=20,
        )
        resp.raise_for_status()
        data = resp.json()["data"]
        # The API preserves input order in practice, but each item
        # carries its own "index" - sort defensively rather than trust that.
        data.sort(key=lambda d: d["index"])
        return [d["embedding"] for d in data]
    except Exception as e:
        print(f"[retrieval] embedding call failed, falling back to TF-IDF: {e}")
        return None


def _rank_by_embeddings(documents: list, query: str):
    """Returns a similarity score array, or None if embeddings aren't available."""
    vectors = _embed_texts(documents + [query])
    if not vectors:
        return None
    job_vectors = np.array(vectors[:-1])
    query_vector = np.array(vectors[-1]).reshape(1, -1)
    return cosine_similarity(job_vectors, query_vector).flatten()


def _rank_by_tfidf(documents: list, query: str):
    """Returns a similarity score array, or None if ranking isn't possible."""
    try:
        vectorizer = TfidfVectorizer(stop_words="english")
        matrix = vectorizer.fit_transform(documents + [query])
        job_vectors = matrix[:-1]
        query_vector = matrix[-1]
        return cosine_similarity(job_vectors, query_vector).flatten()
    except ValueError:
        # Empty vocabulary (e.g. every snippet stripped to nothing).
        return None


def _rank(jobs: list, query: str):
    """
    Scores `jobs` against `query`, embeddings-first with a TF-IDF
    fallback (see module docstring). Returns (scores, method) where
    method is "embeddings" or "tfidf". Never raises - if both ranking
    strategies fail (e.g. empty vocabulary), returns (None, "none")
    and the caller should treat the input order as unranked.
    """
    documents = [_job_to_document(j) for j in jobs]

    if OPENAI_API_KEY:
        scores = _rank_by_embeddings(documents, query)
        if scores is not None:
            return scores, "embeddings"

    scores = _rank_by_tfidf(documents, query)
    if scores is not None:
        return scores, "tfidf"

    return None, "none"


def _fetch_full_description(job: dict, country_domain: str) -> str:
    """
    Fetches the full job description for one posting via Wire's
    in_job_details action. Returns "" on any failure - missing URL,
    WireError, timeout, unrecognized response shape - so the caller
    just keeps using that posting's snippet, exactly like it would
    without enrichment at all.
    """
    url = job.get("url")
    if not url:
        return ""
    try:
        details, _ = get_job_details(
            job_url=url,
            country_domain=country_domain,
            max_wait_seconds=ENRICH_PER_JOB_MAX_WAIT,
            max_retries=ENRICH_PER_JOB_MAX_RETRIES,
            retry_delay=2.0,
        )
        for key in DESCRIPTION_FIELD_CANDIDATES:
            text = details.get(key)
            if text:
                return text
    except (WireError, Exception) as e:
        print(f"[retrieval] full-description fetch failed for {url}: {e}")
    return ""


def enrich_with_full_descriptions(jobs: list, country_domain: str, limit: int = ENRICH_LIMIT, status_callback=None) -> list:
    """
    Fetches full job descriptions for up to `limit` postings (the
    postings the caller passes in - it should already have narrowed
    to its best candidates), concurrently, and returns a new list of
    jobs with `full_description` set wherever the fetch succeeded.

    Best-effort and time-boxed: any individual failure is silently
    skipped (that posting just keeps its snippet), and the whole stage
    gives up after ENRICH_OVERALL_TIMEOUT seconds even if some fetches
    are still in flight - a slow enrichment pass should degrade, not
    block the rest of the gap-check.
    """
    if not jobs or not country_domain:
        return jobs

    to_fetch = jobs[:limit]
    rest = jobs[limit:]
    enriched = [dict(j) for j in to_fetch]
    fetched_texts = {}

    with concurrent.futures.ThreadPoolExecutor(max_workers=ENRICH_WORKERS) as pool:
        future_to_idx = {
            pool.submit(_fetch_full_description, job, country_domain): i
            for i, job in enumerate(enriched)
        }
        try:
            for future in concurrent.futures.as_completed(future_to_idx, timeout=ENRICH_OVERALL_TIMEOUT):
                idx = future_to_idx[future]
                try:
                    fetched_texts[idx] = future.result()
                except Exception:
                    fetched_texts[idx] = ""
                if status_callback:
                    status_callback(f"Fetching full job descriptions ({len(fetched_texts)}/{len(enriched)})...")
        except concurrent.futures.TimeoutError:
            print(
                f"[retrieval] full-description enrichment timed out after {ENRICH_OVERALL_TIMEOUT}s, "
                f"using partial results ({len(fetched_texts)}/{len(enriched)})"
            )

    for idx, text in fetched_texts.items():
        if text:
            enriched[idx]["full_description"] = text

    return enriched + rest


def retrieve_relevant_jobs(
    user_skills: str,
    target_role: str,
    jobs: list,
    top_k: int = 15,
    country_domain: str = None,
    enrich_limit: int = ENRICH_LIMIT,
    status_callback=None,
) -> list:
    """
    Ranks `jobs` (raw postings from wire_client.search_jobs) by
    relevance to the user's skills + target role, and returns the
    top_k most relevant postings, each annotated with:
      - `_relevance`: cosine similarity score against the query
      - `_retrieval_method`: "embeddings" or "tfidf", whichever ranked it
      - `_full_description_used`: True if this posting's ranking (and
        the text sent to the LLM) is based on the full description,
        not just the short snippet

    Two-stage: a cheap snippet-based pass ranks everything, then (if
    `country_domain` is given) the top candidates get their full
    description fetched and are re-ranked with that richer text - see
    module docstring for why this is bounded rather than exhaustive.
    Pass country_domain=None (or enrich_limit=0) to skip stage 2
    entirely and use snippet-only ranking, e.g. if latency matters more
    than ranking precision for a given call site.

    Degrades gracefully at every step: no OPENAI_API_KEY -> TF-IDF.
    Embedding call fails -> TF-IDF. TF-IDF also fails (e.g. empty
    vocabulary) -> original order, unranked. Enrichment fails or times
    out -> falls back to the stage-1 (snippet-based) ranking. A broken
    ranking step should never break the whole gap-check.
    """
    if not jobs:
        return []

    effective_k = min(top_k, len(jobs))
    query = f"{target_role}. Candidate skills and experience: {user_skills}"

    stage1_scores, stage1_method = _rank(jobs, query)
    if stage1_scores is None:
        return jobs[:effective_k]

    stage1_ranked_with_scores = sorted(zip(jobs, stage1_scores), key=lambda p: p[1], reverse=True)
    stage1_ranked = [job for job, _ in stage1_ranked_with_scores]

    if country_domain and enrich_limit > 0:
        candidate_pool_size = max(effective_k, min(enrich_limit, len(stage1_ranked)))
        candidate_pairs = stage1_ranked_with_scores[:candidate_pool_size]
        candidates = [job for job, _ in candidate_pairs]
        enriched_candidates = enrich_with_full_descriptions(
            candidates, country_domain, limit=enrich_limit, status_callback=status_callback
        )
        stage2_scores, stage2_method = _rank(enriched_candidates, query)
        if stage2_scores is not None:
            ranked_pairs = sorted(zip(enriched_candidates, stage2_scores), key=lambda p: p[1], reverse=True)
            method = stage2_method
        else:
            # Re-ranking failed for some reason - fall back to the stage-1
            # scores for these same candidates rather than losing them.
            ranked_pairs = candidate_pairs
            method = stage1_method
    else:
        ranked_pairs = stage1_ranked_with_scores
        method = stage1_method

    results = []
    for job, score in ranked_pairs[:effective_k]:
        annotated = dict(job)
        annotated["_relevance"] = round(float(score), 3)
        annotated["_retrieval_method"] = method
        annotated["_full_description_used"] = bool(job.get("full_description"))
        results.append(annotated)
    return results