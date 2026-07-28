"""
retrieval.py
RAG retrieval layer for GapCheck.

Wire returns live Indeed postings, but not all of them are equally
relevant to a specific user's skills + target role — a broad query like
"developer" can return postings for wildly different seniority levels,
stacks, or sub-roles. Previously analyzer.py just took the first N
postings in whatever order Wire's API happened to return them.

This module turns that into real retrieval: every posting is embedded
and scored against a query built from the user's stated skills and
target role, and only the top-K most relevant postings are handed to
the LLM. That's the "R" in RAG — analyzer.py's Groq call is the "G",
generating grounded in whatever this step retrieves.

Two ranking strategies, in priority order:

1. Semantic (OpenAI text-embedding-3-small), if OPENAI_API_KEY is set.
   Understands "React" ~= "frontend framework", "GCP" ~= "Google Cloud"
   even without literal keyword overlap - TF-IDF can't do that.
2. TF-IDF + cosine similarity, as a zero-dependency, zero-API-key
   fallback. Used automatically if no embedding key is configured, or
   if the embedding API call fails for any reason (network issue, rate
   limit, bad key) - a broken *upgrade* should never break the basic
   gap-check.

Both are query-time retrieval over a small, per-request corpus (the
postings Wire just returned for this one search) - no persistent
vector store is needed since each search's corpus is used once and
then discarded.
"""

import os
import re
import numpy as np
import requests
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")
EMBEDDINGS_URL = "https://api.openai.com/v1/embeddings"


def _strip_html(text):
    """Remove HTML tags and collapse whitespace from a snippet string."""
    if not text:
        return ""
    clean = re.sub(r"<[^>]+>", " ", text)
    clean = re.sub(r"\s+", " ", clean).strip()
    return clean


def _job_to_document(job: dict) -> str:
    """Flatten one job posting into a single text chunk for retrieval."""
    parts = [
        job.get("title") or "",
        job.get("company") or "",
        _strip_html(job.get("snippet"))[:800],
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


def retrieve_relevant_jobs(user_skills: str, target_role: str, jobs: list, top_k: int = 15) -> list:
    """
    Ranks `jobs` (raw postings from wire_client.search_jobs) by
    relevance to the user's skills + target role, and returns the
    top_k most relevant postings, each annotated with:
      - `_relevance`: cosine similarity score against the query
      - `_retrieval_method`: "embeddings" or "tfidf", whichever ranked it

    Degrades gracefully at every step: no OPENAI_API_KEY -> TF-IDF.
    Embedding call fails -> TF-IDF. TF-IDF also fails (e.g. empty
    vocabulary) -> original order, unranked. A broken ranking step
    should never break the whole gap-check.
    """
    if not jobs:
        return []

    effective_k = min(top_k, len(jobs))
    documents = [_job_to_document(j) for j in jobs]
    query = f"{target_role}. Candidate skills and experience: {user_skills}"

    method = "tfidf"
    scores = None
    if OPENAI_API_KEY:
        scores = _rank_by_embeddings(documents, query)
        if scores is not None:
            method = "embeddings"

    if scores is None:
        scores = _rank_by_tfidf(documents, query)
        method = "tfidf"

    if scores is None:
        return jobs[:effective_k]

    ranked = sorted(zip(jobs, scores), key=lambda pair: pair[1], reverse=True)

    results = []
    for job, score in ranked[:effective_k]:
        annotated = dict(job)
        annotated["_relevance"] = round(float(score), 3)
        annotated["_retrieval_method"] = method
        results.append(annotated)
    return results