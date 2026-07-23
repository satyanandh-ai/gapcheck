"""
skill_aliases.py
Normalizes common typos, abbreviations, and shorthand in user-entered
skills before they're sent to the analyzer. Prevents garbage-in like
"could" (typo for "cloud") from being treated as a literal skill.
"""

import re

SKILL_ALIASES = {
    "could": "cloud",
    "cloud comp": "cloud computing",
    "py": "python",
    "js": "javascript",
    "ts": "typescript",
    "ml": "machine learning",
    "dl": "deep learning",
    "nlp": "natural language processing",
    "cv": "computer vision",
    "rl": "reinforcement learning",
    "genai": "generative AI",
    "gen ai": "generative AI",
    "llm": "large language models",
    "llms": "large language models",
    "aws": "AWS",
    "gcp": "Google Cloud Platform",
    "azure": "Microsoft Azure",
    "k8s": "Kubernetes",
    "docker": "Docker",
    "sql": "SQL",
    "nosql": "NoSQL",
    "api": "API development",
    "apis": "API development",
    "rag": "retrieval-augmented generation",
    "oop": "object-oriented programming",
    "dsa": "data structures and algorithms",
    "ci/cd": "CI/CD",
    "cicd": "CI/CD",
    "fe": "frontend",
    "be": "backend",
}


def normalize_skills(raw_text: str) -> str:
    """
    Splits free-text skills input on commas/newlines, corrects known
    typos/abbreviations token-by-token, and rejoins. Leaves anything
    not in the alias map untouched (so real free-text descriptions
    like "built 2 Flask projects" still pass through fine).
    """
    if not raw_text or not raw_text.strip():
        return raw_text

    # Split on commas/newlines/semicolons — keep the rest of each
    # segment intact so descriptive phrases aren't mangled.
    segments = re.split(r"[,\n;]+", raw_text)
    corrected = []
    for seg in segments:
        stripped = seg.strip()
        if not stripped:
            continue
        key = stripped.lower()
        if key in SKILL_ALIASES:
            corrected.append(SKILL_ALIASES[key])
        else:
            corrected.append(stripped)

    return ", ".join(corrected)