K# 🎯 GapCheck

### An honest gap check between your skills and the jobs that actually exist right now.

**Built solo in a single 6-hour sprint for Anakin Blitz 2026** 🚀

![Python](https://img.shields.io/badge/Python-3.10+-blue?style=flat-square)
![Streamlit](https://img.shields.io/badge/Streamlit-UI-FF4B4B?style=flat-square)
![Wire API](https://img.shields.io/badge/Powered%20by-Anakin%20Wire-FF7A3D?style=flat-square)
![Groq](https://img.shields.io/badge/LLM-LLaMA%203.3%20via%20Groq-9FDDB4?style=flat-square)

---

> **Grounded, explainable career intelligence using live job postings.**
> - Retrieves the most relevant live job postings for your specific skills + role.
> - Grounds every recommendation in that retrieved evidence, not generic advice.
> - Validates every citation against trusted retrieval results — the LLM never invents a company name or URL.
> - Shows evidence strength: how much of the retrieved evidence actually backs each claim.

---

## 💡 The Problem

Career advice for students and early-career developers is almost always **generic**:

- 🌀 "Learn more skills"
- 🌀 "Build more projects"
- 🌀 "Improve your resume"

None of it is grounded in what companies are **actually hiring for, today.**

Meanwhile, real job postings are packed with current, specific signal — exact tech stacks, exact experience levels, exact phrasing recruiters use. That signal just sits there, unused, because nobody connects it back to the individual.

> **GapCheck closes that loop.**

---

## ⚙️ What It Does

| Step | Action |
|:---:|---|
| 1️⃣ | You type your **current skills** and/or **upload a resume** (PDF/DOCX/TXT) + a **target role** |
| 2️⃣ | An LLM extracts a **structured profile** — skills, projects, certifications, experience level — shown back to you before analysis runs |
| 3️⃣ | **Wire** pulls **live Indeed postings** for that role — right now, not a stale dataset |
| 4️⃣ | Postings are **ranked by relevance** to your specific profile (semantic embeddings, or TF-IDF as a fallback) |
| 5️⃣ | An **LLM (Groq / LLaMA 3.3)** compares the most relevant real postings against your profile, citing exactly which postings back each claim |
| 6️⃣ | You get a complete, honest, evidence-backed readout |

### 📊 What you get back

- ✅ **Match score** (0–100), with a **breakdown of exactly why** — each area cited against real postings
- 💪 **Strengths** — what you've already got going for you
- ⚠️ **Top gaps** — the 3–5 that actually matter, each with a concrete fix and evidence strength (how much of the retrieved evidence backs it)
- 🚀 **Action plan** — what to do *this week*, not someday
- 🔗 **Live postings** worth applying to right now, plus links to the specific postings behind each claim

No generic roadmap. No "learn AI." Just what's actually blocking you — grounded in jobs that exist *today*.

---

## 🔌 Why Wire Is the Core, Not a Bolt-On

GapCheck has exactly **one** data source, and it's Wire.

Every job posting shown, scored, and reasoned about comes from a **live `in_search_jobs` call** through the Anakin Wire API against Indeed.

- ❌ No scraping code
- ❌ No static dataset
- ❌ No hardcoded fallback list
- ✅ If Wire returns nothing, the app has nothing to analyze

Wire isn't a feature bolted onto an existing idea — **it's the reason the idea is possible at all.**

---

## 🛠️ Tech Stack

| Layer | Technology | Purpose |
|---|---|---|
| 🔌 Data | **Anakin Wire API** | Live job search via `in_search_jobs` (Indeed) |
| 📄 Input parsing | **Groq LLM + pypdf / python-docx** | Extracts a structured profile (skills, projects, certs, experience) from free text or an uploaded resume |
| 🔍 Retrieval | **OpenAI embeddings**, falls back to **scikit-learn TF-IDF** | Ranks live postings by relevance to the user before they reach the LLM (RAG) |
| 🧠 Reasoning | **Groq — LLaMA 3.3 70B** | Fast, structured gap analysis grounded in retrieved postings |
| 🎨 Frontend | **Streamlit** + custom CSS theme | Clean, fast to ship, fully styled |
| 🐍 Backend | **Python** | Wire client, async polling, analysis pipeline |

---

## 🏗️ Architecture

```
User input (skills + target role)
        │
        ▼
Wire API → in_search_jobs (Indeed, live)
        │     [async job, polled until complete]
        ▼
Real job postings (title, company, snippet, salary, url)
        │
        ▼
Retrieval (RAG) → semantic ranking (OpenAI embeddings) if configured,
        │     else TF-IDF + cosine similarity — ranks every live
        │     posting against the user's skills + target role;
        │     only the top-K most relevant postings are kept
        ▼
Groq LLM → compares the retrieved, most-relevant postings vs. stated
        │     skills, citing which postings (by job_index) support
        │     each score_breakdown row and gap
        ▼
Citation resolution → job_index citations are resolved against our
        │     own trusted retrieval results, never the LLM's own text —
        │     eliminates hallucinated company names / URLs
        ▼
Evidence scoring → coverage % of cited postings → evidence tier
        │     (strong / moderate / limited / unsupported)
        ▼
Structured output: score · strengths · gaps (with evidence) · action plan
        │
        ▼
Streamlit UI
```

**Why retrieval matters here:** a broad query like "developer" can return
postings spanning wildly different seniority levels and stacks. Instead of
handing the LLM whatever N postings Wire happened to return first,
`retrieval.py` scores every posting against the user's specific profile and
keeps only the closest matches — so the gap analysis is grounded in postings
that actually apply to *that* user, not just live postings in general.
Semantic embeddings (when `OPENAI_API_KEY` is set) catch matches TF-IDF
would miss — e.g. "React" ≈ "frontend framework" — with TF-IDF as an
always-available fallback so the app works fully without that key.

**Why explainability matters here:** a bare `match_score` is a black box.
Every score breakdown row and gap is grounded in specific retrieved
postings, resolved against trusted data rather than LLM-generated text, and
shown with a measurable evidence tier — so a reader can see exactly *why*
the score is what it is, not just trust it.

---

## 🚀 Running It Locally

```bash
git clone https://github.com/satyanandh-ai/gapcheck.git
cd gapcheck
pip install -r requirements.txt
```

Create a `.env` file in the project root:

```env
WIRE_API_KEY=your_wire_api_key_here
GROQ_API_KEY=your_groq_api_key_here

# Optional: enables semantic (embedding-based) retrieval ranking.
# Without this, retrieval automatically falls back to TF-IDF -
# the app works fully either way.
OPENAI_API_KEY=your_openai_api_key_here
```

Then launch:

```bash
streamlit run app.py
```

---

## 👥 Who This Is For

- 🎓 **Students & freshers** — see exactly what's missing *before* applying, not after rejection
- 🔄 **Career switchers** — find out fast whether a target role is realistic right now
- 🙅 **Anyone tired of generic advice** — get an answer grounded in postings that exist today

---

## ✍️ Author

**Ch. Satyanand**
B.Tech AIML · Andhra Loyola Institute of Engineering & Technology (ALIET), Vijayawada
GitHub: [@satyanandh-ai](https://github.com/satyanandh-ai)

Built live, solo, in 6 hours for **Anakin Blitz 2026.** 🎯