# 🎯 GapCheck

### An honest gap check between your skills and the jobs that actually exist right now.

**Built solo in a single 6-hour sprint for Anakin Blitz 2026** 🚀

![Python](https://img.shields.io/badge/Python-3.10+-blue?style=flat-square)
![Streamlit](https://img.shields.io/badge/Streamlit-UI-FF4B4B?style=flat-square)
![Wire API](https://img.shields.io/badge/Powered%20by-Anakin%20Wire-FF7A3D?style=flat-square)
![Groq](https://img.shields.io/badge/LLM-LLaMA%203.3%20via%20Groq-9FDDB4?style=flat-square)

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
| 1️⃣ | You type your **current skills** + a **target role** |
| 2️⃣ | **Wire** pulls **live Indeed postings** for that role — right now, not a stale dataset |
| 3️⃣ | An **LLM (Groq / LLaMA 3.3)** compares the real postings against what you actually know |
| 4️⃣ | You get a complete, honest readout |

### 📊 What you get back

- ✅ **Match score** (0–100)
- 💪 **Strengths** — what you've already got going for you
- ⚠️ **Top gaps** — the 3–5 that actually matter, each with a concrete fix
- 🚀 **Action plan** — what to do *this week*, not someday
- 🔗 **Live postings** worth applying to right now

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
| 🧠 Reasoning | **Groq — LLaMA 3.3 70B** | Fast, structured gap analysis |
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
Groq LLM → compares postings vs. stated skills
        │
        ▼
Structured output: score · strengths · gaps · action plan
        │
        ▼
Streamlit UI
```

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
