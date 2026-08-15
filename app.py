"""
app.py
GapCheck - skills + target role -> real gap analysis against LIVE
Indeed postings via Wire. Built for Anakin Blitz.
"""

import streamlit as st
from wire_client import search_jobs, WireError, _is_transient_failure
from analyzer import analyze_gap
from location_normalizer import suggestions
from skill_aliases import normalize_skills
from retrieval import retrieve_relevant_jobs
from resume_parser import extract_text_from_upload, parse_profile, profile_to_query_text

st.set_page_config(page_title="GapCheck", page_icon=":dart:", layout="centered")

CUSTOM_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@500;600;700&family=JetBrains+Mono:wght@400;500;600&family=Inter:wght@400;500;600&display=swap');

html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
#MainMenu, footer, header { visibility: hidden; }
.block-container { padding-top: 2.2rem; max-width: 760px; }

.gc-eyebrow {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.72rem;
    letter-spacing: 0.2em;
    text-transform: uppercase;
    color: #FF7A3D;
    margin-bottom: 0.4rem;
    font-weight: 600;
}
.gc-title {
    font-family: 'Space Grotesk', sans-serif;
    font-weight: 700;
    font-size: 2.6rem;
    line-height: 1.05;
    color: #F4F1EA;
    margin-bottom: 0.35rem;
    letter-spacing: -0.02em;
}
.gc-sub {
    color: #9A9690;
    font-size: 1.05rem;
    margin-bottom: 2rem;
    max-width: 480px;
}
.gc-divider { border: none; border-top: 1px solid #262A31; margin: 2rem 0 1.6rem 0; }

div[data-testid="stForm"] {
    background: #1A1C21;
    border: 1px solid #262A31;
    border-radius: 16px;
    padding: 1.6rem 1.6rem 0.8rem 1.6rem;
}
div[data-testid="stForm"] label { color: #C9C6C0 !important; font-size: 0.88rem !important; font-weight: 500 !important; }
.stTextArea textarea, .stTextInput input {
    background: #14161A !important;
    border: 1px solid #2B2F37 !important;
    border-radius: 9px !important;
    color: #E8E6E1 !important;
}
.stButton button, button[kind="formSubmit"] {
    background-color: #FF7A3D !important;
    color: #16181D !important;
    font-weight: 600 !important;
    border: none !important;
    border-radius: 9px !important;
    padding: 0.6rem 0 !important;
    font-size: 0.98rem !important;
    margin-top: 0.4rem;
    transition: opacity 0.15s ease;
}
.stButton button:hover { opacity: 0.88; }

.gc-score-card {
    background: linear-gradient(155deg, #1E2127 0%, #16181C 100%);
    border: 1px solid #2B2F37;
    border-radius: 18px;
    padding: 2rem 2rem 1.7rem 2rem;
    margin-bottom: 1.6rem;
    box-shadow: 0 8px 30px rgba(0,0,0,0.25);
}
.gc-score-top { display: flex; align-items: baseline; justify-content: space-between; }
.gc-score-label {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.74rem;
    letter-spacing: 0.16em;
    text-transform: uppercase;
    color: #7C7973;
    font-weight: 600;
}
.gc-score-number {
    font-family: 'Space Grotesk', sans-serif;
    font-weight: 700;
    font-size: 4rem;
    line-height: 1;
    margin: 0.3rem 0 0.9rem 0;
}
.gc-score-number span { font-size: 1.5rem; color: #5C5954; font-weight: 500; }
.gc-score-summary { color: #C9C6C0; font-size: 1.02rem; line-height: 1.55; }

.gc-section-label {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.74rem;
    letter-spacing: 0.14em;
    text-transform: uppercase;
    color: #7C7973;
    font-weight: 600;
    margin: 1.8rem 0 0.7rem 0;
    display: flex;
    align-items: center;
    gap: 0.5rem;
}
.gc-section-label::after { content: ""; flex: 1; height: 1px; background: #262A31; }

.gc-strength-pill, .gc-pill {
    display: inline-block;
    border-radius: 999px;
    padding: 0.32rem 0.9rem;
    font-size: 0.86rem;
    margin: 0.2rem 0.4rem 0.2rem 0;
    font-weight: 500;
}
.gc-strength-pill { background: rgba(122, 200, 150, 0.10); border: 1px solid rgba(122, 200, 150, 0.32); color: #9FDDB4; }

.gc-job-row {
    border: 1px solid #262A31;
    border-radius: 12px;
    padding: 0.95rem 1.1rem;
    margin-bottom: 0.65rem;
    background: #181A1F;
    transition: border-color 0.15s ease;
}
.gc-job-row:hover { border-color: #FF7A3D55; }
.gc-job-title { font-weight: 600; color: #F4F1EA; font-size: 0.98rem; margin-bottom: 0.15rem; }
.gc-job-company { color: #9A9690; font-size: 0.86rem; }
.gc-job-company a { color: #FF9A6B; text-decoration: none; font-weight: 500; }
.gc-job-company a:hover { text-decoration: underline; }

.gc-breakdown-row {
    display: flex;
    align-items: center;
    justify-content: space-between;
    padding: 0.55rem 0;
    border-bottom: 1px solid #21242b;
    font-size: 0.92rem;
}
.gc-breakdown-row:last-child { border-bottom: none; }
.gc-breakdown-area { color: #E8E6E1; font-weight: 500; }
.gc-breakdown-note { color: #7C7973; font-size: 0.82rem; margin-top: 0.1rem; }
.gc-status-badge {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.72rem;
    font-weight: 600;
    letter-spacing: 0.04em;
    text-transform: uppercase;
    padding: 0.22rem 0.6rem;
    border-radius: 999px;
    white-space: nowrap;
    margin-left: 1rem;
}
.gc-status-strong { background: rgba(122, 200, 150, 0.12); border: 1px solid rgba(122, 200, 150, 0.35); color: #9FDDB4; }
.gc-status-good { background: rgba(154, 200, 122, 0.12); border: 1px solid rgba(154, 200, 122, 0.35); color: #C3DD9F; }
.gc-status-partial { background: rgba(255, 154, 107, 0.12); border: 1px solid rgba(255, 154, 107, 0.35); color: #FF9A6B; }
.gc-status-weak { background: rgba(255, 130, 90, 0.12); border: 1px solid rgba(255, 130, 90, 0.35); color: #FFA37D; }
.gc-status-missing, .gc-status-not_demonstrated { background: rgba(255, 107, 107, 0.12); border: 1px solid rgba(255, 107, 107, 0.35); color: #FF6B6B; }
.gc-priority-high { background: rgba(255, 107, 107, 0.12); border: 1px solid rgba(255, 107, 107, 0.35); color: #FF6B6B; }
.gc-priority-medium { background: rgba(255, 154, 107, 0.12); border: 1px solid rgba(255, 154, 107, 0.35); color: #FF9A6B; }
.gc-priority-low { background: rgba(154, 152, 148, 0.12); border: 1px solid rgba(154, 152, 148, 0.35); color: #A8A5A0; }
.gc-time-badge {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.74rem;
    color: #7C7973;
    margin-left: 0.5rem;
}

.gc-citation {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.72rem;
    color: #FF9A6B;
    margin-top: 0.25rem;
    opacity: 0.85;
}

.gc-resolved-location {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.8rem;
    color: #7C7973;
    margin-top: 0.4rem;
}
.gc-resolved-location b { color: #9FDDB4; }

div[data-testid="stExpander"] {
    background: #181A1F;
    border: 1px solid #262A31;
    border-radius: 10px;
    margin-bottom: 0.5rem;
}

.gc-footer {
    text-align: center;
    color: #5C5954;
    font-size: 0.78rem;
    font-family: 'JetBrains Mono', monospace;
    letter-spacing: 0.05em;
    margin-top: 3rem;
    padding-top: 1.5rem;
    border-top: 1px solid #1F2127;
}
</style>
"""

st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


def market_demand_line(item: dict) -> str:
    """
    Builds the "Market demand" line for a score_breakdown row or gap -
    how many of the retrieved postings actually cite this, from the
    trusted citation data analyzer.py attached (never LLM-generated
    text). Explicitly labeled "Market demand" rather than "evidence" -
    that word doesn't overload with `status`, which is a completely
    separate measurement (the candidate's own skill level, from
    evidence_pct) shown right next to this.

    avg_relevance (retrieval similarity score) is deliberately left out
    of this line - it's useful for debugging retrieval quality, not for
    a candidate deciding what to do about a gap, so it's kept in the
    raw-data expander instead of the primary UI.
    """
    citations = item.get("citations") or []
    if not citations:
        return ""
    companies = []
    for c in citations:
        name = c.get("company")
        if name and name not in companies:
            companies.append(name)
    company_str = ", ".join(companies[:4])
    if len(companies) > 4:
        company_str += f" +{len(companies) - 4} more"

    total = item.get("cited_total")
    count = item.get("cited_count", len(citations))
    pct = item.get("market_demand_pct")
    of_total = f" of {total}" if total else ""
    pct_str = f" ({pct}%)" if pct is not None else ""
    return f"Market demand: {count}{of_total} postings{pct_str} — {company_str}"


STATUS_LABELS = {
    "strong": "Strong",
    "good": "Good",
    "partial": "Partial",
    "weak": "Weak",
    "not_demonstrated": "Not demonstrated",
}

PRIORITY_LABELS = {
    "high": "High priority",
    "medium": "Medium priority",
    "low": "Low priority",
}


st.markdown('<div class="gc-eyebrow">CAREER DIAGNOSTIC</div>', unsafe_allow_html=True)
st.markdown('<div class="gc-title">GapCheck</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="gc-sub">See how your skills stack up against real, live job postings &mdash; not generic advice.</div>',
    unsafe_allow_html=True,
)

with st.form("gap_form"):
    user_skills = st.text_area(
        "Your current skills / experience",
        placeholder="e.g. Python, basic SQL, built 2 small Flask projects, no production experience",
        height=100,
    )
    resume_file = st.file_uploader(
        "Or upload your resume (optional)",
        type=["pdf", "docx", "txt"],
        help="PDF, Word (.docx), or plain text. We'll extract your skills, projects, "
             "certifications, and experience level automatically - add anything extra "
             "in the box above.",
    )
    col1, col2 = st.columns(2)
    with col1:
        target_role = st.text_input("Target role", placeholder="e.g. Backend Developer")
    with col2:
        location = st.text_input(
            "Location",
            placeholder="Example: Bangalore, Hyderabad, Mumbai, Vijayawada...",
            help="Type any Indian city or abbreviation — we'll normalize it automatically. "
                 "(Wire's live job search currently covers India only.)",
        )
    submitted = st.form_submit_button("Run Gap Check", use_container_width=True)

if submitted:
    if not (user_skills or resume_file) or not target_role:
        st.error("Please fill in your skills (or upload a resume) and a target role.")
        st.stop()

    try:
        resume_text = extract_text_from_upload(resume_file) if resume_file else ""
        if resume_file and not resume_text.strip():
            st.warning(
                "Couldn't extract text from **" + resume_file.name + "** "
                "(it may be a scanned/image-only file). Continuing with the skills box only."
            )

        normalized_skills = normalize_skills(user_skills)
        combined_raw = "\n\n".join(t for t in [resume_text, normalized_skills] if t.strip())

        with st.status("Reading your skills/resume...", expanded=False) as status:
            profile = parse_profile(combined_raw)
            status.update(label="Profile parsed")

        query_text = profile_to_query_text(profile) or combined_raw

        with st.expander("What we parsed from your input", expanded=False):
            if profile.get("summary"):
                st.markdown("**Summary:** " + profile["summary"])
            if profile.get("skills"):
                pills = "".join('<span class="gc-strength-pill">' + s + '</span>' for s in profile["skills"])
                st.markdown("**Skills**", unsafe_allow_html=True)
                st.markdown(pills, unsafe_allow_html=True)
            if profile.get("projects"):
                st.markdown("**Projects**")
                for p in profile["projects"]:
                    if isinstance(p, dict):
                        st.markdown("- **" + p.get("name", "") + "** — " + p.get("description", ""))
                    else:
                        st.markdown("- " + str(p))
            if profile.get("certifications"):
                st.markdown("**Certifications:** " + ", ".join(profile["certifications"]))
            if profile.get("years_experience") and profile["years_experience"] != "not stated":
                st.markdown("**Experience:** " + profile["years_experience"])
            if not any([profile.get("skills"), profile.get("projects"), profile.get("certifications")]):
                st.markdown("_Nothing structured was extracted - your raw input will still be used as-is._")

        with st.status("Searching live Indeed postings via Wire...", expanded=False) as status:
            jobs, live, resolved_location, resolved_country = search_jobs(
                target_role, location=location,
                status_callback=lambda msg: status.update(label=msg),
            )
            status.update(label="Found " + str(len(jobs)) + " live postings via Wire")

        st.markdown(
            '<div class="gc-resolved-location">Searched as: <b>' + resolved_location + '</b></div>',
            unsafe_allow_html=True,
        )

        if not jobs:
            st.warning(
                "No live postings found for **" + target_role + "** in **" + resolved_location + "**.\n\n"
                "Try one of these known-good locations, or a broader role title:"
            )
            st.markdown(
                "".join('<span class="gc-pill">' + s + '</span>' for s in suggestions()),
                unsafe_allow_html=True,
            )
            st.stop()

        with st.status("Ranking postings by relevance to your skills...", expanded=False) as status:
            ranked_jobs = retrieve_relevant_jobs(
                query_text, target_role, jobs, top_k=15,
                country_domain=resolved_country,
                status_callback=lambda msg: status.update(label=msg),
            )
            status.update(label="Selected " + str(len(ranked_jobs)) + " most relevant postings")

        with st.status("Analyzing your gap against real postings...", expanded=False) as status:
            result = analyze_gap(query_text, target_role, ranked_jobs)
            status.update(label="Analysis complete")

        st.markdown('<hr class="gc-divider">', unsafe_allow_html=True)

        score = result.get("match_score", 0)
        if score >= 70:
            score_color = "#9FDDB4"
        elif score >= 40:
            score_color = "#FF9A6B"
        else:
            score_color = "#FF6B6B"

        st.markdown(
            '<div class="gc-score-card">'
            '<div class="gc-score-label">REQUIREMENT-ALIGNMENT SCORE &middot; ' + target_role.upper() + '</div>'
            '<div class="gc-score-number" style="color:' + score_color + ';">' + str(score) + '<span>/100</span></div>'
            '<div class="gc-score-summary">' + result.get("summary", "") + '</div>'
            '<div class="gc-citation">Measures alignment between your stated skills and the requirements found '
            'in the retrieved postings. Not a prediction of job-readiness, interview performance, or hiring outcome.</div>'
            '</div>',
            unsafe_allow_html=True,
        )

        if result.get("score_breakdown"):
            st.markdown('<div class="gc-section-label">WHY THIS SCORE</div>', unsafe_allow_html=True)
            rows = ""
            for b in result["score_breakdown"]:
                status = b.get("status", "partial").lower()
                if status not in STATUS_LABELS:
                    status = "partial"
                status_label = STATUS_LABELS.get(status, status)
                evidence_str = ""
                if b.get("evidence_pct") is not None:
                    evidence_str = f" &middot; Candidate evidence: {status_label} ({b['evidence_pct']}%)"
                note_html = '<div class="gc-breakdown-note">' + b.get("note", "") + evidence_str + '</div>'
                demand_line = market_demand_line(b)
                if demand_line:
                    note_html += '<div class="gc-citation">' + demand_line + '</div>'
                rows += (
                    '<div class="gc-breakdown-row">'
                    '<div><div class="gc-breakdown-area">' + b.get("area", "") + '</div>'
                    + note_html + '</div>'
                    '<span class="gc-status-badge gc-status-' + status + '">' + status_label + '</span>'
                    '</div>'
                )
            st.markdown('<div class="gc-score-card">' + rows + '</div>', unsafe_allow_html=True)

        if result.get("strengths"):
            st.markdown('<div class="gc-section-label">WHAT YOU\'VE ALREADY GOT</div>', unsafe_allow_html=True)
            pills = "".join('<span class="gc-strength-pill">' + s + '</span>' for s in result["strengths"])
            st.markdown(pills, unsafe_allow_html=True)

        if result.get("gaps"):
            st.markdown('<div class="gc-section-label">GAPS TO CLOSE</div>', unsafe_allow_html=True)
            for g in result["gaps"]:
                priority = g.get("priority", "medium")
                priority_label = PRIORITY_LABELS.get(priority, "Medium priority")
                label = g.get("skill", "Gap") + "  ·  " + priority_label
                if g.get("time_estimate"):
                    label += "  ·  ~" + g["time_estimate"]
                with st.expander(label):
                    st.write("**Why it matters:** " + g.get("why_it_matters", ""))
                    st.write("**How to fix:** " + g.get("how_to_fix", ""))
                    demand_line = market_demand_line(g)
                    if demand_line:
                        st.markdown('<div class="gc-citation">' + demand_line + '</div>', unsafe_allow_html=True)
                        for c in g.get("citations", []):
                            if c.get("url"):
                                st.markdown(
                                    '<div class="gc-citation">&rarr; <a href="' + c["url"]
                                    + '" target="_blank">' + c.get("title", "posting") + ' @ '
                                    + c.get("company", "") + '</a></div>',
                                    unsafe_allow_html=True,
                                )

        if result.get("top_3_actions"):
            st.markdown('<div class="gc-section-label">DO THIS FIRST</div>', unsafe_allow_html=True)
            for i, action in enumerate(result["top_3_actions"], 1):
                if isinstance(action, dict):
                    text = action.get("action", "")
                    time_est = action.get("time_estimate", "")
                    line = str(i) + ". " + text
                    if time_est:
                        line += "  `~" + time_est + "`"
                    st.markdown(line)
                else:
                    st.markdown(str(i) + ". " + str(action))

        if result.get("matching_jobs"):
            st.markdown('<div class="gc-section-label">LIVE POSTINGS WORTH A LOOK</div>', unsafe_allow_html=True)
            for j in result["matching_jobs"]:
                st.markdown(
                    '<div class="gc-job-row">'
                    '<div class="gc-job-title">' + j.get("title", "") + '</div>'
                    '<div class="gc-job-company">' + j.get("company", "") + ' &middot; <a href="' + j.get("url", "") + '" target="_blank">view posting &rarr;</a></div>'
                    '</div>',
                    unsafe_allow_html=True,
                )

        retrieval_method = ranked_jobs[0].get("_retrieval_method", "tfidf") if ranked_jobs else "tfidf"
        method_labels = {
            "hybrid": "hybrid: keyword + semantic",
            "embeddings": "semantic embeddings",
            "tfidf": "TF-IDF (keyword)",
        }
        method_label = method_labels.get(retrieval_method, "TF-IDF (keyword)")
        with st.expander(
            "Raw data: " + str(len(jobs)) + " live postings pulled via Wire, "
            "top " + str(len(ranked_jobs)) + " used for analysis (ranked by " + method_label + ")"
        ):
            for j in ranked_jobs:
                relevance = j.get("_relevance")
                relevance_str = " · relevance " + f"{relevance:.2f}" if relevance is not None else ""
                full_desc_str = " · full description" if j.get("_full_description_used") else ""
                st.markdown(
                    "- " + j.get("title", "") + " - " + j.get("company", "")
                    + " (" + str(j.get("date_posted", "")) + ")" + relevance_str + full_desc_str
                )

    except WireError as e:
        if _is_transient_failure(str(e)):
            st.error(
                "Wire's live scraper couldn't reach Indeed right now, even after retrying. "
                "This is usually temporary (Indeed rate-limiting the scraper) and typically "
                "clears up within a few minutes — please try again shortly."
            )
        else:
            st.error("Wire API error: " + str(e))
    except Exception as e:
        st.error("Something went wrong: " + str(e))

st.markdown('<div class="gc-footer">BUILT WITH THE ANAKIN WIRE API &middot; ANAKIN BLITZ 2026</div>', unsafe_allow_html=True)