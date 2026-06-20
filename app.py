"""
app.py
GapCheck - skills + target role -> real gap analysis against LIVE
Indeed postings via Wire. Built for Anakin Blitz.
"""

import streamlit as st
from wire_client import search_jobs, WireError
from analyzer import analyze_gap

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
    col1, col2 = st.columns(2)
    with col1:
        target_role = st.text_input("Target role", placeholder="e.g. Backend Developer")
    with col2:
        location = st.text_input("Location", value="Bengaluru, Karnataka")
    submitted = st.form_submit_button("Run Gap Check", use_container_width=True)

if submitted:
    if not user_skills or not target_role:
        st.error("Please fill in your skills and a target role.")
        st.stop()

    try:
        with st.status("Searching live Indeed postings via Wire...", expanded=False) as status:
            jobs, live = search_jobs(target_role, location=location or "Bengaluru, Karnataka")
            status.update(label="Found " + str(len(jobs)) + " live postings via Wire")

        if not jobs:
            st.warning("No live postings found for that role/location. Try a broader role title.")
            st.stop()

        with st.status("Analyzing your gap against real postings...", expanded=False) as status:
            result = analyze_gap(user_skills, target_role, jobs)
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
            '<div class="gc-score-label">MATCH SCORE &middot; ' + target_role.upper() + '</div>'
            '<div class="gc-score-number" style="color:' + score_color + ';">' + str(score) + '<span>/100</span></div>'
            '<div class="gc-score-summary">' + result.get("summary", "") + '</div>'
            '</div>',
            unsafe_allow_html=True,
        )

        if result.get("strengths"):
            st.markdown('<div class="gc-section-label">WHAT YOU\'VE ALREADY GOT</div>', unsafe_allow_html=True)
            pills = "".join('<span class="gc-strength-pill">' + s + '</span>' for s in result["strengths"])
            st.markdown(pills, unsafe_allow_html=True)

        if result.get("gaps"):
            st.markdown('<div class="gc-section-label">GAPS TO CLOSE</div>', unsafe_allow_html=True)
            for g in result["gaps"]:
                with st.expander(g.get("skill", "Gap")):
                    st.write("**Why it matters:** " + g.get("why_it_matters", ""))
                    st.write("**How to fix:** " + g.get("how_to_fix", ""))

        if result.get("top_3_actions"):
            st.markdown('<div class="gc-section-label">DO THIS FIRST</div>', unsafe_allow_html=True)
            for i, action in enumerate(result["top_3_actions"], 1):
                st.markdown(str(i) + ". " + action)

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

        with st.expander("Raw data: " + str(len(jobs)) + " live postings pulled via Wire"):
            for j in jobs:
                st.markdown("- " + j.get("title", "") + " - " + j.get("company", "") + " (" + str(j.get("date_posted", "")) + ")")

    except WireError as e:
        st.error("Wire API error: " + str(e))
    except Exception as e:
        st.error("Something went wrong: " + str(e))

st.markdown('<div class="gc-footer">BUILT WITH THE ANAKIN WIRE API &middot; ANAKIN BLITZ 2026</div>', unsafe_allow_html=True)
