"""
Step 7 - Streamlit dashboard for the Job Search Decision Assistant.

SETUP:
    pip install streamlit requests pandas

RUN (in a SEPARATE terminal from the FastAPI backend - both need to run at once):
    Terminal 1: uvicorn backend.main:app --reload
    Terminal 2: streamlit run frontend/app.py

The dashboard calls your FastAPI backend over HTTP (localhost:8000 by default).
Set API_BASE_URL as an environment variable if the backend runs somewhere else
(e.g. after Cloud Run deployment in Step 8).
"""

import os
import requests
import pandas as pd
import streamlit as st

API_BASE = os.environ.get("API_BASE_URL", "http://127.0.0.1:8000")

st.set_page_config(page_title="Job Search Decision Assistant", layout="wide")
st.title("Job Search Decision Assistant")
st.caption("Ranks real job postings against your profile, flags posting anomalies, and forecasts your application funnel.")


# ---------- Helpers ----------

@st.cache_data(ttl=15)
def fetch_scores(core_weight=None, learning_weight=None, exp_weight=None):
    try:
        params = {}
        if core_weight is not None:
            params["core_weight"] = core_weight
        if learning_weight is not None:
            params["learning_weight"] = learning_weight
        if exp_weight is not None:
            params["exp_weight"] = exp_weight
        r = requests.get(f"{API_BASE}/score", params=params, timeout=10)
        r.raise_for_status()
        return r.json()["jobs"], None
    except Exception as e:
        return None, str(e)

@st.cache_data(ttl=15)
def fetch_forecast(n=10):
    try:
        r = requests.get(f"{API_BASE}/forecast", params={"n": n}, timeout=10)
        r.raise_for_status()
        return r.json(), None
    except Exception as e:
        return None, str(e)

def ask_question(question):
    try:
        r = requests.post(f"{API_BASE}/ask", json={"question": question}, timeout=30)
        r.raise_for_status()
        return r.json(), None
    except requests.exceptions.HTTPError as e:
        detail = e.response.json().get("detail", str(e)) if e.response is not None else str(e)
        return None, detail
    except Exception as e:
        return None, str(e)

def submit_new_job(company, title, jd_text, location):
    try:
        r = requests.post(
            f"{API_BASE}/add_job",
            json={"company": company, "title": title, "jd_text": jd_text, "location": location},
            timeout=30,
        )
        r.raise_for_status()
        return r.json(), None
    except requests.exceptions.HTTPError as e:
        detail = e.response.json().get("detail", str(e)) if e.response is not None else str(e)
        return None, detail
    except Exception as e:
        return None, str(e)

def fetch_gap_advice(job_id):
    try:
        r = requests.post(f"{API_BASE}/skill_gap_advice", json={"job_id": job_id}, timeout=30)
        r.raise_for_status()
        return r.json(), None
    except requests.exceptions.HTTPError as e:
        detail = e.response.json().get("detail", str(e)) if e.response is not None else str(e)
        return None, detail
    except Exception as e:
        return None, str(e)

def fetch_interview_prep(job_id):
    try:
        r = requests.post(f"{API_BASE}/interview_prep", json={"job_id": job_id}, timeout=30)
        r.raise_for_status()
        return r.json(), None
    except requests.exceptions.HTTPError as e:
        detail = e.response.json().get("detail", str(e)) if e.response is not None else str(e)
        return None, detail
    except Exception as e:
        return None, str(e)


# ---------- Layout: three tabs ----------

# on_change="rerun" makes tabs track which one is open, so each tab's body can be
# gated on tab.open below - otherwise every tab's code (including its backend calls)
# runs on every rerun no matter which tab is active, which is what caused the visible
# lag/flash when interacting with widgets in a non-first tab.
tab1, tab2, tab3, tab4, tab5 = st.tabs(
    ["Ranked Jobs", "Application Forecast", "Ask a Question", "Add New Job", "Interview Prep"],
    on_change="rerun",
)


with tab1:
    if tab1.open:
        st.subheader("Jobs ranked by fit score")

        with st.expander("Tune scoring weights"):
            st.caption("Adjust how much each factor matters. These are normalized automatically, so they don't need to add up to 1.")
            col_a, col_b, col_c = st.columns(3)
            core_w = col_a.slider("Core skill match", 0.0, 1.0, 0.5, 0.05)
            learn_w = col_b.slider("Learning skill match", 0.0, 1.0, 0.25, 0.05)
            exp_w = col_c.slider("Experience fit", 0.0, 1.0, 0.25, 0.05)

        jobs, err = fetch_scores(core_w, learn_w, exp_w)
        if err:
            st.error(f"Could not load jobs: {err}\n\nIs the FastAPI backend running? (`uvicorn backend.main:app --reload`)")
        else:
            df = pd.DataFrame(jobs)
            df["anomalies"] = df["anomalies"].apply(lambda a: "; ".join(a) if a else "")
            df = df.rename(columns={
                "company": "Company", "title": "Title", "min_exp": "Min Exp",
                "max_exp": "Max Exp", "fit_score": "Fit Score", "anomalies": "Anomalies Flagged"
            })
            st.dataframe(
                df[["Company", "Title", "Min Exp", "Max Exp", "Fit Score", "Anomalies Flagged"]],
                use_container_width=True, hide_index=True
            )
            st.caption(f"Showing all {len(df)} jobs, ranked highest fit first.")

            st.divider()
            st.subheader("Skill gap advice")
            st.caption("Pick a job to see which skills are genuinely worth learning for it - Gemini filters out near-miss false positives from the raw keyword comparison.")
            job_options = {f"{j['company']} - {j['title']}": j["job_id"] for j in jobs}
            selected_label = st.selectbox("Job", options=list(job_options.keys()), key="gap_job_select")
            selected_job_id = job_options[selected_label]

            if st.session_state.get("gap_last_job_id") != selected_job_id:
                st.session_state["gap_advice_result"] = None
                st.session_state["gap_last_job_id"] = selected_job_id

            if st.button("Get Gap Advice"):
                with st.spinner("Asking Gemini to review the skill gaps..."):
                    advice_result, err = fetch_gap_advice(selected_job_id)
                if err:
                    st.session_state["gap_advice_result"] = None
                    st.error(err)
                else:
                    st.session_state["gap_advice_result"] = advice_result

            advice_result = st.session_state.get("gap_advice_result")
            if advice_result:
                advice = advice_result["advice"]
                st.info(advice["priority_suggestion"])
                col_gap, col_fp = st.columns(2)
                with col_gap:
                    st.markdown("**Genuinely worth learning:**")
                    for skill in advice.get("genuine_gaps", []):
                        st.write(f"- {skill}")
                with col_fp:
                    st.markdown("**Probably already covered (false positives filtered out):**")
                    for skill in advice.get("likely_false_positives", []):
                        st.write(f"- {skill}")


with tab2:
    if tab2.open:
        st.subheader("Application funnel & forecast")
        forecast, err = fetch_forecast()
        if err:
            st.error(f"Could not load forecast: {err}")
        else:
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("Total Applications", forecast["total_applications"])
            col2.metric("Callback Rate", f"{forecast['overall_callback_rate_pct']}%")
            col3.metric("Avg Days to Response", forecast["avg_days_to_response"])
            col4.metric("Trend", forecast["trend"].capitalize())

            st.divider()
            trend_df = pd.DataFrame({
                "Period": ["Early", "Recent"],
                "Callback Rate %": [forecast["early_period_rate_pct"], forecast["recent_period_rate_pct"]],
            })
            st.bar_chart(trend_df.set_index("Period"))

            st.info(
                f"At your current callback rate, applying to {forecast['forecast_next_n']} more "
                f"similar-quality jobs should yield approximately **{forecast['expected_callbacks_next_n']} more callbacks**."
            )
            st.caption("Note: application history is currently synthetic/illustrative data, not a real logged history.")


with tab3:
    if tab3.open:
        st.subheader("Ask a question about the job postings")
        st.caption("Powered by Gemini, which writes SQL against your BigQuery tables in real time.")
        question = st.text_input("Your question", placeholder="e.g. Which companies have inconsistent experience requirements?")
        if st.button("Ask") and question:
            with st.spinner("Asking Gemini..."):
                result, err = ask_question(question)
            if err:
                st.error(err)
            else:
                st.write(f"**Question:** {result['question']}")
                with st.expander("Generated SQL"):
                    st.code(result["generated_sql"], language="sql")
                if result["result"]:
                    st.dataframe(pd.DataFrame(result["result"]), use_container_width=True, hide_index=True)
                else:
                    st.write("No results returned.")


with tab4:
    if tab4.open:
        st.subheader("Add a new job posting")
        st.caption("Paste a job description below. Gemini will extract structured data and score it against your profile automatically.")

        with st.form("add_job_form"):
            new_company = st.text_input("Company")
            new_title = st.text_input("Job Title")
            new_location = st.text_input("Location", value="Not specified")
            new_jd_text = st.text_area("Full Job Description Text", height=250,
                                         placeholder="Paste the complete job description here...")
            submitted = st.form_submit_button("Add Job & Score It")

        if submitted:
            if not new_company or not new_title or not new_jd_text:
                st.warning("Company, Job Title, and Job Description are all required.")
            else:
                with st.spinner("Extracting structured data with Gemini and computing fit score..."):
                    result, err = submit_new_job(new_company, new_title, new_jd_text, new_location)
                if err:
                    st.error(err)
                else:
                    st.success(result["message"])
                    job_result = result["result"]
                    if job_result:
                        col1, col2 = st.columns(2)
                        col1.metric("Fit Score", job_result["fit_score"])
                        col2.metric("Experience Range", f"{job_result['min_exp']} - {job_result['max_exp']}")
                        if job_result["anomalies"]:
                            st.warning("Anomalies flagged: " + "; ".join(job_result["anomalies"]))
                        else:
                            st.info("No anomalies flagged.")
                    st.caption("Switch to the Ranked Jobs tab to see it in context with everything else.")


with tab5:
    if tab5.open:
        st.subheader("Interview prep, on demand")
        st.caption("Pick any job and generate a prep brief - likely questions, real project talking points, and a fit pitch. Works whenever you want, not tied to application status.")

        jobs, err = fetch_scores()
        if err:
            st.error(f"Could not load jobs: {err}")
        else:
            job_options = {f"{j['company']} - {j['title']}": j["job_id"] for j in jobs}
            selected_label = st.selectbox("Job", options=list(job_options.keys()), key="prep_job_select")
            selected_job_id = job_options[selected_label]

            # Clear any previously-shown result the moment the job selection changes,
            # so a new job never shows stale/overlapping content from the last one.
            if st.session_state.get("prep_last_job_id") != selected_job_id:
                st.session_state["prep_result"] = None
                st.session_state["prep_last_job_id"] = selected_job_id

            if st.button("Generate Interview Prep"):
                with st.spinner("Generating prep brief with Gemini..."):
                    prep_result, err = fetch_interview_prep(selected_job_id)
                if err:
                    st.session_state["prep_result"] = None
                    st.error(err)
                else:
                    st.session_state["prep_result"] = prep_result

            prep_result = st.session_state.get("prep_result")
            if prep_result:
                prep = prep_result["prep"]
                st.markdown(f"### {prep_result['title']} at {prep_result['company']}")

                st.markdown("**Fit pitch:**")
                st.info(prep["fit_pitch"])
                st.divider()

                st.markdown("**Likely interview questions:**")
                for i, q in enumerate(prep.get("likely_questions", []), 1):
                    st.write(f"{i}. {q}")
                st.divider()

                st.markdown("**How to frame your real projects for this role:**")
                for mapping in prep.get("star_story_mapping", []):
                    st.write(f"- {mapping}")
                st.divider()

                st.markdown("**If asked about gaps:**")
                st.write(prep.get("gap_talking_points", ""))