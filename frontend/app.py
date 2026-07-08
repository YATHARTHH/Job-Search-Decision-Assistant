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

def fetch_scores():
    try:
        r = requests.get(f"{API_BASE}/score", timeout=10)
        r.raise_for_status()
        return r.json()["jobs"], None
    except Exception as e:
        return None, str(e)

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


# ---------- Layout: three tabs ----------

tab1, tab2, tab3 = st.tabs(["Ranked Jobs", "Application Forecast", "Ask a Question"])


with tab1:
    st.subheader("Jobs ranked by fit score")
    jobs, err = fetch_scores()
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


with tab2:
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