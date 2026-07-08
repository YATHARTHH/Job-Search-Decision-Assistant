"""
Step 6 - FastAPI backend for the Job Search Decision Assistant.

SETUP:
    pip install fastapi uvicorn pandas google-genai google-cloud-bigquery

RUN LOCALLY:
    cd JobSearchDecisionAssistant
    uvicorn backend.main:app --reload

Then open http://127.0.0.1:8000/docs for the interactive Swagger UI - FastAPI builds
this automatically, you can test every endpoint from your browser without writing any
client code.

ENDPOINTS:
    GET  /score     - ranked job list (Step 5 logic: fit score + anomalies)
    GET  /forecast  - application funnel summary + forecast (Step "forecast" logic)
    POST /ask       - natural language question -> Gemini writes SQL -> runs on
                      BigQuery -> returns answer (NEEDS your GEMINI_API_KEY and a
                      BigQuery-authenticated environment - see notes below /ask)
"""

import os
import ast
import json
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "..", "data")
OUTPUT_DIR = os.path.join(BASE_DIR, "..", "outputs")

app = FastAPI(title="Job Search Decision Assistant API")


# ---------- Shared helpers (same logic as step5_fit_score.py) ----------

def safe_list_parse(val):
    if pd.isna(val) or val in ("", "[]"):
        return []
    try:
        result = ast.literal_eval(val)
        return result if isinstance(result, list) else []
    except (ValueError, SyntaxError):
        return []

def skill_match_score(job_skills, profile_skills):
    if not job_skills:
        return 0.0
    job_text = " ".join(job_skills).lower()
    hits = sum(1 for s in profile_skills if s.lower() in job_text)
    return hits / len(profile_skills) if profile_skills else 0.0

def experience_fit(min_exp, max_exp, my_years):
    if pd.isna(min_exp):
        return 0.5
    lo = min_exp
    hi = max_exp if not pd.isna(max_exp) else min_exp + 3
    if lo <= my_years <= hi:
        return 1.0
    gap = min(abs(my_years - lo), abs(my_years - hi))
    return max(0.0, 1 - gap / 3)

def get_anomaly_notes(row):
    notes = []
    if row.get("error") == "could_not_parse":
        notes.append("GEMINI_PARSE_FAILED: structured extraction failed for this posting")
        return notes
    clarity = row.get("requirements_clarity", "")
    if clarity in ("vague", "missing"):
        notes.append(f"REQUIREMENTS_{clarity.upper()}: Gemini flagged this posting's requirements as {clarity}")
    notes.extend(safe_list_parse(row.get("red_flags", "[]")))
    return notes

def compute_ranked_jobs():
    structured = pd.read_csv(os.path.join(DATA_DIR, "jobs_structured.csv"))
    original = pd.read_csv(os.path.join(DATA_DIR, "job_postings.csv"))
    with open(os.path.join(DATA_DIR, "my_profile.json")) as f:
        profile = json.load(f)

    core_skills = profile["core_strength_skills"]
    learning_skills = profile["learning_in_progress_skills"]
    my_years = profile["years_experience"]

    merged = structured.merge(
        original[["job_id", "min_exp_years", "max_exp_years"]],
        on="job_id", suffixes=("_gemini", "_original")
    )
    merged["min_exp_years"] = merged["min_exp_years_gemini"].fillna(merged["min_exp_years_original"])
    merged["max_exp_years"] = merged["max_exp_years_gemini"].fillna(merged["max_exp_years_original"])

    rows = []
    for _, job in merged.iterrows():
        job_skills = safe_list_parse(job["required_skills"]) + safe_list_parse(job["tech_stack"])
        core_score = skill_match_score(job_skills, core_skills)
        learn_score = skill_match_score(job_skills, learning_skills)
        exp_score = experience_fit(job["min_exp_years"], job["max_exp_years"], my_years)
        fit_score = round((0.5 * core_score + 0.25 * learn_score + 0.25 * exp_score) * 100, 1)
        anomalies = get_anomaly_notes(job)

        rows.append({
            "job_id": int(job["job_id"]),
            "company": job["company"],
            "title": job["title"],
            "min_exp": None if pd.isna(job["min_exp_years"]) else job["min_exp_years"],
            "max_exp": None if pd.isna(job["max_exp_years"]) else job["max_exp_years"],
            "fit_score": fit_score,
            "anomalies": anomalies,
        })

    return sorted(rows, key=lambda r: r["fit_score"], reverse=True)


def compute_forecast(n_future=10):
    log = pd.read_csv(os.path.join(DATA_DIR, "applications_log.csv"), parse_dates=["applied_date"])
    log = log.sort_values("applied_date").reset_index(drop=True)

    total = len(log)
    callbacks = int((log["status"] == "callback").sum())
    rejected = int((log["status"] == "rejected").sum())
    no_response = int((log["status"] == "no_response").sum())
    overall_rate = callbacks / total if total else 0

    midpoint = total // 2
    early = log.iloc[:midpoint]
    recent = log.iloc[midpoint:]
    early_rate = (early["status"] == "callback").sum() / len(early) if len(early) else 0
    recent_rate = (recent["status"] == "callback").sum() / len(recent) if len(recent) else 0
    trend = "improving" if recent_rate > early_rate else ("declining" if recent_rate < early_rate else "flat")

    responded = log.dropna(subset=["response_date"]).copy()
    avg_lag = None
    if len(responded):
        responded["response_date"] = pd.to_datetime(responded["response_date"])
        responded["lag_days"] = (responded["response_date"] - responded["applied_date"]).dt.days
        avg_lag = round(responded["lag_days"].mean(), 1)

    return {
        "total_applications": total,
        "callbacks": callbacks,
        "rejected": rejected,
        "no_response": no_response,
        "overall_callback_rate_pct": round(overall_rate * 100, 1),
        "early_period_rate_pct": round(early_rate * 100, 1),
        "recent_period_rate_pct": round(recent_rate * 100, 1),
        "trend": trend,
        "avg_days_to_response": avg_lag,
        "forecast_next_n": n_future,
        "expected_callbacks_next_n": round(overall_rate * n_future, 1),
    }


# ---------- Endpoints ----------

@app.get("/score")
def get_score():
    """Ranked job list with fit scores and anomaly flags."""
    try:
        return {"jobs": compute_ranked_jobs()}
    except FileNotFoundError as e:
        raise HTTPException(status_code=500, detail=f"Missing data file: {e}")


@app.get("/forecast")
def get_forecast(n: int = 10):
    """Application funnel summary + forecast for the next n applications."""
    try:
        return compute_forecast(n_future=n)
    except FileNotFoundError as e:
        raise HTTPException(status_code=500, detail=f"Missing data file: {e}")


class AskRequest(BaseModel):
    question: str


@app.post("/ask")
def ask_question(request: AskRequest):
    """
    Natural language question -> Gemini writes SQL -> runs on BigQuery -> returns answer.

    REQUIRES (not needed for /score or /forecast above):
      1. pip install google-genai google-cloud-bigquery
      2. export GEMINI_API_KEY="your-key"
      3. Authenticate to GCP once: `gcloud auth application-default login`
         (or set GOOGLE_APPLICATION_CREDENTIALS to a service account JSON)
      4. Replace YOUR_PROJECT_ID below with your actual GCP project ID
    """
    try:
        from google import genai
        from google.cloud import bigquery
    except ImportError:
        raise HTTPException(
            status_code=501,
            detail="Missing packages. Run: pip install google-genai google-cloud-bigquery"
        )

    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise HTTPException(status_code=500, detail="GEMINI_API_KEY not set")

    project_id = "job-search-assistant-501709"  # <-- Harry: replace with your real GCP project ID
    dataset = "job_search_data"

    schema_description = f"""
Table `{project_id}.{dataset}.job_postings`:
  job_id, company, title, location, min_exp_years, max_exp_years, source, date_scraped, jd_text

Table `{project_id}.{dataset}.jobs_structured`:
  job_id, company, title, min_exp_years, max_exp_years, seniority,
  required_skills, tech_stack, role_focus, requirements_clarity, red_flags
"""

    client = genai.Client(api_key=api_key)
    sql_prompt = f"""You write BigQuery Standard SQL. Given this schema:
{schema_description}
Write ONE SQL query (no explanation, no markdown fences) that answers: {request.question}
"""
    try:
        response = client.models.generate_content(model="gemini-2.5-flash-lite", contents=sql_prompt)
    except Exception as e:
        error_str = str(e)
        if "RESOURCE_EXHAUSTED" in error_str or "429" in error_str:
            raise HTTPException(
                status_code=429,
                detail=(
                    "Gemini free-tier daily quota (20 requests/day for this model) is used up. "
                    "This resets the next day - try again tomorrow, or switch to a different "
                    "Gemini model (e.g. gemini-2.5-flash-lite) which has a separate quota."
                )
            )
        raise HTTPException(status_code=500, detail=f"Gemini API error: {error_str}")

    sql = response.text.strip().replace("```sql", "").replace("```", "").strip()

    bq_client = bigquery.Client(project=project_id)
    try:
        result_df = bq_client.query(sql).to_dataframe()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"BigQuery error: {e}\nGenerated SQL was: {sql}")

    return {
        "question": request.question,
        "generated_sql": sql,
        "result": result_df.to_dict(orient="records"),
    }