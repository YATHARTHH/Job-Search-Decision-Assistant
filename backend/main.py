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
    POST /add_job   - add a new job posting -> auto-runs Gemini extraction on it ->
                      returns its fit score (NEEDS your GEMINI_API_KEY)
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

def find_skill_gaps(job_skills, profile_skills):
    """Returns the job's required skills that DON'T appear anywhere in the profile's
    skill list. This is the reverse direction of skill_match_score - that function
    asks 'what fraction of MY skills does this job mention', this asks 'which of
    THIS JOB's specific requirements am I missing'. Both are useful, different
    questions - this one directly answers 'what should I learn for this role?'"""
    if not job_skills:
        return []
    profile_skills_lower = [p.lower().strip() for p in profile_skills if p.strip()]
    gaps = []
    for skill in job_skills:
        skill_lower = skill.lower().strip()
        if not skill_lower:
            continue
        # Covered if this job skill contains a profile skill, or a profile skill
        # contains this job skill (handles "Python" vs "Python programming" either way)
        covered = any(
            p in skill_lower or skill_lower in p
            for p in profile_skills_lower
        )
        if not covered:
            gaps.append(skill)
    return gaps

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

def compute_ranked_jobs(core_weight=None, learning_weight=None, exp_weight=None):
    structured = pd.read_csv(os.path.join(DATA_DIR, "jobs_structured.csv"))
    original = pd.read_csv(os.path.join(DATA_DIR, "job_postings.csv"))
    with open(os.path.join(DATA_DIR, "my_profile.json")) as f:
        profile = json.load(f)

    core_skills = profile["core_strength_skills"]
    learning_skills = profile["learning_in_progress_skills"]
    my_years = profile["years_experience"]

    # Use provided weights if given (e.g. from Streamlit sliders), else fall back
    # to profile.json's configured defaults, else the original hardcoded values.
    default_weights = profile.get("scoring_weights", {})
    w_core = core_weight if core_weight is not None else default_weights.get("core_skill_weight", 0.5)
    w_learn = learning_weight if learning_weight is not None else default_weights.get("learning_skill_weight", 0.25)
    w_exp = exp_weight if exp_weight is not None else default_weights.get("experience_fit_weight", 0.25)
    weight_sum = w_core + w_learn + w_exp
    if weight_sum > 0:
        w_core, w_learn, w_exp = w_core / weight_sum, w_learn / weight_sum, w_exp / weight_sum

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
        fit_score = round((w_core * core_score + w_learn * learn_score + w_exp * exp_score) * 100, 1)
        anomalies = get_anomaly_notes(job)
        skill_gaps = find_skill_gaps(job_skills, core_skills + learning_skills)

        rows.append({
            "job_id": int(job["job_id"]),
            "company": job["company"],
            "title": job["title"],
            "min_exp": None if pd.isna(job["min_exp_years"]) else job["min_exp_years"],
            "max_exp": None if pd.isna(job["max_exp_years"]) else job["max_exp_years"],
            "fit_score": fit_score,
            "anomalies": anomalies,
            "skill_gaps": skill_gaps,
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
def get_score(core_weight: float = None, learning_weight: float = None, exp_weight: float = None):
    """Ranked job list with fit scores and anomaly flags.
    Optional query params let you override the default weights from profile.json,
    e.g. /score?core_weight=0.7&learning_weight=0.1&exp_weight=0.2"""
    try:
        return {"jobs": compute_ranked_jobs(core_weight, learning_weight, exp_weight)}
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

    project_id = "YOUR_PROJECT_ID"  # <-- Harry: replace with your real GCP project ID
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
        response = client.models.generate_content(model="gemini-2.5-flash", contents=sql_prompt)
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


class AddJobRequest(BaseModel):
    company: str
    title: str
    jd_text: str
    location: str = "Not specified"
    source: str = "Manual entry"


EXTRACTION_PROMPT = """You are extracting structured data from a job description.
Return ONLY valid JSON, no preamble, no markdown code fences, no explanation.

Schema:
{{
  "min_exp_years": <number or null>,
  "max_exp_years": <number or null>,
  "seniority": "<junior|mid|senior|unclear>",
  "required_skills": [<list of specific technical skills mentioned>],
  "tech_stack": [<list of specific tools/frameworks/platforms mentioned>],
  "role_focus": "<one phrase: e.g. 'LLM application development', 'MLOps', 'computer vision'>",
  "requirements_clarity": "<clear|vague|missing>",
  "red_flags": [<list of short strings for anything inconsistent or suspicious, empty list if none>]
}}

Job Description:
{jd_text}
"""


@app.post("/add_job")
def add_job(request: AddJobRequest):
    """
    Add a new job posting: appends to job_postings.csv, runs Gemini extraction on
    it immediately, appends the result to jobs_structured.csv, and returns the new
    job's fit score using the current scoring weights.

    REQUIRES (same as /ask):
      pip install google-genai
      export GEMINI_API_KEY="your-key"
    """
    try:
        from google import genai
    except ImportError:
        raise HTTPException(
            status_code=501,
            detail="Missing package. Run: pip install google-genai"
        )

    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise HTTPException(status_code=500, detail="GEMINI_API_KEY not set")

    postings_path = os.path.join(DATA_DIR, "job_postings.csv")
    structured_path = os.path.join(DATA_DIR, "jobs_structured.csv")

    postings = pd.read_csv(postings_path)
    new_job_id = int(postings["job_id"].max()) + 1

    # 1. Append to job_postings.csv
    new_row = {
        "job_id": new_job_id, "company": request.company, "title": request.title,
        "location": request.location, "min_exp_years": None, "max_exp_years": None,
        "source": request.source, "date_scraped": pd.Timestamp.now().strftime("%Y-%m-%d"),
        "jd_text": request.jd_text,
    }
    postings = pd.concat([postings, pd.DataFrame([new_row])], ignore_index=True)
    postings.to_csv(postings_path, index=False)

    # 2. Run Gemini extraction on it, same prompt/logic as step2_gemini_extract.py
    client = genai.Client(api_key=api_key)
    try:
        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=EXTRACTION_PROMPT.format(jd_text=request.jd_text)
        )
        raw = response.text.strip().replace("```json", "").replace("```", "").strip()
        extracted = json.loads(raw)
    except json.JSONDecodeError:
        extracted = {"error": "could_not_parse", "raw_response": raw}
    except Exception as e:
        error_str = str(e)
        if "RESOURCE_EXHAUSTED" in error_str or "429" in error_str:
            raise HTTPException(
                status_code=429,
                detail="Gemini free-tier daily quota used up. Job was added to job_postings.csv "
                       "but NOT yet extracted - re-run extraction later with step2_gemini_extract.py."
            )
        raise HTTPException(status_code=500, detail=f"Gemini API error: {error_str}")

    extracted["job_id"] = new_job_id
    extracted["company"] = request.company
    extracted["title"] = request.title

    # 3. Append to jobs_structured.csv
    columns = ["job_id", "company", "title", "min_exp_years", "max_exp_years",
               "seniority", "required_skills", "tech_stack", "role_focus",
               "requirements_clarity", "red_flags", "error", "raw_response"]
    row_df = pd.DataFrame([extracted]).reindex(columns=columns)
    write_header = not os.path.exists(structured_path)
    row_df.to_csv(structured_path, mode="a", header=write_header, index=False)

    # 4. Return this job's fit score using current ranking logic
    all_jobs = compute_ranked_jobs()
    new_job_result = next((j for j in all_jobs if j["job_id"] == new_job_id), None)

    return {
        "job_id": new_job_id,
        "message": f"Job added and extracted (rank {[j['job_id'] for j in all_jobs].index(new_job_id) + 1} of {len(all_jobs)})",
        "result": new_job_result,
    }


class GapAdviceRequest(BaseModel):
    job_id: int


@app.post("/skill_gap_advice")
def skill_gap_advice(request: GapAdviceRequest):
    """
    Takes the raw skill_gaps list (from /score) for a given job_id and has Gemini
    clean it up: remove near-miss false positives (e.g. "professional software
    engineering" when the profile already shows general SWE experience under a
    different label), and return a short, prioritized, actionable suggestion of
    what's genuinely worth learning for this specific role.

    REQUIRES: pip install google-genai, export GEMINI_API_KEY
    """
    try:
        from google import genai
    except ImportError:
        raise HTTPException(status_code=501, detail="Missing package. Run: pip install google-genai")

    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise HTTPException(status_code=500, detail="GEMINI_API_KEY not set")

    all_jobs = compute_ranked_jobs()
    job = next((j for j in all_jobs if j["job_id"] == request.job_id), None)
    if not job:
        raise HTTPException(status_code=404, detail=f"job_id {request.job_id} not found")

    with open(os.path.join(DATA_DIR, "my_profile.json")) as f:
        profile = json.load(f)

    prompt = f"""A candidate has this background:
Core skills: {', '.join(profile['core_strength_skills'])}
Currently learning: {', '.join(profile['learning_in_progress_skills'])}

For the job "{job['title']}" at {job['company']}, a raw keyword-matching script
flagged these as potentially missing skills:
{', '.join(job['skill_gaps']) if job['skill_gaps'] else 'none flagged'}

Some of these may be FALSE POSITIVES - things the candidate likely already has
under a different name or as a natural extension of their existing skills (e.g.
if they have "System Design" and "Microservices", don't flag "software architecture"
as missing). Return ONLY valid JSON, no markdown fences:
{{
  "genuine_gaps": [<short list of skills that are ACTUALLY worth learning for this role>],
  "likely_false_positives": [<skills from the raw list that the candidate probably already covers>],
  "priority_suggestion": "<one sentence: what to focus on first, and why>"
}}
"""

    client = genai.Client(api_key=api_key)
    try:
        response = client.models.generate_content(model="gemini-2.5-flash", contents=prompt)
        raw = response.text.strip().replace("```json", "").replace("```", "").strip()
        advice = json.loads(raw)
    except json.JSONDecodeError:
        raise HTTPException(status_code=500, detail=f"Gemini returned unparseable JSON: {raw}")
    except Exception as e:
        error_str = str(e)
        if "RESOURCE_EXHAUSTED" in error_str or "429" in error_str:
            raise HTTPException(status_code=429, detail="Gemini free-tier daily quota used up. Try again tomorrow.")
        raise HTTPException(status_code=500, detail=f"Gemini API error: {error_str}")

    return {
        "job_id": request.job_id,
        "company": job["company"],
        "title": job["title"],
        "raw_gap_count": len(job["skill_gaps"]),
        "advice": advice,
    }


class InterviewPrepRequest(BaseModel):
    job_id: int


@app.post("/interview_prep")
def interview_prep(request: InterviewPrepRequest):
    """
    Generates an on-demand interview prep brief for any job_id: likely technical
    questions, STAR-story angles mapping the candidate's REAL past projects to this
    role's requirements, and a short "why you're a fit" pitch.

    REQUIRES: pip install google-genai, export GEMINI_API_KEY
    """
    try:
        from google import genai
    except ImportError:
        raise HTTPException(status_code=501, detail="Missing package. Run: pip install google-genai")

    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise HTTPException(status_code=500, detail="GEMINI_API_KEY not set")

    all_jobs = compute_ranked_jobs()
    job = next((j for j in all_jobs if j["job_id"] == request.job_id), None)
    if not job:
        raise HTTPException(status_code=404, detail=f"job_id {request.job_id} not found")

    structured = pd.read_csv(os.path.join(DATA_DIR, "jobs_structured.csv"))
    job_row = structured[structured["job_id"] == request.job_id].iloc[0]

    with open(os.path.join(DATA_DIR, "my_profile.json")) as f:
        profile = json.load(f)

    projects_text = "\n".join(
        f"- {p['title']}: {p['summary']} (demonstrates: {', '.join(p['skills_demonstrated'])})"
        for p in profile.get("notable_projects", [])
    )

    prompt = f"""A candidate is preparing for an interview for "{job['title']}" at {job['company']}.

Job's required skills: {job_row.get('required_skills', 'unknown')}
Job's tech stack: {job_row.get('tech_stack', 'unknown')}
Job's role focus: {job_row.get('role_focus', 'unknown')}
Job's seniority level: {job_row.get('seniority', 'unknown')}

Candidate's real past projects:
{projects_text}

Candidate's core skills: {', '.join(profile['core_strength_skills'])}
Skills the candidate is currently missing for this specific role: {', '.join(job['skill_gaps'][:8])}

Generate an interview prep brief. Return ONLY valid JSON, no markdown fences:
{{
  "likely_questions": [<5 specific technical/behavioral questions this interviewer would plausibly ask, based on the actual job requirements>],
  "star_story_mapping": [<for 2-3 of the candidate's real projects above, one sentence on how to frame it for THIS specific role>],
  "gap_talking_points": "<one sentence on how to honestly address the missing skills if asked, without underselling the candidate>",
  "fit_pitch": "<a 2-3 sentence 'why I'm a strong fit' pitch specific to this role, using the candidate's real background>"
}}
"""

    client = genai.Client(api_key=api_key)
    try:
        response = client.models.generate_content(model="gemini-2.5-flash", contents=prompt)
        raw = response.text.strip().replace("```json", "").replace("```", "").strip()
        prep = json.loads(raw)
    except json.JSONDecodeError:
        raise HTTPException(status_code=500, detail=f"Gemini returned unparseable JSON: {raw}")
    except Exception as e:
        error_str = str(e)
        if "RESOURCE_EXHAUSTED" in error_str or "429" in error_str:
            raise HTTPException(status_code=429, detail="Gemini free-tier daily quota used up. Try again tomorrow.")
        raise HTTPException(status_code=500, detail=f"Gemini API error: {error_str}")

    return {
        "job_id": request.job_id,
        "company": job["company"],
        "title": job["title"],
        "prep": prep,
    }