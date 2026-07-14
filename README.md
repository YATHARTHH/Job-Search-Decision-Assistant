# Job Search Decision Assistant

An AI-powered decision-intelligence tool that ranks real job postings against a
candidate's profile, flags posting anomalies using Gemini, and forecasts application
outcomes. Built for Hackathon under the "AI for Better Living / Decision
Intelligence" track.

## The Problem

Job seekers evaluate dozens of postings with no reliable way to know which ones
truly fit their profile, spend hours re-reading job descriptions, and have no
visibility into whether their search strategy is actually working. This tool
addresses a real, personally-experienced problem: prioritizing which GenAI/ML job
postings to apply to during an active job search.

## What It Does

1. **Ranks job postings by fit score** - combining skill match and experience fit
2. **Flags posting anomalies** using Gemini's reasoning (not rule-based) - catches
   things like contradictory experience requirements or mismatched metadata
3. **Forecasts application outcomes** - callback rate trends and projections
4. **Answers natural language questions** - Gemini writes SQL against BigQuery in
   real time
5. **Adds new job postings on demand** - paste a JD, Gemini extracts structured
   data and scores it against your profile immediately
6. **Gives skill-gap advice** - Gemini reviews the raw keyword-matched skill gaps
   for a job and filters out false positives, returning what's genuinely worth
   learning
7. **Generates interview prep briefs** - likely questions, real-project talking
   points, and a fit pitch tailored to a specific job posting
8. **Semantic fit scoring** - compares profile and job postings by embedding
   similarity (meaning), not just keyword overlap

## Architecture

```
Job postings (CSV) --> Cloud Storage
                            |
                     cudf.pandas cleaning (GPU-accelerated numeric ops)
                            |
                     Gemini structured extraction
                            |
                        BigQuery
                            |
     FastAPI (/score, /forecast, /ask, /add_job,
      /skill_gap_advice, /interview_prep)
                            |
                   Streamlit dashboard
                            |
                       Cloud Run (deployment-ready)
```

**GCP services used:** Cloud Storage, BigQuery, Cloud Run
**NVIDIA acceleration used:** cudf.pandas, cupy (RAPIDS)
**LLM/RAG:** Gemini (structured JD extraction + natural-language-to-SQL)

## Real Data

20 real, currently-active job postings collected from LinkedIn and company career
pages (Deloitte, KPMG, Accenture, Cognite, and others), covering GenAI/ML Engineer
roles. Verified live at collection time.

## The Acceleration Finding (honest, tested, not assumed)

We benchmarked two different workloads on CPU vs. GPU (cudf.pandas/cupy):

| Workload | CPU | GPU | Result |
|---|---|---|---|
| Keyword/string matching, 200k rows | ~13-24s | ~27-36s | GPU slower |
| Vector similarity, 1M embeddings | ~8.4s | ~0.20s | GPU 41x faster |

**Finding:** GPU acceleration isn't automatic - it depends on matching the right
workload (large-scale numeric computation) to the right hardware. String/keyword
search at this scale is dominated by kernel-launch and data-transfer overhead; large
matrix/vector math is exactly what GPUs are built for. This finding directly shaped
the production design choice to move toward embeddings-based similarity scoring.

## Known Limitations (disclosed intentionally)

- 1 of 20 job postings (Persistent Systems) failed Gemini's structured extraction
  due to a malformed JSON response - kept in the dataset as an honest example of
  real-world LLM output variance, not silently dropped
- Application history data is synthetic/illustrative (no real logged history existed
  yet at build time) - the forecast *feature* is real and functional, the specific
  historical numbers are demonstration data
- Embeddings used in the acceleration benchmark are randomly generated, isolating
  "speed of the similarity math" from "speed of generating real embeddings" (a
  separate, API-bound cost)

## Project Structure

```
JobSearchDecisionAssistant/
├── data/                   Real + synthetic datasets, profile, structured Gemini output
├── scripts/                Step 1-5 + benchmark + forecast scripts
├── backend/main.py         FastAPI app (/score, /forecast, /ask, /add_job,
│                            /skill_gap_advice, /interview_prep)
├── frontend/app.py         Streamlit dashboard
├── outputs/                Ranked results from each step
├── docs/                   This README, PPT outline, project status tracker
├── Dockerfile              Container config (Cloud Run / AWS / any Docker host)
├── start.sh                Container startup script (runs backend + frontend together)
└── requirements.txt        Python dependencies
```

## Running Locally

```bash
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt

cp .env.example .env  # then fill in GEMINI_API_KEY

# Terminal 1
uvicorn backend.main:app --reload

# Terminal 2
streamlit run frontend/app.py
```

Requires a `GEMINI_API_KEY` (free at aistudio.google.com) for `/add_job`,
`/skill_gap_advice`, `/interview_prep`, and `/ask`, plus a GCP project with
BigQuery enabled specifically for `/ask`. `/score` and `/forecast` work with no
external credentials. The key is loaded from `.env` via `python-dotenv` - it is
gitignored, so it's never committed.

## Deployment

Container is deployment-ready (Dockerfile + start.sh included) for Cloud Run, AWS
(App Runner/ECS/Fargate), or any Docker-compatible host.

## Tech Stack

Python, FastAPI, Streamlit, pandas, cudf.pandas, cupy, Google Gemini API,
Google BigQuery, Docker
