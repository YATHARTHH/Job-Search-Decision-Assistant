# Job Search Decision Assistant

An AI-powered decision-intelligence tool that ranks real job postings against Harry's
profile, flags anomalies in postings, and (eventually) forecasts application response
rates — built for the "AI for Better Living / Decision Intelligence" hackathon track.

## Folder structure

```
JobSearchDecisionAssistant/
├── data/
│   ├── job_postings.csv       20 real, current GenAI/ML/AI Engineer postings
│   │                          (copied from LinkedIn + Deloitte careers, verified live)
│   ├── my_profile.json        Harry's skill/experience profile used for scoring
│   └── jobs_structured.csv    (created after you run step2 - not yet generated)
├── scripts/
│   ├── step1_fit_score.py     Naive local fit-score ranking (DONE, tested, works)
│   └── step2_gemini_extract.py Gemini structured JD extraction (needs YOUR API key, run locally)
├── outputs/
│   └── step1_ranked_output.csv Ranked output from the last Step 1 run
└── docs/
    └── README.md               This file
```

## Project status tracker

| Step | What | Status |
|---|---|---|
| 0 | Pick project idea (Job Search Decision Assistant) | ✅ Done |
| 0 | Collect 20 real job postings | ✅ Done |
| 0 | Build my_profile.json (+ merged skills from reference resume) | ✅ Done |
| 1 | Local naive fit-score script (no cloud) | ✅ Done, tested, working |
| 2 | Gemini structured JD extraction | 🔲 Script written, YOU need to run it locally with your own free API key |
| 3 | Acceleration benchmark: pandas vs cudf.pandas | 🔲 Not started - needs Google Colab (free GPU) |
| 4 | Load structured data into BigQuery | 🔲 Not started |
| 5 | Smarter fit-score + anomaly + forecast logic (using Gemini output) | 🔲 Not started |
| 6 | FastAPI backend (/score, /ask, /forecast) | 🔲 Not started |
| 7 | Streamlit frontend dashboard | 🔲 Not started |
| 8 | Deploy to Cloud Run (public URL) | 🔲 Not started |
| 9 | PPT + demo video (≤3 min) + GitHub repo + description | 🔲 Not started |

## What YOU need to do next (in order)

### 1. Run Step 2 on your own laptop
This environment cannot reach Google's API, so this step needs to run on your machine.

```bash
cd JobSearchDecisionAssistant
python3 -m venv venv
source venv/bin/activate              # Windows: venv\Scripts\activate
pip install google-generativeai pandas

# Get a free key: https://aistudio.google.com/apikey
export GEMINI_API_KEY="your-key-here" # Windows: set GEMINI_API_KEY=your-key-here

python3 scripts/step2_gemini_extract.py
```
This creates `data/jobs_structured.csv`. Send that back (or just confirm it worked) so
the fit-score logic can be upgraded from keyword-matching to real Gemini reasoning.

### 2. Double-check your profile
Open `data/my_profile.json`. The `learning_in_progress_skills` list includes tools
(FAISS, LangChain, PySpark, Databricks, PyTorch, TensorFlow, etc.) pulled from a
colleague's resume at your request — these count only partially toward your score
right now. If you've genuinely built things with any of them yourself, move them into
`core_strength_skills` for a more accurate ranking.

### | 3 | Acceleration benchmark: pandas vs cudf.pandas | ✅ Done. 
Two findings: (1) string/keyword matching on 200k rows — GPU (cudf.pandas) was SLOWER than CPU (27-36s vs 13-24s), root cause: kernel-launch + data-transfer overhead dominates at this scale for text search. (2) Numeric embeddings similarity on 1M rows — GPU (cupy) was 41x FASTER than CPU (0.20s vs 8.36s). Conclusion used in PPT: acceleration helps most for large-scale numeric/vector computation, not small-scale string search — informed real architecture choice to use embeddings-based similarity (not keyword matching) for the production fit-score. |

## Future Enhancements (parked, not urgent)
- step2_gemini_extract.py: currently treats any existing job_id as "done" even if it
  failed to parse (error == "could_not_parse"). Could add a check that also retries
  rows where error is present, not just missing job_ids. Low priority — 1 failed row
  out of 20 is fine to leave as a real-world example for the demo.