# PPT Outline - Job Search Decision Assistant
(Slide-by-slide content. Build in PowerPoint/Google Slides using this as the script.)

---

## Slide 1 - Title
**Job Search Decision Assistant**
An AI-powered decision-intelligence tool for job seekers
[Your name] | [Hackathon name] | [Date]

---

## Slide 2 - The Problem
- Job seekers evaluate dozens of postings with no way to rank true fit
- Job descriptions often contain hidden inconsistencies (contradictory experience
  requirements, mismatched metadata, vague/generic postings)
- No visibility into whether your application strategy is actually working
- **Real user, real problem:** built for my own active GenAI/ML job search

---

## Slide 3 - What We Built
One-line: A tool that ranks real job postings against your profile, flags posting
anomalies using Gemini, and forecasts your application outcomes - all backed by a
GCP + NVIDIA-accelerated pipeline.

Four outputs (matches the "useful output" rubric directly):
1. Fit-score ranking
2. Anomaly flags (Gemini-detected, not rule-based)
3. Application forecast
4. Natural language Q&A over the data

---

## Slide 4 - Architecture Diagram
[Insert diagram: Cloud Storage -> cudf.pandas cleaning -> Gemini extraction ->
BigQuery -> FastAPI (/score, /forecast, /ask) -> Streamlit dashboard -> Cloud Run]

Call out on this slide:
- GCP layer used: Cloud Storage, BigQuery, Cloud Run (3, exceeds the "2+" requirement)
- NVIDIA layer used: cudf.pandas, cupy (RAPIDS)
- LLM/RAG layer: Gemini (structured extraction + NL-to-SQL)

---

## Slide 5 - Real Data, Not Toy Data
- 20 real, currently-live job postings (Deloitte, KPMG, Accenture, Cognite, etc.)
  collected directly from LinkedIn/company career pages
- Verified live at collection time (one dead link caught and excluded during collection)
- Real profile matching against actual backend/GenAI-transition skill set

---

## Slide 6 - Gemini Found Things a Human Skim-Read Would Miss
Show 2-3 concrete examples (real, from your actual data):
- Accenture in India: contradictory experience requirement ("5 years" vs "3 years"
  in the same posting)
- Infosys: mismatched metadata tags ("Turbomachinery -> Steam Turbine -> Rotor" on
  an AI Engineer posting)
- LTM: correctly identified as a generic recruiter message, not a real JD

This is the "identify patterns and anomalies" rubric requirement, demonstrated with
real, verifiable examples - not a hypothetical.

---

## Slide 7 - Acceleration: An Honest Two-Part Story
**This slide is your strongest differentiator - most teams will show one cherry-picked
speedup number. You tested, found a negative result, understood why, and adapted.**

| Workload | CPU | GPU | Result |
|---|---|---|---|
| Keyword/string matching, 200k rows | 13.4-24.5s | 27-36s | GPU SLOWER (real finding) |
| Vector similarity, 1M embeddings | 8.4s | 0.20s | GPU 41x FASTER |

Takeaway: acceleration isn't automatic - it depends on matching the right workload
(large-scale numeric computation) to the right hardware. This finding directly
shaped a better product design (embeddings-based similarity over naive keyword match).

---

## Slide 8 - Live Demo Screenshots
- Ranked job table (Streamlit)
- Forecast chart + metrics
- /ask natural language query + generated SQL + result

---

## Slide 9 - What's Next / Future Enhancements
- Real embeddings (not simulated) for production semantic similarity scoring
- Retry logic for Gemini JSON parse failures (currently 1/20 postings, known limitation)
- Real application-history logging (currently synthetic/illustrative for demo)
- Multi-user support, browser extension for one-click JD capture

---

## Slide 10 - Thank You / Links
- GitHub repo: [link]
- Demo video: [link]
- Deployment: [link, or note "deployment-ready, config included, not yet live - see README"]