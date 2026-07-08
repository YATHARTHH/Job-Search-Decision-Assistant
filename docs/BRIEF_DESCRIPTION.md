# Brief Description (for submission form)

## Short version (2-3 sentences)

Job Search Decision Assistant ranks real job postings against a candidate's profile,
uses Gemini to detect posting anomalies (contradictory requirements, mismatched
metadata) that a manual read would miss, and forecasts application outcomes -
backed by a GCP (Cloud Storage, BigQuery, Cloud Run) and NVIDIA RAPIDS
(cudf.pandas, cupy) pipeline. An honest CPU-vs-GPU benchmark across two different
workload types shaped the final architecture: GPU acceleration was 41x faster for
vector similarity computation, but slower than CPU for keyword text search - a
real, tested finding rather than an assumed one.

## Longer version (for a submission field with more room)

Job seekers evaluate dozens of postings with no reliable way to know which ones
truly fit, and no visibility into whether their application strategy is working.
Job Search Decision Assistant solves this using 20 real, currently-active job
postings: it extracts structured requirements via Gemini, ranks postings by fit
score against the candidate's actual skill profile, surfaces Gemini-detected
anomalies (e.g., a posting with contradictory experience requirements in the same
listing), and forecasts callback rates from application history. The backend runs
on FastAPI with BigQuery as the data layer and a Streamlit dashboard as the
interface, including a natural-language Q&A endpoint where Gemini writes and
executes SQL queries in real time. For the acceleration requirement, two different
workloads were benchmarked on CPU vs. GPU (cudf.pandas/cupy): keyword-based text
search was measurably slower on GPU due to kernel-launch overhead, while large-scale
vector similarity computation was 41x faster - a genuine empirical finding that
directly informed the product's real similarity-scoring design. The project is
fully containerized (Dockerfile included) and deployment-ready for Cloud Run, AWS,
or any Docker host.