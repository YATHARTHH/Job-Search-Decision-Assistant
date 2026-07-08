"""
Step 2 - Gemini structured extraction (run this on YOUR laptop, not in Claude's sandbox)

SETUP (one-time):
1. Go to https://aistudio.google.com/apikey and create a free API key.
2. In your terminal:
       python -m venv venv
       venv\\Scripts\\activate
       pip install google-genai pandas
3. Set your key:  set GEMINI_API_KEY=your-key-here
4. Run:           python scripts/step2_gemini_extract.py

WHAT THIS DOES:
Reads job_postings.csv, sends each jd_text to Gemini, gets back structured JSON,
and saves to jobs_structured.csv. Resumes from where it left off on re-run.

CHANGELOG (Harry's fixes over the original version):
1. google.generativeai -> google.genai (old package deprecated/dead)
2. gemini-2.0-flash -> gemini-2.5-flash (2.0-flash hit exhausted daily quota)
3. sleep(1) -> sleep(15) (free tier cap is 5 req/min, need >=12s between calls)
4. Retry on 503 (server overload)
5. Retry on 429 (rate limit)
6. Retry on network errors (wifi drops)
7. Incremental save after each job (previously all progress lost on crash)
8. Resume from last saved job on re-run (skip already-completed job_ids)
9. Fixed column order on append (previously caused NaN job_ids from misaligned columns)

KNOWN LIMITATION: "done" = a row with this job_id already exists in the output file.
This includes rows that failed to parse (error == "could_not_parse"), so those are
NOT automatically retried on re-run. Left as-is intentionally for now.
"""

import os
import json
import time
import pandas as pd
from google import genai
from google.genai import errors as genai_errors
import httpx

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "..", "data")

API_KEY = os.environ.get("GEMINI_API_KEY")
if not API_KEY:
    raise SystemExit(
        "GEMINI_API_KEY not set. Run: set GEMINI_API_KEY=your-key-here "
        "(get a free key at https://aistudio.google.com/apikey)"
    )

client = genai.Client(api_key=API_KEY)

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
  "red_flags": [<list of short strings for anything inconsistent or suspicious, e.g. 'says 3-4 years but lists 7+ years worth of tools', empty list if none>]
}}

Job Description:
{jd_text}
"""

def extract_one(jd_text, retries=4):
    prompt = EXTRACTION_PROMPT.format(jd_text=jd_text)
    for attempt in range(retries):
        try:
            response = client.models.generate_content(model="gemini-2.5-flash", contents=prompt)
            raw = response.text.strip()
            raw = raw.replace("```json", "").replace("```", "").strip()
            try:
                return json.loads(raw)
            except json.JSONDecodeError:
                return {"error": "could_not_parse", "raw_response": raw}
        except genai_errors.ServerError:
            wait = 30 * (attempt + 1)
            print(f"  503 server error, retrying in {wait}s... (attempt {attempt+1}/{retries})")
            time.sleep(wait)
        except genai_errors.ClientError as e:
            if "429" in str(e):
                wait = 60
                print(f"  Rate limited, waiting {wait}s...")
                time.sleep(wait)
            else:
                raise
        except (httpx.ConnectError, httpx.TimeoutException, OSError) as e:
            wait = 30 * (attempt + 1)
            print(f"  Network error ({e}), retrying in {wait}s... (attempt {attempt+1}/{retries})")
            time.sleep(wait)
    return {"error": "failed_after_retries"}

COLUMNS = ["job_id", "company", "title", "min_exp_years", "max_exp_years",
           "seniority", "required_skills", "tech_stack", "role_focus",
           "requirements_clarity", "red_flags", "error", "raw_response"]

def main():
    out_path = os.path.join(DATA_DIR, "jobs_structured.csv")
    jobs = pd.read_csv(os.path.join(DATA_DIR, "job_postings.csv"))

    # Resume: skip already-processed job_ids (drop nan rows from corrupt prior runs)
    # FIX: rows that failed to parse (error == "could_not_parse") are now excluded
    # from done_ids so they get RETRIED on re-run, instead of being permanently skipped.
    done_ids = set()
    if os.path.exists(out_path):
        existing = pd.read_csv(out_path)
        existing = existing.dropna(subset=["job_id"])
        failed_mask = existing["error"] == "could_not_parse"
        retry_ids = set(existing.loc[failed_mask, "job_id"].astype(int).tolist())
        existing = existing.loc[~failed_mask]  # drop failed rows - they'll be re-appended if retried
        done_ids = set(existing["job_id"].astype(int).tolist())
        existing.to_csv(out_path, index=False)  # rewrite clean file without nan/failed rows
        if retry_ids:
            print(f"Will retry {len(retry_ids)} previously-failed job(s): {sorted(retry_ids)}")
        print(f"Resuming — {len(done_ids)} jobs already done: {sorted(done_ids)}")

    for _, job in jobs.iterrows():
        if int(job["job_id"]) in done_ids:
            continue
        print(f"Extracting job_id={job['job_id']} ({job['company']})...")
        extracted = extract_one(job["jd_text"])
        extracted["job_id"] = int(job["job_id"])
        extracted["company"] = job["company"]
        extracted["title"] = job["title"]

        # Save with fixed column order so appended rows always align
        row_df = pd.DataFrame([extracted]).reindex(columns=COLUMNS)
        write_header = not os.path.exists(out_path)
        row_df.to_csv(out_path, mode="a", header=write_header, index=False)
        print(f"  Saved.")

        time.sleep(15)  # gemini-2.5-flash free tier: 5 req/min → 15s between calls

    total = len(pd.read_csv(out_path))
    print(f"\nDone. {total} structured rows in {out_path}")

if __name__ == "__main__":
    main()