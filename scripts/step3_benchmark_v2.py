"""
Step 3b v2 - Acceleration benchmark, redesigned after v1 showed GPU slower than CPU.

WHAT CHANGED FROM v1 AND WHY:
1. Added a WARM-UP pass (untimed) before the real timed run. This forces CUDA
   initialization/compilation to happen before we start the clock, so we measure
   actual processing speed, not one-time GPU startup cost.
2. Replaced 20 separate str.contains() calls (20 separate GPU kernel launches)
   with ONE combined regex pattern checked in a single str.count() call. Same
   underlying work (counting skill-keyword occurrences), far fewer kernel launches.

Run locally (plain pandas):
    python3 scripts/step3_benchmark_v2.py

Run in Colab (cudf.pandas):
    cell 1: %load_ext cudf.pandas
    cell 2: paste this entire file and run it
"""

import os
import time
import re
import pandas as pd

# Works both as a local .py file (python scripts/step3_benchmark_v2.py from project root)
# AND when pasted directly into a Colab cell (where __file__ does not exist at all).
try:
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    in_path = os.path.join(BASE_DIR, "..", "data", "job_postings_synthetic_large.csv")
except NameError:
    in_path = "job_postings_synthetic_large.csv"  # Colab: file uploaded next to the notebook

CORE_SKILLS = [
    "python", "fastapi", "docker", "aws", "azure", "gcp", "rag",
    "llm", "langchain", "pytorch", "tensorflow", "pyspark", "kubernetes",
    "vector database", "prompt engineering", "microservices", "ci/cd",
    "mongodb", "postgresql", "airflow",
]

COMBINED_PATTERN = "|".join(re.escape(s) for s in CORE_SKILLS)

def clean_and_score(df):
    """Redesigned: one regex pass instead of 20 separate ones."""
    df["jd_text_clean"] = df["jd_text"].str.lower().str.strip()
    df["word_count"] = df["jd_text_clean"].str.split().str.len()
    df["skill_match_count"] = df["jd_text_clean"].str.count(COMBINED_PATTERN)

    company_avg = df.groupby("company")["skill_match_count"].mean().sort_values(ascending=False)
    return df, company_avg

def main():
    print(f"Loading {in_path} ...")
    df = pd.read_csv(in_path)
    print(f"Loaded {len(df):,} rows")

    # --- WARM-UP (untimed): forces GPU/CUDA init to happen before the clock starts ---
    print("Warming up...")
    warmup_df = df.head(1000).copy()
    clean_and_score(warmup_df)

    # --- TIMED RUN ---
    start = time.time()
    df, company_avg = clean_and_score(df)
    elapsed = time.time() - start

    print(f"\nTop 5 companies by avg skill match:\n{company_avg.head()}")
    print(f"\n=== (warmed up) Cleaning + feature extraction on {len(df):,} rows took {elapsed:.2f} seconds ===")

if __name__ == "__main__":
    main()