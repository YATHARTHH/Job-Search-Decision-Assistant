"""
Step 3b - Acceleration benchmark: cleaning + feature extraction on job postings.

HOW TO USE THIS SAME FILE FOR BOTH TIMINGS:

  Plain pandas (run locally, on your laptop or here):
      python3 scripts/step3_benchmark.py

  cudf.pandas (run in a Google Colab notebook with a free GPU):
      1. Open https://colab.research.google.com, new notebook
      2. Runtime -> Change runtime type -> T4 GPU -> Save
      3. Upload data/job_postings_synthetic_large.csv to the Colab file browser
      4. In the FIRST cell, run:  %load_ext cudf.pandas
      5. In the SECOND cell, paste the entire contents of this file and run it
      (the %load_ext line makes every "import pandas as pd" transparently GPU-backed -
      you do not change a single line of the benchmark code itself)

Both runs print a total time at the end. Put both numbers side by side in your PPT.
"""

import os
import time
import pandas as pd

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "..", "data")

CORE_SKILLS = [
    "python", "fastapi", "docker", "aws", "azure", "gcp", "rag",
    "llm", "langchain", "pytorch", "tensorflow", "pyspark", "kubernetes",
    "vector database", "prompt engineering", "microservices", "ci/cd",
    "mongodb", "postgresql", "airflow",
]

def clean_and_score(df):
    """Same cleaning + feature extraction logic Step 1/5 use, run at scale."""
    df["jd_text_clean"] = df["jd_text"].str.lower().str.strip()
    df["word_count"] = df["jd_text_clean"].str.split().str.len()

    for skill in CORE_SKILLS:
        col_name = "has_" + skill.replace(" ", "_").replace("/", "_")
        df[col_name] = df["jd_text_clean"].str.contains(skill, regex=False)

    skill_cols = ["has_" + s.replace(" ", "_").replace("/", "_") for s in CORE_SKILLS]
    df["skill_match_count"] = df[skill_cols].sum(axis=1)

    company_avg = df.groupby("company")["skill_match_count"].mean().sort_values(ascending=False)
    return df, company_avg

def main():
    in_path = os.path.join(DATA_DIR, "job_postings_synthetic_large.csv")
    print(f"Loading {in_path} ...")
    df = pd.read_csv(in_path)
    print(f"Loaded {len(df):,} rows")

    start = time.time()
    df, company_avg = clean_and_score(df)
    elapsed = time.time() - start

    print(f"\nTop 5 companies by avg skill match:\n{company_avg.head()}")
    print(f"\n=== Cleaning + feature extraction on {len(df):,} rows took {elapsed:.2f} seconds ===")

if __name__ == "__main__":
    main()