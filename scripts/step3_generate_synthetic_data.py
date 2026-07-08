"""
Step 3a - Generate a large synthetic dataset for the acceleration benchmark.

WHY SYNTHETIC: The acceleration benefit of cudf.pandas only shows up at real scale
(100k+ rows). We only have 20 real JDs, so this script expands them into a large
dataset with light text variation - honest to disclose as "synthetic scale-up of real
JD data" in your PPT/demo, which is standard practice for this kind of benchmark.

Run locally (no GPU needed for this part):
    python3 scripts/step3_generate_synthetic_data.py
"""

import os
import random
import pandas as pd

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "..", "data")

N_ROWS = 200_000  # large enough for cudf.pandas to show a real speedup

FILLER_SENTENCES = [
    "This role requires strong collaboration across cross-functional teams.",
    "Candidates should be comfortable working in a fast-paced environment.",
    "Prior experience with agile methodologies is a plus.",
    "This position offers growth opportunities within the organization.",
    "Strong communication skills are essential for this role.",
    "The team follows a hybrid working model with flexible hours.",
    "Candidates will work closely with product and engineering stakeholders.",
    "This role may require occasional travel to client sites.",
]

def vary_text(text, seed):
    rng = random.Random(seed)
    filler = rng.choice(FILLER_SENTENCES)
    # light variation: append a random filler sentence, occasionally shuffle case of one word
    return f"{text} {filler}"

def main():
    real_jobs = pd.read_csv(os.path.join(DATA_DIR, "job_postings.csv"))
    real_jobs = real_jobs.dropna(subset=["jd_text"]).reset_index(drop=True)

    rows = []
    for i in range(N_ROWS):
        base = real_jobs.iloc[i % len(real_jobs)]
        rows.append({
            "job_id": i + 1,
            "company": base["company"],
            "title": base["title"],
            "jd_text": vary_text(base["jd_text"], seed=i),
        })

    synthetic = pd.DataFrame(rows)
    out_path = os.path.join(DATA_DIR, "job_postings_synthetic_large.csv")
    synthetic.to_csv(out_path, index=False)
    print(f"Wrote {len(synthetic)} synthetic rows to {out_path}")
    print(f"File size: {os.path.getsize(out_path) / 1_000_000:.1f} MB")

if __name__ == "__main__":
    main()