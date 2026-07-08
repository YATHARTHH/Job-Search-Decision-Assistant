"""
Generate applications_log.csv - SYNTHETIC data (Harry has no real application history
logged yet), correlated with Step 5 fit scores so the demo tells a coherent story:
"jobs our tool scored higher also tended to get better real-world responses."

Disclose this as synthetic/illustrative in the PPT - same honesty standard as the
acceleration benchmark's synthetic scale-up data.

Run locally:
    python3 scripts/generate_applications_log.py
"""

import os
import random
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "..", "data")
OUTPUT_DIR = os.path.join(BASE_DIR, "..", "outputs")

random.seed(42)

def main():
    ranked = pd.read_csv(os.path.join(OUTPUT_DIR, "step5_ranked_output.csv"))

    # Simulate applying to 16 of the 20 jobs, weighted toward higher-fit ones
    # (a real job seeker would naturally prioritize applying to better-fit roles)
    np.random.seed(42)
    weights = ranked["fit_score"] / ranked["fit_score"].sum()
    chosen_idx = np.random.choice(ranked.index, size=16, replace=False, p=weights)
    applied_jobs = ranked.loc[chosen_idx]

    today = datetime.now()
    rows = []
    for _, job in applied_jobs.iterrows():
        days_ago = random.randint(3, 60)
        applied_date = today - timedelta(days=days_ago)

        # Higher fit_score -> higher chance of callback (bounded probability)
        callback_prob = min(0.15 + (job["fit_score"] / 100) * 0.5, 0.75)
        rejected_prob = 0.20
        roll = random.random()

        if roll < callback_prob:
            status = "callback"
            response_lag = random.randint(3, 15)
        elif roll < callback_prob + rejected_prob:
            status = "rejected"
            response_lag = random.randint(5, 20)
        else:
            status = "no_response"
            response_lag = None

        response_date = (applied_date + timedelta(days=response_lag)) if response_lag else None
        # don't let a response_date land in the future
        if response_date and response_date > today:
            response_date = None
            status = "no_response"

        rows.append({
            "job_id": job["job_id"],
            "company": job["company"],
            "fit_score": job["fit_score"],
            "applied_date": applied_date.strftime("%Y-%m-%d"),
            "status": status,
            "response_date": response_date.strftime("%Y-%m-%d") if response_date else "",
        })

    log = pd.DataFrame(rows).sort_values("applied_date").reset_index(drop=True)
    out_path = os.path.join(DATA_DIR, "applications_log.csv")
    log.to_csv(out_path, index=False)
    print(f"Wrote {len(log)} synthetic application records to {out_path}")
    print(log.to_string(index=False))

if __name__ == "__main__":
    main()