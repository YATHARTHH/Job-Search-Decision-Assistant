"""
Forecast script - the "predict outcomes" piece of the decision-intelligence pipeline.

Answers: "At my current rate, what should I expect if I apply to N more jobs?"

Run locally:
    python3 scripts/step_forecast.py
"""

import os
import pandas as pd

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "..", "data")

N_FUTURE_APPLICATIONS = 10  # forecast horizon - tweak as needed

def main():
    log = pd.read_csv(os.path.join(DATA_DIR, "applications_log.csv"), parse_dates=["applied_date"])
    log = log.sort_values("applied_date").reset_index(drop=True)

    total = len(log)
    callbacks = (log["status"] == "callback").sum()
    rejected = (log["status"] == "rejected").sum()
    no_response = (log["status"] == "no_response").sum()

    overall_callback_rate = callbacks / total

    # Recent trend: last half of applications vs first half (simple, honest, no fancy modeling)
    midpoint = total // 2
    early = log.iloc[:midpoint]
    recent = log.iloc[midpoint:]
    early_rate = (early["status"] == "callback").sum() / len(early) if len(early) else 0
    recent_rate = (recent["status"] == "callback").sum() / len(recent) if len(recent) else 0

    trend = "improving" if recent_rate > early_rate else ("declining" if recent_rate < early_rate else "flat")

    expected_callbacks = round(overall_callback_rate * N_FUTURE_APPLICATIONS, 1)

    # Average time-to-response, for context
    responded = log.dropna(subset=["response_date"]).copy()
    if len(responded):
        responded["response_date"] = pd.to_datetime(responded["response_date"])
        responded["lag_days"] = (responded["response_date"] - responded["applied_date"]).dt.days
        avg_lag = round(responded["lag_days"].mean(), 1)
    else:
        avg_lag = None

    print("=== APPLICATION FUNNEL SUMMARY ===")
    print(f"Total applications logged: {total}")
    print(f"  Callbacks: {callbacks} ({callbacks/total*100:.1f}%)")
    print(f"  Rejected: {rejected} ({rejected/total*100:.1f}%)")
    print(f"  No response yet: {no_response} ({no_response/total*100:.1f}%)")
    print(f"Average time to hear back (when a response came): {avg_lag} days" if avg_lag else "No responses yet to measure lag")
    print(f"\nEarly-period callback rate: {early_rate*100:.1f}%")
    print(f"Recent-period callback rate: {recent_rate*100:.1f}%")
    print(f"Trend: {trend}")
    print(f"\n=== FORECAST ===")
    print(f"At your current overall callback rate ({overall_callback_rate*100:.1f}%),")
    print(f"applying to {N_FUTURE_APPLICATIONS} more similar-quality jobs should yield")
    print(f"approximately {expected_callbacks} more callbacks.")

if __name__ == "__main__":
    main()