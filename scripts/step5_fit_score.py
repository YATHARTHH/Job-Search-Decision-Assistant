"""
Step 5 - Smarter fit-score, anomaly detection, using Gemini's REAL structured output
(jobs_structured.csv from Step 2), not naive keyword matching on raw JD text (Step 1).

WHAT'S BETTER THAN STEP 1:
- Skill matching now compares against Gemini's cleaned required_skills + tech_stack
  lists, not noisy substring search across raw paragraph text.
- Anomaly detection now uses Gemini's own red_flags + requirements_clarity fields -
  real reasoning-based flags (e.g. "says 5 years in one place, 3 years in another"),
  not a naive word-count rule.
- Falls back to job_postings.csv's original min/max experience for job_id 5, since
  that row's Gemini extraction failed to parse (known, documented limitation).

Run locally:
    python3 scripts/step5_fit_score.py
"""

import os
import json
import ast
import pandas as pd

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "..", "data")

def safe_list_parse(val):
    """Gemini's list columns are stored as Python-literal strings like "['Python', 'AWS']".
    Parse with ast.literal_eval (safe), return [] on any failure (e.g. job 5's empty cells)."""
    if pd.isna(val) or val in ("", "[]"):
        return []
    try:
        result = ast.literal_eval(val)
        return result if isinstance(result, list) else []
    except (ValueError, SyntaxError):
        return []

def skill_match_score(job_skills, profile_skills):
    """job_skills: Gemini-extracted list for this job. profile_skills: Harry's skill list.
    Fraction of PROFILE skills that appear (as substring) somewhere in job_skills."""
    if not job_skills:
        return 0.0
    job_text = " ".join(job_skills).lower()
    hits = sum(1 for s in profile_skills if s.lower() in job_text)
    return hits / len(profile_skills) if profile_skills else 0.0

def experience_fit(min_exp, max_exp, my_years):
    if pd.isna(min_exp):
        return 0.5
    lo = min_exp
    hi = max_exp if not pd.isna(max_exp) else min_exp + 3
    if lo <= my_years <= hi:
        return 1.0
    gap = min(abs(my_years - lo), abs(my_years - hi))
    return max(0.0, 1 - gap / 3)

def get_anomaly_notes(row):
    """Use Gemini's OWN reasoning instead of naive rules."""
    notes = []
    if row.get("error") == "could_not_parse":
        notes.append("GEMINI_PARSE_FAILED: structured extraction failed for this posting")
        return notes
    clarity = row.get("requirements_clarity", "")
    if clarity in ("vague", "missing"):
        notes.append(f"REQUIREMENTS_{clarity.upper()}: Gemini flagged this posting's requirements as {clarity}")
    red_flags = safe_list_parse(row.get("red_flags", "[]"))
    notes.extend(red_flags)
    return notes

def main():
    structured = pd.read_csv(os.path.join(DATA_DIR, "jobs_structured.csv"))
    original = pd.read_csv(os.path.join(DATA_DIR, "job_postings.csv"))
    with open(os.path.join(DATA_DIR, "my_profile.json")) as f:
        profile = json.load(f)

    core_skills = profile["core_strength_skills"]
    learning_skills = profile["learning_in_progress_skills"]
    my_years = profile["years_experience"]

    # Merge: fall back to original CSV's exp range when Gemini's extraction is missing (job 5)
    merged = structured.merge(
        original[["job_id", "min_exp_years", "max_exp_years"]],
        on="job_id", suffixes=("_gemini", "_original")
    )
    merged["min_exp_years"] = merged["min_exp_years_gemini"].fillna(merged["min_exp_years_original"])
    merged["max_exp_years"] = merged["max_exp_years_gemini"].fillna(merged["max_exp_years_original"])

    rows = []
    for _, job in merged.iterrows():
        job_skills = safe_list_parse(job["required_skills"]) + safe_list_parse(job["tech_stack"])

        core_score = skill_match_score(job_skills, core_skills)
        learn_score = skill_match_score(job_skills, learning_skills)
        exp_score = experience_fit(job["min_exp_years"], job["max_exp_years"], my_years)

        fit_score = round((0.5 * core_score + 0.25 * learn_score + 0.25 * exp_score) * 100, 1)
        anomalies = get_anomaly_notes(job)

        rows.append({
            "job_id": job["job_id"],
            "company": job["company"],
            "title": job["title"],
            "min_exp": job["min_exp_years"],
            "max_exp": job["max_exp_years"],
            "core_skill_%": round(core_score * 100, 1),
            "learning_skill_%": round(learn_score * 100, 1),
            "exp_fit_%": round(exp_score * 100, 1),
            "fit_score": fit_score,
            "anomaly_count": len(anomalies),
            "anomalies": " | ".join(anomalies) if anomalies else "",
        })

    result = pd.DataFrame(rows).sort_values("fit_score", ascending=False).reset_index(drop=True)
    result.index += 1

    pd.set_option("display.width", 200)
    pd.set_option("display.max_colwidth", 60)
    print("\n=== STEP 5 RANKED JOB LIST (using real Gemini structured data) ===\n")
    print(result.to_string())

    out_path = os.path.join(DATA_DIR, "..", "outputs", "step5_ranked_output.csv")
    result.to_csv(out_path, index=False)
    print(f"\nSaved to {out_path}")

if __name__ == "__main__":
    main()