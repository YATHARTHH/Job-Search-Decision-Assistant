import json
import os
import pandas as pd

# --- Load data (paths are relative to this script's location) ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "..", "data")
OUTPUT_DIR = os.path.join(BASE_DIR, "..", "outputs")

jobs = pd.read_csv(os.path.join(DATA_DIR, "job_postings.csv"))
with open(os.path.join(DATA_DIR, "my_profile.json")) as f:
    profile = json.load(f)

core_skills = [s.lower() for s in profile["core_strength_skills"]]
learning_skills = [s.lower() for s in profile["learning_in_progress_skills"]]
my_years = profile["years_experience"]

def skill_match_score(jd_text, skill_list):
    """Fraction of skills in skill_list that appear in the JD text."""
    text = jd_text.lower()
    hits = sum(1 for s in skill_list if s in text)
    return hits / len(skill_list) if skill_list else 0.0

def experience_fit(min_exp, max_exp, my_years):
    """1.0 if squarely in range, partial credit if close, 0 if way off."""
    if pd.isna(min_exp):
        return 0.5  # unknown requirement -> neutral score, not a penalty
    lo = min_exp
    hi = max_exp if not pd.isna(max_exp) else min_exp + 3
    if lo <= my_years <= hi:
        return 1.0
    gap = min(abs(my_years - lo), abs(my_years - hi))
    return max(0.0, 1 - gap / 3)  # lose credit gradually over ~3 years' gap

def anomaly_flag(jd_text, min_exp):
    """Very light rule-based anomaly check (naive - Gemini will replace this in Step 2)."""
    text = jd_text.lower()
    word_count = len(text.split())
    if word_count < 40:
        return "LOW_INFO: JD too short / missing structured requirements"
    if pd.isna(min_exp) and any(k in text for k in ["required", "requirements", "qualifications"]):
        return "MISSING_EXP: has requirements section but no clear experience range"
    return ""

rows = []
for _, job in jobs.iterrows():
    core_score = skill_match_score(job["jd_text"], core_skills)
    learn_score = skill_match_score(job["jd_text"], learning_skills)
    exp_score = experience_fit(job["min_exp_years"], job["max_exp_years"], my_years)

    # Weighting: core skills matter most, learning skills are a bonus,
    # experience fit gates whether it's realistic to even apply.
    fit_score = round((0.5 * core_score + 0.25 * learn_score + 0.25 * exp_score) * 100, 1)

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
        "anomaly": anomaly_flag(job["jd_text"], job["min_exp_years"]),
    })

result = pd.DataFrame(rows).sort_values("fit_score", ascending=False).reset_index(drop=True)
result.index += 1

pd.set_option("display.width", 160)
print("\n=== RANKED JOB LIST (Step 1 - naive local prototype, no cloud/LLM yet) ===\n")
print(result.to_string())

out_path = os.path.join(OUTPUT_DIR, "step1_ranked_output.csv")
result.to_csv(out_path, index=False)
print(f"\nSaved full ranked output to {out_path}")
