"""
Semantic fit-score - upgrades step5_fit_score.py by blending in REAL embedding
similarity (from step_embeddings_generate.py) alongside the existing skill-list
matching and experience fit.

GRACEFUL FALLBACK: if data/embeddings.json doesn't exist yet (you haven't run
step_embeddings_generate.py), this script still runs fine - it just skips the
semantic similarity component and behaves identically to step5_fit_score.py.
This means you can drop this file in immediately without breaking anything.

Run locally (after running step_embeddings_generate.py at least once):
    python3 scripts/step5_semantic_fit_score.py
"""

import os
import json
import ast
import math
import pandas as pd

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "..", "data")
OUTPUT_DIR = os.path.join(BASE_DIR, "..", "outputs")


def safe_list_parse(val):
    if pd.isna(val) or val in ("", "[]"):
        return []
    try:
        result = ast.literal_eval(val)
        return result if isinstance(result, list) else []
    except (ValueError, SyntaxError):
        return []


def skill_match_score(job_skills, profile_skills):
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


def cosine_similarity(vec_a, vec_b):
    dot = sum(a * b for a, b in zip(vec_a, vec_b))
    norm_a = math.sqrt(sum(a * a for a in vec_a))
    norm_b = math.sqrt(sum(b * b for b in vec_b))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


def get_anomaly_notes(row):
    notes = []
    if row.get("error") == "could_not_parse":
        notes.append("GEMINI_PARSE_FAILED: structured extraction failed for this posting")
        return notes
    clarity = row.get("requirements_clarity", "")
    if clarity in ("vague", "missing"):
        notes.append(f"REQUIREMENTS_{clarity.upper()}: Gemini flagged this posting's requirements as {clarity}")
    notes.extend(safe_list_parse(row.get("red_flags", "[]")))
    return notes


def main():
    structured = pd.read_csv(os.path.join(DATA_DIR, "jobs_structured.csv"))
    original = pd.read_csv(os.path.join(DATA_DIR, "job_postings.csv"))
    with open(os.path.join(DATA_DIR, "my_profile.json")) as f:
        profile = json.load(f)

    core_skills = profile["core_strength_skills"]
    learning_skills = profile["learning_in_progress_skills"]
    my_years = profile["years_experience"]

    # Try to load real embeddings - gracefully skip semantic scoring if not present yet
    embeddings_path = os.path.join(DATA_DIR, "embeddings.json")
    semantic_available = os.path.exists(embeddings_path)
    profile_embedding = None
    job_embeddings = {}
    if semantic_available:
        with open(embeddings_path) as f:
            emb_data = json.load(f)
        profile_embedding = emb_data["profile_embedding"]
        job_embeddings = {int(k): v for k, v in emb_data["job_embeddings"].items()}
        print(f"Loaded real embeddings for {len(job_embeddings)} jobs (model: {emb_data.get('model', 'unknown')})")
    else:
        print("No embeddings.json found - run scripts/step_embeddings_generate.py first for semantic scoring.")
        print("Continuing WITHOUT semantic similarity (same behavior as step5_fit_score.py).\n")

    merged = structured.merge(
        original[["job_id", "min_exp_years", "max_exp_years"]],
        on="job_id", suffixes=("_gemini", "_original")
    )
    merged["min_exp_years"] = merged["min_exp_years_gemini"].fillna(merged["min_exp_years_original"])
    merged["max_exp_years"] = merged["max_exp_years_gemini"].fillna(merged["max_exp_years_original"])

    # Compute raw cosine similarity for every job first, THEN normalize relative to
    # this candidate set (min -> 0, max -> 100). Raw cosine similarity values tend to
    # cluster in a narrow absolute range when all postings share a domain (as here -
    # everything is AI/ML related), which makes the raw numbers poor at discriminating
    # between jobs. Rescaling relative to the set turns that narrow band into a full
    # 0-100 signal that actually differentiates - same underlying data, useful scale.
    raw_semantic_scores = {}
    if semantic_available:
        for _, job in merged.iterrows():
            job_id = int(job["job_id"])
            if job_id in job_embeddings:
                raw_semantic_scores[job_id] = cosine_similarity(profile_embedding, job_embeddings[job_id])

    normalized_semantic_scores = {}
    if raw_semantic_scores:
        lo, hi = min(raw_semantic_scores.values()), max(raw_semantic_scores.values())
        spread = hi - lo
        for job_id, raw in raw_semantic_scores.items():
            normalized_semantic_scores[job_id] = (raw - lo) / spread if spread > 0 else 0.5

    rows = []
    for _, job in merged.iterrows():
        job_id = int(job["job_id"])
        job_skills = safe_list_parse(job["required_skills"]) + safe_list_parse(job["tech_stack"])

        core_score = skill_match_score(job_skills, core_skills)
        learn_score = skill_match_score(job_skills, learning_skills)
        exp_score = experience_fit(job["min_exp_years"], job["max_exp_years"], my_years)

        semantic_score = normalized_semantic_scores.get(job_id)

        if semantic_score is not None:
            # Blend: semantic similarity replaces half the core-skill weight,
            # since it captures meaning-based matches keyword search would miss
            fit_score = round((0.25 * core_score + 0.25 * semantic_score +
                                0.25 * learn_score + 0.25 * exp_score) * 100, 1)
        else:
            fit_score = round((0.5 * core_score + 0.25 * learn_score + 0.25 * exp_score) * 100, 1)

        anomalies = get_anomaly_notes(job)

        rows.append({
            "job_id": job_id,
            "company": job["company"],
            "title": job["title"],
            "min_exp": job["min_exp_years"],
            "max_exp": job["max_exp_years"],
            "core_skill_%": round(core_score * 100, 1),
            "semantic_similarity_%": round(semantic_score * 100, 1) if semantic_score is not None else "N/A",
            "exp_fit_%": round(exp_score * 100, 1),
            "fit_score": fit_score,
            "anomalies": " | ".join(anomalies) if anomalies else "",
        })

    result = pd.DataFrame(rows).sort_values("fit_score", ascending=False).reset_index(drop=True)
    result.index += 1

    pd.set_option("display.width", 200)
    pd.set_option("display.max_colwidth", 60)
    print("\n=== SEMANTIC FIT-SCORE RANKED JOB LIST ===\n")
    print(result.to_string())

    out_path = os.path.join(OUTPUT_DIR, "step5_semantic_ranked_output.csv")
    result.to_csv(out_path, index=False)
    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()