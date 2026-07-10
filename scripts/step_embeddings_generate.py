"""
Generate REAL embeddings (not simulated - these are actual Gemini embeddings) for
every job posting and for your profile, then save them for use in semantic
similarity scoring (see step5_semantic_fit_score.py).

WHY THIS MATTERS: Steps 1 and 5 so far compare skill LISTS as text (does "Python"
appear in both lists?). That's still just string matching, even on Gemini's cleaned
data. Real embeddings let us compare MEANING - e.g. "Kubernetes" and "container
orchestration" would be recognized as related concepts, not just missed because
the exact words differ. This is the "real embeddings for production semantic
similarity scoring" enhancement from the roadmap.

SETUP (same as step2):
    pip install google-genai
    export GEMINI_API_KEY="your-key"

Run locally:
    python3 scripts/step_embeddings_generate.py

NOTE: embeddings share the SAME daily free-tier quota family as generate_content
calls on this API key/project - if you've used up today's quota on Step 2 or /ask,
this may also be rate-limited. Wait a day or use a fresh key if needed.
"""

import os
import json
import time
import ast
import pandas as pd
from google import genai

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "..", "data")

API_KEY = os.environ.get("GEMINI_API_KEY")
if not API_KEY:
    raise SystemExit("GEMINI_API_KEY not set. export GEMINI_API_KEY='your-key'")

client = genai.Client(api_key=API_KEY)
EMBEDDING_MODEL = "gemini-embedding-001"  # stable, text-only, 3072-dim vectors


def safe_list_parse(val):
    if pd.isna(val) or val in ("", "[]"):
        return []
    try:
        result = ast.literal_eval(val)
        return result if isinstance(result, list) else []
    except (ValueError, SyntaxError):
        return []


def embed_text(text, retries=3):
    """Call Gemini's embedding endpoint. Returns a list of floats (the vector)."""
    for attempt in range(retries):
        try:
            result = client.models.embed_content(model=EMBEDDING_MODEL, contents=text)
            # result.embeddings is a list (one per input); we send one string at a time
            return result.embeddings[0].values
        except Exception as e:
            error_str = str(e)
            if "RESOURCE_EXHAUSTED" in error_str or "429" in error_str:
                wait = 20 * (attempt + 1)
                print(f"  Rate limited, waiting {wait}s...")
                time.sleep(wait)
            else:
                raise
    raise RuntimeError(f"Failed to embed after {retries} retries")


def build_job_text(row):
    """Combine the most meaningful fields into one string to embed per job."""
    skills = safe_list_parse(row.get("required_skills", "[]"))
    tech = safe_list_parse(row.get("tech_stack", "[]"))
    role_focus = row.get("role_focus", "") or ""
    return f"Role focus: {role_focus}. Required skills: {', '.join(skills)}. Tech stack: {', '.join(tech)}."


def build_profile_text(profile):
    core = ", ".join(profile["core_strength_skills"])
    learning = ", ".join(profile["learning_in_progress_skills"])
    titles = ", ".join(profile["target_titles"])
    return f"Target roles: {titles}. Core skills: {core}. Currently building: {learning}."


def main():
    structured = pd.read_csv(os.path.join(DATA_DIR, "jobs_structured.csv"))
    with open(os.path.join(DATA_DIR, "my_profile.json")) as f:
        profile = json.load(f)

    print("Embedding profile...")
    profile_text = build_profile_text(profile)
    profile_embedding = embed_text(profile_text)
    time.sleep(2)

    job_embeddings = {}
    for _, row in structured.iterrows():
        job_id = int(row["job_id"])
        if row.get("error") == "could_not_parse":
            print(f"Skipping job_id={job_id} ({row['company']}) - no structured data to embed")
            continue
        print(f"Embedding job_id={job_id} ({row['company']})...")
        job_text = build_job_text(row)
        job_embeddings[job_id] = embed_text(job_text)
        time.sleep(2)  # be gentle with rate limits

    output = {
        "profile_embedding": profile_embedding,
        "job_embeddings": job_embeddings,
        "model": EMBEDDING_MODEL,
    }
    out_path = os.path.join(DATA_DIR, "embeddings.json")
    with open(out_path, "w") as f:
        json.dump(output, f)

    print(f"\nSaved profile embedding + {len(job_embeddings)} job embeddings to {out_path}")


if __name__ == "__main__":
    main()