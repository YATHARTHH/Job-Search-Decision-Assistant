"""
Step 3 v3 - Acceleration benchmark using vector similarity (not string search).

WHY THE PIVOT:
v1 and v2 both showed cudf.pandas GPU slower than CPU pandas for string/keyword
matching at 200k rows. That's a real, valid finding - text search at this scale is
dominated by GPU launch/data-transfer overhead, not parallel compute. GPUs earn their
keep on large numeric matrix/vector math instead - which is exactly what a REAL
semantic similarity score (comparing job description embeddings to a profile
embedding) needs. This is also a more realistic upgrade path for the actual product:
"which jobs are semantically similar to my profile" instead of naive keyword overlap.

WHAT THIS DOES:
Simulates N job description embeddings (384-dim, the size real sentence-transformer
models use) and one profile embedding, then computes cosine similarity between the
profile and every job in one vectorized matrix operation - the kind of workload GPUs
are built for.

NOTE: embeddings here are randomly generated, NOT real Gemini/sentence-transformer
output - this isolates and measures "how fast is the similarity MATH", independent of
"how fast is generating embeddings" (a separate, API-bound cost). Be upfront about
this in your PPT: it is a synthetic-but-representative compute benchmark.

Run locally (CPU / numpy):
    python3 scripts/step3_benchmark_v3_embeddings.py

Run in Colab (GPU / cupy) - only ONE line changes:
    cell 1: !pip install cupy-cuda12x -q
    cell 2: paste this file, but change "import numpy as np" to "import cupy as np"
    (cupy is RAPIDS' drop-in GPU replacement for numpy, same API, same code otherwise)
"""

import time
import numpy as np   # <-- for Colab GPU run, change this single line to: import cupy as np

N_JOBS = 1_000_000       # larger scale than the string benchmark - GPUs need volume to shine
EMBEDDING_DIM = 384       # typical real-world sentence-embedding size

def cosine_similarity(job_embeddings, profile_embedding):
    dot_products = job_embeddings @ profile_embedding
    job_norms = np.linalg.norm(job_embeddings, axis=1)
    profile_norm = np.linalg.norm(profile_embedding)
    return dot_products / (job_norms * profile_norm)

def main():
    rng = np.random.default_rng(42) if hasattr(np.random, "default_rng") else None
    if rng is not None:
        job_embeddings = rng.random((N_JOBS, EMBEDDING_DIM), dtype=np.float32)
        profile_embedding = rng.random(EMBEDDING_DIM, dtype=np.float32)
    else:  # cupy doesn't have default_rng the same way in all versions - fallback
        job_embeddings = np.random.rand(N_JOBS, EMBEDDING_DIM).astype(np.float32)
        profile_embedding = np.random.rand(EMBEDDING_DIM).astype(np.float32)

    print(f"Simulated {N_JOBS:,} job embeddings x {EMBEDDING_DIM} dimensions")

    # --- WARM-UP (untimed): forces GPU/CUDA init before the clock starts ---
    print("Warming up...")
    _ = cosine_similarity(job_embeddings[:1000], profile_embedding)

    # --- TIMED RUN ---
    start = time.time()
    similarities = cosine_similarity(job_embeddings, profile_embedding)
    top5_idx = np.argsort(similarities)[-5:][::-1]
    elapsed = time.time() - start

    print(f"Top 5 similarity scores: {similarities[top5_idx]}")
    print(f"\n=== Cosine similarity across {N_JOBS:,} embeddings took {elapsed:.4f} seconds ===")

if __name__ == "__main__":
    main()