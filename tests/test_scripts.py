import os

import pandas as pd

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "..", "data")
OUTPUT_DIR = os.path.join(BASE_DIR, "..", "output")


def test_data_files_exist():
    """Verify that core data files exist in the data directory."""
    job_postings = os.path.join(DATA_DIR, "job_postings.csv")
    jobs_structured = os.path.join(DATA_DIR, "jobs_structured.csv")
    my_profile = os.path.join(DATA_DIR, "my_profile.json")

    assert os.path.exists(job_postings), "job_postings.csv missing"
    assert os.path.exists(jobs_structured), "jobs_structured.csv missing"
    assert os.path.exists(my_profile), "my_profile.json missing"


def test_job_postings_data_structure():
    """Verify schema of job_postings.csv."""
    df = pd.read_csv(os.path.join(DATA_DIR, "job_postings.csv"))
    expected_cols = {"job_id", "company", "title", "jd_text"}
    assert expected_cols.issubset(set(df.columns))
    assert len(df) > 0


def test_jobs_structured_data_structure():
    """Verify schema of jobs_structured.csv."""
    df = pd.read_csv(os.path.join(DATA_DIR, "jobs_structured.csv"))
    expected_cols = {"job_id", "company", "title", "required_skills"}
    assert expected_cols.issubset(set(df.columns))
    assert len(df) > 0
