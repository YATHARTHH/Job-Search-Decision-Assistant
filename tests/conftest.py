import os

import pytest
from fastapi.testclient import TestClient

# Ensure BASE_DIR resolution works in tests
os.environ["GEMINI_API_KEY"] = "mock_key_for_testing"

from backend.main import app


@pytest.fixture
def client():
    """FastAPI TestClient fixture."""
    return TestClient(app)


@pytest.fixture
def sample_profile_skills():
    return ["Python", "FastAPI", "SQL", "Pandas", "Docker", "Machine Learning"]


@pytest.fixture
def sample_job_skills():
    return ["Python", "FastAPI", "Docker", "Kubernetes", "PyTorch"]
