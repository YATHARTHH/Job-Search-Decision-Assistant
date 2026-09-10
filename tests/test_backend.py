from unittest.mock import MagicMock, patch

from backend.main import (
    compute_forecast,
    compute_ranked_jobs,
    experience_fit,
    find_skill_gaps,
    get_anomaly_notes,
    safe_list_parse,
    skill_match_score,
)

# ---------- Helper Unit Tests ----------


def test_safe_list_parse():
    assert safe_list_parse("['Python', 'FastAPI']") == ["Python", "FastAPI"]
    assert safe_list_parse("") == []
    assert safe_list_parse("[]") == []
    assert safe_list_parse(None) == []
    assert safe_list_parse("invalid string") == []
    assert safe_list_parse("123") == []


def test_skill_match_score():
    profile = ["Python", "FastAPI", "SQL"]
    job = ["Python", "FastAPI", "Docker"]
    assert round(skill_match_score(job, profile), 2) == 0.67
    assert skill_match_score([], profile) == 0.0
    assert skill_match_score(job, []) == 0.0


def test_find_skill_gaps():
    profile = ["Python", "FastAPI"]
    job = ["Python", "Kubernetes", "Docker"]
    gaps = find_skill_gaps(job, profile)
    assert "Kubernetes" in gaps
    assert "Docker" in gaps
    assert "Python" not in gaps


def test_experience_fit():
    assert experience_fit(2, 5, 3) == 1.0
    assert experience_fit(None, None, 3) == 0.5
    fit = experience_fit(5, 7, 3)
    assert 0.0 <= fit < 1.0


def test_get_anomaly_notes():
    row_error = {"error": "could_not_parse"}
    notes = get_anomaly_notes(row_error)
    assert any("GEMINI_PARSE_FAILED" in n for n in notes)

    row_vague = {"requirements_clarity": "vague", "red_flags": "['Unclear responsibilities']"}
    notes_vague = get_anomaly_notes(row_vague)
    assert any("REQUIREMENTS_VAGUE" in n for n in notes_vague)
    assert "Unclear responsibilities" in notes_vague


def test_compute_ranked_jobs():
    jobs = compute_ranked_jobs()
    assert isinstance(jobs, list)
    assert len(jobs) > 0
    first = jobs[0]
    assert "job_id" in first
    assert "fit_score" in first
    assert "anomalies" in first
    assert "skill_gaps" in first


def test_compute_forecast():
    forecast = compute_forecast(n_future=5)
    assert "total_applications" in forecast
    assert "overall_callback_rate_pct" in forecast
    assert forecast["forecast_next_n"] == 5


# ---------- FastAPI Endpoint Tests ----------


def test_get_score_endpoint(client):
    response = client.get("/score")
    assert response.status_code == 200
    data = response.json()
    assert "jobs" in data
    assert len(data["jobs"]) > 0

    response_custom = client.get("/score?core_weight=0.6&learning_weight=0.2&exp_weight=0.2")
    assert response_custom.status_code == 200
    assert "jobs" in response_custom.json()


def test_get_score_file_not_found(client):
    with patch("backend.main.compute_ranked_jobs", side_effect=FileNotFoundError("missing file")):
        response = client.get("/score")
        assert response.status_code == 500
        assert "Missing data file" in response.json()["detail"]


def test_get_forecast_endpoint(client):
    response = client.get("/forecast?n=15")
    assert response.status_code == 200
    data = response.json()
    assert data["forecast_next_n"] == 15
    assert "expected_callbacks_next_n" in data


def test_get_forecast_file_not_found(client):
    with patch("backend.main.compute_forecast", side_effect=FileNotFoundError("missing log")):
        response = client.get("/forecast")
        assert response.status_code == 500
        assert "Missing data file" in response.json()["detail"]


# ---------- /ask Endpoint Tests ----------


def test_ask_endpoint_missing_api_key(client, monkeypatch):
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    response = client.post("/ask", json={"question": "How many jobs are in London?"})
    assert response.status_code == 500
    assert "GEMINI_API_KEY not set" in response.json()["detail"]


@patch("google.genai.Client")
@patch("google.cloud.bigquery.Client")
def test_ask_endpoint_success(mock_bq, mock_genai, client):
    mock_genai_instance = MagicMock()
    mock_genai.return_value = mock_genai_instance
    mock_response = MagicMock()
    mock_response.text = "SELECT COUNT(*) FROM `table`"
    mock_genai_instance.models.generate_content.return_value = mock_response

    mock_bq_instance = MagicMock()
    mock_bq.return_value = mock_bq_instance
    mock_query_job = MagicMock()
    mock_df = MagicMock()
    mock_df.to_dict.return_value = [{"count": 20}]
    mock_query_job.to_dataframe.return_value = mock_df
    mock_bq_instance.query.return_value = mock_query_job

    response = client.post("/ask", json={"question": "Count jobs"})
    assert response.status_code == 200
    data = response.json()
    assert data["generated_sql"] == "SELECT COUNT(*) FROM `table`"
    assert data["result"] == [{"count": 20}]


@patch("google.genai.Client")
def test_ask_endpoint_quota_error(mock_genai, client):
    mock_genai_instance = MagicMock()
    mock_genai.return_value = mock_genai_instance
    mock_genai_instance.models.generate_content.side_effect = Exception("429 RESOURCE_EXHAUSTED")

    response = client.post("/ask", json={"question": "Count jobs"})
    assert response.status_code == 429
    assert "Gemini free-tier daily quota" in response.json()["detail"]


@patch("google.genai.Client")
def test_ask_endpoint_generic_error(mock_genai, client):
    mock_genai_instance = MagicMock()
    mock_genai.return_value = mock_genai_instance
    mock_genai_instance.models.generate_content.side_effect = Exception("Unknown API error")

    response = client.post("/ask", json={"question": "Count jobs"})
    assert response.status_code == 500
    assert "Gemini API error" in response.json()["detail"]


@patch("google.genai.Client")
@patch("google.cloud.bigquery.Client")
def test_ask_endpoint_bigquery_error(mock_bq, mock_genai, client):
    mock_genai_instance = MagicMock()
    mock_genai.return_value = mock_genai_instance
    mock_response = MagicMock()
    mock_response.text = "SELECT invalid"
    mock_genai_instance.models.generate_content.return_value = mock_response

    mock_bq_instance = MagicMock()
    mock_bq.return_value = mock_bq_instance
    mock_bq_instance.query.side_effect = Exception("Syntax error in SQL")

    response = client.post("/ask", json={"question": "Count jobs"})
    assert response.status_code == 500
    assert "BigQuery error" in response.json()["detail"]


# ---------- /add_job Endpoint Tests ----------


def test_add_job_endpoint_missing_key(client, monkeypatch):
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    payload = {"company": "Test", "title": "Dev", "jd_text": "Python"}
    response = client.post("/add_job", json=payload)
    assert response.status_code == 500
    assert "GEMINI_API_KEY not set" in response.json()["detail"]


@patch("google.genai.Client")
def test_add_job_endpoint_json_parse_fallback(mock_genai, client):
    mock_genai_instance = MagicMock()
    mock_genai.return_value = mock_genai_instance
    mock_response = MagicMock()
    mock_response.text = "Invalid Non-JSON response"
    mock_genai_instance.models.generate_content.return_value = mock_response

    payload = {"company": "TestCorp", "title": "Dev", "jd_text": "Python JD"}
    response = client.post("/add_job", json=payload)
    assert response.status_code == 200
    assert "job_id" in response.json()


@patch("google.genai.Client")
def test_add_job_endpoint_quota_error(mock_genai, client):
    mock_genai_instance = MagicMock()
    mock_genai.return_value = mock_genai_instance
    mock_genai_instance.models.generate_content.side_effect = Exception("429 RESOURCE_EXHAUSTED")

    payload = {"company": "TestCorp", "title": "Dev", "jd_text": "Python JD"}
    response = client.post("/add_job", json=payload)
    assert response.status_code == 429
    assert "quota used up" in response.json()["detail"]


@patch("google.genai.Client")
def test_add_job_endpoint_generic_error(mock_genai, client):
    mock_genai_instance = MagicMock()
    mock_genai.return_value = mock_genai_instance
    mock_genai_instance.models.generate_content.side_effect = Exception("API connection dropped")

    payload = {"company": "TestCorp", "title": "Dev", "jd_text": "Python JD"}
    response = client.post("/add_job", json=payload)
    assert response.status_code == 500
    assert "Gemini API error" in response.json()["detail"]


# ---------- /skill_gap_advice Endpoint Tests ----------


def test_skill_gap_advice_missing_key(client, monkeypatch):
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    response = client.post("/skill_gap_advice", json={"job_id": 1})
    assert response.status_code == 500
    assert "GEMINI_API_KEY not set" in response.json()["detail"]


@patch("google.genai.Client")
def test_skill_gap_advice_json_decode_error(mock_genai, client):
    mock_genai_instance = MagicMock()
    mock_genai.return_value = mock_genai_instance
    mock_response = MagicMock()
    mock_response.text = "Not JSON"
    mock_genai_instance.models.generate_content.return_value = mock_response

    jobs = compute_ranked_jobs()
    valid_id = jobs[0]["job_id"]

    response = client.post("/skill_gap_advice", json={"job_id": valid_id})
    assert response.status_code == 500
    assert "unparseable JSON" in response.json()["detail"]


@patch("google.genai.Client")
def test_skill_gap_advice_quota_error(mock_genai, client):
    mock_genai_instance = MagicMock()
    mock_genai.return_value = mock_genai_instance
    mock_genai_instance.models.generate_content.side_effect = Exception("429 Quota Exceeded")

    jobs = compute_ranked_jobs()
    valid_id = jobs[0]["job_id"]

    response = client.post("/skill_gap_advice", json={"job_id": valid_id})
    assert response.status_code == 429


@patch("google.genai.Client")
def test_skill_gap_advice_generic_error(mock_genai, client):
    mock_genai_instance = MagicMock()
    mock_genai.return_value = mock_genai_instance
    mock_genai_instance.models.generate_content.side_effect = Exception("Unexpected Error")

    jobs = compute_ranked_jobs()
    valid_id = jobs[0]["job_id"]

    response = client.post("/skill_gap_advice", json={"job_id": valid_id})
    assert response.status_code == 500


# ---------- /interview_prep Endpoint Tests ----------


def test_interview_prep_missing_key(client, monkeypatch):
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    response = client.post("/interview_prep", json={"job_id": 1})
    assert response.status_code == 500
    assert "GEMINI_API_KEY not set" in response.json()["detail"]


@patch("google.genai.Client")
def test_interview_prep_json_decode_error(mock_genai, client):
    mock_genai_instance = MagicMock()
    mock_genai.return_value = mock_genai_instance
    mock_response = MagicMock()
    mock_response.text = "Not JSON"
    mock_genai_instance.models.generate_content.return_value = mock_response

    jobs = compute_ranked_jobs()
    valid_id = jobs[0]["job_id"]

    response = client.post("/interview_prep", json={"job_id": valid_id})
    assert response.status_code == 500
    assert "unparseable JSON" in response.json()["detail"]


@patch("google.genai.Client")
def test_interview_prep_quota_error(mock_genai, client):
    mock_genai_instance = MagicMock()
    mock_genai.return_value = mock_genai_instance
    mock_genai_instance.models.generate_content.side_effect = Exception("429 RESOURCE_EXHAUSTED")

    jobs = compute_ranked_jobs()
    valid_id = jobs[0]["job_id"]

    response = client.post("/interview_prep", json={"job_id": valid_id})
    assert response.status_code == 429


@patch("google.genai.Client")
def test_interview_prep_generic_error(mock_genai, client):
    mock_genai_instance = MagicMock()
    mock_genai.return_value = mock_genai_instance
    mock_genai_instance.models.generate_content.side_effect = Exception("Failed connection")

    jobs = compute_ranked_jobs()
    valid_id = jobs[0]["job_id"]

    response = client.post("/interview_prep", json={"job_id": valid_id})
    assert response.status_code == 500
