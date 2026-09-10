# Production CI/CD Architecture & Operations Guide

This document outlines the Continuous Integration and Continuous Deployment (CI/CD) architecture, local development verification workflows, security checks, and automation policies for **Job Search Decision Assistant**.

---

## 1. Overview & Architecture

The CI/CD pipeline enforces enterprise-grade code quality, automated test coverage ($\ge 80\%$), static type safety, security vulnerability auditing, and container build integrity across all pull requests and branch merges.

```
[ Push / PR ] ──► [ GitHub Actions ]
                       ├── 1. Code Quality & Typing (Ruff + Mypy)
                       ├── 2. Unit & Integration Tests (Pytest + Coverage Matrix: 3.10, 3.11, 3.12)
                       ├── 3. Security Audits (Bandit + Pip-Audit + Hadolint)
                       └── 4. Docker Build & Vulnerability Scan (Buildx + Trivy)
                                 │
                     [ Tag Release v*.*.* ]
                                 ▼
                     [ Release CD Pipeline ] ──► GitHub Container Registry (ghcr.io)
```

---

## 2. Local Developer Workflows

Before pushing changes or submitting a Pull Request, developers can execute all CI checks locally.

### Setup Development Environment
```bash
python -m venv venv
# Windows:
venv\Scripts\activate
# Linux/macOS:
source venv/bin/activate

pip install -r requirements.txt
pip install -r requirements-dev.txt
```

### 1. Code Style & Linting (`ruff`)
```bash
# Check code for lint errors
ruff check .

# Check formatting
ruff format --check .

# Auto-fix formatting if needed
ruff format .
```

### 2. Static Type Checking (`mypy`)
```bash
mypy backend scripts
```

### 3. Unit Testing & Coverage (`pytest`)
```bash
# Run pytest with coverage report
pytest --cov=backend --cov-report=term-missing

# Enforce minimum 80% coverage threshold
pytest --cov=backend --cov-fail-under=80
```

### 4. Security Auditing (`bandit` & `pip-audit`)
```bash
# Audit Python source code for security flaws
bandit -r backend scripts frontend -ll

# Audit Python dependencies for known CVE vulnerabilities
pip-audit
```

### 5. Docker Build Verification
```bash
docker build -t job-search-assistant:local .
```

---

## 3. GitHub Actions Pipelines

### CI Pipeline (`.github/workflows/ci.yml`)
- **Triggers**: `push` to `main`/`master`/`develop`, `pull_request` to `main`/`master`/`develop`.
- **Jobs**:
  1. `lint-and-typecheck`: Runs `ruff check`, `ruff format --check`, and `mypy`.
  2. `unit-tests`: Executes pytest matrix on Python 3.10, 3.11, and 3.12 with XML coverage artifact generation.
  3. `security-scan`: Scans code for security vulnerabilities using `bandit`, tests dependencies with `pip-audit`, and lints `Dockerfile` with `hadolint`.
  4. `docker-build-scan`: Validates container build with GHA layer caching and scans image using Trivy.

### Release CD Pipeline (`.github/workflows/release-cd.yml`)
- **Triggers**: Push of semantic version tags (`v*.*.*`) or manual trigger (`workflow_dispatch`).
- **Actions**: Builds Docker container and publishes tagged image to GitHub Container Registry (`ghcr.io`).

### Dependency Review & Dependabot (`.github/dependabot.yml`)
- Automatically checks weekly for PyPI and GitHub Actions security updates.
- Blocks pull requests introducing high-risk dependencies via `dependency-review.yml`.

---

## 4. Configuration Files Reference

- **[`pyproject.toml`](file:///d:/JobSearchDecisionAssistant/pyproject.toml)**: Central configuration file for `ruff`, `pytest`, `mypy`, and `coverage`.
- **[`requirements-dev.txt`](file:///d:/JobSearchDecisionAssistant/requirements-dev.txt)**: Development and CI tools specification.
- **[`.gitignore`](file:///d:/JobSearchDecisionAssistant/.gitignore)**: Configured to exclude test caches (`.pytest_cache`, `.coverage`, `.ruff_cache`, `.mypy_cache`).
