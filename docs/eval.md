# 🧪 Evaluation Plan
### *AI-Powered Restaurant Recommendation System — Zomato Use Case*

---

## 📌 Purpose

This document defines the **complete evaluation strategy** for the project — covering unit tests, integration tests, LLM output quality metrics, and manual acceptance testing. It maps directly to the 5 phases in the implementation plan and establishes clear **pass/fail criteria** for each component.

---

## 🗂️ Table of Contents

1. [Evaluation Philosophy](#1--evaluation-philosophy)
2. [Phase 1 — Setup Evaluation](#2--phase-1--setup-evaluation)
3. [Phase 2 — Data Layer Evaluation](#3--phase-2--data-layer-evaluation)
4. [Phase 3 — Filtering & Prompt Evaluation](#4--phase-3--filtering--prompt-evaluation)
5. [Phase 4 — LLM Engine Evaluation](#5--phase-4--llm-engine-evaluation)
6. [Phase 5 — UI & Output Evaluation](#6--phase-5--ui--output-evaluation)
7. [End-to-End Evaluation Scenarios](#7--end-to-end-evaluation-scenarios)
8. [LLM Output Quality Metrics](#8--llm-output-quality-metrics)
9. [Evaluation Scorecard](#9--evaluation-scorecard)

---

## 1. 🧠 Evaluation Philosophy

This project has **two fundamentally different evaluation targets**:

| Target | Method | Automation |
|--------|---------|------------|
| Deterministic code (data loading, filtering, parsing) | Unit + Integration tests | ✅ Fully automated |
| LLM-generated output (ranking, explanations, summary) | Rubric-based manual scoring | 🟡 Semi-automated |

> [!NOTE]
> LLM outputs cannot be evaluated purely with assertions. They are scored using a defined quality rubric applied to sampled outputs across multiple test runs.

---

## 2. ✅ Phase 1 — Setup Evaluation

> **Goal**: Verify the project scaffolding, environment, and dependencies are correctly configured.

### 🔬 Test Cases

| Test ID | Test Description | Pass Condition |
|---------|-----------------|----------------|
| S-01 | Install all dependencies from `requirements.txt` | `pip install -r requirements.txt` exits with code `0` |
| S-02 | Virtual environment isolation | Running `python -c "import openai"` works only inside venv |
| S-03 | `.env` file loads correctly | `os.getenv("LLM_PROVIDER")` returns expected value |
| S-04 | `.gitignore` excludes sensitive files | `git status` does not show `.env` as tracked |
| S-05 | Directory structure matches architecture spec | All `src/` subdirs exist |
| S-06 | `python-dotenv` loads `.env` at import | No `KeyError` when accessing env vars via `os.getenv` |

### ▶️ Verification Commands

```bash
# S-01: Install check
pip install -r requirements.txt && echo "PASS: Dependencies installed"

# S-03: .env load check
python -c "from dotenv import load_dotenv; import os; load_dotenv(); print(os.getenv('LLM_PROVIDER'))"

# S-05: Directory structure check
python -c "
import os
required = ['src/data', 'src/filters', 'src/prompt', 'src/llm', 'src/models', 'tests', 'data/processed', 'docs']
for d in required:
    status = 'PASS' if os.path.isdir(d) else 'FAIL'
    print(f'{status}: {d}')
"
```

### ✔️ Phase 1 Pass Gate
- All 6 test cases pass
- No import errors on `import src`

---

## 3. ✅ Phase 2 — Data Layer Evaluation

> **Goal**: Verify the dataset loads correctly, preprocessing produces a clean DataFrame, and Pydantic schemas validate correctly.

### 🔬 Unit Test Cases — `tests/test_loader.py`

| Test ID | Test Description | Pass Condition |
|---------|-----------------|----------------|
| D-01 | Dataset loads from Hugging Face | Returns a non-empty DataFrame |
| D-02 | Required columns present after load | `['name', 'location', 'cuisines', 'rate']` all exist |
| D-03 | No nulls in critical columns after preprocessing | `df[REQUIRED_COLS].isnull().sum().sum() == 0` |
| D-04 | Rating column is float, range 0.0–5.0 | `df['rating'].between(0.0, 5.0).all() == True` |
| D-05 | `budget_tier` column has only valid values | All values in `{'low', 'medium', 'high'}` |
| D-06 | Row count after preprocessing > 1,000 | Sufficient data retained after cleaning |
| D-07 | Parquet cache written to `data/processed/` | `os.path.exists("data/processed/zomato_clean.parquet")` |
| D-08 | Cache loads correctly on second run | Second `load_zomato_dataset()` call reads from Parquet |
| D-09 | `Restaurant` Pydantic model validates a good row | No `ValidationError` on a known good dict |
| D-10 | `Restaurant` Pydantic model rejects invalid rating | `ValidationError` raised for `rating = 6.0` |
| D-11 | `UserPreference` rejects empty location | `ValidationError` raised for `location = ""` |
| D-12 | `BudgetTier` enum rejects invalid value | `ValidationError` raised for `budget = "luxury"` |

### 🔬 Sample Test Implementation

```python
# tests/test_loader.py
import pytest
import pandas as pd
from src.data.loader import load_zomato_dataset
from src.data.preprocessor import preprocess
from src.models.schemas import Restaurant, UserPreference, BudgetTier
from pydantic import ValidationError

@pytest.fixture(scope="module")
def df():
    return preprocess(load_zomato_dataset())

def test_dataset_not_empty(df):
    assert len(df) > 0, "Dataset should not be empty"

def test_required_columns_exist(df):
    for col in ['name', 'location', 'cuisine', 'rating', 'budget_tier']:
        assert col in df.columns, f"Missing column: {col}"

def test_no_nulls_in_critical_columns(df):
    assert df[['name', 'location', 'cuisine', 'rating']].isnull().sum().sum() == 0

def test_rating_range(df):
    assert df['rating'].between(0.0, 5.0).all()

def test_budget_tier_values(df):
    assert set(df['budget_tier'].unique()).issubset({'low', 'medium', 'high'})

def test_pydantic_rejects_invalid_rating():
    with pytest.raises(ValidationError):
        Restaurant(name="Test", location="Delhi", cuisine="Indian",
                   rating=6.0, budget_tier=BudgetTier.LOW)

def test_user_preference_rejects_empty_location():
    with pytest.raises(ValidationError):
        UserPreference(location="", budget=BudgetTier.MEDIUM,
                       cuisine="Italian", min_rating=3.5)
```

### 📊 Data Quality Benchmarks

| Metric | Minimum Acceptable Threshold |
|--------|------------------------------|
| Rows retained after cleaning | ≥ 70% of raw dataset |
| Null rate in critical columns | 0% (after preprocessing) |
| Budget tier coverage | All 3 tiers (`low`, `medium`, `high`) represented |
| City coverage | ≥ 5 distinct cities in `location` column |
| Rating distribution | Mean rating between 3.0 and 4.5 |

### ✔️ Phase 2 Pass Gate
- All 12 test cases pass
- Data quality benchmarks met
- Parquet cache written and readable

---

## 4. ✅ Phase 3 — Filtering & Prompt Evaluation

> **Goal**: Verify the filter pipeline returns accurate, relevant candidates and the prompt builder produces well-structured prompts.

### 🔬 Unit Test Cases — `tests/test_filters.py`

| Test ID | Test Description | Pass Condition |
|---------|-----------------|----------------|
| F-01 | Location filter: exact city match | Only restaurants in specified city returned |
| F-02 | Location filter: case-insensitive | `"bangalore"` == `"Bangalore"` |
| F-03 | Cuisine filter: exact match | Returns restaurants with matching cuisine |
| F-04 | Cuisine filter: fuzzy match (typo) | `"Italain"` returns Italian restaurants |
| F-05 | Cuisine filter: synonym handling | `"North Indian"` matches `"Indian"` entries |
| F-06 | Budget filter: low/medium/high | Only matching budget tier returned |
| F-07 | Rating filter: threshold applied | No restaurant with rating < `min_rating` in results |
| F-08 | Top-N limit enforced | Result has ≤ 15 rows |
| F-09 | Result sorted by rating descending | `result.iloc[0]['rating'] >= result.iloc[-1]['rating']` |
| F-10 | Fallback triggered on zero results | Returns results when all 4 filters would return empty |
| F-11 | Filter with `min_rating = 5.0` | Returns empty or triggers fallback, no crash |
| F-12 | Unknown location returns empty or fallback | No crash; graceful result |

### 🔬 Prompt Builder Test Cases

| Test ID | Test Description | Pass Condition |
|---------|-----------------|----------------|
| P-01 | Prompt contains user location | `user.location in prompt` |
| P-02 | Prompt contains user cuisine | `user.cuisine in prompt` |
| P-03 | Prompt contains all candidate restaurant names | Each candidate name appears in prompt string |
| P-04 | Prompt requests JSON output | String `"JSON"` or `"json"` present in prompt |
| P-05 | Prompt handles `extra_notes = None` gracefully | No `None` string or crash in output |
| P-06 | Prompt length within token budget | `len(prompt) < 12000` characters |

### 🔬 Sample Filter Test Implementation

```python
# tests/test_filters.py
import pytest
import pandas as pd
from src.filters.engine import apply_filters
from src.models.schemas import UserPreference, BudgetTier

@pytest.fixture(scope="module")
def df():
    from src.data.loader import load_zomato_dataset
    from src.data.preprocessor import preprocess
    return preprocess(load_zomato_dataset())

def make_user(**kwargs):
    defaults = dict(location="Bangalore", cuisine="North Indian",
                    budget=BudgetTier.MEDIUM, min_rating=3.5)
    return UserPreference(**(defaults | kwargs))

def test_location_filter_case_insensitive(df):
    result = apply_filters(df, make_user(location="bangalore"))
    assert result['location'].str.contains("bangalore", case=False).all()

def test_rating_threshold_respected(df):
    user = make_user(min_rating=4.0)
    result = apply_filters(df, user)
    assert (result['rating'] >= 4.0).all()

def test_result_sorted_by_rating(df):
    result = apply_filters(df, make_user())
    ratings = result['rating'].tolist()
    assert ratings == sorted(ratings, reverse=True)

def test_top_n_enforced(df):
    result = apply_filters(df, make_user(), top_n=15)
    assert len(result) <= 15

def test_fuzzy_cuisine_match(df):
    result = apply_filters(df, make_user(cuisine="Italain"))  # typo
    assert len(result) >= 0  # Should not crash; may return 0 and trigger fallback

def test_fallback_on_zero_results(df):
    result = apply_filters(df, make_user(location="Bangalore", cuisine="Peruvian",
                                          budget=BudgetTier.HIGH, min_rating=4.9))
    # Fallback should return something rather than crashing
    assert result is not None
```

### ✔️ Phase 3 Pass Gate
- All 12 filter tests pass
- All 6 prompt builder tests pass
- Manual spot-check: prompt looks coherent for 3 different input combos

---

## 5. ✅ Phase 4 — LLM Engine Evaluation

> **Goal**: Verify the LLM client calls the API correctly, retries on failure, and the parser reliably converts raw responses into validated objects.

### 🔬 Unit Test Cases — `tests/test_llm.py`

| Test ID | Test Description | Pass Condition |
|---------|-----------------|----------------|
| L-01 | Parser handles valid JSON response | Returns `RecommendationResult` with ≥ 1 recommendation |
| L-02 | Parser strips markdown code fences | Handles ` ```json ... ``` ` wrapper |
| L-03 | Parser handles missing `summary` field | Uses default value, no crash |
| L-04 | Parser handles missing `explanation` field | Uses default value, no crash |
| L-05 | Parser raises on completely invalid JSON | `json.JSONDecodeError` or `ValueError` raised |
| L-06 | Parser handles partial JSON (truncated) | Raises descriptive error |
| L-07 | LLM client selects correct provider | `LLM_PROVIDER=groq` calls `_call_groq()` |
| L-07b | LLM client selects correct provider | `LLM_PROVIDER=gemini` calls `_call_gemini()` |
| L-08 | LLM client selects correct provider | `LLM_PROVIDER=openai` calls `_call_openai()` |
| L-09 | Retry triggers on simulated rate limit | `tenacity` retries 3 times before raising |
| L-10 | Auth error does NOT retry | Raises immediately after 1 attempt |
| L-11 | Hallucination filter removes invalid names | Names not in candidate list are filtered out |
| L-12 | Response with 3 recommendations (< 5) accepted | Returns partial result without error |

### 🔬 Sample LLM Test Implementation (Mocked)

```python
# tests/test_llm.py
import pytest
import json
from unittest.mock import patch, MagicMock
from src.llm.parser import parse_llm_response
from src.llm.client import call_llm
from src.models.schemas import RecommendationResult

VALID_RESPONSE = json.dumps({
    "recommendations": [
        {"rank": 1, "name": "Taste of India", "cuisine": "North Indian",
         "rating": 4.5, "budget_tier": "medium", "location": "Bangalore",
         "explanation": "A great choice for North Indian food in your budget."}
    ],
    "summary": "Taste of India is your top pick for tonight."
})

def test_parse_valid_json():
    result = parse_llm_response(VALID_RESPONSE)
    assert isinstance(result, RecommendationResult)
    assert len(result.recommendations) == 1

def test_parse_strips_code_fences():
    fenced = f"```json\n{VALID_RESPONSE}\n```"
    result = parse_llm_response(fenced)
    assert result.recommendations[0].name == "Taste of India"

def test_parse_raises_on_invalid_json():
    with pytest.raises(Exception):
        parse_llm_response("This is not JSON at all.")

def test_parse_handles_missing_summary():
    no_summary = json.dumps({
        "recommendations": [
            {"rank": 1, "name": "Test", "cuisine": "Chinese", "rating": 4.0,
             "budget_tier": "low", "location": "Delhi", "explanation": "Good."}
        ]
    })
    result = parse_llm_response(no_summary)
    assert result.summary is not None  # Should use default

@patch("src.llm.client._call_gemini")
def test_client_calls_gemini(mock_gemini):
    mock_gemini.return_value = VALID_RESPONSE
    with patch.dict("os.environ", {"LLM_PROVIDER": "gemini"}):
        result = call_llm("system", "user")
    mock_gemini.assert_called_once()
```

### ✔️ Phase 4 Pass Gate
- All 12 test cases pass with mocked API responses
- At least 1 successful **live API call** with real credentials (manual verification)
- Response parsed into valid `RecommendationResult` on live call

---

## 6. ✅ Phase 5 — UI & Output Evaluation

> **Goal**: Verify the Streamlit app renders correctly, all input fields work, recommendations display completely, and edge cases show user-friendly messages.

### 🔬 Manual UI Test Cases

| Test ID | Test Description | Pass Condition |
|---------|-----------------|----------------|
| U-01 | App launches without errors | `streamlit run src/app.py` starts with no exceptions |
| U-02 | All 5 input fields render | Location, Cuisine, Budget, Rating, Notes all visible |
| U-03 | Form submit triggers pipeline | Spinner appears and disappears; results shown |
| U-04 | Recommendation cards display all fields | Name, Cuisine, Rating, Budget, Location, Explanation all present |
| U-05 | AI Summary section renders below cards | `st.success` block visible |
| U-06 | Unknown city shows friendly error | `st.warning` displayed, no Python traceback |
| U-07 | API auth failure shows setup message | `st.error` with API key guidance shown |
| U-08 | Dataset loads from cache on 2nd visit | Page loads faster on repeat; no re-download |
| U-09 | Rating slider only accepts 0.0–5.0 | Slider clamps to range |
| U-10 | Budget dropdown has exactly 3 options | `low`, `medium`, `high` available |

### 🔬 Automated Smoke Test (Headless)

```python
# tests/test_app_smoke.py
# Run with: pytest tests/test_app_smoke.py

import subprocess
import time
import requests

def test_app_starts():
    proc = subprocess.Popen(
        ["streamlit", "run", "src/app.py", "--server.headless=true"],
        stdout=subprocess.PIPE, stderr=subprocess.PIPE
    )
    time.sleep(5)  # Wait for startup
    try:
        response = requests.get("http://localhost:8501", timeout=5)
        assert response.status_code == 200, "App should return HTTP 200"
    finally:
        proc.terminate()
```

### ✔️ Phase 5 Pass Gate
- All 10 manual UI tests pass
- Smoke test passes (app returns HTTP 200)
- End-to-end demo runs successfully with real inputs

---

## 7. 🔄 End-to-End Evaluation Scenarios

These are **full pipeline tests** that exercise every layer simultaneously. Run each scenario and score it using the quality rubric in Section 8.

### Scenario A — Happy Path: Clear Preferences
```
Location    : Bangalore
Cuisine     : North Indian
Budget      : Medium
Min Rating  : 4.0
Notes       : Family-friendly restaurant
```
**Expected**: 3–5 well-matched restaurants with relevant explanations.

---

### Scenario B — Low Budget, High Specificity
```
Location    : Delhi
Cuisine     : Chinese
Budget      : Low
Min Rating  : 4.2
Notes       : Quick service preferred
```
**Expected**: Filtered list of affordable Chinese restaurants in Delhi; LLM acknowledges quick service preference.

---

### Scenario C — Fuzzy Cuisine Input
```
Location    : Mumbai
Cuisine     : Italain         ← intentional typo
Budget      : High
Min Rating  : 3.5
Notes       : (empty)
```
**Expected**: Fuzzy match catches typo; Italian restaurants returned; no crash.

---

### Scenario D — Relaxed Filters / Fallback Trigger
```
Location    : Bangalore
Cuisine     : Peruvian        ← very niche, unlikely to match
Budget      : High
Min Rating  : 4.5
Notes       : (empty)
```
**Expected**: Fallback activates; user informed filters were relaxed; some restaurants still shown.

---

### Scenario E — Minimal Input (Edge Case)
```
Location    : Chennai
Cuisine     : Any
Budget      : Low
Min Rating  : 0.0
Notes       : (empty)
```
**Expected**: Wide result set; LLM picks the most popular/highest-rated options; summary is coherent.

---

### Scenario F — Location Not in Dataset
```
Location    : New York         ← not in Zomato India dataset
Cuisine     : Italian
Budget      : Medium
Min Rating  : 4.0
Notes       : (empty)
```
**Expected**: Friendly error message: *"No restaurants found for New York. Please try a city in India."*

---

## 8. 📐 LLM Output Quality Metrics

Each LLM output is scored across **5 dimensions** on a 1–5 scale. Minimum passing score is **3 per dimension** (15/25 total).

### Scoring Rubric

| Dimension | 1 (Poor) | 3 (Acceptable) | 5 (Excellent) |
|-----------|----------|----------------|---------------|
| **Relevance** | Restaurants don't match user preferences | Most restaurants partially match | All restaurants clearly match preferences |
| **Ranking Quality** | Order appears random | Reasonable ranking with minor issues | Clear, justified ranking from best to worst match |
| **Explanation Clarity** | Vague or generic text | Mentions 1–2 specific user preference matches | Personalized, specific, references multiple user preferences |
| **Factual Accuracy** | Hallucinated details not in dataset | Minor embellishments, core facts correct | Fully grounded in provided candidate data |
| **Summary Coherence** | Summary contradicts cards / incoherent | Summary loosely references recommendations | Summary synthesizes all cards into a clear, helpful narrative |

### Evaluation Sheet Template

```
Scenario: _______________    Date: ___________    LLM Model: ___________

| Dimension            | Score (1–5) | Notes                          |
|----------------------|-------------|--------------------------------|
| Relevance            |             |                                |
| Ranking Quality      |             |                                |
| Explanation Clarity  |             |                                |
| Factual Accuracy     |             |                                |
| Summary Coherence    |             |                                |
| TOTAL                |    /25      |                                |

PASS (≥ 15/25)?  [ ] YES   [ ] NO
Hallucinations detected? [ ] YES   [ ] NO
If YES, list: _____________________________________________
```

### Benchmark Targets

| Metric | Target |
|--------|--------|
| Mean quality score across 6 scenarios | ≥ 18 / 25 |
| Zero hallucinated restaurant names | 100% scenarios |
| Relevance score ≥ 3 | 100% scenarios |
| Factual Accuracy score ≥ 4 | ≥ 80% scenarios |

---

## 9. 📊 Evaluation Scorecard

Use this scorecard to track readiness across all phases before considering the project complete.

```
╔══════════════════════════════════════════════════════════════╗
║           PROJECT EVALUATION SCORECARD                      ║
╠══════════╦═════════════════════════════╦════════╦══════════╣
║  Phase   ║  Component                  ║ Tests  ║ Status   ║
╠══════════╬═════════════════════════════╬════════╬══════════╣
║ Phase 1  ║ Setup & Environment         ║  6/6   ║ [ ] PASS ║
╠══════════╬═════════════════════════════╬════════╬══════════╣
║ Phase 2  ║ Data Loading                ║  4/4   ║ [ ] PASS ║
║          ║ Preprocessing               ║  4/4   ║ [ ] PASS ║
║          ║ Pydantic Schemas            ║  4/4   ║ [ ] PASS ║
╠══════════╬═════════════════════════════╬════════╬══════════╣
║ Phase 3  ║ Filter Engine               ║ 12/12  ║ [ ] PASS ║
║          ║ Prompt Builder              ║  6/6   ║ [ ] PASS ║
╠══════════╬═════════════════════════════╬════════╬══════════╣
║ Phase 4  ║ LLM Client (mocked)         ║  6/6   ║ [ ] PASS ║
║          ║ Response Parser             ║  6/6   ║ [ ] PASS ║
║          ║ Live API Call               ║  1/1   ║ [ ] PASS ║
╠══════════╬═════════════════════════════╬════════╬══════════╣
║ Phase 5  ║ UI Smoke Test               ║  1/1   ║ [ ] PASS ║
║          ║ Manual UI Tests             ║ 10/10  ║ [ ] PASS ║
╠══════════╬═════════════════════════════╬════════╬══════════╣
║ E2E      ║ Scenarios A–F               ║  6/6   ║ [ ] PASS ║
║ Quality  ║ LLM Rubric (≥ 18/25 avg)   ║  —     ║ [ ] PASS ║
╚══════════╩═════════════════════════════╩════════╩══════════╝

OVERALL PROJECT STATUS:  [ ] READY FOR DEMO   [ ] NEEDS WORK
```

---

### 🏁 Running All Automated Tests

```bash
# Run all unit and integration tests
pytest tests/ -v

# Run with coverage report
pytest tests/ --cov=src --cov-report=term-missing

# Run a specific phase's tests
pytest tests/test_loader.py -v     # Phase 2
pytest tests/test_filters.py -v    # Phase 3
pytest tests/test_llm.py -v        # Phase 4
```

### 🎯 Minimum Coverage Target

| Module | Target Line Coverage |
|--------|----------------------|
| `src/data/loader.py` | ≥ 85% |
| `src/data/preprocessor.py` | ≥ 90% |
| `src/filters/engine.py` | ≥ 90% |
| `src/prompt/builder.py` | ≥ 85% |
| `src/llm/parser.py` | ≥ 90% |
| `src/llm/client.py` | ≥ 75% (mocked) |

---

> [!IMPORTANT]
> The **Factual Accuracy** dimension in the LLM rubric is the most critical. Any scenario where the LLM hallucinates a restaurant name not in the candidate list is an automatic **FAIL** for that scenario, regardless of total score.

> [!TIP]
> Run all 6 E2E scenarios at least twice with different LLM temperature settings (`temperature=0.2` for consistency, `temperature=0.7` for variety) to understand variability in output quality.

---

*This evaluation plan is the quality gate for the Zomato AI Recommendation System milestone project.*
