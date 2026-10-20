# 📋 Phase-Wise Implementation Plan
### *AI-Powered Restaurant Recommendation System — Zomato Use Case*

---

## 🗂️ Plan Overview

This document breaks the project into **5 well-defined implementation phases**, each building on the previous one. Every phase includes tasks, deliverables, files to create, and acceptance criteria.

| Phase | Name                          | Focus Area                            | Est. Effort |
|-------|-------------------------------|---------------------------------------|-------------|
| 1     | 🏗️ Project Setup & Foundation  | Environment, structure, dependencies  | ~1–2 hrs    |
| 2     | 🗄️ Data Ingestion & Processing | Dataset loading, cleaning, filtering  | ~2–3 hrs    |
| 3     | 🔗 Integration & Prompt Layer  | Candidate filtering, prompt builder (Groq-ready) | ~2–3 hrs    |
| 4     | 🤖 LLM Recommendation Engine  | API integration (Groq / OpenAI / Gemini), ranking | ~2–3 hrs    |
| 5     | 🖥️ UI & Output Display         | Streamlit app, cards, polish          | ~2–3 hrs    |

---

## ✅ Phase 1 — Project Setup & Foundation

> **Goal**: Establish the project skeleton, configure the environment, and install all dependencies so subsequent phases can build on a clean foundation.

### 📌 Tasks

- [x] **1.1** Initialize project directory structure as defined in `architecture.md`
- [x] **1.2** Create and activate a Python virtual environment
- [x] **1.3** Write `requirements.txt` with all pinned dependencies
- [x] **1.4** Create `.env` file template with required environment variables
- [x] **1.5** Create a `README.md` with project overview and setup instructions
- [x] **1.6** Set up `.gitignore` (exclude `.env`, `__pycache__`, `data/processed/`, `.venv/`)

### 📁 Files to Create

```
Milestone-Zomato/
├── .env                    ← API keys (GROQ_API_KEY / OPENAI_API_KEY / GEMINI_API_KEY)
├── .env.example            ← Template for .env (safe to commit)
├── .gitignore
├── README.md
└── requirements.txt
```

### 📦 `requirements.txt` Content

```txt
# Data
datasets>=2.14.0
pandas>=2.0.0
rapidfuzz>=3.0.0

# LLM & AI
groq>=0.4.0
openai>=1.0.0
google-generativeai>=0.3.0
langchain>=0.1.0
tenacity>=8.0.0

# Validation
pydantic>=2.0.0

# Frontend
streamlit>=1.28.0

# Utilities
python-dotenv>=1.0.0
```

### `.env.example` Template

```env
# Choose one: groq | openai | gemini
LLM_PROVIDER=groq

# Groq (if LLM_PROVIDER=groq)
GROQ_API_KEY=your_groq_api_key_here
GROQ_MODEL=llama-3.3-70b-versatile

# OpenAI (if LLM_PROVIDER=openai)
OPENAI_API_KEY=your_openai_key_here
OPENAI_MODEL=gpt-4o

# Google Gemini (if LLM_PROVIDER=gemini)
GEMINI_API_KEY=your_gemini_key_here
GEMINI_MODEL=gemini-1.5-pro

# Filtering config
TOP_N_CANDIDATES=15
```

### ✔️ Acceptance Criteria

- [x] `pip install -r requirements.txt` completes without errors
- [x] All directories from the architecture exist (even if empty)
- [x] `.env` is loaded correctly via `python-dotenv`
- [x] `.gitignore` prevents `.env` from being tracked

---

## ✅ Phase 2 — Data Ingestion & Processing

> **Goal**: Load the Zomato dataset from Hugging Face, clean and normalize it, and expose it as a queryable in-memory DataFrame ready for filtering.

### 📌 Tasks

- [x] **2.1** Implement `src/data/loader.py` — load dataset from Hugging Face Hub
- [x] **2.2** Explore the raw dataset schema; identify relevant columns
- [x] **2.3** Implement `src/data/preprocessor.py` — clean, normalize, and enrich data
- [x] **2.4** Implement `src/models/schemas.py` — define Pydantic models for `Restaurant` and `UserPreference`
- [x] **2.5** Cache the processed dataset to `data/processed/zomato_clean.parquet` for fast reloads
- [x] **2.6** Write `tests/test_loader.py` — verify data loads and schema is correct

### 📁 Files to Create / Modify

```
src/
├── data/
│   ├── loader.py           ← NEW
│   └── preprocessor.py     ← NEW
├── models/
│   └── schemas.py          ← NEW
data/
└── processed/
    └── zomato_clean.parquet  ← AUTO-GENERATED (on first run)
tests/
└── test_loader.py          ← NEW
```

### 🔑 Key Implementation Details

**`src/data/loader.py`**
```python
from datasets import load_dataset
import pandas as pd

DATASET_NAME = "ManikaSaini/zomato-restaurant-recommendation"

def load_zomato_dataset() -> pd.DataFrame:
    """Load raw Zomato dataset from Hugging Face Hub."""
    dataset = load_dataset(DATASET_NAME, split="train")
    return dataset.to_pandas()
```

**`src/data/preprocessor.py`** — Key steps:
- Drop rows where `name`, `location`, `cuisines`, or `rate` are null
- Strip whitespace and lowercase text fields
- Parse `rate` column (e.g., `"4.1/5"` → `4.1`)
- Map `approx_cost` → budget tier:
  - `≤ 300` → `"low"`, `301–700` → `"medium"`, `> 700` → `"high"`
- Rename columns to a consistent snake_case schema

**`src/models/schemas.py`** — Pydantic models:
```python
from pydantic import BaseModel, Field
from typing import Optional
from enum import Enum

class BudgetTier(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"

class Restaurant(BaseModel):
    name: str
    location: str
    cuisine: str
    rating: float
    budget_tier: BudgetTier
    approx_cost: Optional[int] = None
    votes: Optional[int] = None

class UserPreference(BaseModel):
    location: str
    budget: BudgetTier
    cuisine: str
    min_rating: float = Field(ge=0.0, le=5.0, default=3.5)
    extra_notes: Optional[str] = None
```

### ✔️ Acceptance Criteria

- [x] Dataset loads without errors from Hugging Face
- [x] Processed DataFrame has no nulls in key columns
- [x] `budget_tier` column is correctly mapped for all rows
- [x] `rating` is a float between 0.0 and 5.0 for all rows
- [x] Parquet cache file written to `data/processed/`
- [x] `test_loader.py` passes all assertions

---

## ✅ Phase 3 — Filtering Engine & Integration Layer

> **Goal**: Build the rule-based filtering engine that narrows the full dataset to relevant candidates, and the prompt builder that serializes those candidates into a structured LLM prompt.

### 📌 Tasks

- [x] **3.1** Implement `src/filters/engine.py` — sequential filter pipeline
- [x] **3.2** Handle edge cases (no results, partial matches, fallback behavior)
- [x] **3.3** Implement `src/prompt/builder.py` — prompt template and serialization
- [x] **3.4** Write `tests/test_filters.py` — unit tests for each filter step
- [x] **3.5** Test prompt output manually with sample inputs

### 📁 Files to Create

```
src/
├── filters/
│   └── engine.py           ← NEW
└── prompt/
    └── builder.py          ← NEW
tests/
└── test_filters.py         ← NEW
```

### 🔑 Key Implementation Details

**`src/filters/engine.py`** — Filter Pipeline:

```python
from rapidfuzz import fuzz
import pandas as pd
from src.models.schemas import UserPreference

def apply_filters(df: pd.DataFrame, user: UserPreference, top_n: int = 15) -> pd.DataFrame:
    # 1. Location filter
    result = df[df['location'].str.contains(user.location, case=False, na=False)]

    # 2. Cuisine filter (fuzzy match, threshold 65)
    result = result[result['cuisine'].apply(
        lambda c: fuzz.partial_ratio(str(c).lower(), user.cuisine.lower()) >= 65
    )]

    # 3. Budget filter
    result = result[result['budget_tier'] == user.budget.value]

    # 4. Minimum rating filter
    result = result[result['rating'] >= user.min_rating]

    # 5. Fallback: if fewer than 3 results, relax budget filter
    if len(result) < 3:
        result = df[
            df['location'].str.contains(user.location, case=False, na=False) &
            (df['rating'] >= user.min_rating)
        ]

    return result.sort_values('rating', ascending=False).head(top_n)
```

**`src/prompt/builder.py`** — Prompt Template:

> [!TIP]
> **Groq & LLaMA Compatibility**:
> Groq's JSON mode requires that the prompt explicitly asks for JSON. `SYSTEM_PROMPT` and `build_prompt()` specify strict JSON format with explicit schema, making the output directly compatible with Groq's `response_format={"type": "json_object"}` and models like `llama-3.3-70b-versatile`. Phase 3 is completely LLM-agnostic and produces prompts fully tested for Groq.

```python
SYSTEM_PROMPT = """You are an expert restaurant recommendation assistant for Zomato.
Your goal is to rank and explain restaurant recommendations based on user preferences.
Always respond in valid JSON only."""

def build_prompt(user: UserPreference, candidates: list[dict]) -> str:
    """Build a structured LLM prompt from user preferences and filtered candidates."""
    restaurants_text = "\n".join([
        f"{i+1}. {r['name']} | {r['cuisine']} | ⭐{r['rating']} | 💰{r['budget_tier']} | 📍{r['location']}"
        for i, r in enumerate(candidates)
    ])
    return f"""
USER PREFERENCES:
- Location: {user.location}
- Budget: {user.budget.value}
- Cuisine: {user.cuisine}
- Minimum Rating: {user.min_rating}
- Additional Notes: {user.extra_notes or 'None'}

CANDIDATE RESTAURANTS:
{restaurants_text}

TASK:
Rank the top 5 restaurants. For each, provide a 2-3 sentence personalized explanation.
Also write a brief summary of your overall recommendations.

Respond ONLY in this JSON structure:
{{
  "recommendations": [
    {{
      "rank": 1,
      "name": "...",
      "cuisine": "...",
      "rating": 4.5,
      "budget_tier": "...",
      "location": "...",
      "explanation": "..."
    }}
  ],
  "summary": "..."
}}
"""
```

### ⚠️ Edge Cases to Handle

| Scenario                         | Fallback Behavior                                         |
|----------------------------------|-----------------------------------------------------------|
| No restaurants match all filters | Relax budget filter; use location + rating only           |
| Cuisine not found in dataset     | Return top-rated restaurants in that location             |
| Fewer than 5 candidates          | Return all available; LLM ranks whatever is passed        |
| Empty location input             | Return error: "Please provide a valid location"           |

### ✔️ Acceptance Criteria

- [x] Filter pipeline correctly narrows dataset by all 4 criteria
- [x] Fuzzy cuisine matching works for common variations
- [x] Fallback logic triggers when < 3 results found
- [x] Prompt output is valid string with all user preferences injected
- [x] `test_filters.py` passes for at least 3 different input scenarios

---

## ✅ Phase 4 — LLM Recommendation Engine

> **Goal**: Connect to an LLM API (Groq as primary, OpenAI or Gemini as alternatives), send the crafted prompt, handle the response, and parse it into a validated structured output object.

### 📌 Tasks

- [x] **4.1** Implement `src/llm/client.py` — LLM API wrapper (Groq primary, multi-provider)
- [x] **4.2** Implement `src/llm/parser.py` — parse and validate LLM JSON response
- [x] **4.3** Add retry logic using `tenacity` for rate limits / timeouts
- [x] **4.4** Add `RecommendationResult` Pydantic model to `schemas.py`
- [x] **4.5** Write `tests/test_llm.py` — mock API responses and test parsing
- [x] **4.6** End-to-end test: pass real user preferences and get recommendations

### 📁 Files to Create / Modify

```
src/
├── llm/
│   ├── client.py           ← NEW
│   └── parser.py           ← NEW
└── models/
    └── schemas.py          ← MODIFY (add RecommendationResult)
tests/
└── test_llm.py             ← NEW
```

### 🔑 Key Implementation Details

**`src/llm/client.py`** — Provider-agnostic LLM wrapper:

```python
import os
from tenacity import retry, stop_after_attempt, wait_exponential
from groq import Groq

LLM_PROVIDER = os.getenv("LLM_PROVIDER", "groq")

@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
def call_llm(system_prompt: str, user_prompt: str) -> str:
    """Call the configured LLM and return raw text response."""
    if LLM_PROVIDER == "groq":
        return _call_groq(system_prompt, user_prompt)
    elif LLM_PROVIDER == "openai":
        return _call_openai(system_prompt, user_prompt)
    elif LLM_PROVIDER == "gemini":
        return _call_gemini(system_prompt, user_prompt)
    else:
        raise ValueError(f"Unsupported LLM_PROVIDER: {LLM_PROVIDER}")

def _call_groq(system_prompt: str, user_prompt: str) -> str:
    """Call Groq API using LLaMA 3.3 70B with JSON mode."""
    client = Groq(api_key=os.getenv("GROQ_API_KEY"))
    model = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        response_format={"type": "json_object"},
        temperature=0.3,
    )
    return response.choices[0].message.content
```

**`src/llm/parser.py`** — Response parsing with fallback:

```python
import json
from src.models.schemas import RecommendationResult

def parse_llm_response(raw_response: str) -> RecommendationResult:
    """Parse the LLM JSON string into a validated RecommendationResult."""
    # Strip markdown code fences if present
    cleaned = raw_response.strip().removeprefix("```json").removesuffix("```").strip()
    data = json.loads(cleaned)
    return RecommendationResult(**data)
```

**Additional schemas in `src/models/schemas.py`:**

```python
class RecommendedRestaurant(BaseModel):
    rank: int
    name: str
    cuisine: str
    rating: float
    budget_tier: BudgetTier
    location: str
    explanation: str

class RecommendationResult(BaseModel):
    recommendations: list[RecommendedRestaurant]
    summary: str
```

### 🔁 Retry Strategy

| Failure Type          | Action                                      |
|-----------------------|---------------------------------------------|
| Rate limit (429)      | Exponential backoff, retry up to 3 times    |
| Timeout               | Retry immediately once, then backoff        |
| Invalid JSON response | Re-prompt with explicit JSON correction     |
| API auth error        | Raise immediately (no retry)                |

### ✔️ Acceptance Criteria

- [x] `call_llm()` works with Groq (primary), OpenAI, and Gemini providers
- [x] Retry logic activates on simulated rate-limit errors
- [x] `parse_llm_response()` correctly parses valid JSON
- [x] Invalid/malformed JSON raises a descriptive error
- [x] `test_llm.py` passes with mocked API responses
- [x] End-to-end: given real user preferences, receive ≥ 3 ranked recommendations

---

## ✅ Phase 5 — UI, Output Display & Final Polish

> **Goal**: Build the Streamlit frontend, render AI recommendations as styled cards, wire all modules together in `app.py`, and polish the user experience.

### 📌 Tasks

- [x] **5.1** Implement `src/app.py` — main orchestration entry point
- [x] **5.2** Build Streamlit UI with input form (location, budget, cuisine, rating, notes)
- [x] **5.3** Render recommendation output as styled restaurant cards
- [x] **5.4** Display AI summary section below the cards
- [x] **5.5** Add loading spinner during LLM inference
- [x] **5.6** Add error handling & user-friendly error messages in UI
- [x] **5.7** Optimize: cache preprocessed DataFrame with `@st.cache_data`
- [x] **5.8** Final end-to-end manual test with 3–5 diverse input scenarios

### 📁 Files to Create / Modify

```
src/
└── app.py              ← NEW (main Streamlit app)
```

### 🔑 Key Implementation Details

**`src/app.py`** — Orchestration flow:

```python
import streamlit as st
from src.data.loader import load_zomato_dataset
from src.data.preprocessor import preprocess
from src.filters.engine import apply_filters
from src.prompt.builder import SYSTEM_PROMPT, build_prompt
from src.llm.client import call_llm
from src.llm.parser import parse_llm_response
from src.models.schemas import UserPreference, BudgetTier

@st.cache_data
def get_dataset():
    return preprocess(load_zomato_dataset())

def main():
    st.set_page_config(page_title="🍽️ Zomato AI Recommender", layout="wide")
    st.title("🍽️ AI-Powered Restaurant Finder")

    df = get_dataset()

    with st.form("preferences_form"):
        location = st.text_input("📍 Location", placeholder="e.g., Bangalore")
        cuisine  = st.text_input("🍜 Cuisine", placeholder="e.g., Italian")
        budget   = st.selectbox("💰 Budget", ["low", "medium", "high"])
        rating   = st.slider("⭐ Minimum Rating", 0.0, 5.0, 3.5, 0.1)
        notes    = st.text_area("💬 Additional Preferences (optional)")
        submitted = st.form_submit_button("🔍 Find Restaurants")

    if submitted:
        user = UserPreference(
            location=location, cuisine=cuisine,
            budget=BudgetTier(budget), min_rating=rating, extra_notes=notes
        )
        candidates = apply_filters(df, user)

        if candidates.empty:
            st.warning("No restaurants found. Try relaxing your filters.")
            return

        prompt = build_prompt(user, candidates.to_dict(orient="records"))

        with st.spinner("🤖 AI is finding your perfect restaurant..."):
            raw = call_llm(SYSTEM_PROMPT, prompt)
            result = parse_llm_response(raw)

        # Render recommendations
        st.subheader(f"🏆 Top {len(result.recommendations)} Recommendations")
        for rec in result.recommendations:
            with st.container():
                st.markdown(f"### {rec.rank}. {rec.name}")
                col1, col2, col3 = st.columns(3)
                col1.metric("⭐ Rating", rec.rating)
                col2.metric("💰 Budget", rec.budget_tier.capitalize())
                col3.metric("🍽️ Cuisine", rec.cuisine)
                st.markdown(f"📍 **Location**: {rec.location}")
                st.info(f"🤖 **AI Says**: {rec.explanation}")
                st.divider()

        # Summary
        st.subheader("📝 AI Summary")
        st.success(result.summary)

if __name__ == "__main__":
    main()
```

### 🎨 UI Checklist

| UI Element              | Implementation                                  |
|-------------------------|-------------------------------------------------|
| Input Form              | `st.form` with all 5 preference fields          |
| Loading Indicator       | `st.spinner` wrapping LLM call                  |
| Recommendation Cards    | `st.container` + `st.columns` for metrics       |
| AI Explanation          | `st.info` block per restaurant                  |
| Summary Block           | `st.success` for the overall summary            |
| Error Messages          | `st.warning` / `st.error` for edge cases        |
| Performance             | `@st.cache_data` on dataset loading             |

### ▶️ Run Command

```bash
streamlit run src/app.py
```

### ✔️ Acceptance Criteria

- [x] Streamlit app launches without errors on `streamlit run src/app.py`
- [x] All 5 input fields render correctly
- [x] Submitting form triggers filtering → prompt → LLM → display pipeline
- [x] At least 3 ranked restaurant cards render with all fields populated
- [x] AI summary appears below the cards
- [x] Loading spinner appears during LLM call
- [x] Graceful error message shown if no restaurants match
- [x] Dataset loads once and is cached (fast on subsequent queries)

---

## 📊 Overall Progress Tracker

```
Phase 1 — Setup & Foundation        [x] [x] [x] [x] [x] [x]
Phase 2 — Data Ingestion            [x] [x] [x] [x] [x] [x]
Phase 3 — Filtering & Prompt        [x] [x] [x] [x] [x]
Phase 4 — LLM Engine                [x] [x] [x] [x] [x] [x]
Phase 5 — UI & Display              [x] [x] [x] [x] [x] [x] [x] [x]
```

---

## 🔗 Cross-References

| Document                | Purpose                                         |
|-------------------------|-------------------------------------------------|
| [problemStatement.md](problemStatement.md) | Project objectives and requirements |
| [architecture.md](architecture.md)         | Technical design and module specs   |

---

> [!TIP]
> Work through phases sequentially. Each phase has a clear set of acceptance criteria — don't move to the next phase until the current one's criteria are met.

> [!IMPORTANT]
> Phase 2 (Data) and Phase 3 (Prompt) can be validated entirely **without an LLM API key**. Keep them decoupled to make testing faster and cheaper.

---

*This implementation plan is the actionable roadmap for building the Zomato AI Recommendation System milestone project.*
