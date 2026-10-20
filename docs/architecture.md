# 🏗️ System Architecture — AI-Powered Restaurant Recommendation System
### *Zomato Use Case — Detailed Technical Architecture*

---

## 📐 Architecture Overview

This document describes the **end-to-end technical architecture** for the AI-powered Zomato Restaurant Recommendation System. The system is designed as a **modular, layered pipeline** that combines structured data filtering with LLM-driven reasoning to deliver personalized restaurant recommendations.

---

## 🗺️ High-Level Architecture Diagram

```
┌─────────────────────────────────────────────────────────────────────┐
│                        USER INTERFACE LAYER                         │
│               (Streamlit / Gradio / Web Frontend)                   │
│    [Location] [Budget] [Cuisine] [Min Rating] [Extra Preferences]   │
└──────────────────────────────┬──────────────────────────────────────┘
                               │  User Preferences (JSON/Dict)
                               ▼
┌─────────────────────────────────────────────────────────────────────┐
│                       INPUT PROCESSING LAYER                        │
│              Validate → Normalize → Structure Inputs                │
└──────────────────────────────┬──────────────────────────────────────┘
                               │  Structured Query Object
                               ▼
┌─────────────────────────────────────────────────────────────────────┐
│                       DATA LAYER                                    │
│  ┌─────────────────────┐    ┌──────────────────────────────────┐   │
│  │  Hugging Face       │───▶│  Data Preprocessing & Cleaning   │   │
│  │  Zomato Dataset     │    │  (Pandas / Polars)               │   │
│  └─────────────────────┘    └──────────────┬─────────────────  ┘   │
│                                            │ Cleaned DataFrame      │
│                             ┌──────────────▼─────────────────  ┐   │
│                             │  Filtering Engine                 │   │
│                             │  (Location, Budget, Cuisine,      │   │
│                             │   Rating filters applied)         │   │
│                             └──────────────┬─────────────────  ┘   │
└──────────────────────────────┬─────────────┘────────────────────────┘
                               │  Top-N Filtered Restaurants
                               ▼
┌─────────────────────────────────────────────────────────────────────┐
│                     INTEGRATION / PROMPT LAYER                      │
│  ┌──────────────────────────────────────────────────────────────┐  │
│  │  Prompt Builder                                              │  │
│  │  - Serialize filtered restaurants into structured context    │  │
│  │  - Inject user preferences into prompt template             │  │
│  │  - Add ranking + explanation instructions                    │  │
│  └──────────────────────────────┬───────────────────────────── ┘  │
└──────────────────────────────────┬──────────────────────────────────┘
                               │  Crafted LLM Prompt
                               ▼
┌─────────────────────────────────────────────────────────────────────┐
│                     LLM RECOMMENDATION ENGINE                       │
│  ┌──────────────────────────────────────────────────────────────┐  │
│  │  LLM API (Groq LLaMA 3.3 / OpenAI GPT / Google Gemini)       │  │
│  │  - Reasons over filtered candidates                          │  │
│  │  - Ranks restaurants                                         │  │
│  │  - Generates personalized explanations                       │  │
│  │  - Optionally summarizes top picks                           │  │
│  └──────────────────────────────┬───────────────────────────── ┘  │
└──────────────────────────────────┬──────────────────────────────────┘
                               │  Ranked JSON Response
                               ▼
┌─────────────────────────────────────────────────────────────────────┐
│                        OUTPUT / DISPLAY LAYER                       │
│   Restaurant Cards: Name | Cuisine | Rating | Cost | AI Explanation │
└─────────────────────────────────────────────────────────────────────┘
```

---

## 📦 Module Breakdown

### 1. 🗄️ Data Layer

**Purpose**: Load, clean, and store the Zomato restaurant dataset for fast querying.

| Component              | Technology                | Responsibility                                      |
|------------------------|---------------------------|-----------------------------------------------------|
| Dataset Loader         | `datasets` (Hugging Face) | Load raw Zomato dataset from the Hub                |
| Data Preprocessor      | `pandas` / `polars`       | Clean nulls, normalize columns, map budget tiers    |
| In-Memory Store        | Pandas DataFrame / SQLite | Store processed data for low-latency lookups        |
| Schema Validator       | `pydantic`                | Validate records conform to expected schema         |

**Key Preprocessing Steps:**
- Drop rows with missing `name`, `location`, `cuisine`, or `rating`
- Normalize text fields (lowercase, strip whitespace)
- Map raw cost values → `low` / `medium` / `high` budget tiers
- Standardize ratings to a 0–5 scale

---

### 2. 🧑‍💻 User Input Layer

**Purpose**: Collect, validate, and normalize user preferences before querying the dataset.

| Input Field         | Type     | Validation Rule                              |
|---------------------|----------|----------------------------------------------|
| Location            | `string` | Must match a known city in dataset           |
| Budget              | `enum`   | One of: `low`, `medium`, `high`              |
| Cuisine             | `string` | Fuzzy-matched against dataset cuisine list   |
| Minimum Rating      | `float`  | Between `0.0` and `5.0`                      |
| Additional Notes    | `string` | Free-text, passed directly to LLM context    |

**Input Handling Flow:**
```
Raw Input → Pydantic Validation → Normalization → UserPreference Object
```

---

### 3. 🔍 Filtering Engine

**Purpose**: Reduce the full dataset to a manageable shortlist of relevant candidates before passing to the LLM.

**Filter Pipeline (Applied Sequentially):**

```python
# Step 1: Location filter (exact or partial match)
df = df[df['location'].str.contains(user.location, case=False)]

# Step 2: Cuisine filter (fuzzy match using rapidfuzz)
df = df[df['cuisine'].apply(lambda c: fuzz.partial_ratio(c, user.cuisine) > 70)]

# Step 3: Budget filter
df = df[df['budget_tier'] == user.budget]

# Step 4: Minimum rating filter
df = df[df['rating'] >= user.min_rating]

# Step 5: Limit to Top-N candidates for LLM context
df = df.sort_values('rating', ascending=False).head(15)
```

> [!NOTE]
> Fuzzy matching (via `rapidfuzz`) is used for cuisine to handle variations like *"North Indian"* vs *"Indian"*.

---

### 4. 🔗 Prompt Engineering Layer

**Purpose**: Construct a well-structured LLM prompt from filtered data and user preferences.

**Prompt Template Structure:**

```
[SYSTEM ROLE]
You are an expert restaurant recommendation assistant.
Your goal is to help users find the perfect restaurant.

[USER PREFERENCES]
- Location: {location}
- Budget: {budget}
- Cuisine: {cuisine}
- Minimum Rating: {min_rating}
- Additional Notes: {extra_preferences}

[CANDIDATE RESTAURANTS]
{serialized_restaurant_list}

[TASK]
From the candidate restaurants above:
1. Rank the top 5 restaurants based on how well they match the user's preferences.
2. For each, provide a short personalized explanation (2–3 sentences).
3. If possible, provide a brief overall summary of the recommendations.

Respond in valid JSON with the structure:
{
  "recommendations": [...],
  "summary": "..."
}
```

---

### 5. 🤖 LLM Recommendation Engine

**Purpose**: Use an LLM to perform intelligent ranking and explanation generation.

| Component          | Technology                        | Notes                                        |
|--------------------|-----------------------------------|----------------------------------------------|
| LLM Client         | `groq` / `openai` / `google-generativeai` | Primary: Groq LLaMA 3.3 70B, GPT-4o, Gemini |
| LangChain (opt.)   | `langchain`                       | For chain orchestration and prompt management|
| Response Parser    | `json` / `pydantic`               | Parse and validate structured LLM output     |
| Retry Handler      | `tenacity`                        | Auto-retry on API rate limits or timeouts    |

**LLM Call Flow:**
```
Prompt String
     │
     ▼
LLM API (Groq LLaMA 3.3 / GPT-4o / Gemini 1.5)
     │
     ▼
Raw JSON Response
     │
     ▼
Response Parser → Validated RecommendationResult Object
```

---

### 6. 🖥️ Output / Display Layer

**Purpose**: Render the final recommendations in a clean, user-friendly interface.

**Output Card Structure (per restaurant):**

```
┌──────────────────────────────────────────────────┐
│  🏠  The Great Kabab Factory          ⭐ 4.6 / 5  │
│  🍽️  Cuisine: North Indian, Mughlai               │
│  💵  Est. Cost for Two: ₹1,200 (Medium)           │
│  📍  Location: Connaught Place, Delhi             │
├──────────────────────────────────────────────────┤
│  🤖 AI Explanation:                              │
│  "A top-rated North Indian restaurant perfectly  │
│   suited for family dining in your budget range, │
│   known for its iconic kebabs and warm ambience."│
└──────────────────────────────────────────────────┘
```

---

## 🗂️ Project Directory Structure

```
Milestone-Zomato/
│
├── docs/
│   ├── problemStatement.md       # Project problem statement
│   └── architecture.md          # This document
│
├── data/
│   └── processed/               # Cached/preprocessed dataset (CSV/Parquet)
│
├── src/
│   ├── data/
│   │   ├── loader.py            # Load dataset from Hugging Face
│   │   └── preprocessor.py     # Clean and normalize data
│   │
│   ├── filters/
│   │   └── engine.py            # Filtering logic (location, budget, cuisine)
│   │
│   ├── prompt/
│   │   └── builder.py           # Prompt template construction
│   │
│   ├── llm/
│   │   ├── client.py            # LLM API wrapper (Groq / OpenAI / Gemini)
│   │   └── parser.py            # Parse and validate LLM JSON responses
│   │
│   ├── models/
│   │   └── schemas.py           # Pydantic models (UserPreference, Restaurant, etc.)
│   │
│   └── app.py                   # Main application entry point
│
├── tests/
│   ├── test_loader.py
│   ├── test_filters.py
│   └── test_llm.py
│
├── .env                         # API keys (GROQ_API_KEY / OPENAI_API_KEY / GEMINI_API_KEY)
├── requirements.txt             # Python dependencies
└── README.md                    # Project overview
```

---

## 🔄 End-to-End Data Flow

```
1. App Startup
   └──▶ Load Zomato dataset from Hugging Face Hub
   └──▶ Preprocess & cache as DataFrame

2. User Submits Preferences
   └──▶ Validate inputs (Pydantic)
   └──▶ Build UserPreference object

3. Filtering Stage
   └──▶ Apply Location → Cuisine → Budget → Rating filters
   └──▶ Return Top-15 candidates

4. Prompt Construction
   └──▶ Serialize candidates to JSON/text
   └──▶ Inject into prompt template with user preferences

5. LLM Inference
   └──▶ Send prompt to LLM API
   └──▶ Receive ranked + explained results

6. Output Rendering
   └──▶ Parse LLM JSON response
   └──▶ Render restaurant cards in UI
   └──▶ Display AI summary paragraph
```

---

## 📚 Dependencies

```txt
# Data
datasets>=2.14.0          # Hugging Face dataset loading
pandas>=2.0.0             # Data manipulation
rapidfuzz>=3.0.0          # Fuzzy string matching for cuisine

# LLM & AI
groq>=0.4.0               # Groq API client (ultra-fast LLaMA 3.3 inference)
openai>=1.0.0             # OpenAI GPT API client
google-generativeai       # Gemini Pro API client (alternative)
langchain>=0.1.0          # Optional: chain orchestration
tenacity>=8.0.0           # Retry logic for API calls

# Validation & Typing
pydantic>=2.0.0           # Schema validation

# Frontend
streamlit>=1.28.0         # Web interface (or gradio)

# Utilities
python-dotenv             # Load API keys from .env
```

---

## 🔐 Configuration & Environment

| Variable           | Description                          | Required |
|--------------------|--------------------------------------|----------|
| `GROQ_API_KEY`     | API key for Groq Cloud models        | Required if `LLM_PROVIDER=groq` |
| `GROQ_MODEL`       | Groq model (e.g., `llama-3.3-70b-versatile`) | Optional |
| `OPENAI_API_KEY`   | API key for OpenAI GPT models        | Required if `LLM_PROVIDER=openai` |
| `GEMINI_API_KEY`   | API key for Google Gemini models     | Required if `LLM_PROVIDER=gemini` |
| `LLM_PROVIDER`     | `groq` (primary), `openai`, or `gemini` | Yes      |
| `TOP_N_CANDIDATES` | Number of restaurants sent to LLM    | Default: 15 |

> [!IMPORTANT]
> At least one valid API key (`GROQ_API_KEY`, `OPENAI_API_KEY`, or `GEMINI_API_KEY`) matching `LLM_PROVIDER` must be set in the `.env` file for the recommendation engine to function.

---

## ⚠️ Key Design Decisions

| Decision                          | Rationale                                                                 |
|-----------------------------------|---------------------------------------------------------------------------|
| Filter before LLM                 | Reduces token usage and improves relevance; LLM sees only good candidates |
| Structured JSON LLM output        | Enables reliable parsing without regex hacks                             |
| Fuzzy matching for cuisine        | Handles real-world typos and synonym variations in user input            |
| Pydantic for all schemas          | Ensures type safety and clear error messages at every boundary           |
| Modular `src/` layout             | Each concern is isolated; easy to swap LLM providers or data sources     |

---

*This architecture document is the engineering blueprint for the Zomato AI Recommendation System milestone project.*
