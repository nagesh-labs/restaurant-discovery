# 🍽️ AI-Powered Restaurant Recommendation System
### *Zomato Use Case — Hyperlocal AI Discovery & Recommendation Engine*

---

## 📌 Overview

An AI-powered restaurant recommendation service inspired by Zomato. The system combines **structured rule-based data filtering** with a **Large Language Model (LLM)** to deliver personalized, human-like restaurant recommendations based on user preferences.

By pre-filtering thousands of restaurant records down to an optimal candidate shortlist before invoking the LLM, the system achieves sub-second latency, eliminates token waste, avoids hallucinations, and produces strictly validated, structured JSON recommendations.

---

## 🎯 Features

- 🔍 **Multi-Stage Rule Filtering**: Filter restaurants by location, fuzzy cuisine matching, budget tier, and minimum rating.
- 🤖 **LLM-Powered Ranking & Reasoning**: Context-aware ranking that weighs user preferences, cuisine nuances, and free-text notes.
- 💬 **Personalized Explanations**: Generates natural, compelling 2–3 sentence rationales for every recommended pick.
- 📝 **AI Executive Summary**: Synthesizes the overall recommendation set into a coherent dining overview.
- 🛡️ **Multi-Tier Scope Limits**: Strict controls on candidate limits, output counts, and input token budgets.
- 🔄 **Fault-Tolerant Parsing**: Resilient JSON parser with markdown fence stripping, regex extraction, and schema fallbacks.
- ⚡ **Multi-Provider Support**: Seamless switching between Groq (LLaMA 3.3 70B), OpenAI (GPT-4o), and Google Gemini (Gemini 1.5 Pro).
- 🖥️ **Dual Frontend Experience**: Modern, fast ASGI Web App (Stitch Hyperlocal Sensory UI) plus a Streamlit exploratory UI.

---

## 🧠 System Prompt

The system prompt defines the AI's role, behavioral boundaries, output style, and strict JSON output requirement. It is defined in [`src/prompt/builder.py`](file:///c:/GenAI_Projects/Milestone-Zomato/src/prompt/builder.py):

```python
SYSTEM_PROMPT = """You are an expert restaurant recommendation assistant for Zomato.
Your goal is to rank and explain restaurant recommendations based on user preferences.
Be warm, specific, and helpful. Highlight what makes each restaurant a great fit.
Always respond in valid JSON only — no markdown, no prose outside the JSON."""
```

### Breakdown of System Prompt Directives:

| Directive Component | Prompt Phrase | Purpose & Justification |
|---------------------|---------------|-------------------------|
| **Persona & Domain** | `You are an expert restaurant recommendation assistant for Zomato.` | Establishes the agent's domain context, expertise, and authority in culinary discovery. |
| **Core Objective** | `Your goal is to rank and explain restaurant recommendations based on user preferences.` | Directs the model to evaluate the candidate list against the user's explicit criteria. |
| **Tone & Style** | `Be warm, specific, and helpful. Highlight what makes each restaurant a great fit.` | Prevents dry, robotic listings; drives the LLM to highlight standout attributes (signature dishes, vibe, value). |
| **Negative Formatting Constraints** | `Always respond in valid JSON only — no markdown, no prose outside the JSON.` | Crucial for API JSON modes (especially Groq & OpenAI); prevents chat conversational prefixes (e.g., *"Here are your recommendations:"*) and markdown wrappers (```` ```json ... ``` ````) that cause JSON deserialization errors. |

---

## 📋 The Response Schema

The recommendation engine requires the LLM to produce a strictly structured JSON response. This schema is formally specified in [`src/models/schemas.py`](file:///c:/GenAI_Projects/Milestone-Zomato/src/models/schemas.py) using **Pydantic v2** models (`RecommendationResult` and `RecommendedRestaurant`).

### 1. JSON Wire Schema

```json
{
  "recommendations": [
    {
      "rank": 1,
      "name": "Truffles",
      "cuisine": "American, Fast Food, Burger",
      "rating": 4.7,
      "budget_tier": "medium",
      "location": "Koramangala",
      "explanation": "Famous for high-quality gourmet burgers and decadent desserts, Truffles delivers exceptional value and a vibrant atmosphere that fits your medium budget perfectly."
    }
  ],
  "summary": "Both Truffles and Meghana Foods offer top-tier culinary experiences in Koramangala, combining popular signature dishes with great value."
}
```

### 2. Pydantic Domain Models

```python
class BudgetTier(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"

class RecommendedRestaurant(BaseModel):
    rank: int = Field(..., ge=1, description="Ranking position (1 = best)")
    name: str = Field(..., description="Restaurant name matching candidate list")
    cuisine: str = Field(..., description="Cuisine types served")
    rating: float = Field(..., ge=0.0, le=5.0, description="Customer rating on 0-5 scale")
    budget_tier: BudgetTier = Field(..., description="Cost classification: low | medium | high")
    location: str = Field(..., description="Neighborhood or area")
    explanation: str = Field(
        default="Recommended match based on your preferences.",
        description="Personalized 2-3 sentence AI explanation"
    )

class RecommendationResult(BaseModel):
    recommendations: list[RecommendedRestaurant] = Field(
        ..., min_length=1, description="Ranked list of recommended restaurants"
    )
    summary: str = Field(
        default="Here are the top restaurant recommendations based on your preferences.",
        description="Overall AI-generated synthesis"
    )
```

### 3. Fault-Tolerant Parsing & Self-Healing Pipeline

In [`src/llm/parser.py`](file:///c:/GenAI_Projects/Milestone-Zomato/src/llm/parser.py), incoming LLM strings pass through a robust normalization and validation pipeline before schema instantiation:
- **Markdown Fence Stripping**: Automatically removes leading/trailing ```` ```json ```` and ```` ``` ```` tags.
- **Regex Outermost Object Extraction**: If conversational text precedes or succeeds the JSON block, regex extracts `{ ... }`.
- **Alternative Key Normalization**: Detects and maps common variations (`restaurants`, `results`, `picks`, `data`) into `recommendations`.
- **Case-Insensitive Enum Mapping**: Sanitizes budget values like `"Medium"`, `"MEDIUM"`, or `"low "`.
- **Missing Field Fallbacks**: Injects contextual defaults for missing `explanation` or `summary` without failing the request.

---

## 📈 What Changed Across Prompt Versions and Why

During development and evaluation ([`docs/eval.md`](file:///c:/GenAI_Projects/Milestone-Zomato/docs/eval.md) and [`docs/edge-cases.md`](file:///c:/GenAI_Projects/Milestone-Zomato/docs/edge-cases.md)), the prompt evolved through three major iterations to resolve hallucinations, formatting inconsistencies, and token inefficiencies:

```
┌─────────────────────────────────┐
│     Prompt Version 1 (v1)       │  • Generic assistant prompt
│       Initial Baseline          │  • Fixed "Rank top 5" instruction
│                                 │  • Minimal candidate string
└───────────────┬─────────────────┘
                │ ⚠️ Issues: Hallucinations on <5 candidates; markdown backticks broke json.loads;
                │    missing pricing details led to vague explanations.
                ▼
┌─────────────────────────────────┐
│     Prompt Version 2 (v2)       │  • Introduced structured schema block
│      Schema Stabilization       │  • Standardized Candidate serialization line
│                                 │  • Added Groq JSON mode compatibility
└───────────────┬─────────────────┘
                │ ⚠️ Issues: Fixed "top 5" still forced LLM to invent missing records if only
                │    2 or 3 candidates survived strict filters; "None" text leaked when notes were empty.
                ▼
┌─────────────────────────────────┐
│     Prompt Version 3 (v3)       │  • Dynamic candidate cap: min(5, len(candidates))
│     Current Production Prompt   │  • Strict negative formatting constraints
│                                 │  • Added exact ₹ cost for two in candidate serialization
│                                 │  • Explicit handling for empty cuisine/notes defaults
└─────────────────────────────────┘
```

### Version Comparison & Design Rationale

| Feature / Aspect | Version 1 (Baseline) | Version 2 (Schema Stabilization) | Version 3 (Current Production) | Rationale & Problem Solved |
|------------------|----------------------|-----------------------------------|--------------------------------|----------------------------|
| **Target Count** | Fixed *"Rank the top 5 restaurants"* | Fixed *"Rank the top 5 restaurants"* | **Dynamic**: `top_k = min(5, n_candidates)` | When strict filters yield 1–4 candidates, asking for "top 5" forces LLMs to hallucinate restaurants not in the candidate list. |
| **System Directive** | *"You are a helpful assistant..."* | *"You are an expert restaurant recommendation assistant for Zomato..."* | **+ Warm persona & strict negative constraint**: *"valid JSON only — no markdown, no prose outside the JSON"* | Prevents conversational preambles and markdown fences that break automated `json.loads()`. Compatible with Groq JSON mode. |
| **Candidate Serialization** | Unformatted text dump | `1. Name \| Cuisine \| ⭐Rating \| 💰Budget \| 📍Location` | **Enriched**: `1. Name \| Cuisine \| ⭐Rating \| 💰Budget (₹Cost for two) \| 📍Location` | Including rupee cost for two enables the LLM to give grounded pricing explanations (e.g., *"At ₹600 for two..."*). |
| **User Preference Fallbacks** | Passed raw string or `None` | Passed raw string | **Sanitized**: Title-cased inputs; `"Any (all cuisines)"` if empty; `"None"` for blank notes | Prevents string conversion crashes and avoids confusing the model with raw `None` objects. |
| **Candidate Count Header** | `CANDIDATE RESTAURANTS:` | `CANDIDATE RESTAURANTS:` | `CANDIDATE RESTAURANTS ({n_candidates} options):` | Explicitly reminds the LLM of the candidate boundaries to prevent hallucinating extra venues. |

---

## 🛡️ How the Scope Limit is Enforced

The system enforces scope limits across **5 distinct architectural layers**, ensuring token efficiency, bounded latency, factual grounding, and protection against prompt injection or out-of-distribution queries:

```
[ Incoming User Request ]
           │
           ▼
┌─────────────────────────────────────────────────────────────┐
│ 1. Input Boundary Validation                                │
│    • Location required (min length 1)                       │
│    • Rating clamped to [0.0, 5.0]                           │
│    • Budget constrained to BudgetTier Enum                 │
│    • Text fields stripped, lowercased, and sanitized        │
└──────────────────────────────┬──────────────────────────────┘
                               ▼
┌─────────────────────────────────────────────────────────────┐
│ 2. Pre-LLM Structured Filtering (Candidate Scope Cap)      │
│    • Hard cap: TOP_N_CANDIDATES = 15 (default)             │
│    • Sort by rating DESC -> .head(top_n)                    │
│    • Narrows 10,000+ dataset rows to <= 15 items            │
│    • Keeps prompt token footprint under ~1,500 tokens       │
└──────────────────────────────┬──────────────────────────────┘
                               ▼
┌─────────────────────────────────────────────────────────────┐
│ 3. Progressive Fallback Ladder                              │
│    • If matches < 3: Relax Cuisine constraint               │
│    • If still < 3: Relax Budget constraint                  │
│    • Hard Boundary: Location NEVER relaxed (raises 404)     │
└──────────────────────────────┬──────────────────────────────┘
                               ▼
┌─────────────────────────────────────────────────────────────┐
│ 4. Dynamic LLM Output Scope Constraint                      │
│    • top_k = min(5, len(candidates))                        │
│    • Prompt: "rank the top {top_k} that best match..."      │
│    • Model cannot recommend more than available candidates  │
└──────────────────────────────┬──────────────────────────────┘
                               ▼
┌─────────────────────────────────────────────────────────────┐
│ 5. Post-Inference Validation & Grounding Enforcement        │
│    • Pydantic schema validation (types, ranges, enums)      │
│    • Candidate name cross-referencing                       │
│    • Automatic rejection of ungrounded / hallucinated items │
└─────────────────────────────────────────────────────────────┘
```

1. **Pre-LLM Structured Filtering (Candidate Cap)**:
   - The raw Zomato dataset contains thousands of restaurants. Sending all records to an LLM would blow through context limits and incur massive token costs.
   - [`src/filters/engine.py`](file:///c:/GenAI_Projects/Milestone-Zomato/src/filters/engine.py) applies fast rule-based Pandas filters (Location $\rightarrow$ Cuisine $\rightarrow$ Budget $\rightarrow$ Rating).
   - Results are capped using `top_n = DEFAULT_TOP_N` (15 candidates maximum, configurable via `.env`). This limits prompt size to $\approx$ 1,500 tokens.

2. **Progressive Relaxation Bounds (Fallback Ladder)**:
   - If strict criteria return fewer than 3 candidates (`FALLBACK_MIN_RESULTS = 3`), the engine automatically triggers controlled relaxation:
     - *Fallback 1*: Relaxes cuisine matching while preserving location and budget.
     - *Fallback 2*: Relaxes budget while preserving location and minimum rating.
   - **Hard Boundary**: Location is **never** relaxed. If a location does not exist in the dataset, a `ValueError` is raised immediately.

3. **Dynamic Prompt Output Scope (`top_k = min(5, n_candidates)`)**:
   - In [`src/prompt/builder.py`](file:///c:/GenAI_Projects/Milestone-Zomato/src/prompt/builder.py), the prompt dynamically computes `top_k = min(5, len(candidates))` and instructs: *"From the candidate restaurants above, rank the top {top_k}..."*. The LLM is never asked to produce more recommendations than candidates provided.

4. **Input Field Scope & Injection Defense**:
   - Pydantic models validate input types: `min_rating` is bounded between `0.0` and `5.0`; `budget` is constrained to the `BudgetTier` enum (`low`, `medium`, `high`).
   - Free-text `extra_notes` are sanitized and injected into a clearly delimited section, preventing prompt injection attacks from altering system instructions.

5. **Anti-Hallucination Post-Validation Grounding**:
   - The parser verifies each recommendation against the known candidate pool. If an LLM attempts to inject an ungrounded restaurant name, it is flagged and filtered out before rendering.

---

## 🛠️ Tech Stack

```
──────────────────────────────────────────────────────────────────────────────
Component                 Technology & Version        Role in Architecture
──────────────────────────────────────────────────────────────────────────────
Primary LLM Provider      Groq (LLaMA 3.3 70B)        Ultra-fast inference & reasoning
Alternative LLMs          OpenAI (GPT-4o)             High-accuracy multimodal / reasoning
                          Google Gemini (1.5 Pro)     Long-context reasoning fallback
Resilience / Retry        Tenacity (>=8.0.0)          Exponential backoff on 429/503 errors
Data Processing           Pandas (>=2.0.0)            In-memory candidate filtering
Fuzzy Matching            RapidFuzz (>=3.0.0)         Partial ratio matching for cuisines
Schema Validation         Pydantic v2 (>=2.0.0)       Domain modeling & strict JSON validation
Data Source & Storage     Hugging Face Datasets       Zomato restaurant dataset ingestion
                          Apache Arrow / Parquet      Sub-10ms cached dataset loading
Async Backend             Starlette / Uvicorn         High-throughput ASGI REST API server
Modern Web App            HTML5 / Vanilla CSS / JS   Glassmorphic "Stitch" UI (port 8000)
Exploratory UI            Streamlit (>=1.28.0)        Data science dashboard & inspector
Testing Framework         Pytest & Pytest-Cov         Unit, integration, and smoke tests
Cloud Deployments         Railway & Vercel            Containerized backend & CDN frontend
──────────────────────────────────────────────────────────────────────────────
```

---

## 🗂️ Project Structure

```
Milestone-Zomato/
│
├── docs/
│   ├── problemStatement.md       # Project objectives & requirements
│   ├── architecture.md           # Detailed system architecture & data flow
│   ├── implementation-plan.md    # Phase-by-phase implementation guide
│   ├── edge-cases.md             # 20+ edge cases and mitigation strategies
│   └── eval.md                   # Evaluation rubrics, test cases & scorecard
│
├── data/
│   └── processed/                # Cached clean dataset (zomato_clean.parquet)
│
├── frontend/                     # Modern Hyperlocal Sensory AI Web Interface
│   ├── index.html                # Single-page discovery app
│   ├── app.js                    # Interactive state, API client & modal logic
│   └── styles.css                # Glassmorphism, animations & design tokens
│
├── src/
│   ├── data/
│   │   ├── loader.py             # Ingest dataset from Hugging Face Hub
│   │   └── preprocessor.py       # Clean, normalize & map budget tiers
│   │
│   ├── filters/
│   │   └── engine.py             # 4-stage filter pipeline & fallback ladder
│   │
│   ├── prompt/
│   │   └── builder.py            # System prompt & dynamic prompt constructor
│   │
│   ├── llm/
│   │   ├── client.py             # Multi-provider client (Groq, OpenAI, Gemini)
│   │   └── parser.py             # Fault-tolerant JSON parser & validator
│   │
│   ├── models/
│   │   └── schemas.py            # Pydantic v2 domain schemas & enums
│   │
│   ├── server.py                 # Starlette REST backend API server
│   └── app.py                    # Streamlit fallback application
│
├── tests/
│   ├── test_loader.py            # Phase 2: Ingestion & preprocessing tests
│   ├── test_filters.py           # Phase 3: Filter & prompt builder tests
│   ├── test_llm.py               # Phase 4: Parser, client & retry tests
│   ├── test_server.py            # Phase 5: REST API endpoints tests
│   └── test_app.py               # Phase 5: Streamlit integration tests
│
├── run_web.py                    # Entry point for production Starlette web app
├── Procfile                      # Process configuration for PaaS (Railway)
├── railway.json                  # Railway deployment configuration
├── vercel.json                   # Vercel frontend reverse proxy configuration
├── requirements.txt              # Pinned Python dependencies
├── .env.example                  # Environment configuration template
└── README.md                     # Project documentation
```

---

## ⚡ Quick Start

### 1. Clone the Repository

```bash
git clone <your-repo-url>
cd Milestone-Zomato
```

### 2. Set Up Virtual Environment

```bash
# Windows
python -m venv .venv
.venv\Scripts\activate

# macOS / Linux
python -m venv .venv
source .venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Configure Environment Variables

```bash
# Copy template
cp .env.example .env
```

Edit `.env` with your API credentials:
```env
LLM_PROVIDER=groq
GROQ_API_KEY=gsk_your_actual_groq_key_here
GROQ_MODEL=llama-3.3-70b-versatile
TOP_N_CANDIDATES=15
```

### 5. Run the Application

#### 🚀 Option A: Hyperlocal Sensory Web App (Recommended)
Launch the modern, responsive web application:

```bash
python run_web.py
```
Open **http://localhost:8000** in your browser.

#### 🖥️ Option B: Streamlit Exploratory Interface
```bash
streamlit run src/app.py
```
Open **http://localhost:8501** in your browser.

---

## 🧪 Running Tests

The test suite covers data loading, preprocessing, filter accuracy, fallback logic, prompt assembly, and mocked LLM responses:

```bash
# Run all unit and integration tests
pytest tests/ -v

# Run with line coverage report
pytest tests/ --cov=src --cov-report=term-missing
```

---

## 🔐 Environment Variables

| Variable | Description | Default | Required |
|----------|-------------|---------|----------|
| `LLM_PROVIDER` | Active provider: `groq`, `openai`, or `gemini` | `groq` | ✅ Yes |
| `GROQ_API_KEY` | API key from [Groq Console](https://console.groq.com) | — | If using Groq |
| `GROQ_MODEL` | Groq model identifier | `llama-3.3-70b-versatile` | ❌ No |
| `OPENAI_API_KEY` | API key from OpenAI Platform | — | If using OpenAI |
| `OPENAI_MODEL` | OpenAI model identifier | `gpt-4o` | ❌ No |
| `GEMINI_API_KEY` | API key from Google AI Studio | — | If using Gemini |
| `GEMINI_MODEL` | Gemini model identifier | `gemini-1.5-pro` | ❌ No |
| `TOP_N_CANDIDATES` | Candidate pool limit sent to LLM | `15` | ❌ No |
| `PORT` | ASGI server listening port | `8000` | ❌ No |
| `HOST` | ASGI server listening host | `127.0.0.1` | ❌ No |

---

## 📚 Documentation Links

| Document | Description |
|----------|-------------|
| [Problem Statement](docs/problemStatement.md) | Business requirements, objectives, and success criteria |
| [Architecture Document](docs/architecture.md) | Detailed technical architecture, pipeline data flow, and module contracts |
| [Implementation Plan](docs/implementation-plan.md) | Phase-by-phase build log, milestones, and testing gates |
| [Edge Cases Document](docs/edge-cases.md) | 20+ documented edge cases across data, filters, LLM, and UI |
| [Evaluation Plan](docs/eval.md) | Test cases, quality evaluation rubrics, and project scorecard |

---

## 📦 Dataset

- **Name**: `ManikaSaini/zomato-restaurant-recommendation`
- **Source**: [Hugging Face Hub](https://huggingface.co/datasets/ManikaSaini/zomato-restaurant-recommendation)
- **Local Cache**: Auto-downloaded and stored as an optimized Parquet file at `data/processed/zomato_clean.parquet`.

---

*Built with passion as an AI Engineering Milestone Project.*

