# 🍽️ AI-Powered Restaurant Recommendation System
### *Zomato Use Case — Milestone Project*

---

## 📌 Overview

An AI-powered restaurant recommendation service inspired by Zomato. The system combines **structured data filtering** with a **Large Language Model (LLM)** to deliver personalized, human-like restaurant recommendations based on user preferences.

---

## 🎯 Features

- 🔍 Filter restaurants by **location**, **cuisine**, **budget**, and **minimum rating**
- 🤖 LLM-powered **ranking and explanation** of recommendations
- 💬 Personalized, human-readable explanations for each pick
- 📝 AI-generated **summary** of the overall recommendations
- 🖥️ Clean **Streamlit** web interface

---

## 🗂️ Project Structure

```
Milestone-Zomato/
│
├── docs/
│   ├── problemStatement.md       # Project requirements
│   ├── architecture.md           # System design & module specs
│   ├── implementation-plan.md    # Phase-wise build plan
│   ├── edge-cases.md             # Edge cases & handling strategies
│   └── eval.md                   # Evaluation plan & test cases
│
├── data/
│   └── processed/                # Auto-generated: cleaned dataset cache
│
├── src/
│   ├── data/
│   │   ├── loader.py             # Load dataset from Hugging Face
│   │   └── preprocessor.py      # Clean and normalize data
│   │
│   ├── filters/
│   │   └── engine.py             # Filtering pipeline (location, budget, cuisine)
│   │
│   ├── prompt/
│   │   └── builder.py            # Prompt template construction
│   │
│   ├── llm/
│   │   ├── client.py             # LLM API wrapper (Groq / OpenAI / Gemini)
│   │   └── parser.py             # Parse and validate LLM JSON responses
│   │
│   ├── models/
│   │   └── schemas.py            # Pydantic models
│   │
│   └── app.py                    # Main Streamlit application
│
├── tests/
│   ├── test_loader.py
│   ├── test_filters.py
│   └── test_llm.py
│
├── .env                          # Your API keys (NOT committed)
├── .env.example                  # Template for .env
├── .gitignore
├── requirements.txt
└── README.md
```

---

## ⚡ Quick Start

### 1. Clone the Repository

```bash
git clone <your-repo-url>
cd Milestone-Zomato
```

### 2. Create & Activate Virtual Environment

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
# Copy the template
cp .env.example .env

# Edit .env and add your API key
# Set LLM_PROVIDER to "groq", "openai", or "gemini"
```

Your `.env` should look like:
```env
LLM_PROVIDER=groq
GROQ_API_KEY=your_actual_key_here
GROQ_MODEL=llama-3.3-70b-versatile
TOP_N_CANDIDATES=15
```

### 5. Run the Application

#### ✨ Option A: New Full-Stack Web App (Stitch "Hyperlocal Sensory AI" Design)
Launch the modern, responsive web application matching the Stitch design system:

```bash
python run_web.py
```
Open your browser at **http://localhost:8000** 🚀

#### 🖥️ Option B: Streamlit Web UI (Fallback)
```bash
streamlit run src/app.py
```
Open your browser at **http://localhost:8501**

---

## 🧪 Running Tests

```bash
# Run all tests
pytest tests/ -v

# Run with coverage report
pytest tests/ --cov=src --cov-report=term-missing
```

---

## 🛠️ Tech Stack

| Component            | Technology                            |
|----------------------|---------------------------------------|
| Dataset              | Hugging Face — `ManikaSaini/zomato-restaurant-recommendation` |
| LLM                  | Groq (LLaMA 3.3 70B) / OpenAI GPT-4o / Google Gemini Pro |
| Data Processing      | Pandas, RapidFuzz                     |
| Schema Validation    | Pydantic v2                           |
| Frontend             | Streamlit                             |
| Retry Logic          | Tenacity                              |
| Testing              | Pytest + pytest-cov                   |

---

## 🔐 Environment Variables

| Variable           | Description                          | Required |
|--------------------|--------------------------------------|----------|
| `LLM_PROVIDER`     | `groq` (primary), `openai`, or `gemini` | ✅ Yes   |
| `GROQ_API_KEY`     | API key for Groq Cloud               | If using Groq |
| `GROQ_MODEL`       | e.g., `llama-3.3-70b-versatile`      | If using Groq |
| `OPENAI_API_KEY`   | API key for OpenAI GPT models        | If using OpenAI |
| `OPENAI_MODEL`     | e.g., `gpt-4o`                       | If using OpenAI |
| `GEMINI_API_KEY`   | API key for Google Gemini models     | If using Gemini |
| `GEMINI_MODEL`     | e.g., `gemini-1.5-pro`              | If using Gemini |
| `TOP_N_CANDIDATES` | Max restaurants passed to LLM        | ❌ Default: 15 |

---

## 📚 Documentation

| Document | Description |
|----------|-------------|
| [Problem Statement](docs/problemStatement.md) | Project objectives and requirements |
| [Architecture](docs/architecture.md) | System design, data flow, module specs |
| [Implementation Plan](docs/implementation-plan.md) | Phase-wise build guide with tasks |
| [Edge Cases](docs/edge-cases.md) | Corner scenarios and handling strategies |
| [Evaluation Plan](docs/eval.md) | Test cases, quality rubric, scorecard |

---

## 📦 Dataset

- **Name**: `ManikaSaini/zomato-restaurant-recommendation`
- **Source**: [Hugging Face Hub](https://huggingface.co/datasets/ManikaSaini/zomato-restaurant-recommendation)
- **Auto-loaded** on first run and cached to `data/processed/zomato_clean.parquet`

---

*Built as a Generative AI Milestone Project.*
