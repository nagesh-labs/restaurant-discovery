"""
src/app.py
──────────
Main Streamlit application for the AI-Powered Restaurant Recommendation System.
Integrates data loading, rule-based filtering, prompt construction, and Groq LLM inference.
"""

import logging
import os
import sys
import time
from pathlib import Path
from typing import Optional

import pandas as pd
import streamlit as st
from dotenv import load_dotenv

# Ensure repo root is on sys.path
BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

# Load environment variables
load_dotenv()

from src.data.loader import load_zomato_dataset
from src.data.preprocessor import preprocess
from src.filters.engine import apply_filters
from src.llm.client import call_llm
from src.llm.parser import parse_llm_response
from src.models.schemas import BudgetTier, RecommendationResult, UserPreference
from src.prompt.builder import SYSTEM_PROMPT, build_prompt

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ── Page Configuration ────────────────────────────────────────────────────────

st.set_page_config(
    page_title="🍽️ Zomato AI Restaurant Recommender",
    page_icon="🍽️",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ── Custom CSS for Premium Visual Styling ─────────────────────────────────────

CUSTOM_CSS = """
<style>
/* Global styles */
.main-header {
    background: linear-gradient(135deg, #FF4B2B 0%, #FF416C 100%);
    padding: 28px;
    border-radius: 16px;
    color: white;
    margin-bottom: 24px;
    box-shadow: 0 8px 24px rgba(255, 75, 43, 0.25);
}
.main-header h1 {
    color: white !important;
    font-size: 2.3rem;
    font-weight: 800;
    margin-bottom: 6px;
}
.main-header p {
    color: rgba(255, 255, 255, 0.92) !important;
    font-size: 1.05rem;
    margin: 0;
}

/* Card container */
.restaurant-card {
    background: var(--background-color, #ffffff);
    border: 1px solid rgba(128, 128, 128, 0.2);
    border-radius: 14px;
    padding: 20px;
    margin-bottom: 18px;
    box-shadow: 0 4px 14px rgba(0, 0, 0, 0.05);
    transition: transform 0.2s ease, box-shadow 0.2s ease;
}
.restaurant-card:hover {
    transform: translateY(-2px);
    box-shadow: 0 8px 20px rgba(0, 0, 0, 0.09);
}

/* Badges */
.badge {
    display: inline-block;
    padding: 4px 10px;
    border-radius: 20px;
    font-size: 0.85rem;
    font-weight: 600;
    margin-right: 6px;
}
.badge-rating {
    background-color: #2e7d32;
    color: white;
}
.badge-budget {
    background-color: #0288d1;
    color: white;
}
.badge-cuisine {
    background-color: #ed6c02;
    color: white;
}
.badge-rank {
    background: linear-gradient(135deg, #FF4B2B, #FF416C);
    color: white;
    font-size: 0.95rem;
    padding: 4px 12px;
    border-radius: 12px;
}

/* AI explanation block */
.ai-quote-box {
    background-color: rgba(2, 136, 209, 0.08);
    border-left: 4px solid #0288d1;
    border-radius: 0 8px 8px 0;
    padding: 12px 16px;
    margin-top: 12px;
    font-style: normal;
}
.ai-quote-title {
    font-weight: 700;
    color: #0288d1;
    font-size: 0.85rem;
    text-transform: uppercase;
    letter-spacing: 0.5px;
    margin-bottom: 4px;
}
</style>
"""

st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


# ── Cached Data Loader ────────────────────────────────────────────────────────

@st.cache_data(show_spinner=False)
def get_dataset() -> pd.DataFrame:
    """Load and preprocess the Zomato dataset once with Streamlit caching."""
    logger.info("Loading Zomato dataset into memory...")
    raw_df = load_zomato_dataset()
    clean_df = preprocess(raw_df)
    logger.info("Loaded %d cleaned restaurant records.", len(clean_df))
    return clean_df


@st.cache_data(show_spinner=False)
def get_available_locations(df: pd.DataFrame) -> list[str]:
    """Return sorted, title-cased unique locations from the dataset."""
    raw_locs = sorted(df["location"].dropna().unique())
    acronyms = {
        "btm": "BTM",
        "hsr": "HSR",
        "mg road": "MG Road",
        "residency road": "Residency Road",
        "cv raman nagar": "CV Raman Nagar",
        "rt nagar": "RT Nagar",
        "jp nagar": "JP Nagar",
        "kr puram": "KR Puram",
        "hbr layout": "HBR Layout",
        "new bel road": "New BEL Road",
    }
    return [acronyms.get(l, l.title()) for l in raw_locs]


# ── Helpers ───────────────────────────────────────────────────────────────────

def validate_api_key() -> tuple[bool, str]:
    """Check whether the active LLM provider has a valid API key configured."""
    provider = os.getenv("LLM_PROVIDER", "groq").lower().strip()
    if provider == "groq":
        key = os.getenv("GROQ_API_KEY", "")
        if not key or key == "your_groq_api_key_here":
            return False, "GROQ_API_KEY is not configured in .env."
    elif provider == "openai":
        key = os.getenv("OPENAI_API_KEY", "")
        if not key or key == "your_openai_api_key_here":
            return False, "OPENAI_API_KEY is not configured in .env."
    elif provider == "gemini":
        key = os.getenv("GEMINI_API_KEY", "")
        if not key or key == "your_gemini_api_key_here":
            return False, "GEMINI_API_KEY is not configured in .env."
    else:
        return False, f"Unsupported LLM_PROVIDER '{provider}' configured in .env."
    return True, ""


# ── Presets Configuration ─────────────────────────────────────────────────────

PRESETS = {
    "🍕 Casual Italian (Indiranagar)": {
        "location": "Indiranagar",
        "cuisine": "Italian",
        "budget": "medium",
        "rating": 4.0,
        "notes": "Wood-fired pizza, cozy ambience for dinner with friends",
    },
    "🍛 Authentic South Indian (Jayanagar)": {
        "location": "Jayanagar",
        "cuisine": "South Indian",
        "budget": "low",
        "rating": 4.0,
        "notes": "Traditional filter coffee, crispy dosas, budget-friendly",
    },
    "🥘 Family North Indian (Koramangala)": {
        "location": "Koramangala",
        "cuisine": "North Indian",
        "budget": "medium",
        "rating": 4.0,
        "notes": "Family-friendly dining, rich gravies and fresh paneer dishes",
    },
    "☕ Rooftop Cafe & Continental (Whitefield)": {
        "location": "Whitefield",
        "cuisine": "Continental",
        "budget": "high",
        "rating": 3.8,
        "notes": "Rooftop dining with great music and dessert options",
    },
}


# ── Main UI ───────────────────────────────────────────────────────────────────

def main() -> None:
    # ── Sidebar: Metadata, Configuration & Quick Presets ──────────────────────
    with st.sidebar:
        st.markdown("## ⚙️ Configuration")

        active_provider = os.getenv("LLM_PROVIDER", "groq").upper()
        if active_provider == "GROQ":
            active_model = os.getenv("GROQ_MODEL", "openai/gpt-oss-120b")
        elif active_provider == "OPENAI":
            active_model = os.getenv("OPENAI_MODEL", "gpt-4o")
        else:
            active_model = os.getenv("GEMINI_MODEL", "gemini-1.5-pro")

        st.info(f"**LLM Provider**: {active_provider}\n\n**Model**: `{active_model}`")

        st.divider()

        st.markdown("### ⚡ Quick Presets")
        st.caption("Click any preset to prefill the preference form:")

        for preset_name, preset_vals in PRESETS.items():
            if st.button(preset_name, use_container_width=True):
                st.session_state["pref_location"] = preset_vals["location"]
                st.session_state["pref_cuisine"] = preset_vals["cuisine"]
                st.session_state["pref_budget"] = preset_vals["budget"]
                st.session_state["pref_rating"] = preset_vals["rating"]
                st.session_state["pref_notes"] = preset_vals["notes"]
                st.rerun()

        st.divider()
        st.markdown("### 📊 Dataset Info")
        st.caption(
            "• **City**: Bangalore, India\n"
            "• **Total Restaurants**: 41,654\n"
            "• **Source**: Hugging Face Hub\n"
            "• **Filtering**: RapidFuzz + Rules\n"
            "• **AI Reasoning**: Groq Cloud"
        )

    # ── Hero Banner ───────────────────────────────────────────────────────────
    st.markdown(
        """
        <div class="main-header">
            <h1>🍽️ AI-Powered Restaurant Finder</h1>
            <p>Smart, personalized dining recommendations powered by structured filtering and Groq AI reasoning.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Check API key configuration
    is_valid, key_msg = validate_api_key()
    if not is_valid:
        st.error(
            f"⚠️ **Configuration Error**: {key_msg}\n\n"
            "Please check your `.env` file and make sure your API key is correctly defined."
        )
        st.stop()

    # Load dataset
    with st.spinner("📦 Initializing restaurant database..."):
        try:
            df = get_dataset()
        except Exception as exc:
            st.error(f"❌ Failed to load restaurant dataset: {exc}")
            st.stop()

    # ── Input Form ────────────────────────────────────────────────────────────
    st.markdown("### 🎯 Your Dining Preferences")

    # Retrieve unique locations for dropdown
    available_locations = get_available_locations(df)

    # Initialize session state defaults if absent or invalid
    if "pref_location" not in st.session_state or st.session_state["pref_location"] not in available_locations:
        curr_loc = str(st.session_state.get("pref_location", "Koramangala")).lower()
        matched = next((loc for loc in available_locations if loc.lower() == curr_loc), available_locations[0])
        st.session_state["pref_location"] = matched
    if "pref_cuisine" not in st.session_state:
        st.session_state["pref_cuisine"] = "North Indian"
    if "pref_budget" not in st.session_state:
        st.session_state["pref_budget"] = "medium"
    if "pref_rating" not in st.session_state:
        st.session_state["pref_rating"] = 3.8
    if "pref_notes" not in st.session_state:
        st.session_state["pref_notes"] = "Family dining with vegetarian options"

    with st.form("preference_form", clear_on_submit=False):
        col1, col2 = st.columns(2)

        with col1:
            location_input = st.selectbox(
                "📍 Location / Area in Bangalore",
                options=available_locations,
                key="pref_location",
                help="Select a neighborhood or area in Bangalore.",
            )
            cuisine_input = st.text_input(
                "🍜 Preferred Cuisine (Optional)",
                key="pref_cuisine",
                placeholder="e.g., North Indian, Italian, Chinese (leave blank for any)",
                help="Optional: Enter a cuisine, or leave blank to search across all cuisines.",
            )

        with col2:
            budget_options = ["low", "medium", "high"]
            budget_input = st.selectbox(
                "💰 Budget Tier",
                options=budget_options,
                key="pref_budget",
                format_func=lambda b: {
                    "low": "Low (≤ ₹300 for two)",
                    "medium": "Medium (₹301 – ₹700 for two)",
                    "high": "High (> ₹700 for two)",
                }.get(b, b),
                help="Select your approximate spending tier.",
            )

            rating_input = st.slider(
                "⭐ Minimum Rating",
                min_value=0.0,
                max_value=5.0,
                key="pref_rating",
                step=0.1,
                help="Only show restaurants with at least this rating (out of 5.0).",
            )

        notes_input = st.text_area(
            "💬 Additional Notes / Special Preferences (Optional)",
            key="pref_notes",
            placeholder="e.g., quiet ambience for date night, pet-friendly outdoor seating, authentic spicy Andhra meals...",
            help="Free-text instructions passed directly to the AI for personalized ranking.",
            height=85,
        )

        top_n_candidates = int(os.getenv("TOP_N_CANDIDATES", "15"))

        submitted = st.form_submit_button(
            "🔍 Find Recommendations",
            use_container_width=True,
            type="primary",
        )

    # ── Submission & Processing Pipeline ──────────────────────────────────────
    if submitted:
        # Validate required inputs (location selected from dropdown; cuisine is optional)
        loc_clean = location_input.strip()
        cui_clean = cuisine_input.strip()

        if not loc_clean:
            st.warning("⚠️ Please select a location in Bangalore to search in.")
            return

        try:
            user_pref = UserPreference(
                location=loc_clean,
                budget=BudgetTier(budget_input),
                cuisine=cui_clean,
                min_rating=rating_input,
                extra_notes=notes_input.strip() if notes_input.strip() else None,
            )
        except Exception as err:
            st.error(f"❌ Input validation error: {err}")
            return

        # 1. Apply Rule-Based Filtering
        with st.spinner("🔍 Filtering candidates from 41,600+ restaurants..."):
            try:
                candidate_df = apply_filters(df, user_pref, top_n=top_n_candidates)
            except ValueError as val_err:
                st.warning(f"📍 {val_err}")
                return
            except Exception as exc:
                st.error(f"❌ Error during filtering: {exc}")
                return

        if candidate_df.empty:
            st.warning(
                "No restaurants found matching your exact criteria. "
                "Try lowering the minimum rating or broadening your cuisine choice."
            )
            return

        candidates_list = candidate_df.to_dict(orient="records")

        # 2. Build Structured LLM Prompt
        try:
            prompt_str = build_prompt(user_pref, candidates_list)
        except Exception as exc:
            st.error(f"❌ Error building prompt: {exc}")
            return

        # 3. Invoke LLM Inference
        start_time = time.time()
        with st.spinner(f"🤖 Consulting {active_provider} AI to rank and explain your picks..."):
            try:
                raw_response = call_llm(SYSTEM_PROMPT, prompt_str)
                duration = time.time() - start_time
            except Exception as exc:
                st.error(
                    f"❌ LLM API call failed: {exc}\n\n"
                    "Please check your API key, network connection, or try again in a few moments."
                )
                return

        # 4. Parse & Validate Structured JSON
        try:
            result: RecommendationResult = parse_llm_response(raw_response)
        except Exception as exc:
            st.error(
                f"❌ Failed to parse AI response: {exc}\n\n"
                "The model returned an unexpected format. Please try again."
            )
            with st.expander("Show raw model output for debugging"):
                st.code(raw_response)
            return

        # ── Output Display ────────────────────────────────────────────────────
        st.divider()

        # AI Summary
        st.markdown("### 📝 AI Recommendation Summary")
        st.success(f"🤖 **Overview**: {result.summary}")
        st.caption(f"⚡ *Response generated in {duration:.2f}s using {active_provider} ({active_model})*")

        # Top Recommendations Header
        st.markdown(f"### 🏆 Top {len(result.recommendations)} Personalized Recommendations")

        # Map candidate details by name (case-insensitive) for enriching display
        candidate_meta = {
            str(r.get("name", "")).lower().strip(): r for r in candidates_list
        }

        # Render cards
        for rec in result.recommendations:
            # Look up metadata from filtered dataset
            meta = candidate_meta.get(rec.name.lower().strip(), {})
            approx_cost = meta.get("approx_cost")
            cost_label = f"₹{approx_cost} for two" if approx_cost else f"{rec.budget_tier.value.capitalize()} budget"
            votes = meta.get("votes")
            votes_label = f"{votes} votes" if votes is not None else ""

            st.markdown(
                f"""
                <div class="restaurant-card">
                    <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 8px;">
                        <div>
                            <span class="badge badge-rank">#{rec.rank}</span>
                            <span style="font-size: 1.35rem; font-weight: 700; margin-left: 8px;">{rec.name}</span>
                        </div>
                        <div>
                            <span class="badge badge-rating">⭐ {rec.rating:.1f} / 5</span>
                            <span class="badge badge-budget">💰 {cost_label}</span>
                            <span class="badge badge-cuisine">🍽️ {rec.cuisine}</span>
                        </div>
                    </div>
                    <div style="color: #666; font-size: 0.95rem; margin-bottom: 8px;">
                        📍 <strong>Location:</strong> {rec.location.title()}
                        {f' &nbsp;•&nbsp; 👥 <em>{votes_label}</em>' if votes_label else ''}
                    </div>
                    <div class="ai-quote-box">
                        <div class="ai-quote-title">💡 Why this is a great fit for you</div>
                        <div>{rec.explanation}</div>
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        # Transparency expander
        with st.expander("🔍 View Candidates & Prompt Details (Transparency & Debug)"):
            st.markdown(f"**Total candidates pre-filtered from dataset:** {len(candidates_list)}")
            st.dataframe(
                pd.DataFrame(candidates_list)[
                    ["name", "location", "cuisine", "rating", "budget_tier", "approx_cost"]
                ],
                use_container_width=True,
            )
            st.markdown("**Constructed System Prompt:**")
            st.code(SYSTEM_PROMPT)
            st.markdown("**Constructed User Prompt:**")
            st.code(prompt_str)


if __name__ == "__main__":
    main()
