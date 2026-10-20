"""
scripts/test_groq_live.py
─────────────────────────
Performs exactly ONE live API call to Groq to verify credentials,
end-to-end prompt processing, and JSON response parsing into RecommendationResult.
"""

import os
import sys
import time
from pathlib import Path
from dotenv import load_dotenv

# Ensure repo root is on sys.path
BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

load_dotenv()

from src.data.loader import load_zomato_dataset
from src.data.preprocessor import preprocess
from src.filters.engine import apply_filters
from src.models.schemas import UserPreference, BudgetTier
from src.prompt.builder import SYSTEM_PROMPT, build_prompt
from src.llm.client import call_llm
from src.llm.parser import parse_llm_response


def test_single_groq_call():
    print("=" * 60)
    print("🚀 GROQ LIVE CONNECTION VERIFICATION (EXACTLY 1 CALL)")
    print("=" * 60)

    # 1. Verify environment configuration
    provider = os.getenv("LLM_PROVIDER", "groq")
    model = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")
    api_key = os.getenv("GROQ_API_KEY", "")

    if not api_key:
        print("❌ ERROR: GROQ_API_KEY is missing from .env")
        return False

    masked_key = api_key[:6] + "..." + api_key[-4:] if len(api_key) > 10 else "***"
    print(f"• LLM Provider : {provider}")
    print(f"• Model Target : {model}")
    print(f"• API Key      : {masked_key}")

    # 2. Build candidates from local dataset
    print("\n📂 Loading and filtering local candidates for prompt...")
    df = preprocess(load_zomato_dataset())
    user = UserPreference(
        location="koramangala",
        budget=BudgetTier.MEDIUM,
        cuisine="north indian",
        min_rating=4.0,
        extra_notes="Family dining, vegetarian options preferred",
    )
    candidates = apply_filters(df, user, top_n=5)
    candidates_list = candidates.to_dict(orient="records")
    prompt = build_prompt(user, candidates_list)
    print(f"• Filtered {len(candidates_list)} candidate restaurants.")

    # 3. Make EXACTLY ONE live API call
    print("\n⚡ Sending request to Groq Cloud API (Single call)...")
    start_time = time.time()
    try:
        raw_response = call_llm(SYSTEM_PROMPT, prompt, provider="groq", model=model)
        duration = time.time() - start_time
        print(f"✅ Groq responded in {duration:.2f} seconds!")
    except Exception as exc:
        print(f"❌ Groq API Call failed: {exc}")
        return False

    # 4. Parse response into RecommendationResult
    print("\n🔍 Parsing and validating structured response...")
    try:
        result = parse_llm_response(raw_response)
        print("✅ Response successfully parsed into RecommendationResult Pydantic schema!")
        print(f"\n📝 AI Summary:\n{result.summary}")
        print("\n🏆 Ranked Recommendations:")
        for rec in result.recommendations:
            print(
                f"  {rec.rank}. {rec.name} — ⭐{rec.rating} | 💰{rec.budget_tier.value.capitalize()} | 📍{rec.location}"
            )
            print(f"     💬 Explanation: {rec.explanation}")
        return True
    except Exception as exc:
        print(f"❌ Parsing failed: {exc}")
        print(f"Raw response was:\n{raw_response}")
        return False


if __name__ == "__main__":
    success = test_single_groq_call()
    exit(0 if success else 1)
