"""
src/prompt/builder.py
──────────────────────
Constructs the LLM prompt from filtered restaurant candidates and the
validated user preferences.

Exports:
  SYSTEM_PROMPT  — fixed system role string sent to the LLM
  build_prompt() — builds the per-request user-turn prompt string
"""

from src.models.schemas import UserPreference

# ── System prompt ─────────────────────────────────────────────────────────────

SYSTEM_PROMPT = """You are an expert restaurant recommendation assistant for Zomato.
Your goal is to rank and explain restaurant recommendations based on user preferences.
Be warm, specific, and helpful. Highlight what makes each restaurant a great fit.
Always respond in valid JSON only — no markdown, no prose outside the JSON."""


# ── Internal helpers ──────────────────────────────────────────────────────────

def _serialize_restaurants(candidates: list[dict]) -> str:
    """Format a list of restaurant dicts into a numbered plain-text block.

    Each line: ``1. Name | Cuisine | ⭐Rating | 💰budget_tier | 📍location``

    Args:
        candidates: List of row dicts from the filtered DataFrame
                    (e.g. ``df.to_dict(orient='records')``).

    Returns:
        A multi-line string with one restaurant per line.
    """
    lines = []
    for i, r in enumerate(candidates, start=1):
        name = str(r.get("name", "Unknown")).title()
        cuisine = str(r.get("cuisine", "N/A")).title()
        rating = r.get("rating", "N/A")
        budget = str(r.get("budget_tier", "N/A")).capitalize()
        location = str(r.get("location", "N/A")).title()
        cost = r.get("approx_cost")
        cost_str = f" (₹{cost} for two)" if cost else ""
        lines.append(
            f"{i}. {name} | {cuisine} | ⭐{rating} | 💰{budget}{cost_str} | 📍{location}"
        )
    return "\n".join(lines)


# ── Public API ────────────────────────────────────────────────────────────────

def build_prompt(user: UserPreference, candidates: list[dict]) -> str:
    """Build the complete user-turn LLM prompt.

    Injects the user's preferences and serialized candidate restaurants into
    a structured template that instructs the LLM to return valid JSON with
    ranked recommendations and a summary.

    Args:
        user:       Validated ``UserPreference`` object.
        candidates: List of restaurant row dicts from the filtered DataFrame.

    Returns:
        A formatted prompt string ready to be sent as the user turn to the LLM.

    Raises:
        ValueError: If *candidates* is empty (nothing to rank).
    """
    if not candidates:
        raise ValueError(
            "Cannot build a prompt with zero candidates. "
            "Run the filter pipeline first and ensure it returns results."
        )

    restaurants_text = _serialize_restaurants(candidates)
    n_candidates = len(candidates)
    top_k = min(5, n_candidates)   # ask for at most 5, or however many we have

    prompt = f"""USER PREFERENCES:
- Location: {user.location.title()}
- Budget: {user.budget.value.capitalize()}
- Cuisine: {user.cuisine.title() if user.cuisine else "Any (all cuisines)"}
- Minimum Rating: {user.min_rating}
- Additional Notes: {user.extra_notes or "None"}

CANDIDATE RESTAURANTS ({n_candidates} options):
{restaurants_text}

TASK:
From the candidate restaurants above, rank the top {top_k} that best match the user's preferences.
For each restaurant, write a 2–3 sentence personalised explanation of why it suits this user.
Also write a brief overall summary (2–3 sentences) of your recommendations.

Respond ONLY in this exact JSON structure — no markdown, no extra text:
{{
  "recommendations": [
    {{
      "rank": 1,
      "name": "...",
      "cuisine": "...",
      "rating": 4.5,
      "budget_tier": "medium",
      "location": "...",
      "explanation": "..."
    }}
  ],
  "summary": "..."
}}"""

    return prompt
