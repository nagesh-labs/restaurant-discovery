"""
src/filters/engine.py
──────────────────────
Rule-based filter pipeline that narrows the full Zomato DataFrame to a
manageable shortlist of candidates before they are passed to the LLM.

Pipeline (applied sequentially):
  1. Location filter   — partial string match (case-insensitive)
  2. Cuisine filter    — fuzzy match via rapidfuzz (threshold: 65)
  3. Budget filter     — exact budget_tier match
  4. Rating filter     — minimum rating threshold
  5. Fallback level 1  — if < 3 results, relax cuisine filter
  6. Fallback level 2  — if still < 3 results, relax budget filter too
  7. Sort by rating DESC and return top-N candidates

Edge cases handled:
  - No location match → raise ValueError with user-friendly message
  - Empty cuisine field → skip cuisine filter step
  - Fewer than 5 candidates → return all; LLM handles the rest
"""

import logging
from typing import Optional

import pandas as pd
from rapidfuzz import fuzz

from src.models.schemas import UserPreference

logger = logging.getLogger(__name__)

# ── Constants ─────────────────────────────────────────────────────────────────

CUISINE_FUZZY_THRESHOLD = 65     # minimum rapidfuzz score to count as a match
FALLBACK_MIN_RESULTS = 3         # trigger fallback when fewer results than this
DEFAULT_TOP_N = 15               # default maximum candidates sent to LLM


# ── Internal helpers ──────────────────────────────────────────────────────────

def _cuisine_matches(cuisine_str: str, query: str, threshold: int = CUISINE_FUZZY_THRESHOLD) -> bool:
    """Return True if *query* fuzzy-matches any cuisine in a comma-separated string.

    Uses ``rapidfuzz.fuzz.partial_ratio`` so that "Indian" will match
    "North Indian, Chinese" without requiring an exact string.

    Args:
        cuisine_str: Comma-separated cuisine string from the dataset, e.g.
                     ``"north indian, mughlai, chinese"``.
        query:       User-supplied cuisine preference (already lowercased).
        threshold:   Minimum fuzzy score (0–100) to count as a match.
    """
    if not isinstance(cuisine_str, str) or not cuisine_str.strip():
        return False
    score = fuzz.partial_ratio(query, cuisine_str.lower())
    return score >= threshold


def _location_filter(df: pd.DataFrame, location: str) -> pd.DataFrame:
    """Keep rows whose *location* column contains the query string."""
    return df[df["location"].str.contains(location, case=False, na=False, regex=False)]


def _cuisine_filter(df: pd.DataFrame, cuisine: str) -> pd.DataFrame:
    """Keep rows whose *cuisine* column fuzzy-matches the query."""
    if not cuisine.strip():
        return df  # no cuisine preference → skip
    mask = df["cuisine"].apply(lambda c: _cuisine_matches(c, cuisine))
    return df[mask]


def _budget_filter(df: pd.DataFrame, budget: str) -> pd.DataFrame:
    """Keep rows matching the budget_tier exactly."""
    return df[df["budget_tier"] == budget]


def _rating_filter(df: pd.DataFrame, min_rating: float) -> pd.DataFrame:
    """Keep rows with rating >= min_rating."""
    return df[df["rating"] >= min_rating]


# ── Public API ────────────────────────────────────────────────────────────────

def apply_filters(
    df: pd.DataFrame,
    user: UserPreference,
    top_n: int = DEFAULT_TOP_N,
) -> pd.DataFrame:
    """Apply the full sequential filter pipeline and return top-N candidates.

    The pipeline applies four filters in order (location → cuisine → budget →
    rating).  If fewer than ``FALLBACK_MIN_RESULTS`` remain at any point, the
    pipeline gradually relaxes constraints:

    * **Fallback 1** (< 3 after all 4 filters): re-run without cuisine filter.
    * **Fallback 2** (< 3 after fallback 1): re-run with only location + rating.

    Args:
        df:     Cleaned Zomato DataFrame (output of ``preprocess()``).
        user:   Validated user preferences.
        top_n:  Maximum number of candidates to return (default: 15).

    Returns:
        A DataFrame of up to *top_n* rows sorted by rating (descending).

    Raises:
        ValueError: If the location string matches zero restaurants in the
                    dataset (hard failure — user must correct input).
    """
    budget_value = user.budget.value

    # ── Step 1: Location (hard filter — always applied) ───────────────────────
    loc_filtered = _location_filter(df, user.location)

    if loc_filtered.empty:
        raise ValueError(
            f"No restaurants found in location '{user.location}'. "
            "Please try a different area or check your spelling."
        )

    logger.debug("After location filter: %d rows", len(loc_filtered))

    # ── Step 2–4: Cuisine → Budget → Rating ───────────────────────────────────
    result = _cuisine_filter(loc_filtered, user.cuisine)
    logger.debug("After cuisine filter: %d rows", len(result))

    result = _budget_filter(result, budget_value)
    logger.debug("After budget filter: %d rows", len(result))

    result = _rating_filter(result, user.min_rating)
    logger.debug("After rating filter: %d rows", len(result))

    # ── Fallback 1: relax cuisine filter ──────────────────────────────────────
    if len(result) < FALLBACK_MIN_RESULTS:
        logger.info(
            "Only %d results after full filter — relaxing cuisine constraint",
            len(result),
        )
        result = _budget_filter(loc_filtered, budget_value)
        result = _rating_filter(result, user.min_rating)

    # ── Fallback 2: relax budget filter as well ───────────────────────────────
    if len(result) < FALLBACK_MIN_RESULTS:
        logger.info(
            "Still only %d results — relaxing budget constraint too",
            len(result),
        )
        result = _rating_filter(loc_filtered, user.min_rating)

    # ── Sort and cap ──────────────────────────────────────────────────────────
    result = result.sort_values("rating", ascending=False).head(top_n)
    logger.info(
        "Filter pipeline complete: %d candidates returned (top_n=%d)",
        len(result),
        top_n,
    )
    return result.reset_index(drop=True)


def get_filter_summary(
    df: pd.DataFrame,
    user: UserPreference,
    top_n: int = DEFAULT_TOP_N,
) -> dict:
    """Return a diagnostic summary of what each filter step produces.

    Useful for debugging and testing — shows the row count after each stage
    without triggering fallback logic.

    Args:
        df:     Cleaned Zomato DataFrame.
        user:   Validated user preferences.
        top_n:  Candidate cap (same as in ``apply_filters``).

    Returns:
        A dict with keys: ``location``, ``cuisine``, ``budget``, ``rating``,
        ``final``.  Each value is the row count after that filter step.
    """
    after_location = _location_filter(df, user.location)
    after_cuisine = _cuisine_filter(after_location, user.cuisine)
    after_budget = _budget_filter(after_cuisine, user.budget.value)
    after_rating = _rating_filter(after_budget, user.min_rating)

    return {
        "location": len(after_location),
        "cuisine": len(after_cuisine),
        "budget": len(after_budget),
        "rating": len(after_rating),
        "final": min(len(after_rating), top_n),
    }
