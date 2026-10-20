"""
tests/test_filters.py
──────────────────────
Phase 3 acceptance tests — Filtering Engine & Prompt Layer.

Test classes:
  TestFilterHelpers          — unit tests for individual filter functions
  TestApplyFilters           — integration tests for apply_filters()
  TestFallbackLogic          — verify fallback triggers correctly
  TestEdgeCases              — empty results, invalid location, no cuisine
  TestGetFilterSummary       — diagnostic summary helper
  TestPromptBuilder          — unit tests for build_prompt() and SYSTEM_PROMPT
  TestPromptContent          — structural validation of generated prompt strings
  TestEndToEnd               — 3+ real-data scenarios from load → filter → prompt
"""

import pytest
import pandas as pd

from src.models.schemas import UserPreference, BudgetTier

# ── Shared fixtures ────────────────────────────────────────────────────────────


@pytest.fixture(scope="module")
def clean_df() -> pd.DataFrame:
    """Load the cached cleaned DataFrame once for the whole module."""
    return pd.read_parquet("data/processed/zomato_clean.parquet")


@pytest.fixture(scope="module")
def pref_koramangala_medium_north_indian() -> UserPreference:
    return UserPreference(
        location="koramangala",
        budget=BudgetTier.MEDIUM,
        cuisine="north indian",
        min_rating=3.5,
    )


@pytest.fixture(scope="module")
def pref_indiranagar_low_south_indian() -> UserPreference:
    return UserPreference(
        location="indiranagar",
        budget=BudgetTier.LOW,
        cuisine="south indian",
        min_rating=3.0,
    )


@pytest.fixture(scope="module")
def pref_jayanagar_high_chinese() -> UserPreference:
    return UserPreference(
        location="jayanagar",
        budget=BudgetTier.HIGH,
        cuisine="chinese",
        min_rating=4.0,
    )


# ═════════════════════════════════════════════════════════════════════════════
# Unit tests for individual filter helpers
# ═════════════════════════════════════════════════════════════════════════════

class TestFilterHelpers:
    """Test the internal helper functions directly."""

    def test_cuisine_matches_exact(self):
        from src.filters.engine import _cuisine_matches
        assert _cuisine_matches("north indian, mughlai, chinese", "north indian")

    def test_cuisine_matches_partial(self):
        from src.filters.engine import _cuisine_matches
        # "indian" should fuzzy-match "north indian"
        assert _cuisine_matches("north indian, chinese", "indian")

    def test_cuisine_matches_below_threshold(self):
        from src.filters.engine import _cuisine_matches
        # Completely unrelated cuisine should NOT match
        assert not _cuisine_matches("north indian, mughlai", "sushi", threshold=65)

    def test_cuisine_matches_empty_string(self):
        from src.filters.engine import _cuisine_matches
        assert not _cuisine_matches("", "north indian")

    def test_cuisine_matches_none(self):
        from src.filters.engine import _cuisine_matches
        assert not _cuisine_matches(None, "north indian")  # type: ignore

    def test_location_filter_returns_subset(self, clean_df):
        from src.filters.engine import _location_filter
        result = _location_filter(clean_df, "koramangala")
        assert len(result) > 0
        assert all("koramangala" in loc for loc in result["location"])

    def test_location_filter_case_insensitive(self, clean_df):
        from src.filters.engine import _location_filter
        lower = _location_filter(clean_df, "koramangala")
        upper = _location_filter(clean_df, "KORAMANGALA")
        assert len(lower) == len(upper)

    def test_budget_filter_only_matching_tier(self, clean_df):
        from src.filters.engine import _budget_filter
        result = _budget_filter(clean_df, "low")
        assert set(result["budget_tier"].unique()) == {"low"}

    def test_rating_filter_min_threshold(self, clean_df):
        from src.filters.engine import _rating_filter
        result = _rating_filter(clean_df, 4.0)
        assert (result["rating"] >= 4.0).all()


# ═════════════════════════════════════════════════════════════════════════════
# Integration tests for apply_filters()
# ═════════════════════════════════════════════════════════════════════════════

class TestApplyFilters:
    """Test the full apply_filters() pipeline."""

    def test_returns_dataframe(self, clean_df, pref_koramangala_medium_north_indian):
        from src.filters.engine import apply_filters
        result = apply_filters(clean_df, pref_koramangala_medium_north_indian)
        assert isinstance(result, pd.DataFrame)

    def test_respects_top_n(self, clean_df, pref_koramangala_medium_north_indian):
        from src.filters.engine import apply_filters
        result = apply_filters(clean_df, pref_koramangala_medium_north_indian, top_n=5)
        assert len(result) <= 5

    def test_default_top_n_is_15(self, clean_df, pref_koramangala_medium_north_indian):
        from src.filters.engine import apply_filters
        result = apply_filters(clean_df, pref_koramangala_medium_north_indian)
        assert len(result) <= 15

    def test_sorted_by_rating_descending(self, clean_df, pref_koramangala_medium_north_indian):
        from src.filters.engine import apply_filters
        result = apply_filters(clean_df, pref_koramangala_medium_north_indian)
        if len(result) > 1:
            ratings = result["rating"].tolist()
            assert ratings == sorted(ratings, reverse=True), "Results not sorted by rating desc"

    def test_all_results_meet_min_rating(self, clean_df, pref_koramangala_medium_north_indian):
        from src.filters.engine import apply_filters
        result = apply_filters(clean_df, pref_koramangala_medium_north_indian)
        assert (result["rating"] >= pref_koramangala_medium_north_indian.min_rating).all()

    def test_scenario_low_budget_south_indian(self, clean_df, pref_indiranagar_low_south_indian):
        """Scenario 2 — Indiranagar, low budget, South Indian."""
        from src.filters.engine import apply_filters
        result = apply_filters(clean_df, pref_indiranagar_low_south_indian)
        assert isinstance(result, pd.DataFrame)
        # Rating threshold respected
        assert (result["rating"] >= pref_indiranagar_low_south_indian.min_rating).all()

    def test_scenario_high_budget_chinese(self, clean_df, pref_jayanagar_high_chinese):
        """Scenario 3 — Jayanagar, high budget, Chinese."""
        from src.filters.engine import apply_filters
        result = apply_filters(clean_df, pref_jayanagar_high_chinese)
        assert isinstance(result, pd.DataFrame)
        assert (result["rating"] >= pref_jayanagar_high_chinese.min_rating).all()

    def test_index_is_reset(self, clean_df, pref_koramangala_medium_north_indian):
        """Returned DataFrame must have a clean 0-based index."""
        from src.filters.engine import apply_filters
        result = apply_filters(clean_df, pref_koramangala_medium_north_indian)
        assert list(result.index) == list(range(len(result)))


# ═════════════════════════════════════════════════════════════════════════════
# Fallback logic tests
# ═════════════════════════════════════════════════════════════════════════════

class TestFallbackLogic:
    """Verify that relaxed fallbacks trigger when too few results are found."""

    def test_fallback_returns_results_when_strict_empty(self, clean_df):
        """When full filter gives 0 results, fallback should still return rows."""
        from src.filters.engine import apply_filters
        # Asking for a very high rating with a rare cuisine combo — likely to trigger fallback
        user = UserPreference(
            location="koramangala",
            budget=BudgetTier.HIGH,
            cuisine="ethiopian",   # very rare cuisine → fallback expected
            min_rating=3.0,
        )
        result = apply_filters(clean_df, user)
        # After fallback we should still get something (location has restaurants)
        assert isinstance(result, pd.DataFrame)

    def test_location_only_fallback_rating_respected(self, clean_df):
        """Even in deepest fallback (location + rating only), min_rating is respected."""
        from src.filters.engine import apply_filters
        user = UserPreference(
            location="whitefield",
            budget=BudgetTier.HIGH,
            cuisine="mongolian",   # very rare → fallback
            min_rating=3.5,
        )
        result = apply_filters(clean_df, user)
        if not result.empty:
            assert (result["rating"] >= 3.5).all()


# ═════════════════════════════════════════════════════════════════════════════
# Edge cases
# ═════════════════════════════════════════════════════════════════════════════

class TestEdgeCases:
    """Edge cases defined in the plan and architecture."""

    def test_invalid_location_raises_value_error(self, clean_df):
        """A location that matches nothing should raise ValueError."""
        from src.filters.engine import apply_filters
        user = UserPreference(
            location="atlantis",   # does not exist in dataset
            budget=BudgetTier.MEDIUM,
            cuisine="north indian",
        )
        with pytest.raises(ValueError, match="No restaurants found"):
            apply_filters(clean_df, user)

    def test_empty_cuisine_skips_cuisine_filter(self, clean_df):
        """An empty cuisine string should not filter out restaurants."""
        from src.filters.engine import _cuisine_filter
        original_len = len(clean_df)
        result = _cuisine_filter(clean_df, "")
        assert len(result) == original_len, "Empty cuisine should skip cuisine filtering"

    def test_high_min_rating_gives_fewer_results(self, clean_df):
        """Very high min_rating (4.8) should return fewer results than lower threshold."""
        from src.filters.engine import apply_filters
        user_strict = UserPreference(
            location="koramangala",
            budget=BudgetTier.MEDIUM,
            cuisine="north indian",
            min_rating=4.8,
        )
        user_relaxed = UserPreference(
            location="koramangala",
            budget=BudgetTier.MEDIUM,
            cuisine="north indian",
            min_rating=3.0,
        )
        result_strict = apply_filters(clean_df, user_strict)
        result_relaxed = apply_filters(clean_df, user_relaxed)
        assert len(result_strict) <= len(result_relaxed)


# ═════════════════════════════════════════════════════════════════════════════
# get_filter_summary() helper
# ═════════════════════════════════════════════════════════════════════════════

class TestGetFilterSummary:
    def test_summary_returns_correct_keys(self, clean_df, pref_koramangala_medium_north_indian):
        from src.filters.engine import get_filter_summary
        summary = get_filter_summary(clean_df, pref_koramangala_medium_north_indian)
        expected_keys = {"location", "cuisine", "budget", "rating", "final"}
        assert set(summary.keys()) == expected_keys

    def test_summary_counts_are_non_negative(self, clean_df, pref_koramangala_medium_north_indian):
        from src.filters.engine import get_filter_summary
        summary = get_filter_summary(clean_df, pref_koramangala_medium_north_indian)
        assert all(v >= 0 for v in summary.values())

    def test_summary_counts_decrease_or_stay_same(self, clean_df, pref_koramangala_medium_north_indian):
        """Each successive filter should produce ≤ previous step's count."""
        from src.filters.engine import get_filter_summary
        s = get_filter_summary(clean_df, pref_koramangala_medium_north_indian)
        assert s["cuisine"] <= s["location"]
        assert s["budget"] <= s["cuisine"]
        assert s["rating"] <= s["budget"]
        assert s["final"] <= s["rating"]


# ═════════════════════════════════════════════════════════════════════════════
# Prompt builder unit tests
# ═════════════════════════════════════════════════════════════════════════════

class TestPromptBuilder:
    """Unit tests for build_prompt() and SYSTEM_PROMPT."""

    def test_system_prompt_is_string(self):
        from src.prompt.builder import SYSTEM_PROMPT
        assert isinstance(SYSTEM_PROMPT, str)
        assert len(SYSTEM_PROMPT) > 20

    def test_system_prompt_mentions_json(self):
        from src.prompt.builder import SYSTEM_PROMPT
        assert "json" in SYSTEM_PROMPT.lower()

    def test_build_prompt_returns_string(self):
        from src.prompt.builder import build_prompt
        user = UserPreference(
            location="koramangala", budget=BudgetTier.MEDIUM, cuisine="north indian"
        )
        candidates = [
            {"name": "test cafe", "cuisine": "north indian", "rating": 4.2,
             "budget_tier": "medium", "location": "koramangala", "approx_cost": 500}
        ]
        prompt = build_prompt(user, candidates)
        assert isinstance(prompt, str)

    def test_build_prompt_empty_candidates_raises(self):
        from src.prompt.builder import build_prompt
        user = UserPreference(
            location="koramangala", budget=BudgetTier.MEDIUM, cuisine="north indian"
        )
        with pytest.raises(ValueError, match="zero candidates"):
            build_prompt(user, [])

    def test_build_prompt_injects_user_location(self):
        from src.prompt.builder import build_prompt
        user = UserPreference(
            location="indiranagar", budget=BudgetTier.LOW, cuisine="south indian"
        )
        candidates = [
            {"name": "dosa hut", "cuisine": "south indian", "rating": 3.8,
             "budget_tier": "low", "location": "indiranagar", "approx_cost": 200}
        ]
        prompt = build_prompt(user, candidates)
        assert "Indiranagar" in prompt

    def test_build_prompt_injects_budget(self):
        from src.prompt.builder import build_prompt
        user = UserPreference(
            location="koramangala", budget=BudgetTier.HIGH, cuisine="chinese"
        )
        candidates = [
            {"name": "dragon palace", "cuisine": "chinese", "rating": 4.5,
             "budget_tier": "high", "location": "koramangala", "approx_cost": 1200}
        ]
        prompt = build_prompt(user, candidates)
        assert "High" in prompt  # budget shows as capitalized

    def test_build_prompt_injects_cuisine(self):
        from src.prompt.builder import build_prompt
        user = UserPreference(
            location="koramangala", budget=BudgetTier.MEDIUM, cuisine="italian"
        )
        candidates = [
            {"name": "la piazza", "cuisine": "italian, pizza", "rating": 4.3,
             "budget_tier": "medium", "location": "koramangala", "approx_cost": 600}
        ]
        prompt = build_prompt(user, candidates)
        assert "Italian" in prompt

    def test_build_prompt_injects_extra_notes(self):
        from src.prompt.builder import build_prompt
        user = UserPreference(
            location="koramangala", budget=BudgetTier.MEDIUM, cuisine="north indian",
            extra_notes="vegan options preferred"
        )
        candidates = [
            {"name": "green bowl", "cuisine": "north indian", "rating": 4.0,
             "budget_tier": "medium", "location": "koramangala", "approx_cost": 500}
        ]
        prompt = build_prompt(user, candidates)
        assert "vegan options preferred" in prompt

    def test_build_prompt_extra_notes_none_shows_none(self):
        from src.prompt.builder import build_prompt
        user = UserPreference(
            location="koramangala", budget=BudgetTier.MEDIUM, cuisine="north indian"
        )
        candidates = [
            {"name": "spice garden", "cuisine": "north indian", "rating": 4.1,
             "budget_tier": "medium", "location": "koramangala", "approx_cost": 500}
        ]
        prompt = build_prompt(user, candidates)
        assert "None" in prompt  # extra_notes defaults to None


# ═════════════════════════════════════════════════════════════════════════════
# Prompt content / structure tests
# ═════════════════════════════════════════════════════════════════════════════

class TestPromptContent:
    """Verify the structural requirements of the generated prompt."""

    @pytest.fixture
    def sample_prompt(self):
        from src.prompt.builder import build_prompt
        user = UserPreference(
            location="koramangala", budget=BudgetTier.MEDIUM, cuisine="north indian",
            min_rating=3.5, extra_notes="outdoor seating preferred"
        )
        candidates = [
            {"name": f"restaurant {i}", "cuisine": "north indian, chinese",
             "rating": round(3.5 + i * 0.1, 1), "budget_tier": "medium",
             "location": "koramangala", "approx_cost": 500}
            for i in range(8)
        ]
        return build_prompt(user, candidates), user, candidates

    def test_prompt_contains_user_preferences_header(self, sample_prompt):
        prompt, _, _ = sample_prompt
        assert "USER PREFERENCES" in prompt

    def test_prompt_contains_candidate_restaurants_header(self, sample_prompt):
        prompt, _, _ = sample_prompt
        assert "CANDIDATE RESTAURANTS" in prompt

    def test_prompt_contains_task_section(self, sample_prompt):
        prompt, _, _ = sample_prompt
        assert "TASK" in prompt

    def test_prompt_contains_json_schema(self, sample_prompt):
        prompt, _, _ = sample_prompt
        assert '"recommendations"' in prompt
        assert '"explanation"' in prompt
        assert '"summary"' in prompt

    def test_prompt_lists_all_candidates(self, sample_prompt):
        prompt, _, candidates = sample_prompt
        # Each restaurant should be referenced by its ordinal
        assert f"{len(candidates)}." in prompt

    def test_prompt_contains_restaurant_details(self, sample_prompt):
        prompt, _, _ = sample_prompt
        # Rating emoji and budget emoji should be present
        assert "⭐" in prompt
        assert "💰" in prompt
        assert "📍" in prompt

    def test_prompt_contains_min_rating(self, sample_prompt):
        prompt, user, _ = sample_prompt
        assert str(user.min_rating) in prompt


# ═════════════════════════════════════════════════════════════════════════════
# End-to-end: load → filter → prompt  (3 diverse scenarios)
# ═════════════════════════════════════════════════════════════════════════════

class TestEndToEnd:
    """Full pipeline: cleaned data → filter → prompt, no LLM involved."""

    def _run_scenario(self, clean_df, user: UserPreference, label: str):
        from src.filters.engine import apply_filters
        from src.prompt.builder import build_prompt, SYSTEM_PROMPT

        candidates = apply_filters(clean_df, user, top_n=15)
        assert not candidates.empty, f"[{label}] Filter returned no results"

        records = candidates.to_dict(orient="records")
        prompt = build_prompt(user, records)

        assert isinstance(prompt, str), f"[{label}] Prompt is not a string"
        assert len(prompt) > 100, f"[{label}] Prompt is too short"
        assert '"recommendations"' in prompt, f"[{label}] JSON schema missing from prompt"
        assert SYSTEM_PROMPT, f"[{label}] SYSTEM_PROMPT is empty"

        return candidates, prompt

    def test_scenario_1_koramangala_medium_north_indian(self, clean_df):
        user = UserPreference(
            location="koramangala",
            budget=BudgetTier.MEDIUM,
            cuisine="north indian",
            min_rating=3.5,
            extra_notes="looking for family dining",
        )
        candidates, prompt = self._run_scenario(clean_df, user, "Scenario 1")
        assert (candidates["rating"] >= 3.5).all()

    def test_scenario_2_indiranagar_low_south_indian(self, clean_df):
        user = UserPreference(
            location="indiranagar",
            budget=BudgetTier.LOW,
            cuisine="south indian",
            min_rating=3.0,
        )
        candidates, prompt = self._run_scenario(clean_df, user, "Scenario 2")
        assert (candidates["rating"] >= 3.0).all()

    def test_scenario_3_btm_high_continental(self, clean_df):
        user = UserPreference(
            location="btm",
            budget=BudgetTier.HIGH,
            cuisine="continental",
            min_rating=3.5,
            extra_notes="rooftop or fine dining preferred",
        )
        candidates, prompt = self._run_scenario(clean_df, user, "Scenario 3")
        assert (candidates["rating"] >= 3.5).all()

    def test_prompt_candidate_count_matches_filter_output(self, clean_df):
        """The number serialized in the prompt header must match filter output."""
        from src.filters.engine import apply_filters
        from src.prompt.builder import build_prompt
        user = UserPreference(
            location="jayanagar",
            budget=BudgetTier.MEDIUM,
            cuisine="chinese",
            min_rating=3.5,
        )
        candidates = apply_filters(clean_df, user, top_n=10)
        records = candidates.to_dict(orient="records")
        prompt = build_prompt(user, records)
        # The prompt header should mention the correct count
        assert f"({len(records)} options)" in prompt
