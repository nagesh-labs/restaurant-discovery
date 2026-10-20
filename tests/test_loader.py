"""
tests/test_loader.py
─────────────────────
Phase 2 acceptance tests — Data Ingestion & Processing.

Covers:
  - Loader: dataset shape, column presence, raw dtypes
  - Preprocessor: no nulls in key cols, rating range, budget_tier values,
    output columns, parquet cache written
  - Schemas: Restaurant and UserPreference Pydantic validation
"""

import os
import re
from pathlib import Path

import pandas as pd
import pytest

# ── Path helpers ──────────────────────────────────────────────────────────────

BASE_DIR = Path(__file__).resolve().parent.parent
PROCESSED_CACHE = BASE_DIR / "data" / "processed" / "zomato_clean.parquet"

# ── Fixtures ──────────────────────────────────────────────────────────────────


@pytest.fixture(scope="module")
def raw_df():
    """Load the raw dataset once for the whole module."""
    from src.data.loader import load_zomato_dataset
    return load_zomato_dataset()


@pytest.fixture(scope="module")
def clean_df(raw_df):
    """Run preprocessing on the raw dataset once for the whole module."""
    from src.data.preprocessor import preprocess
    return preprocess(raw_df, save_cache=True)


# ═════════════════════════════════════════════════════════════════════════════
# 2.1  Loader tests
# ═════════════════════════════════════════════════════════════════════════════

class TestLoader:
    def test_dataset_loads(self, raw_df):
        """Dataset loads and is a non-empty DataFrame."""
        assert isinstance(raw_df, pd.DataFrame)
        assert len(raw_df) > 0, "Dataset is empty"

    def test_minimum_row_count(self, raw_df):
        """Dataset should contain at least 10 000 rows."""
        assert len(raw_df) >= 10_000, f"Only {len(raw_df)} rows loaded"

    def test_required_raw_columns(self, raw_df):
        """All columns needed for preprocessing must be present."""
        required = {"name", "location", "cuisines", "rate", "votes",
                    "approx_cost(for two people)"}
        missing = required - set(raw_df.columns)
        assert not missing, f"Missing raw columns: {missing}"

    def test_rate_column_format(self, raw_df):
        """Non-null rate values follow 'X.X/5', 'X.X /5' (space variant), 'NEW', or '-'."""
        non_null = raw_df["rate"].dropna()
        # Dataset has values like "4.1/5", "3.8 /5" (space before slash), "NEW", "-"
        pattern = re.compile(r"^(\d+\.?\d*\s*(/\d)?|NEW|-)$", re.IGNORECASE)
        bad = [v for v in non_null.unique() if not pattern.match(str(v).strip())]
        assert not bad, f"Unexpected rate values: {bad[:5]}"

    def test_cache_written_after_load(self):
        """Raw parquet cache must exist after load_zomato_dataset() is called."""
        raw_cache = BASE_DIR / "data" / "processed" / "zomato_raw.parquet"
        assert raw_cache.exists(), "Raw parquet cache was not written"


# ═════════════════════════════════════════════════════════════════════════════
# 2.3  Preprocessor tests
# ═════════════════════════════════════════════════════════════════════════════

class TestPreprocessor:
    def test_output_is_dataframe(self, clean_df):
        """Preprocessor returns a DataFrame."""
        assert isinstance(clean_df, pd.DataFrame)

    def test_output_has_rows(self, clean_df):
        """Cleaned DataFrame should retain a meaningful number of rows."""
        assert len(clean_df) >= 5_000, f"Too few rows after preprocessing: {len(clean_df)}"

    def test_required_output_columns(self, clean_df):
        """All expected output columns must exist."""
        expected = {"name", "location", "cuisine", "rating", "budget_tier",
                    "approx_cost", "votes"}
        missing = expected - set(clean_df.columns)
        assert not missing, f"Missing output columns: {missing}"

    def test_no_nulls_in_critical_columns(self, clean_df):
        """Critical columns must have zero null values after preprocessing."""
        for col in ["name", "location", "cuisine", "rating", "budget_tier"]:
            nulls = clean_df[col].isnull().sum()
            assert nulls == 0, f"Column '{col}' has {nulls} null values"

    def test_rating_range(self, clean_df):
        """All ratings must be in [0.0, 5.0]."""
        out_of_range = clean_df[(clean_df["rating"] < 0.0) | (clean_df["rating"] > 5.0)]
        assert len(out_of_range) == 0, (
            f"{len(out_of_range)} rows have ratings outside [0.0, 5.0]"
        )

    def test_rating_dtype(self, clean_df):
        """Rating column must be float dtype."""
        assert pd.api.types.is_float_dtype(clean_df["rating"]), (
            f"Rating dtype is {clean_df['rating'].dtype}, expected float"
        )

    def test_budget_tier_values(self, clean_df):
        """budget_tier must only contain valid tier strings."""
        valid = {"low", "medium", "high"}
        actual = set(clean_df["budget_tier"].unique())
        unexpected = actual - valid
        assert not unexpected, f"Unexpected budget_tier values: {unexpected}"

    def test_budget_tier_low_mapping(self, clean_df):
        """Rows with approx_cost ≤ 300 should map to 'low'."""
        low_rows = clean_df[clean_df["approx_cost"].notna() & (clean_df["approx_cost"] <= 300)]
        if len(low_rows):
            assert (low_rows["budget_tier"] == "low").all(), "budget_tier mismatch for low cost"

    def test_budget_tier_medium_mapping(self, clean_df):
        """Rows with 301 ≤ approx_cost ≤ 700 should map to 'medium'."""
        med_rows = clean_df[
            clean_df["approx_cost"].notna()
            & (clean_df["approx_cost"] > 300)
            & (clean_df["approx_cost"] <= 700)
        ]
        if len(med_rows):
            assert (med_rows["budget_tier"] == "medium").all(), "budget_tier mismatch for medium cost"

    def test_budget_tier_high_mapping(self, clean_df):
        """Rows with approx_cost > 700 should map to 'high'."""
        high_rows = clean_df[clean_df["approx_cost"].notna() & (clean_df["approx_cost"] > 700)]
        if len(high_rows):
            assert (high_rows["budget_tier"] == "high").all(), "budget_tier mismatch for high cost"

    def test_text_fields_lowercased(self, clean_df):
        """name, location, cuisine should be lowercase strings."""
        for col in ["name", "location", "cuisine"]:
            has_upper = clean_df[col].str.contains(r"[A-Z]", regex=True, na=False).any()
            assert not has_upper, f"Column '{col}' contains uppercase characters"

    def test_text_fields_no_leading_trailing_spaces(self, clean_df):
        """name, location, cuisine should have no leading/trailing whitespace."""
        for col in ["name", "location", "cuisine"]:
            has_spaces = (clean_df[col] != clean_df[col].str.strip()).any()
            assert not has_spaces, f"Column '{col}' contains leading/trailing spaces"

    def test_votes_is_integer(self, clean_df):
        """votes column must be integer dtype."""
        assert pd.api.types.is_integer_dtype(clean_df["votes"]), (
            f"votes dtype is {clean_df['votes'].dtype}, expected int"
        )

    def test_parquet_cache_written(self):
        """Processed parquet cache must exist after preprocess() is called."""
        assert PROCESSED_CACHE.exists(), (
            f"Processed parquet cache not found at {PROCESSED_CACHE}"
        )

    def test_parquet_cache_readable(self):
        """Processed parquet cache must be readable as a DataFrame."""
        df = pd.read_parquet(PROCESSED_CACHE)
        assert len(df) > 0, "Parquet cache is empty"


# ═════════════════════════════════════════════════════════════════════════════
# 2.4  Pydantic schema tests
# ═════════════════════════════════════════════════════════════════════════════

class TestSchemas:
    def test_restaurant_model_valid(self):
        """Valid Restaurant dict should construct without error."""
        from src.models.schemas import Restaurant, BudgetTier
        r = Restaurant(
            name="Test Cafe",
            location="bangalore",
            cuisine="north indian",
            rating=4.2,
            budget_tier=BudgetTier.MEDIUM,
            approx_cost=500,
            votes=120,
        )
        assert r.name == "Test Cafe"
        assert r.budget_tier == BudgetTier.MEDIUM

    def test_restaurant_rating_out_of_range(self):
        """Restaurant with rating > 5.0 should raise ValidationError."""
        from pydantic import ValidationError
        from src.models.schemas import Restaurant, BudgetTier
        with pytest.raises(ValidationError):
            Restaurant(
                name="Bad Cafe", location="x", cuisine="y",
                rating=6.0, budget_tier=BudgetTier.LOW
            )

    def test_user_preference_valid(self):
        """Valid UserPreference should be constructed correctly."""
        from src.models.schemas import UserPreference, BudgetTier
        pref = UserPreference(
            location="  Bangalore  ",
            budget=BudgetTier.HIGH,
            cuisine="  North Indian  ",
            min_rating=4.0,
        )
        # Validator should strip and lowercase
        assert pref.location == "bangalore"
        assert pref.cuisine == "north indian"

    def test_user_preference_min_rating_default(self):
        """min_rating should default to 3.5."""
        from src.models.schemas import UserPreference, BudgetTier
        pref = UserPreference(location="x", budget=BudgetTier.LOW, cuisine="y")
        assert pref.min_rating == 3.5

    def test_user_preference_invalid_min_rating(self):
        """min_rating > 5.0 should raise ValidationError."""
        from pydantic import ValidationError
        from src.models.schemas import UserPreference, BudgetTier
        with pytest.raises(ValidationError):
            UserPreference(location="x", budget=BudgetTier.LOW, cuisine="y", min_rating=6.0)

    def test_budget_tier_enum_values(self):
        """BudgetTier enum must have exactly low/medium/high."""
        from src.models.schemas import BudgetTier
        assert {t.value for t in BudgetTier} == {"low", "medium", "high"}

    def test_recommendation_result_model(self):
        """RecommendationResult model constructs correctly from valid data."""
        from src.models.schemas import RecommendationResult, RecommendedRestaurant, BudgetTier
        result = RecommendationResult(
            recommendations=[
                RecommendedRestaurant(
                    rank=1, name="A", cuisine="B", rating=4.5,
                    budget_tier=BudgetTier.HIGH, location="C",
                    explanation="Great place!"
                )
            ],
            summary="Overall good picks.",
        )
        assert len(result.recommendations) == 1
        assert result.recommendations[0].rank == 1

    def test_recommended_restaurant_budget_tier_case_insensitive(self):
        """RecommendedRestaurant should normalize 'Medium', 'LOW', 'High' from LLM output."""
        from src.models.schemas import RecommendedRestaurant, BudgetTier
        rec = RecommendedRestaurant(
            rank=1, name="A", cuisine="B", rating=4.5,
            budget_tier="Medium", location="C", explanation="Great place!"
        )
        assert rec.budget_tier == BudgetTier.MEDIUM
        assert rec.budget_tier.value == "medium"


# ═════════════════════════════════════════════════════════════════════════════
# 2.5  Integration: preprocessed df → schema round-trip
# ═════════════════════════════════════════════════════════════════════════════

class TestPreprocessorSchemaIntegration:
    def test_sample_rows_validate_as_restaurant(self, clean_df):
        """A sample of cleaned rows should be parseable as Restaurant models."""
        from src.models.schemas import Restaurant
        sample = clean_df.sample(min(20, len(clean_df)), random_state=42)
        errors = []
        for _, row in sample.iterrows():
            try:
                Restaurant(
                    name=row["name"],
                    location=row["location"],
                    cuisine=row["cuisine"],
                    rating=row["rating"],
                    budget_tier=row["budget_tier"],
                    approx_cost=row.get("approx_cost"),
                    votes=row.get("votes"),
                )
            except Exception as exc:
                errors.append(f"Row '{row['name']}': {exc}")
        assert not errors, "Schema validation failed for rows:\n" + "\n".join(errors)
