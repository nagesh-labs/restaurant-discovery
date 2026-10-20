"""
src/models/schemas.py
──────────────────────
Pydantic v2 domain models for the Zomato AI Recommendation System.

Models defined here:
  - BudgetTier        (enum)
  - Restaurant        (a single cleaned restaurant record)
  - UserPreference    (validated user input)
  - RecommendedRestaurant  (one ranked LLM recommendation)
  - RecommendationResult   (full LLM response, parsed & validated)
"""

from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field, field_validator


# ── Enums ─────────────────────────────────────────────────────────────────────

class BudgetTier(str, Enum):
    """Three-tier cost classification derived from approx_cost."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


# ── Data layer models ─────────────────────────────────────────────────────────

class Restaurant(BaseModel):
    """A single cleaned restaurant record from the Zomato dataset."""

    name: str
    location: str
    cuisine: str
    rating: float = Field(ge=0.0, le=5.0)
    budget_tier: BudgetTier
    approx_cost: Optional[int] = None
    votes: Optional[int] = None
    rest_type: Optional[str] = None
    online_order: Optional[bool] = None
    book_table: Optional[bool] = None

    model_config = {"from_attributes": True}


# ── Input layer models ────────────────────────────────────────────────────────

class UserPreference(BaseModel):
    """Validated, normalised user preference input."""

    location: str = Field(..., min_length=1, description="City or area to search in")
    budget: BudgetTier = Field(..., description="Cost tier: low | medium | high")
    cuisine: str = Field(default="", description="Preferred cuisine type (optional)")
    min_rating: float = Field(default=3.5, ge=0.0, le=5.0, description="Minimum acceptable rating")
    extra_notes: Optional[str] = Field(default=None, description="Free-text extra preferences")

    @field_validator("location", "cuisine", mode="before")
    @classmethod
    def strip_and_lower(cls, v: Optional[str]) -> str:
        """Normalise text inputs: strip whitespace and lowercase."""
        if v is None:
            return ""
        return v.strip().lower() if isinstance(v, str) else str(v)


# ── LLM output models (Phase 4) ───────────────────────────────────────────────

class RecommendedRestaurant(BaseModel):
    """A single ranked restaurant returned by the LLM."""

    rank: int = Field(..., ge=1, description="Ranking position (1 = best)")
    name: str
    cuisine: str
    rating: float = Field(ge=0.0, le=5.0)
    budget_tier: BudgetTier
    location: str
    explanation: str = Field(
        default="Recommended match based on your preferences.",
        description="Personalised AI explanation"
    )

    @field_validator("budget_tier", mode="before")
    @classmethod
    def normalize_budget_tier(cls, v: str | BudgetTier) -> str | BudgetTier:
        """Allow case-insensitive budget tier parsing from LLM output."""
        return v.strip().lower() if isinstance(v, str) else v


class RecommendationResult(BaseModel):
    """Complete structured response from the LLM recommendation engine."""

    recommendations: list[RecommendedRestaurant] = Field(
        ..., min_length=1, description="Ranked list of recommended restaurants"
    )
    summary: str = Field(
        default="Here are the top restaurant recommendations based on your preferences.",
        description="Overall AI-generated recommendation summary"
    )
