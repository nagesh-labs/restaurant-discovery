"""
src/llm/parser.py
─────────────────
Parses raw LLM string responses into validated RecommendationResult Pydantic models.
Handles code fences, edge cases (missing explanations/summaries), and malformed JSON.
"""

import json
import logging
import re
from typing import Any, Optional

from src.models.schemas import RecommendationResult, RecommendedRestaurant, BudgetTier

logger = logging.getLogger(__name__)


def parse_llm_response(raw_response: str) -> RecommendationResult:
    """Parse the raw LLM JSON response into a validated RecommendationResult.

    Steps:
      1. Strip whitespace and markdown code fences (```json ... ``` or ``` ... ```).
      2. Extract valid JSON using regex fallback if extra conversational text exists.
      3. Deserialize via json.loads().
      4. Normalize missing optional fields (e.g. summary, explanations, ranks, budget_tier).
      5. Construct and validate RecommendationResult.

    Args:
        raw_response: The raw string returned by the LLM.

    Returns:
        A validated RecommendationResult instance.

    Raises:
        ValueError: If response is empty, contains no JSON, or fails validation.
    """
    if not raw_response or not raw_response.strip():
        raise ValueError("Received empty response from LLM.")

    cleaned = raw_response.strip()

    # Strip markdown code fences if present
    if cleaned.startswith("```"):
        lines = cleaned.splitlines()
        # Drop the first line (e.g., ```json or ```)
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        # Drop the last line if it closes the fence
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        cleaned = "\n".join(lines).strip()

    # Also handle trailing markdown fences if any remain
    if cleaned.endswith("```"):
        cleaned = cleaned[:-3].strip()

    # Attempt direct json.loads
    try:
        data = json.loads(cleaned)
    except json.JSONDecodeError:
        # Fallback: try finding outermost JSON object { ... }
        match = re.search(r"\{.*\}", cleaned, re.DOTALL)
        if match:
            try:
                data = json.loads(match.group(0))
            except json.JSONDecodeError as err:
                raise ValueError(
                    f"Failed to parse LLM response as JSON: {err}\nRaw snippet: {cleaned[:200]}"
                ) from err
        else:
            raise ValueError(
                f"No valid JSON object found in LLM response.\nRaw snippet: {cleaned[:200]}"
            )

    if not isinstance(data, dict):
        raise ValueError(
            f"Expected JSON object at top level, got {type(data).__name__}."
        )

    # Ensure 'recommendations' key is present
    if "recommendations" not in data:
        # Check common alternative keys
        for alt_key in ("restaurants", "results", "picks", "data"):
            if alt_key in data and isinstance(data[alt_key], list):
                data["recommendations"] = data.pop(alt_key)
                break
        else:
            raise ValueError("LLM JSON response missing required 'recommendations' field.")

    if not isinstance(data["recommendations"], list) or len(data["recommendations"]) == 0:
        raise ValueError("LLM JSON response contains an empty or invalid 'recommendations' list.")

    # Normalize recommendations items
    for idx, item in enumerate(data["recommendations"], start=1):
        if not isinstance(item, dict):
            continue
        if "rank" not in item or not item["rank"]:
            item["rank"] = idx
        if "explanation" not in item or not item["explanation"]:
            item["explanation"] = "Recommended match based on your preferences."
        if "budget_tier" not in item or not item["budget_tier"]:
            item["budget_tier"] = "medium"

    # Normalize summary
    if "summary" not in data or not data["summary"]:
        count = len(data["recommendations"])
        data["summary"] = f"Here are the top {count} personalized restaurant recommendations based on your preferences."

    try:
        return RecommendationResult(**data)
    except Exception as exc:
        raise ValueError(f"RecommendationResult validation error: {exc}") from exc
