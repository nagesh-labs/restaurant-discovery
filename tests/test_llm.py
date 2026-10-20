"""
tests/test_llm.py
─────────────────
Phase 4 unit and integration tests (mocked).
Covers:
  - Parser: valid JSON, code fences, missing fields, invalid JSON
  - Client: provider selection (Groq, OpenAI, Gemini), missing API keys
  - Retry logic: retry on rate limits, no retry on auth errors
"""

import json
from unittest.mock import MagicMock, patch

import pytest
from tenacity import RetryError

from src.llm.client import _is_transient_error, call_llm
from src.llm.parser import parse_llm_response
from src.models.schemas import BudgetTier, RecommendationResult


# ── Sample responses ──────────────────────────────────────────────────────────

VALID_RAW_RESPONSE = json.dumps(
    {
        "recommendations": [
            {
                "rank": 1,
                "name": "Truffles",
                "cuisine": "American, Fast Food",
                "rating": 4.7,
                "budget_tier": "medium",
                "location": "Koramangala",
                "explanation": "Famous for burgers and desserts with great value.",
            },
            {
                "rank": 2,
                "name": "Meghana Foods",
                "cuisine": "Biryani, Andhra",
                "rating": 4.5,
                "budget_tier": "medium",
                "location": "Koramangala",
                "explanation": "Legendary Andhra-style biryani with authentic spice levels.",
            },
        ],
        "summary": "Both Truffles and Meghana Foods are top-rated spots in Koramangala.",
    }
)


# ── Parser Tests ──────────────────────────────────────────────────────────────

class TestLLMParser:
    """Tests for parse_llm_response()."""

    def test_parse_valid_json(self):
        result = parse_llm_response(VALID_RAW_RESPONSE)
        assert isinstance(result, RecommendationResult)
        assert len(result.recommendations) == 2
        assert result.recommendations[0].name == "Truffles"
        assert result.recommendations[0].rank == 1
        assert result.recommendations[0].budget_tier == BudgetTier.MEDIUM
        assert "Truffles" in result.summary

    def test_parse_strips_markdown_code_fences(self):
        fenced = f"```json\n{VALID_RAW_RESPONSE}\n```"
        result = parse_llm_response(fenced)
        assert len(result.recommendations) == 2
        assert result.recommendations[0].name == "Truffles"

    def test_parse_strips_generic_code_fences(self):
        fenced = f"```\n{VALID_RAW_RESPONSE}\n```"
        result = parse_llm_response(fenced)
        assert len(result.recommendations) == 2

    def test_parse_extracts_json_with_surrounding_text(self):
        noisy = f"Here are your recommendations:\n{VALID_RAW_RESPONSE}\nHope you enjoy!"
        result = parse_llm_response(noisy)
        assert len(result.recommendations) == 2

    def test_parse_handles_missing_summary(self):
        data = {
            "recommendations": [
                {
                    "rank": 1,
                    "name": "Cafe Felix",
                    "cuisine": "Continental",
                    "rating": 4.3,
                    "budget_tier": "high",
                    "location": "MG Road",
                    "explanation": "Chic rooftop cafe.",
                }
            ]
        }
        result = parse_llm_response(json.dumps(data))
        assert result.summary is not None
        assert len(result.summary) > 0

    def test_parse_handles_missing_explanation(self):
        data = {
            "recommendations": [
                {
                    "rank": 1,
                    "name": "Cafe Felix",
                    "cuisine": "Continental",
                    "rating": 4.3,
                    "budget_tier": "high",
                    "location": "MG Road",
                }
            ],
            "summary": "Great picks.",
        }
        result = parse_llm_response(json.dumps(data))
        assert result.recommendations[0].explanation is not None
        assert len(result.recommendations[0].explanation) > 0

    def test_parse_handles_case_insensitive_budget_tier(self):
        data = {
            "recommendations": [
                {
                    "rank": 1,
                    "name": "Corner House",
                    "cuisine": "Desserts",
                    "rating": 4.6,
                    "budget_tier": "Low",
                    "location": "Indiranagar",
                    "explanation": "Best ice cream.",
                }
            ],
            "summary": "Sweet spot.",
        }
        result = parse_llm_response(json.dumps(data))
        assert result.recommendations[0].budget_tier == BudgetTier.LOW

    def test_parse_handles_alternative_key_restaurants(self):
        data = {
            "restaurants": [
                {
                    "rank": 1,
                    "name": "Nagarjuna",
                    "cuisine": "Andhra",
                    "rating": 4.4,
                    "budget_tier": "medium",
                    "location": "Residency Road",
                    "explanation": "Traditional Andhra meals.",
                }
            ],
            "summary": "Iconic meal.",
        }
        result = parse_llm_response(json.dumps(data))
        assert len(result.recommendations) == 1

    def test_parse_raises_on_empty_string(self):
        with pytest.raises(ValueError, match="empty response"):
            parse_llm_response("")

    def test_parse_raises_on_invalid_json(self):
        with pytest.raises(ValueError, match="JSON"):
            parse_llm_response("Not a JSON document at all.")

    def test_parse_raises_when_no_recommendations_key(self):
        with pytest.raises(ValueError, match="recommendations"):
            parse_llm_response(json.dumps({"random_key": "some value"}))

    def test_parse_raises_when_recommendations_list_empty(self):
        with pytest.raises(ValueError, match="empty or invalid"):
            parse_llm_response(json.dumps({"recommendations": []}))


# ── Client Provider Selection Tests (Mocked) ──────────────────────────────────

class TestLLMClient:
    """Tests for call_llm() provider selection and error handling."""

    @patch("src.llm.client._call_groq")
    def test_client_selects_groq_by_default(self, mock_groq):
        mock_groq.return_value = VALID_RAW_RESPONSE
        with patch.dict("os.environ", {"LLM_PROVIDER": "groq"}):
            result = call_llm("sys", "user")
        assert result == VALID_RAW_RESPONSE
        mock_groq.assert_called_once()

    @patch("src.llm.client._call_groq")
    def test_client_selects_groq_explicit_provider(self, mock_groq):
        mock_groq.return_value = VALID_RAW_RESPONSE
        result = call_llm("sys", "user", provider="groq")
        assert result == VALID_RAW_RESPONSE
        mock_groq.assert_called_once()

    @patch("src.llm.client._call_openai")
    def test_client_selects_openai(self, mock_openai):
        mock_openai.return_value = VALID_RAW_RESPONSE
        with patch.dict("os.environ", {"LLM_PROVIDER": "openai"}):
            result = call_llm("sys", "user")
        assert result == VALID_RAW_RESPONSE
        mock_openai.assert_called_once()

    @patch("src.llm.client._call_gemini")
    def test_client_selects_gemini(self, mock_gemini):
        mock_gemini.return_value = VALID_RAW_RESPONSE
        with patch.dict("os.environ", {"LLM_PROVIDER": "gemini"}):
            result = call_llm("sys", "user")
        assert result == VALID_RAW_RESPONSE
        mock_gemini.assert_called_once()

    def test_client_raises_on_unsupported_provider(self):
        with pytest.raises(ValueError, match="Unsupported LLM_PROVIDER"):
            call_llm("sys", "user", provider="unsupported_provider")

    def test_groq_raises_on_missing_api_key(self):
        from src.llm.client import _call_groq
        with patch.dict("os.environ", {"GROQ_API_KEY": ""}, clear=False):
            with pytest.raises(ValueError, match="GROQ_API_KEY"):
                _call_groq("sys", "user")

    def test_openai_raises_on_missing_api_key(self):
        from src.llm.client import _call_openai
        with patch.dict("os.environ", {"OPENAI_API_KEY": ""}, clear=False):
            with pytest.raises(ValueError, match="OPENAI_API_KEY"):
                _call_openai("sys", "user")

    def test_gemini_raises_on_missing_api_key(self):
        from src.llm.client import _call_gemini
        with patch.dict("os.environ", {"GEMINI_API_KEY": ""}, clear=False):
            with pytest.raises(ValueError, match="GEMINI_API_KEY"):
                _call_gemini("sys", "user")


# ── Retry Logic Tests (Mocked) ────────────────────────────────────────────────

class TestRetryLogic:
    """Tests for tenacity retry behaviour."""

    def test_is_transient_error_detects_rate_limit(self):
        class RateLimitException(Exception):
            pass

        assert _is_transient_error(RateLimitException("Rate limit reached")) is True

    def test_is_transient_error_detects_timeout(self):
        class TimeoutException(Exception):
            pass

        assert _is_transient_error(TimeoutException("Operation timed out")) is True

    def test_is_transient_error_rejects_auth_error(self):
        class AuthenticationError(Exception):
            pass

        assert _is_transient_error(AuthenticationError("Invalid API key")) is False

    def test_is_transient_error_rejects_value_error(self):
        assert _is_transient_error(ValueError("Bad input")) is False

    @patch("src.llm.client._call_groq")
    def test_retry_on_transient_error_succeeds_eventually(self, mock_groq):
        """Simulate 2 rate-limit failures followed by success."""
        class RateLimitError(Exception):
            pass

        mock_groq.side_effect = [
            RateLimitError("Rate limit exceeded 429"),
            RateLimitError("Rate limit exceeded 429"),
            VALID_RAW_RESPONSE,
        ]

        result = call_llm("sys", "user", provider="groq")
        assert result == VALID_RAW_RESPONSE
        assert mock_groq.call_count == 3

    @patch("src.llm.client._call_groq")
    def test_no_retry_on_authentication_error(self, mock_groq):
        """Authentication error must fail immediately without retrying."""
        class AuthenticationError(Exception):
            pass

        mock_groq.side_effect = AuthenticationError("Unauthorized: Invalid API key")

        with pytest.raises(AuthenticationError):
            call_llm("sys", "user", provider="groq")

        assert mock_groq.call_count == 1
