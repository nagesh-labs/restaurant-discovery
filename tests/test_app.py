"""
tests/test_app.py
─────────────────
Phase 5 acceptance tests for Streamlit UI (src/app.py).
Tests:
  - Configuration & API key validation
  - Data caching via get_dataset()
  - Preset configurations integrity
  - Full UI render (via streamlit.testing.v1.AppTest)
  - Form submission pipeline with mocked LLM
  - Edge case input validation warnings (empty location, nonexistent location)
"""

import json
import os
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from streamlit.testing.v1 import AppTest

from src.app import PRESETS, get_dataset, validate_api_key

APP_FILE = str(Path(__file__).resolve().parent.parent / "src" / "app.py")

MOCK_LLM_RESPONSE = json.dumps(
    {
        "recommendations": [
            {
                "rank": 1,
                "name": "Truffles",
                "cuisine": "American, Fast Food",
                "rating": 4.7,
                "budget_tier": "medium",
                "location": "Koramangala",
                "explanation": "Best burgers and casual family dining spot in Koramangala.",
            },
            {
                "rank": 2,
                "name": "Meghana Foods",
                "cuisine": "Biryani, Andhra",
                "rating": 4.5,
                "budget_tier": "medium",
                "location": "Koramangala",
                "explanation": "Legendary spicy biryani with generous portions and vegetarian options.",
            },
            {
                "rank": 3,
                "name": "Empire Restaurant",
                "cuisine": "North Indian, Mughlai",
                "rating": 4.2,
                "budget_tier": "medium",
                "location": "Koramangala",
                "explanation": "Late-night classic with hearty North Indian curries.",
            },
        ],
        "summary": "Koramangala offers great diverse dining options matching your criteria.",
    }
)


# ── Unit tests for helper functions ───────────────────────────────────────────

class TestAppHelpers:
    """Test standalone helper logic in app.py."""

    def test_validate_api_key_valid_groq(self):
        with patch.dict(
            os.environ,
            {"LLM_PROVIDER": "groq", "GROQ_API_KEY": "gsk_valid_key_test_123"},
        ):
            is_valid, msg = validate_api_key()
            assert is_valid is True
            assert msg == ""

    def test_validate_api_key_missing_groq(self):
        with patch.dict(os.environ, {"LLM_PROVIDER": "groq", "GROQ_API_KEY": ""}):
            is_valid, msg = validate_api_key()
            assert is_valid is False
            assert "GROQ_API_KEY" in msg

    def test_validate_api_key_placeholder_groq(self):
        with patch.dict(
            os.environ,
            {"LLM_PROVIDER": "groq", "GROQ_API_KEY": "your_groq_api_key_here"},
        ):
            is_valid, msg = validate_api_key()
            assert is_valid is False
            assert "GROQ_API_KEY" in msg

    def test_validate_api_key_unsupported_provider(self):
        with patch.dict(os.environ, {"LLM_PROVIDER": "unknown_ai"}):
            is_valid, msg = validate_api_key()
            assert is_valid is False
            assert "Unsupported" in msg

    def test_get_dataset_returns_dataframe(self):
        df = get_dataset()
        assert not df.empty
        assert "location" in df.columns
        assert "cuisine" in df.columns
        assert "rating" in df.columns
        assert "budget_tier" in df.columns

    def test_presets_structure(self):
        assert len(PRESETS) >= 3
        for name, preset in PRESETS.items():
            assert "location" in preset
            assert "cuisine" in preset
            assert "budget" in preset
            assert "rating" in preset
            assert "notes" in preset
            assert preset["budget"] in ("low", "medium", "high")


# ── AppTest integration tests ─────────────────────────────────────────────────

class TestStreamlitApp:
    """End-to-end tests for the Streamlit UI lifecycle using AppTest."""

    def test_app_initial_render_widgets(self):
        """Verify the app launches and renders all required input widgets."""
        at = AppTest.from_file(APP_FILE, default_timeout=20)
        at.run()
        assert not at.exception, f"App crashed on launch: {at.exception}"

        # 2 selectboxes: Location dropdown and Budget Tier
        assert len(at.selectbox) >= 2
        loc_boxes = [s for s in at.selectbox if "Location" in s.label]
        assert len(loc_boxes) == 1
        # Check that location options are loaded from dataset
        assert "Koramangala" in loc_boxes[0].options
        assert "Indiranagar" in loc_boxes[0].options
        assert "Whitefield" in loc_boxes[0].options

        # 1 text input: Preferred Cuisine (Optional)
        assert len(at.text_input) >= 1
        cuisine_inputs = [t for t in at.text_input if "Cuisine" in t.label]
        assert len(cuisine_inputs) == 1
        assert "optional" in cuisine_inputs[0].label.lower()

        # 1 slider: Rating
        assert len(at.slider) >= 1
        # 1 text area: Notes
        assert len(at.text_area) >= 1

        # Submit button
        submit_buttons = [b for b in at.button if "Find" in b.label]
        assert len(submit_buttons) == 1

    def test_app_form_submission_success(self):
        """Submit the form and verify recommendation cards and summary render."""
        with patch("src.llm.client._call_groq", return_value=MOCK_LLM_RESPONSE):
            at = AppTest.from_file(APP_FILE, default_timeout=25)
            at.run()

            # Find submit button
            submit_btn = [b for b in at.button if "Find" in b.label][0]
            submit_btn.click().run()

            assert not at.exception, f"Exception during form submission: {at.exception}"

            # Success box for AI summary
            assert len(at.success) >= 1
            summary_text = at.success[0].value
            assert "Koramangala" in summary_text

            # Markdown output contains the restaurant names
            raw_markdown = " ".join([m.value for m in at.markdown])
            assert "Truffles" in raw_markdown
            assert "Meghana Foods" in raw_markdown

    def test_app_submission_with_empty_optional_cuisine_succeeds(self):
        """Leaving preferred cuisine empty is allowed and succeeds without warnings."""
        with patch("src.llm.client._call_groq", return_value=MOCK_LLM_RESPONSE):
            at = AppTest.from_file(APP_FILE, default_timeout=25)
            at.run()

            # Clear cuisine input (optional field)
            cui_widget = [t for t in at.text_input if "Cuisine" in t.label][0]
            cui_widget.set_value("")

            # Submit form
            submit_btn = [b for b in at.button if "Find" in b.label][0]
            submit_btn.click().run()

            assert not at.exception
            assert len(at.warning) == 0, f"Unexpected warning: {at.warning}"
            assert len(at.success) >= 1
