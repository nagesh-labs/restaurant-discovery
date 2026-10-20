"""
src/llm/client.py
─────────────────
Provider-agnostic LLM client supporting Groq (primary), OpenAI, and Google Gemini.
Includes automatic exponential backoff retry logic via tenacity for rate limits and timeouts.
"""

import logging
import os
from typing import Optional

from dotenv import load_dotenv
from tenacity import retry, retry_if_exception, stop_after_attempt, wait_exponential

# Ensure .env is loaded
load_dotenv()

logger = logging.getLogger(__name__)


def _is_transient_error(exception: BaseException) -> bool:
    """Determine whether an error is transient and should trigger a retry.

    Retries on:
      - Rate limit (429)
      - Network/connection errors
      - Timeouts
      - Server errors (500, 502, 503, 504)

    Does NOT retry on:
      - Authentication/Authorization errors (invalid or missing API key)
      - Invalid inputs / ValueErrors
    """
    if isinstance(exception, (ValueError, TypeError)):
        return False

    exc_name = type(exception).__name__
    exc_str = str(exception).lower()

    # Authentication errors should fail immediately
    if (
        "auth" in exc_name.lower()
        or "permissiondenied" in exc_name.lower()
        or "unauthorized" in exc_str
        or "invalid api key" in exc_str
        or "invalid_api_key" in exc_str
    ):
        logger.error("Authentication error encountered (%s). Not retrying.", exc_name)
        return False

    # Transient conditions to retry
    if (
        "ratelimit" in exc_name.lower()
        or "timeout" in exc_name.lower()
        or "connection" in exc_name.lower()
        or "apierror" in exc_name.lower()
        or "overloaded" in exc_str
        or "429" in exc_str
        or "503" in exc_str
    ):
        logger.warning("Transient LLM error encountered (%s): %s. Retrying...", exc_name, exception)
        return True

    status_code = getattr(exception, "status_code", None)
    if status_code in (429, 500, 502, 503, 504):
        logger.warning("Transient HTTP status %d. Retrying...", status_code)
        return True

    return False


def _call_groq(
    system_prompt: str,
    user_prompt: str,
    model: Optional[str] = None,
    timeout: float = 30.0,
) -> str:
    """Call Groq API with JSON mode enabled."""
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key or api_key == "your_groq_api_key_here":
        raise ValueError(
            "GROQ_API_KEY is not set or contains default placeholder. "
            "Please configure a valid GROQ_API_KEY in your .env file."
        )

    from groq import Groq

    client = Groq(api_key=api_key, timeout=timeout)
    model_name = model or os.getenv("GROQ_MODEL", "openai/gpt-oss-120b")

    logger.info("Calling Groq API (model=%s, timeout=%.1fs)...", model_name, timeout)
    response = client.chat.completions.create(
        model=model_name,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        response_format={"type": "json_object"},
        temperature=0.3,
    )
    return response.choices[0].message.content


def _call_openai(
    system_prompt: str,
    user_prompt: str,
    model: Optional[str] = None,
    timeout: float = 30.0,
) -> str:
    """Call OpenAI API with JSON mode enabled."""
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key or api_key == "your_openai_key_here":
        raise ValueError(
            "OPENAI_API_KEY is not set or contains default placeholder. "
            "Please configure a valid OPENAI_API_KEY in your .env file."
        )

    from openai import OpenAI

    client = OpenAI(api_key=api_key, timeout=timeout)
    model_name = model or os.getenv("OPENAI_MODEL", "gpt-4o")

    logger.info("Calling OpenAI API (model=%s, timeout=%.1fs)...", model_name, timeout)
    response = client.chat.completions.create(
        model=model_name,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        response_format={"type": "json_object"},
        temperature=0.3,
    )
    return response.choices[0].message.content


def _call_gemini(
    system_prompt: str,
    user_prompt: str,
    model: Optional[str] = None,
    timeout: float = 30.0,
) -> str:
    """Call Google Gemini API with JSON output enabled."""
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key or api_key == "your_gemini_key_here":
        raise ValueError(
            "GEMINI_API_KEY is not set or contains default placeholder. "
            "Please configure a valid GEMINI_API_KEY in your .env file."
        )

    import google.generativeai as genai

    genai.configure(api_key=api_key)
    model_name = model or os.getenv("GEMINI_MODEL", "gemini-1.5-pro")

    logger.info("Calling Gemini API (model=%s, timeout=%.1fs)...", model_name, timeout)
    gen_model = genai.GenerativeModel(
        model_name=model_name,
        system_instruction=system_prompt,
        generation_config={
            "response_mime_type": "application/json",
            "temperature": 0.3,
        },
    )
    response = gen_model.generate_content(user_prompt)
    return response.text


@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=2, max=10),
    retry=retry_if_exception(_is_transient_error),
    reraise=True,
)
def call_llm(
    system_prompt: str,
    user_prompt: str,
    provider: Optional[str] = None,
    model: Optional[str] = None,
    timeout: float = 30.0,
) -> str:
    """Call the configured LLM provider and return raw text response.

    Automatically retries on rate limits, connection errors, and timeouts
    with exponential backoff (max 3 attempts).

    Args:
        system_prompt: System role instructions for the LLM.
        user_prompt:   User prompt containing preferences and candidate list.
        provider:      LLM provider name ('groq', 'openai', 'gemini').
                       Defaults to LLM_PROVIDER in .env (or 'groq').
        model:         Specific model name override.
        timeout:       API call timeout in seconds (default 30.0s).

    Returns:
        Raw string response from the LLM.

    Raises:
        ValueError: If provider is unsupported or credentials are missing.
        Exception:  If all retries are exhausted.
    """
    selected_provider = (provider or os.getenv("LLM_PROVIDER", "groq")).strip().lower()

    if selected_provider == "groq":
        return _call_groq(system_prompt, user_prompt, model=model, timeout=timeout)
    elif selected_provider == "openai":
        return _call_openai(system_prompt, user_prompt, model=model, timeout=timeout)
    elif selected_provider == "gemini":
        return _call_gemini(system_prompt, user_prompt, model=model, timeout=timeout)
    else:
        raise ValueError(
            f"Unsupported LLM_PROVIDER: '{selected_provider}'. "
            "Supported providers are: 'groq', 'openai', 'gemini'."
        )
