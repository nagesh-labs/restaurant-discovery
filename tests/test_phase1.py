import os
import subprocess
from pathlib import Path
from dotenv import load_dotenv

BASE_DIR = Path(__file__).resolve().parent.parent

def test_directory_structure():
    """Verify all directories defined in architecture.md exist."""
    required_dirs = [
        BASE_DIR / "docs",
        BASE_DIR / "data",
        BASE_DIR / "data" / "processed",
        BASE_DIR / "src",
        BASE_DIR / "src" / "data",
        BASE_DIR / "src" / "filters",
        BASE_DIR / "src" / "prompt",
        BASE_DIR / "src" / "llm",
        BASE_DIR / "src" / "models",
        BASE_DIR / "tests",
    ]
    for directory in required_dirs:
        assert directory.is_dir(), f"Directory missing: {directory}"

def test_required_files_exist():
    """Verify all Phase 1 root files exist."""
    required_files = [
        BASE_DIR / ".env",
        BASE_DIR / ".env.example",
        BASE_DIR / ".gitignore",
        BASE_DIR / "README.md",
        BASE_DIR / "requirements.txt",
    ]
    for file_path in required_files:
        assert file_path.is_file(), f"File missing: {file_path}"

def test_dotenv_loading():
    """Verify .env loads correctly using python-dotenv."""
    env_loaded = load_dotenv(BASE_DIR / ".env", override=True)
    assert env_loaded is True, "Failed to load .env file"
    assert os.getenv("LLM_PROVIDER") in ["openai", "gemini", "groq"], "LLM_PROVIDER not set or invalid"
    assert os.getenv("TOP_N_CANDIDATES") is not None, "TOP_N_CANDIDATES not found in .env"

def test_env_example_template():
    """Verify .env.example contains expected template fields without actual secrets."""
    example_path = BASE_DIR / ".env.example"
    content = example_path.read_text(encoding="utf-8")
    assert "LLM_PROVIDER=" in content
    assert "GROQ_API_KEY=" in content
    assert "OPENAI_API_KEY=" in content
    assert "GEMINI_API_KEY=" in content
    assert "TOP_N_CANDIDATES=" in content

def test_gitignore_rules():
    """Verify .gitignore contains necessary exclusion patterns."""
    gitignore_path = BASE_DIR / ".gitignore"
    content = gitignore_path.read_text(encoding="utf-8")
    assert ".env" in content
    assert "__pycache__" in content
    assert ".venv" in content
    assert "data/processed" in content

def test_git_ignores_env_file():
    """Verify git actually ignores .env file."""
    result = subprocess.run(
        ["git", "check-ignore", ".env"],
        cwd=BASE_DIR,
        capture_output=True,
        text=True
    )
    assert result.returncode == 0, "Git is not ignoring .env"
    assert ".env" in result.stdout

def test_installed_dependencies():
    """Verify all core dependencies can be imported successfully."""
    import datasets
    import pandas
    import rapidfuzz
    import groq
    import openai
    import google.generativeai
    import pydantic
    import streamlit
    import tenacity

    assert datasets is not None
    assert pandas is not None
    assert rapidfuzz is not None
    assert groq is not None
    assert openai is not None
    assert google.generativeai is not None
    assert pydantic is not None
    assert streamlit is not None
    assert tenacity is not None
