"""
src/data/preprocessor.py
─────────────────────────
Cleans and normalises the raw Zomato DataFrame produced by ``loader.py``.

Pipeline (applied in order):
  1. Select and rename relevant columns to a consistent snake_case schema.
  2. Drop rows with nulls in critical fields (name, location, cuisines, rate).
  3. Parse ``rate`` from "4.1/5" format → float.
  4. Drop rows where parsed rating is outside [0.0, 5.0].
  5. Parse ``approx_cost`` string → int (strip commas, whitespace).
  6. Map ``approx_cost`` → ``budget_tier`` (low / medium / high).
  7. Normalise free-text fields (strip, lowercase).
  8. Parse boolean-string columns (online_order, book_table).
  9. Reset index and cache to ``data/processed/zomato_clean.parquet``.
"""

import logging
import re
from pathlib import Path

import pandas as pd

from src.models.schemas import BudgetTier

logger = logging.getLogger(__name__)

# ── Constants ─────────────────────────────────────────────────────────────────

PROCESSED_PATH = (
    Path(__file__).resolve().parents[2] / "data" / "processed" / "zomato_clean.parquet"
)

# Budget tier thresholds (cost for two people, in INR)
BUDGET_LOW_MAX = 300
BUDGET_MEDIUM_MAX = 700

# Columns kept from the raw dataset and their target snake_case names
COLUMN_MAP = {
    "name": "name",
    "location": "location",
    "cuisines": "cuisine",
    "rate": "rate_raw",
    "votes": "votes",
    "approx_cost(for two people)": "cost_raw",
    "rest_type": "rest_type",
    "online_order": "online_order_raw",
    "book_table": "book_table_raw",
    "listed_in(type)": "listed_type",
    "listed_in(city)": "listed_city",
}

# ── Internal helpers ──────────────────────────────────────────────────────────

def _parse_rating(raw: str) -> float | None:
    """Extract numeric rating from strings like '4.1/5' or '4.1'.

    Returns None for 'NEW', '-', or any unparseable value.
    """
    if not isinstance(raw, str):
        return None
    cleaned = raw.strip().upper()
    if cleaned in {"NEW", "-", ""}:
        return None
    # Match leading float, e.g. "4.1/5" → "4.1"
    match = re.match(r"(\d+\.?\d*)", cleaned)
    if match:
        value = float(match.group(1))
        return value if 0.0 <= value <= 5.0 else None
    return None


def _parse_cost(raw: str) -> int | None:
    """Convert cost strings like '800', '1,200' to an integer.

    Returns None for missing or unparseable values.
    """
    if not isinstance(raw, str):
        return None
    cleaned = raw.strip().replace(",", "")
    try:
        return int(float(cleaned))
    except (ValueError, TypeError):
        return None


def _map_budget_tier(cost: int | None) -> str | None:
    """Classify cost into low / medium / high budget tier."""
    if cost is None:
        return None
    if cost <= BUDGET_LOW_MAX:
        return BudgetTier.LOW.value
    if cost <= BUDGET_MEDIUM_MAX:
        return BudgetTier.MEDIUM.value
    return BudgetTier.HIGH.value


def _parse_bool_col(series: pd.Series) -> pd.Series:
    """Map 'Yes'/'No' strings to True/False booleans."""
    return series.str.strip().str.upper().map({"YES": True, "NO": False})


# ── Public API ────────────────────────────────────────────────────────────────

def preprocess(df: pd.DataFrame, save_cache: bool = True) -> pd.DataFrame:
    """Clean and normalise the raw Zomato DataFrame.

    Args:
        df:          Raw DataFrame as returned by ``load_zomato_dataset()``.
        save_cache:  If True, persist the cleaned DataFrame to parquet.

    Returns:
        A cleaned :class:`pandas.DataFrame` with the following columns::

            name, location, cuisine, rating, budget_tier,
            approx_cost, votes, rest_type, online_order, book_table,
            listed_type, listed_city
    """
    logger.info("Starting preprocessing  (input rows: %d)", len(df))

    # ── Step 1: Select and rename columns ────────────────────────────────────
    # Only keep columns that exist in this dataset version
    available = {k: v for k, v in COLUMN_MAP.items() if k in df.columns}
    df = df[list(available.keys())].rename(columns=available).copy()

    # ── Step 2: Drop rows with nulls in critical fields ───────────────────────
    critical = ["name", "location", "cuisine", "rate_raw"]
    before = len(df)
    df.dropna(subset=critical, inplace=True)
    logger.info("Dropped %d rows with nulls in critical fields", before - len(df))

    # ── Step 3: Parse rating ──────────────────────────────────────────────────
    df["rating"] = df["rate_raw"].apply(_parse_rating)
    df.drop(columns=["rate_raw"], inplace=True)

    # Drop rows where rating couldn't be parsed or is out of range
    before = len(df)
    df.dropna(subset=["rating"], inplace=True)
    logger.info("Dropped %d rows with unparseable/invalid rating", before - len(df))

    # ── Step 4: Parse approx_cost ─────────────────────────────────────────────
    if "cost_raw" in df.columns:
        df["approx_cost"] = df["cost_raw"].apply(_parse_cost)
        df.drop(columns=["cost_raw"], inplace=True)
    else:
        df["approx_cost"] = None

    # ── Step 5: Map budget tier ───────────────────────────────────────────────
    df["budget_tier"] = df["approx_cost"].apply(_map_budget_tier)

    # Drop rows where budget_tier couldn't be determined (cost was null)
    before = len(df)
    df.dropna(subset=["budget_tier"], inplace=True)
    logger.info("Dropped %d rows with unknown budget tier", before - len(df))

    # ── Step 6: Normalise text fields ─────────────────────────────────────────
    for col in ["name", "location", "cuisine"]:
        df[col] = df[col].astype(str).str.strip().str.lower()

    if "rest_type" in df.columns:
        df["rest_type"] = df["rest_type"].astype(str).str.strip().str.lower()

    # ── Step 7: Parse boolean columns ─────────────────────────────────────────
    for raw_col, target_col in [
        ("online_order_raw", "online_order"),
        ("book_table_raw", "book_table"),
    ]:
        if raw_col in df.columns:
            df[target_col] = _parse_bool_col(df[raw_col].astype(str))
            df.drop(columns=[raw_col], inplace=True)

    # ── Step 8: Final type coercions ──────────────────────────────────────────
    df["rating"] = df["rating"].astype(float).round(1)
    df["votes"] = pd.to_numeric(df.get("votes"), errors="coerce").fillna(0).astype(int)
    df["approx_cost"] = df["approx_cost"].astype("Int64")  # nullable int

    # ── Step 9: Reset index ───────────────────────────────────────────────────
    df.reset_index(drop=True, inplace=True)
    logger.info("Preprocessing complete  (output rows: %d)", len(df))

    # ── Step 10: Cache to parquet ─────────────────────────────────────────────
    if save_cache:
        PROCESSED_PATH.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(PROCESSED_PATH, index=False)
        logger.info("Cleaned dataset cached to %s", PROCESSED_PATH)

    return df


def load_processed(force_reprocess: bool = False) -> pd.DataFrame:
    """Convenience helper: return the processed DataFrame.

    Reads from the parquet cache if available; otherwise runs the full
    load → preprocess pipeline.

    Args:
        force_reprocess: Ignore the cache and rerun preprocessing.
    """
    if not force_reprocess and PROCESSED_PATH.exists():
        logger.info("Loading cleaned dataset from cache: %s", PROCESSED_PATH)
        return pd.read_parquet(PROCESSED_PATH)

    from src.data.loader import load_zomato_dataset  # local import avoids circular deps
    raw = load_zomato_dataset()
    return preprocess(raw)
