"""
src/data/loader.py
──────────────────
Loads the Zomato restaurant dataset from Hugging Face Hub.

On first run:  downloads from Hub → writes parquet cache.
On subsequent runs: reads from the local parquet cache instantly.
"""

import logging
from pathlib import Path

import pandas as pd
from datasets import load_dataset

logger = logging.getLogger(__name__)

# ── Constants ─────────────────────────────────────────────────────────────────

DATASET_NAME = "ManikaSaini/zomato-restaurant-recommendation"
CACHE_PATH = Path(__file__).resolve().parents[2] / "data" / "processed" / "zomato_raw.parquet"


# ── Public API ────────────────────────────────────────────────────────────────

def load_zomato_dataset(force_reload: bool = False) -> pd.DataFrame:
    """Load the raw Zomato dataset from Hugging Face Hub.

    On the first call the dataset is downloaded and cached to a local parquet
    file at ``data/processed/zomato_raw.parquet``.  Subsequent calls read from
    that cache for near-instant startup, unless *force_reload* is True.

    Args:
        force_reload: If True, bypass the cache and re-download from the Hub.

    Returns:
        A :class:`pandas.DataFrame` with the raw, unprocessed dataset rows.
    """
    if not force_reload and CACHE_PATH.exists():
        logger.info("Loading Zomato dataset from local cache: %s", CACHE_PATH)
        return pd.read_parquet(CACHE_PATH)

    logger.info("Downloading Zomato dataset from Hugging Face Hub: %s", DATASET_NAME)
    dataset = load_dataset(DATASET_NAME, split="train")
    df = dataset.to_pandas()

    # Persist raw data so future calls skip the network round-trip
    CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(CACHE_PATH, index=False)
    logger.info("Raw dataset cached to %s  (%d rows)", CACHE_PATH, len(df))

    return df
