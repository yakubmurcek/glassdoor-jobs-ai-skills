#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Utilities for loading, preparing, and saving job data."""

import logging
from pathlib import Path
from typing import Sequence

import pandas as pd

from .config import INPUT_CSV, OUTPUT_CSV, PREFERRED_COLUMN_ORDER

logger = logging.getLogger(__name__)

REQUIRED_COLUMNS: Sequence[str] = ("skills", "job_desc_text")


def load_input_data(path=INPUT_CSV) -> pd.DataFrame:
    """Read the raw CSV and normalize required columns."""
    csv_path = Path(path)
    logger.info(f"Loading input data from {csv_path}")
    
    try:
        df = pd.read_csv(csv_path, sep=";", dtype=str, low_memory=False, encoding="utf-8-sig")
    except FileNotFoundError:
        logger.error(f"Input file not found: {csv_path}")
        raise
    except Exception as e:
        logger.error(f"Failed to read input CSV: {e}")
        raise

    return _ensure_required_columns(df)


def _ensure_required_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Ensure all required columns exist and are treated as strings."""
    for column in REQUIRED_COLUMNS:
        if column not in df.columns:
            logger.warning(f"Missing required column '{column}' - filling with empty strings.")
            df[column] = ""
        df[column] = df[column].fillna("").astype(str)
    return df


def reorder_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Place the preferred columns first while preserving the rest."""
    remaining_columns = [col for col in df.columns if col not in PREFERRED_COLUMN_ORDER]
    return df[PREFERRED_COLUMN_ORDER + remaining_columns]


def save_results(df: pd.DataFrame, path=OUTPUT_CSV) -> None:
    """Persist the final DataFrame as a CSV file."""
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    logger.info(f"Saving {len(df)} rows to {output_path}")
    df.to_csv(output_path, sep=";", index=False, encoding="utf-8-sig")
