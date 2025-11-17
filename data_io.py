#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Utilities for loading, preparing, and saving job data."""

from typing import Sequence

import pandas as pd

from config import INPUT_CSV, OUTPUT_CSV, PREFERRED_COLUMN_ORDER

REQUIRED_COLUMNS: Sequence[str] = ("skills", "job_desc_text")


def load_input_data(path=INPUT_CSV) -> pd.DataFrame:
    """Read the raw CSV and normalize required columns."""
    df = pd.read_csv(path, sep=";", dtype=str, low_memory=False)

    for column in REQUIRED_COLUMNS:
        if column not in df.columns:
            df[column] = ""
        df[column] = df[column].fillna("").astype(str)

    return df


def reorder_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Place the preferred columns first while preserving the rest."""
    remaining_columns = [col for col in df.columns if col not in PREFERRED_COLUMN_ORDER]
    return df[PREFERRED_COLUMN_ORDER + remaining_columns]


def save_results(df: pd.DataFrame, path=OUTPUT_CSV) -> None:
    """Persist the final DataFrame as a CSV file."""
    df.to_csv(path, sep=";", index=False)

