#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""High-level orchestration of the AI skill analysis workflow."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Callable, List

import pandas as pd

from .data_io import load_input_data, reorder_columns, save_results
from .models import JobAnalysisResult
from .openai_analyzer import OpenAIJobAnalyzer
from .skill_processing import annotate_declared_skills

logger = logging.getLogger(__name__)


class JobAnalysisPipeline:
    """Coordinates each step of data ingestion and enrichment."""

    def __init__(self, *, analyzer: OpenAIJobAnalyzer | None = None) -> None:
        self.analyzer = analyzer or OpenAIJobAnalyzer()

    def run(
        self,
        *,
        progress_callback: Callable[[int, int], None] | None = None,
        input_csv: Path | str | None = None,
        output_csv: Path | str | None = None,
    ) -> pd.DataFrame:
        """Execute the full pipeline and return the final DataFrame."""
        logger.info("Starting job analysis pipeline...")
        
        df = (
            load_input_data(path=input_csv)
            if input_csv is not None
            else load_input_data()
        )
        logger.info(f"Loaded {len(df)} records.")

        df = annotate_declared_skills(df)
        logger.info("Annotated declared skills.")

        df = self._annotate_job_descriptions(df, progress_callback)
        logger.info("Finished OpenAI analysis.")

        df = reorder_columns(df)
        
        if output_csv is not None:
            save_results(df, path=output_csv)
            logger.info(f"Saved results to {output_csv}")
        else:
            save_results(df)
            logger.info("Saved results to default output path.")
            
        return df

    def _annotate_job_descriptions(
        self, df: pd.DataFrame, progress_callback: Callable[[int, int], None] | None = None
    ) -> pd.DataFrame:
        """Apply the OpenAI analyzer to every job description."""
        annotated_df = df.copy()
        job_texts = [
            None if pd.isna(text) else str(text)
            for text in annotated_df["job_desc_text"].tolist()
        ]
        job_titles = [
            "" if pd.isna(title) else str(title)
            for title in annotated_df["job_title"].tolist()
        ]
        
        logger.info(f"Analyzing {len(job_texts)} job descriptions with OpenAI...")
        results: List[JobAnalysisResult] = self.analyzer.analyze_texts(
            job_texts, job_titles=job_titles, progress_callback=progress_callback
        )

        # Convert results to a list of dicts for efficient DataFrame creation
        # We start with the dict form, but we might want to respect the None logic from before
        # if preserving "sparsity" is important. However, explicit "none" is usually better for data analysis.
        # I will use the robust values.
        
        result_dicts = [r.as_columns() for r in results]
        results_df = pd.DataFrame(result_dicts)
        
        # Concatenate columns. We reset index to ensure alignment, although strictly lists are ordered.
        # Using simple assignment is safer if indices match.
        for col in results_df.columns:
            annotated_df[col] = results_df[col].values

        # Compute agreement between hard-coded skill matcher and OpenAI classification
        # A job is considered "AI" if tier is not "none"
        annotated_df["AI_skill_agreement"] = (
            annotated_df["AI_skill_hard"] == (annotated_df["AI_tier_openai"] != "none").astype(int)
        ).astype(int)
        
        return annotated_df

