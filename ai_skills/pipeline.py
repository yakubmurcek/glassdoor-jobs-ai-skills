#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""High-level orchestration of the AI skill analysis workflow."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Callable, List

import pandas as pd

from .data_io import load_input_data, reorder_columns, save_results
from .deterministic_extractor import (
    extract_hardskills_deterministic,
    extract_softskills_deterministic,
    merge_skills,
    format_skills_string,
)
from .models import JobAnalysisResult
from .openai_analyzer import OpenAIJobAnalyzer
from .skill_normalizer import normalize_hardskills, normalize_softskills
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

        return self._merge_results_into_df(annotated_df, results)

    def _merge_results_into_df(
        self, df: pd.DataFrame, results: List[JobAnalysisResult]
    ) -> pd.DataFrame:
        """Merge analysis results into the DataFrame.
        
        This implements hybrid skill extraction:
        1. Dictionary-based extraction (deterministic, comprehensive)
        2. LLM-based extraction (context-aware for ambiguous terms)
        3. Merged results (union of both for maximum recall)
        """
        result_dicts = [r.as_columns() for r in results]
        results_df = pd.DataFrame(result_dicts)
        
        for col in results_df.columns:
            df[col] = results_df[col].values

        # Get job description texts for dictionary extraction
        job_texts = df["job_desc_text"].fillna("").astype(str).tolist()
        
        # --- DICTIONARY-BASED EXTRACTION ---
        hardskills_dict = [
            extract_hardskills_deterministic(text) for text in job_texts
        ]
        softskills_dict = [
            extract_softskills_deterministic(text) for text in job_texts
        ]
        
        # --- LLM-BASED EXTRACTION (normalized) ---
        hardskills_llm = [
            normalize_hardskills(r.hardskills_raw).split(", ") 
            if normalize_hardskills(r.hardskills_raw) else []
            for r in results
        ]
        softskills_llm = [
            normalize_softskills(r.softskills_raw).split(", ")
            if normalize_softskills(r.softskills_raw) else []
            for r in results
        ]
        
        # --- MERGE: Union of both methods ---
        hardskills_merged = [
            merge_skills(dict_skills, llm_skills)
            for dict_skills, llm_skills in zip(hardskills_dict, hardskills_llm)
        ]
        softskills_merged = [
            merge_skills(dict_skills, llm_skills)
            for dict_skills, llm_skills in zip(softskills_dict, softskills_llm)
        ]
        
        # Add all columns to DataFrame
        df["hardskills_dict"] = [format_skills_string(s) for s in hardskills_dict]
        df["hardskills_llm"] = [format_skills_string(s) for s in hardskills_llm]
        df["hardskills"] = [format_skills_string(s) for s in hardskills_merged]
        
        df["softskills_dict"] = [format_skills_string(s) for s in softskills_dict]
        df["softskills_llm"] = [format_skills_string(s) for s in softskills_llm]
        df["softskills"] = [format_skills_string(s) for s in softskills_merged]

        # Compute agreement between hard-coded skill matcher and OpenAI classification
        # A job is considered "AI" if tier is not "none"
        df["AI_skill_agreement"] = (
            df["AI_skill_hard"] == (df["AI_tier_openai"] != "none").astype(int)
        ).astype(int)
        
        return df

