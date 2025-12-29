#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""High-level orchestration of the AI skill analysis workflow."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Callable, List

import pandas as pd

from .config import REAL_AI_SKILLS
from .data_io import load_input_data, reorder_columns, save_results
from .deterministic_extractor import (
    extract_hardskills_deterministic,
    extract_softskills_deterministic,
    merge_skills,
    format_skills_string,
    format_skills_by_family,
)
from .education_extractor import extract_education_from_row
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
        hardskills_llm = []
        for r in results:
            norm_h = normalize_hardskills(r.hardskills_raw)
            hardskills_llm.append(norm_h.split(", ") if norm_h else [])
            
        softskills_llm = []
        for r in results:
            norm_s = normalize_softskills(r.softskills_raw)
            softskills_llm.append(norm_s.split(", ") if norm_s else [])
        
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
        df["desc_hard_det"] = [format_skills_string(s) for s in hardskills_dict]
        df["desc_hard_llm"] = [format_skills_string(s) for s in hardskills_llm]
        df["hardskills"] = [format_skills_string(s) for s in hardskills_merged]
        df["skill_cluster"] = [format_skills_by_family(s) for s in hardskills_merged]
        
        df["desc_soft_det"] = [format_skills_string(s) for s in softskills_dict]
        df["desc_soft_llm"] = [format_skills_string(s) for s in softskills_llm]
        df["softskills"] = [format_skills_string(s) for s in softskills_merged]

        # Compute agreement between dictionary matcher and LLM classification
        # A job is considered "AI" if tier is not "none"
        df["ai_det_llm_match"] = (
            df["skills_hasai_det"] == (df["desc_tier_llm"] != "none").astype(int)
        ).astype(int)
        
        # is_real_ai: Binary indicator for "real AI" jobs (building/training/deploying)
        # Hybrid approach: tier-based OR skill-based detection
        
        def is_real_ai_job(tier: str, skills_str: str) -> int:
            # Check tier first
            if tier in ["core_ai", "applied_ai"]:
                return 1
            # Check skills intersection
            if skills_str:
                detected_skills = {s.strip().lower() for s in skills_str.split(",")}
                if detected_skills & REAL_AI_SKILLS:
                    return 1
            return 0
        
        df["is_real_ai"] = df.apply(
            lambda row: is_real_ai_job(row["desc_tier_llm"], row.get("hardskills", "")),
            axis=1
        )
        
        # --- EDUCATION EXTRACTION ---
        # edu_level_det: Deterministic extraction from existing 'educations' column
        if "educations" in df.columns:
            df["edu_level_det"] = df["educations"].apply(
                lambda x: extract_education_from_row(x if pd.notna(x) else None)
            )
        else:
            df["edu_level_det"] = ""
            logger.warning("'educations' column not found, edu_level_det will be empty.")
        
        # edu_req_llm: Already added from LLM results via as_columns()
        
        return df

