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
        skip_llm: bool = False,
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

        df = self._annotate_job_descriptions(df, progress_callback, skip_llm=skip_llm)
        logger.info("Finished OpenAI analysis (or skipped with hydration).")

        df = reorder_columns(df)
        
        if output_csv is not None:
            save_results(df, path=output_csv)
            logger.info(f"Saved results to {output_csv}")
        else:
            save_results(df)
            logger.info("Saved results to default output path.")
            
        # Update server cache with new embeddings found
        try:
            self.save_embeddings_cache()
        except Exception as e:
            logger.error(f"Failed to save embeddings cache: {e}")
            
        return df

    def _annotate_job_descriptions(
        self, 
        df: pd.DataFrame, 
        progress_callback: Callable[[int, int], None] | None = None,
        skip_llm: bool = False,
    ) -> pd.DataFrame:
        """Apply the OpenAI analyzer to every job description."""
        annotated_df = df.copy()
        
        # --- HYDRATION MODE ---
        if skip_llm:
            logger.info("Skipping LLM calls. Hydrating from input CSV columns...")
            results = []
            
            # Required columns for hydration
            required_cols = [
                "desc_tier_llm", "desc_ai_llm", "desc_conf_llm", 
                "desc_rationale_llm", "edureq_llm", "desc_hard_llm", "desc_soft_llm"
            ]
            
            # Check availability
            missing = [c for c in required_cols if c not in annotated_df.columns]
            if missing:
                logger.warning(
                    f"Cannot hydrate LLM results: missing columns {missing}. "
                    "Returning empty default results for these rows."
                )
                from .models import JobAnalysisResult
                # Return empty results equal to row count
                results = [JobAnalysisResult() for _ in range(len(annotated_df))]
            else:
                from .models import JobAnalysisResult, AITier
                
                for _, row in annotated_df.iterrows():
                    # Parse education (- means None)
                    edu_val = row["edureq_llm"]
                    edu_req = None
                    if pd.notna(edu_val) and str(edu_val).strip() != "-":
                        try:
                            edu_req = int(float(edu_val))
                        except ValueError:
                            pass
                            
                    # Parse lists
                    hard = str(row.get("desc_hard_llm", "")).split(", ") if pd.notna(row.get("desc_hard_llm")) and row.get("desc_hard_llm") else []
                    soft = str(row.get("desc_soft_llm", "")).split(", ") if pd.notna(row.get("desc_soft_llm")) and row.get("desc_soft_llm") else []
                    ai_skills = str(row.get("desc_ai_llm", "")).split(", ") if pd.notna(row.get("desc_ai_llm")) and row.get("desc_ai_llm") else []

                    # Parse Tier
                    tier_str = str(row.get("desc_tier_llm", "none"))
                    try:
                        tier = AITier(tier_str)
                    except ValueError:
                        tier = AITier.NONE

                    res = JobAnalysisResult(
                        ai_tier=tier,
                        ai_skills_mentioned=[s for s in ai_skills if s],
                        confidence=float(row.get("desc_conf_llm", 0.0)),
                        rationale=str(row.get("desc_rationale_llm", "")),
                        hardskills_raw=[s for s in hard if s],
                        softskills_raw=[s for s in soft if s],
                        education_required=edu_req
                    )
                    results.append(res)
            
            return self._merge_results_into_df(annotated_df, results)

        # --- NORMAL LLM COMPUTE MODE ---
        job_texts = [
            None if pd.isna(text) else str(text)
            for text in annotated_df["job_desc_text"].tolist()
        ]
        job_titles = [
            "" if pd.isna(title) else str(title)
            for title in annotated_df["job_title"].tolist()
        ]
        
        # Extract educations column for education requirement task
        educations = []
        if "educations" in annotated_df.columns:
            educations = [
                "" if pd.isna(edu) else str(edu)
                for edu in annotated_df["educations"].tolist()
            ]
        
        logger.info(f"Analyzing {len(job_texts)} job descriptions with OpenAI...")
        results: List[JobAnalysisResult] = self.analyzer.analyze_texts(
            job_texts, 
            job_titles=job_titles, 
            educations=educations if educations else None,
            progress_callback=progress_callback
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
        
        # --- DICTIONARY-BASED EXTRACTION FROM JOB DESCRIPTION ---
        hardskills_desc = [
            extract_hardskills_deterministic(text) for text in job_texts
        ]
        softskills_desc = [
            extract_softskills_deterministic(text) for text in job_texts
        ]
        
        # --- DICTIONARY-BASED EXTRACTION FROM SKILLS COLUMN ---
        # The skills column is already comma-separated, join for dictionary matching
        skills_texts = df["skills"].fillna("").astype(str).tolist()
        hardskills_skills = [
            extract_hardskills_deterministic(text) for text in skills_texts
        ]
        softskills_skills = [
            extract_softskills_deterministic(text) for text in skills_texts
        ]
        
        # Merge skills column + job description extraction (deterministic)
        hardskills_dict = [
            merge_skills(desc, skills)
            for desc, skills in zip(hardskills_desc, hardskills_skills)
        ]
        softskills_dict = [
            merge_skills(desc, skills)
            for desc, skills in zip(softskills_desc, softskills_skills)
        ]
        
        # --- LLM-BASED EXTRACTION (normalized) ---
        hardskills_llm = []
        for r in results:
            # Enable semantic matching for LLM extraction to catch variations
            norm_h = normalize_hardskills(r.hardskills_raw, use_semantic=True)
            hardskills_llm.append(norm_h.split(", ") if norm_h else [])
            
        softskills_llm = []
        for r in results:
            norm_s = normalize_softskills(r.softskills_raw, use_semantic=True)
            softskills_llm.append(norm_s.split(", ") if norm_s else [])
        

        # --- MERGE: Union of deterministic + LLM ---
        hardskills_merged = [
            merge_skills(dict_skills, llm_skills)
            for dict_skills, llm_skills in zip(hardskills_dict, hardskills_llm)
        ]
        softskills_merged = [
            merge_skills(dict_skills, llm_skills)
            for dict_skills, llm_skills in zip(softskills_dict, softskills_llm)
        ]
        
        # --- AUTO-CATEGORIZATION (Semantic Families) ---
        from .skill_normalizer import get_semantic_normalizer
        
        # Collect all unique hard skills to categorize
        all_hardskills = set()
        for skills in hardskills_merged:
            all_hardskills.update(skills)
            
        # Categorize them
        try:
            normalizer = get_semantic_normalizer()
            # This returns {skill: "Family Name"}
            semantic_mapping = normalizer.categorize_skills(list(all_hardskills))
        except Exception as e:
            logger.warning(f"Auto-categorization failed: {e}")
            semantic_mapping = {}

        def format_with_semantic_clusters(skills: List[str]) -> str:
            if not skills: return ""
            # Group by family
            families = {}
            for skill in skills:
                # Use semantic mapping, fallback to unknown
                family = semantic_mapping.get(skill, "Uncategorized")
                # Or fallback to dictionary lookup if semantic failed/was skipped
                if family == "Uncategorized":
                    from .skills_dictionary import get_skill_family
                    family = get_skill_family(skill) or "Other"
                    
                if family not in families:
                    families[family] = []
                families[family].append(skill)
                
            # Format string
            parts = []
            for fam in sorted(families.keys()):
                fam_skills = sorted(families[fam])
                parts.append(f"{fam}: {', '.join(fam_skills)}")
            return "; ".join(parts)

        # Add all columns to DataFrame
        df["desc_hard_det"] = [format_skills_string(s) for s in hardskills_dict]
        df["desc_hard_llm"] = [format_skills_string(s) for s in hardskills_llm]
        df["hardskills"] = [format_skills_string(s) for s in hardskills_merged]
        
        # Use new semantic formatter
        df["skill_cluster"] = [format_with_semantic_clusters(s) for s in hardskills_merged]
        
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
        
        # edureq_llm: Already added from LLM results via as_columns()
        
        self.last_semantic_mapping = semantic_mapping  # Store for saving later
        self.last_unique_skills = list(all_hardskills)
        
        return df

    def save_embeddings_cache(self):
        """Save computed embeddings to JSON for server usage."""
        if not hasattr(self, 'last_unique_skills') or not self.last_unique_skills:
            return
            
        import json
        import numpy as np
        from sklearn.manifold import TSNE
        from .skill_normalizer import get_semantic_normalizer, get_unique_skills
        
        logger.info("Updating skills_embeddings.json with new skills...")
        
        # Get embeddings
        normalizer = get_semantic_normalizer()
        embeddings = normalizer.embedding_service.embed_batch(self.last_unique_skills)
        
        # Compute TSNE (Re-compute for the whole set? Or just new ones? 
        # For simplicity and consistence, we re-compute the whole set of UNIQUE skills found)
        # But wait, self.last_unique_skills only has current CSV skills. 
        # We should merge with global unique skills if we want a comprehensive map.
        # For now, let's just save the ones we found, as they are the "Active" ones.
        
        if len(embeddings) > 5:
            n_samples = len(embeddings)
            perplexity = min(30, max(5, int(n_samples/4))) 
            tsne = TSNE(n_components=2, perplexity=perplexity, random_state=42, init='pca', learning_rate='auto')
            coords = tsne.fit_transform(np.array(embeddings))
            # Normalize (-400, 400)
            max_val = np.max(np.abs(coords))
            if max_val > 0:
                coords = (coords / max_val) * 400
        else:
            coords = np.zeros((len(embeddings), 2))
            
        # Build Output
        output_data = {}
        for i, skill in enumerate(self.last_unique_skills):
            family = self.last_semantic_mapping.get(skill, "Other")
            
            # Formatting embedding for list
            emb_list = embeddings[i]
            if hasattr(emb_list, 'tolist'):
                emb_list = emb_list.tolist()
                
            output_data[skill] = {
                "type": "item", # General type
                "family": family,
                "count": 1, # We don't have global counts here easily, passing 1
                "embedding": emb_list,
                "x": float(coords[i, 0]),
                "y": float(coords[i, 1])
            }
            
        # Write
        data_dir = Path("data")
        data_dir.mkdir(exist_ok=True)
        with open(data_dir / "skills_embeddings.json", "w") as f:
            json.dump(output_data, f)
        logger.info(f"Saved {len(output_data)} skills to cache.")

