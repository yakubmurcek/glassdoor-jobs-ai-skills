#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""High-level helpers for annotating AI skills in job listings."""

import re
from typing import List, Set

import pandas as pd

from .config import AI_SKILLS

# Lowercase version for reliable matching
AI_SKILLS_LOWER = [skill.lower() for skill in AI_SKILLS]
AI_SKILLS_SET: Set[str] = set(AI_SKILLS_LOWER)


def tokenize_skills(skill_string: str) -> List[str]:
    """Split comma-delimited skills into normalized tokens."""
    if not isinstance(skill_string, str):
        return []
    parts = re.split(r",|\n", skill_string)
    return [part.strip().lower() for part in parts if part.strip()]


def find_ai_matches(skill_string: str) -> str:
    """Return comma-separated AI skills found in the string."""
    tokens = tokenize_skills(skill_string)
    matches = sorted(set(t for t in tokens if t in AI_SKILLS_SET))
    return ", ".join(matches)


def annotate_declared_skills(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add AI-specific columns derived from the explicit `skills` field.

    Returns a copy of the input DataFrame with two new columns:
        - AI_skills_found: comma-separated list of detected AI skills
        - AI_skill_hard: binary indicator if any AI skill is present
    """
    annotated_df = df.copy()
    annotated_df["AI_skills_found"] = annotated_df["skills"].apply(find_ai_matches)
    annotated_df["AI_skill_hard"] = annotated_df["AI_skills_found"].apply(
        lambda s: int(bool(s))
    )
    return annotated_df
