#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Education level extraction from the existing educations column.

Implements deterministic parsing per professor's methodology:
- EDUCATION2: Lowest explicitly mentioned education level
- Hierarchy: highschool < associate < bachelor < master < phd
"""

from __future__ import annotations

import re
from enum import Enum
from typing import Optional


class EducationLevel(str, Enum):
    """Education levels in ascending order."""
    
    HIGHSCHOOL = "highschool"
    ASSOCIATE = "associate"
    BACHELOR = "bachelor"
    MASTER = "master"
    PHD = "phd"
    NONE = ""
    
    @classmethod
    def hierarchy_order(cls) -> list["EducationLevel"]:
        """Return levels in ascending order (lowest to highest)."""
        return [cls.HIGHSCHOOL, cls.ASSOCIATE, cls.BACHELOR, cls.MASTER, cls.PHD]


# Regex patterns for each education level (case-insensitive)
EDUCATION_PATTERNS: dict[EducationLevel, list[re.Pattern]] = {
    EducationLevel.HIGHSCHOOL: [
        re.compile(r"\bhigh\s*school\b", re.IGNORECASE),
        re.compile(r"\bhigh-school\b", re.IGNORECASE),
        re.compile(r"\bged\b", re.IGNORECASE),
        re.compile(r"\bhsd\b", re.IGNORECASE),
    ],
    EducationLevel.ASSOCIATE: [
        re.compile(r"\bassociate['']?s?\s*degree\b", re.IGNORECASE),
        re.compile(r"\bassociate['']?s?\b", re.IGNORECASE),
        re.compile(r"\b(?:a\.?a\.?|a\.?s\.?)\b", re.IGNORECASE),
    ],
    EducationLevel.BACHELOR: [
        re.compile(r"\bbachelor['']?s?\s*degree\b", re.IGNORECASE),
        re.compile(r"\bbachelor['']?s?\b", re.IGNORECASE),
        re.compile(r"\bundergraduate\s*degree\b", re.IGNORECASE),
        re.compile(r"\b(?:b\.?a\.?|b\.?s\.?|b\.?sc\.?)\b", re.IGNORECASE),
    ],
    EducationLevel.MASTER: [
        re.compile(r"\bmaster['']?s?\s*degree\b", re.IGNORECASE),
        re.compile(r"\bmaster['']?s?\b", re.IGNORECASE),
        re.compile(r"\bgraduate\s*degree\b", re.IGNORECASE),
        re.compile(r"\bmba\b", re.IGNORECASE),
        re.compile(r"\b(?:m\.?a\.?|m\.?s\.?|m\.?sc\.?)\b", re.IGNORECASE),
    ],
    EducationLevel.PHD: [
        re.compile(r"\bph\.?d\.?\b", re.IGNORECASE),
        re.compile(r"\bdoctorate\b", re.IGNORECASE),
        re.compile(r"\bdoctoral\s*degree\b", re.IGNORECASE),
        re.compile(r"\bdoctor\s*of\b", re.IGNORECASE),
    ],
}


def detect_education_levels(text: str) -> set[EducationLevel]:
    """Detect all education levels mentioned in text."""
    if not text or not isinstance(text, str):
        return set()
    
    found = set()
    for level, patterns in EDUCATION_PATTERNS.items():
        for pattern in patterns:
            if pattern.search(text):
                found.add(level)
                break  # One match per level is enough
    return found


def extract_lowest_education(text: str) -> str:
    """Extract the LOWEST education level from text.
    
    Args:
        text: The educations column value (e.g., "Bachelor's degree, Master's degree")
        
    Returns:
        The lowest education level found, or empty string if none.
    """
    found = detect_education_levels(text)
    
    if not found:
        return ""
    
    # Return the lowest level in hierarchy
    for level in EducationLevel.hierarchy_order():
        if level in found:
            return level.value
    
    return ""


def extract_education_from_row(educations_value: Optional[str]) -> str:
    """Process a single row's educations value.
    
    Args:
        educations_value: Value from the 'educations' column (may be None/NaN)
        
    Returns:
        Lowest education level as string, or empty string if none/invalid.
    """
    if educations_value is None:
        return ""
    if not isinstance(educations_value, str):
        return ""
    
    # Normalize: handle OCR errors, punctuation variations
    normalized = educations_value.strip()
    normalized = re.sub(r"\s+", " ", normalized)  # Collapse whitespace
    
    return extract_lowest_education(normalized)
