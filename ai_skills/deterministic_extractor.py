#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Deterministic skill extraction using dictionary matching.

This module provides functions to extract hardskills and softskills from text
using word-boundary-aware regex matching against a comprehensive dictionary.
"""

from __future__ import annotations

import re
from typing import List

from .skills_dictionary import (
    HARDSKILL_VARIANTS,
    SOFTSKILL_VARIANTS,
    get_all_hardskill_patterns,
    get_all_softskill_patterns,
    format_skills_by_family,  # re-export for pipeline
)


def _create_word_boundary_pattern(skill: str) -> re.Pattern:
    """Create a regex pattern with word boundaries for a skill.
    
    Handles special characters in skill names (like c++, .net, c#).
    """
    # Escape special regex characters
    escaped = re.escape(skill)
    # Create pattern with word boundaries
    # Use lookbehind/lookahead for more flexible matching
    pattern = rf'(?<![a-zA-Z0-9_]){escaped}(?![a-zA-Z0-9_])'
    return re.compile(pattern, re.IGNORECASE)


# Pre-compile patterns for performance (lazy loading)
_hardskill_patterns: list[tuple[re.Pattern, str]] | None = None
_softskill_patterns: list[tuple[re.Pattern, str]] | None = None


def _get_hardskill_patterns() -> list[tuple[re.Pattern, str]]:
    """Get compiled hardskill patterns (cached)."""
    global _hardskill_patterns
    if _hardskill_patterns is None:
        patterns = []
        for skill in get_all_hardskill_patterns():
            canonical = HARDSKILL_VARIANTS[skill]
            pattern = _create_word_boundary_pattern(skill)
            patterns.append((pattern, canonical))
        _hardskill_patterns = patterns
    return _hardskill_patterns


def _get_softskill_patterns() -> list[tuple[re.Pattern, str]]:
    """Get compiled softskill patterns (cached)."""
    global _softskill_patterns
    if _softskill_patterns is None:
        patterns = []
        for skill in get_all_softskill_patterns():
            canonical = SOFTSKILL_VARIANTS[skill]
            pattern = _create_word_boundary_pattern(skill)
            patterns.append((pattern, canonical))
        _softskill_patterns = patterns
    return _softskill_patterns


def extract_hardskills_deterministic(text: str) -> List[str]:
    """Extract hardskills from text using dictionary matching.
    
    Args:
        text: The job description text to extract skills from.
        
    Returns:
        Sorted list of unique canonical hardskill names found in the text.
    """
    if not text or not isinstance(text, str):
        return []
    
    found_skills: set[str] = set()
    patterns = _get_hardskill_patterns()
    
    for pattern, canonical in patterns:
        if pattern.search(text):
            found_skills.add(canonical)
    
    return sorted(found_skills)


def extract_softskills_deterministic(text: str) -> List[str]:
    """Extract softskills from text using dictionary matching.
    
    Args:
        text: The job description text to extract skills from.
        
    Returns:
        Sorted list of unique canonical softskill names found in the text.
    """
    if not text or not isinstance(text, str):
        return []
    
    found_skills: set[str] = set()
    patterns = _get_softskill_patterns()
    
    for pattern, canonical in patterns:
        if pattern.search(text):
            found_skills.add(canonical)
    
    return sorted(found_skills)


def merge_skills(dict_skills: List[str], llm_skills: List[str]) -> List[str]:
    """Merge skills from dictionary and LLM extraction.
    
    Takes the union of both lists, normalizes, deduplicates, and sorts.
    
    Args:
        dict_skills: Skills extracted by dictionary matching.
        llm_skills: Skills extracted by LLM.
        
    Returns:
        Sorted list of unique merged skills.
    """
    # Normalize both to lowercase for comparison
    dict_normalized = {s.lower().strip() for s in dict_skills if s}
    llm_normalized = {s.lower().strip() for s in llm_skills if s}
    
    # Union
    merged = dict_normalized | llm_normalized
    
    return sorted(merged)


def format_skills_string(skills: List[str]) -> str:
    """Format a list of skills as a comma-separated string."""
    if not skills:
        return ""
    return ", ".join(skills)
