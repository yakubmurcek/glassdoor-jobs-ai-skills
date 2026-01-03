#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Deterministic skill extraction using dictionary matching.

This module provides functions to extract hardskills and softskills from text
using word-boundary-aware regex matching against a comprehensive dictionary.
"""

from __future__ import annotations

import re
from typing import List, Tuple

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


def _extract_skills_with_spans(text: str, patterns: List[tuple[re.Pattern, str]]) -> List[str]:
    """Extract skills using span-based matching to resolve overlaps.
    
    Algorithm:
    1. Find all matches for all skills in the dictionary.
    2. Collect (start, end, lengths, canonical_name).
    3. Sort matches by length (descending) to prioritize specific terms (e.g. "SQL Server" > "SQL").
    4. Fill a boolean mask for the text length. If a match fits in free space, valid. Else, ignore.
    """
    if not text or not isinstance(text, str):
        return []
    
    # Track all candidate matches: (start, end, length, canonical)
    candidates = []
    
    # We scan for ALL patterns. 
    # n.b. efficient enough for typical job descriptions (<10k chars).
    for pattern, canonical in patterns:
        for match in pattern.finditer(text):
            candidates.append((match.start(), match.end(), match.end() - match.start(), canonical))
            
    # Sort by length DESC, then by start position
    candidates.sort(key=lambda x: (x[2], x[0]), reverse=True)
    
    found_skills = set()
    occupied = set() # Track occupied indices
    
    # Resolution Strategy: Longest Match Wins (Greedy)
    # Because we sorted by length, we process "SQL Server" before "SQL".
    for start, end, length, canonical in candidates:
        # Check if any character position is already occupied
        is_overlap = False
        for i in range(start, end):
            if i in occupied:
                is_overlap = True
                break
        
        if not is_overlap:
            # Valid match - take it
            found_skills.add(canonical)
            # Mark positions as occupied
            for i in range(start, end):
                occupied.add(i)
                
    return sorted(found_skills)


def extract_hardskills_deterministic(text: str) -> List[str]:
    """Extract hardskills from text using dictionary matching with overlap resolution.
    
    Ensures "SQL Server" doesn't also trigger "SQL", "React Native" doesn't trigger "React".
    
    Args:
        text: The job description text to extract skills from.
        
    Returns:
        Sorted list of unique canonical hardskill names found in the text.
    """
    patterns = _get_hardskill_patterns()
    return _extract_skills_with_spans(text, patterns)


def extract_softskills_deterministic(text: str) -> List[str]:
    """Extract softskills from text using dictionary matching with overlap resolution.
    
    Args:
        text: The job description text to extract skills from.
        
    Returns:
        Sorted list of unique canonical softskill names found in the text.
    """
    patterns = _get_softskill_patterns()
    return _extract_skills_with_spans(text, patterns)


def merge_skills(dict_skills: List[str], llm_skills: List[str]) -> List[str]:
    """Merge skills from dictionary and LLM extraction.
    
    Takes the union of both lists, normalizes, deduplicates, and sorts.
    
    Args:
        dict_skills: Skills extracted by dictionary matching.
        llm_skills: Skills extracted by LLM.
        
    Returns:
        Sorted list of unique merged skills.
    """
    # Create a map of lowercase -> original casing
    # Priority: dict_skills (canonical) > llm_skills (variable)
    case_map = {}
    
    # Process LLM skills first (lower priority casing)
    for skill in llm_skills:
        if skill:
            case_map[skill.lower().strip()] = skill.strip()
            
    # Process Dictionary skills second (overwrite with canonical casing)
    for skill in dict_skills:
        if skill:
            case_map[skill.lower().strip()] = skill.strip()
    
    # Sort by the final case-preserved strings
    merged = sorted(case_map.values())
    
    return merged


def format_skills_string(skills: List[str]) -> str:
    """Format a list of skills as a comma-separated string."""
    if not skills:
        return ""
    return ", ".join(skills)
