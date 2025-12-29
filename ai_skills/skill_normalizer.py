#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Deterministic skill normalization for extracted skills.

This module handles the post-processing of LLM-extracted skills:
- Canonicalization (restful -> rest, .net -> dotnet)
- Deduplication
- Alphabetical sorting

The LLM extracts raw skill mentions; this module normalizes them
for consistent output.
"""

import re
from typing import List

# ============================================================================
# CANONICALIZATION MAPPINGS
# Based on professor's prompt, with additions for common variations
# ============================================================================

HARDSKILL_CANONICALIZATION: dict[str, str] = {
    # REST variants
    "restful": "rest",
    "restlets": "rest",
    "restlet": "rest",
    # API variants
    "apis": "api",
    "web api": "api",
    "web apis": "api",
    # .NET variants (handled by regex too)
    ".net": "dotnet",
    ".net core": "dotnet",
    ".net 6": "dotnet",
    ".net 7": "dotnet",
    ".net 8": "dotnet",
    "asp.net": "dotnet",
    "asp.net core": "dotnet",
    # Cloud providers
    "amazon web services": "aws",
    "google cloud": "gcp",
    "google cloud platform": "gcp",
    # Office tools
    "ms excel": "excel",
    "microsoft excel": "excel",
    "ms word": "word",
    "microsoft word": "word",
    # JS frameworks
    "vue": "vue.js",
    "vuejs": "vue.js",
    "react.js": "react",
    "reactjs": "react",
    "node": "node.js",
    "nodejs": "node.js",
    # HTTP
    "http": "http/https",
    "https": "http/https",
    # SRE
    "sre": "site reliability engineering",
    # Java
    "j2ee": "java ee",
    "jakarta ee": "java ee",
    # OOP
    "oop": "object-oriented programming",
    "ooa&d": "object-oriented analysis and design",
    # TDD
    "tdd": "test-driven development",
    # Microservices
    "micro-service": "microservices",
    "micro-services": "microservices",
    "microservice": "microservices",
    # AI/ML
    "artificial intelligence": "ai",
    "machine learning": "ml",
    "deep learning": "dl",
    "natural language processing": "nlp",
    # Containers
    "k8s": "kubernetes",
    # Databases
    "postgres": "postgresql",
    "mongo": "mongodb",
    "mssql": "sql server",
    "microsoft sql server": "sql server",
}

SOFTSKILL_CANONICALIZATION: dict[str, str] = {
    # Communication
    "communication": "communication skills",
    "verbal communication": "communication skills",
    "written communication": "communication skills",
    # Teamwork
    "team player": "teamwork",
    "team work": "teamwork",
    "team-oriented": "teamwork",
    "team environments": "teamwork",
    # Collaboration
    "cross-functional teams": "collaboration",
    "cross functional teams": "collaboration",
    "cross-functional collaboration": "collaboration",
    # Problem solving
    "problem solving": "problem-solving",
    "analytical thinking": "analytical skills",
    "critical thinking": "analytical skills",
    # Attention
    "detail-oriented": "attention to detail",
    "detail oriented": "attention to detail",
    # Leadership
    "team lead": "leadership",
    "team leadership": "leadership",
    # Flexibility
    "flexible": "flexibility",
    "adaptable": "adaptability",
    # Organization
    "organizational skills": "organization skills",
    "organized": "organization skills",
    # Self-management
    "self-starter": "initiative",
    "self-motivated": "initiative",
    "proactive": "initiative",
    # Independence
    "autonomous": "independence",
    "work independently": "independence",
    "independent work": "independence",
}


def _apply_regex_canonicalization(skill: str) -> str:
    """Apply regex-based canonicalization for complex patterns."""
    # .NET pattern: .net, . net, .net core, .net 6, etc.
    # Note: Regex must be anchored to avoid matching words starting with "net" like "network"
    if re.match(r"^\.?\s*net(\s+(core|\d+))?$", skill, re.IGNORECASE):
        return "dotnet"
    return skill


def normalize_hardskills(raw_skills: List[str]) -> str:
    """Normalize hardskills: canonicalize, dedupe, sort, return comma-separated.
    
    Args:
        raw_skills: List of raw skill strings from LLM extraction
        
    Returns:
        Comma-separated, lowercase, deduplicated, alphabetically sorted skills
    """
    if not raw_skills:
        return ""
    
    normalized = set()
    for skill in raw_skills:
        if not skill or not isinstance(skill, str):
            continue
        # Lowercase and strip
        skill = skill.lower().strip()
        if not skill:
            continue
        # Apply regex canonicalization first
        skill = _apply_regex_canonicalization(skill)
        # Apply dict canonicalization
        skill = HARDSKILL_CANONICALIZATION.get(skill, skill)
        normalized.add(skill)
    
    # Sort alphabetically and join
    return ", ".join(sorted(normalized))


def normalize_softskills(raw_skills: List[str]) -> str:
    """Normalize softskills: canonicalize, dedupe, sort, return comma-separated.
    
    Args:
        raw_skills: List of raw skill strings from LLM extraction
        
    Returns:
        Comma-separated, lowercase, deduplicated, alphabetically sorted skills
    """
    if not raw_skills:
        return ""
    
    normalized = set()
    for skill in raw_skills:
        if not skill or not isinstance(skill, str):
            continue
        # Lowercase and strip
        skill = skill.lower().strip()
        if not skill:
            continue
        # Apply dict canonicalization
        skill = SOFTSKILL_CANONICALIZATION.get(skill, skill)
        normalized.add(skill)
    
    # Sort alphabetically and join
    return ", ".join(sorted(normalized))
