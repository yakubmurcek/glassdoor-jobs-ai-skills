# -*- coding: utf-8 -*-
"""Deterministic & Semantic skill normalization.

This module handles the post-processing of LLM-extracted skills using a hybrid approach:
1. Regex/Dictionary Canonicalization (Fast, deterministic)
2. Semantic Matching via Embeddings (Slow, robust fallback)

The LLM extracts raw skill mentions; this module normalizes them for consistent output.
"""

import re
import logging
from typing import List, Dict, Tuple, Optional
import numpy as np

# Lazy load to avoid circular imports or slow startup if not used
from .embeddings import EmbeddingService

logger = logging.getLogger(__name__)

from .skills_dictionary import (
    HARDSKILL_VARIANTS as HARDSKILL_CANONICALIZATION,
    SOFTSKILL_VARIANTS as SOFTSKILL_CANONICALIZATION,
)



# Singleton instance
_SEMANTIC_NORMALIZER = None

def get_semantic_normalizer() -> SemanticSkillNormalizer:
    global _SEMANTIC_NORMALIZER
    if _SEMANTIC_NORMALIZER is None:
        _SEMANTIC_NORMALIZER = SemanticSkillNormalizer()
    return _SEMANTIC_NORMALIZER


class SemanticSkillNormalizer:
    """Uses vector embeddings to fuzzy-match skills and auto-categorize them."""
    
    _instance = None
    
    def __init__(self) -> None:
        self.embedding_service = EmbeddingService()
        
        # Load all canonical skills
        self.canonical_hardskills = sorted(list(set(HARDSKILL_CANONICALIZATION.values())))
        self.canonical_softskills = sorted(list(set(SOFTSKILL_CANONICALIZATION.values())))
        
        logger.info("Initializing SemanticSkillNormalizer with vectorization...")
        
        # Pre-compute matrices
        self.hard_matrix, self.hard_map = self._build_matrix(self.canonical_hardskills)
        self.soft_matrix, self.soft_map = self._build_matrix(self.canonical_softskills)
        
        # Pre-compute Family Centroids for Auto-Categorization
        # We need to import SKILL_TO_FAMILY structure to build centroids
        from .skills_dictionary import SKILL_TO_FAMILY
        self.family_centroids, self.family_names = self._build_family_centroids(SKILL_TO_FAMILY)
        
        logger.info("SemanticSkillNormalizer initialized.")

    def _build_matrix(self, skills: List[str]) -> Tuple[np.ndarray, List[str]]:
        """Build a normalized embedding matrix for a list of skills."""
        if not skills:
            return np.array([]), []
            
        embeddings = self.embedding_service.embed_batch(skills)
        if not embeddings:
            return np.array([]), []
            
        # Convert to numpy and normalize (L2) for fast cosine similarity
        matrix = np.array(embeddings)
        norms = np.linalg.norm(matrix, axis=1, keepdims=True)
        # Avoid division by zero
        norms[norms == 0] = 1.0
        normalized_matrix = matrix / norms
        
        return normalized_matrix, skills

    def _build_family_centroids(self, skill_to_family: Dict[str, str]) -> Tuple[np.ndarray, List[str]]:
        """Compute the average embedding (centroid) for each skill family."""
        family_skills = {}
        for skill, family in skill_to_family.items():
            if family not in family_skills:
                family_skills[family] = []
            family_skills[family].append(skill)
            
        family_names = sorted(family_skills.keys())
        centroids = []
        
        for family in family_names:
            skills = family_skills[family]
            # Embed all skills in the family
            embs = self.embedding_service.embed_batch(skills)
            if not embs:
                centroids.append(np.zeros(384)) # Fallback assumption: 384 dim for all-MiniLM-L6-v2
                continue
                
            # Average them
            mat = np.array(embs)
            centroid = np.mean(mat, axis=0)
            
            # Normalize centroid
            norm = np.linalg.norm(centroid)
            if norm > 0:
                centroid = centroid / norm
            centroids.append(centroid)
            
        return np.array(centroids), family_names

    def find_matches(self, raw_skills: List[str], skill_type: str = "hard", threshold: float = 0.85) -> List[str]:
        """Find semantic matches for a list of raw skills using vectorized search.
        
        Args:
            raw_skills: List of skills to normalize
            skill_type: "hard" or "soft"
            threshold: Similarity threshold (0.0 to 1.0). 
                       Suggest 0.9+ for strict replacement, 0.7+ for loose matching.
        """
        if not raw_skills:
            return []
            
        matrix = self.hard_matrix if skill_type == "hard" else self.soft_matrix
        candidates = self.hard_map if skill_type == "hard" else self.soft_map
        
        if matrix.size == 0:
            return raw_skills
            
        # 1. Embed raw skills
        raw_embeddings = self.embedding_service.embed_batch(raw_skills)
        if not raw_embeddings:
            return raw_skills
            
        raw_mat = np.array(raw_embeddings)
        # Normalize raw vectors
        raw_norms = np.linalg.norm(raw_mat, axis=1, keepdims=True)
        raw_norms[raw_norms == 0] = 1.0
        raw_mat_norm = raw_mat / raw_norms
        
        # 2. Compute Cosine Similarity (Dot Product of Normalized Matrix)
        # Shape: (n_raw, n_candidates)
        scores = np.dot(raw_mat_norm, matrix.T)
        
        # 3. Find best matches
        best_indices = np.argmax(scores, axis=1)
        best_scores = np.max(scores, axis=1)
        
        matches = []
        for i, score in enumerate(best_scores):
            if score >= threshold:
                matches.append(candidates[best_indices[i]])
            else:
                matches.append(raw_skills[i]) # Keep original if no good match
                
        return matches

    def categorize_skills(self, skills: List[str]) -> Dict[str, str]:
        """Categorize a list of skills into families using semantic centroids.
        
        Returns:
            Dict mapping skill -> family name
        """
        if not skills or self.family_centroids.size == 0:
            return {s: "Uncategorized" for s in skills}
            
        # 1. Embed skills
        embs = self.embedding_service.embed_batch(skills)
        if not embs:
            return {s: "Uncategorized" for s in skills}
            
        mat = np.array(embs)
        norms = np.linalg.norm(mat, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        mat_norm = mat / norms
        
        # 2. Compare to Family Centroids
        # Shape: (n_skills, n_families)
        scores = np.dot(mat_norm, self.family_centroids.T)
        
        best_indices = np.argmax(scores, axis=1)
        best_scores = np.max(scores, axis=1)
        
        results = {}
        for i, skill in enumerate(skills):
            # Only categorize if reasonable similarity, else 'Uncategorized'
            if best_scores[i] > 0.4: # Low threshold for broad categorization
                results[skill] = self.family_names[best_indices[i]]
            else:
                results[skill] = "Other"
                
        return results

    @classmethod
    def get_instance(cls) -> "SemanticSkillNormalizer":
        if cls._instance is None:
            cls._instance = SemanticSkillNormalizer()
        return cls._instance


def _apply_regex_canonicalization(skill: str) -> str:
    """Apply regex-based canonicalization for complex patterns."""
    skill = skill.lower().strip()
    # .NET pattern
    if re.match(r"^\.?\s*net(\s+(core|\d+))?$", skill, re.IGNORECASE):
        return "dotnet"
    return skill


def normalize_hardskills(raw_skills: List[str], use_semantic: bool = False) -> str:
    """Normalize hardskills: canonicalize, dedupe, sort.
    
    Args:
        raw_skills: List of raw skill strings
        use_semantic: If True, uses vector embeddings for fuzzy matching (slower)
    """
    if not raw_skills:
        return ""
    
    # 1. First pass: Basic Cleanup & Deterministic Dict Match
    pre_normalized = []
    for skill in raw_skills:
        if not skill or not isinstance(skill, str): continue
        s = _apply_regex_canonicalization(skill)
        s = HARDSKILL_CANONICALIZATION.get(s, s) # Dict lookup
        pre_normalized.append(s)
        
    final_skills = set(pre_normalized)

    # 2. Semantic De-duplication (Variant Matching)
    if use_semantic:
        try:
            normalizer = get_semantic_normalizer()
            unique_list = list(final_skills)
            # Use high threshold for replacement to avoid false positives
            # e.g. don't replace "Java" with "JavaScript"
            matched = normalizer.find_matches(unique_list, skill_type="hard", threshold=0.90)
            final_skills = set(matched)
        except Exception as e:
            logger.warning(f"Semantic normalization failed: {e}")

    # Sort and join
    return ", ".join(sorted([s for s in final_skills if s]))


def normalize_softskills(raw_skills: List[str], use_semantic: bool = False) -> str:
    """Normalize softskills: canonicalize, dedupe, sort."""
    if not raw_skills:
        return ""
        
    pre_normalized = []
    for skill in raw_skills:
        if not skill or not isinstance(skill, str): continue
        s = skill.lower().strip()
        s = SOFTSKILL_CANONICALIZATION.get(s, s)
        pre_normalized.append(s)
        
    final_skills = set(pre_normalized)
    
    if use_semantic:
        try:
            normalizer = get_semantic_normalizer()
            unique_list = list(final_skills)
            matched = normalizer.find_matches(unique_list, skill_type="soft", threshold=0.85)
            final_skills = set(matched)
        except Exception as e:
            logger.warning(f"Semantic normalization failed: {e}")

    return ", ".join(sorted([s for s in final_skills if s]))


def get_unique_skills() -> Dict[str, Dict[str, any]]:
    """Returns a dict of unique_skill -> {type, count}."""
    skills = {}
    
    # helper to add/increment
    def add(canonical, sk_type):
        if canonical not in skills:
            skills[canonical] = {"type": sk_type, "count": 0}
        skills[canonical]["count"] += 1

    for variants in HARDSKILL_CANONICALIZATION.values():
        add(variants, "hard")
    for variants in SOFTSKILL_CANONICALIZATION.values():
        add(variants, "soft")
            
    return skills
