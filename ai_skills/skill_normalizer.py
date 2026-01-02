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

# ============================================================================
# CANONICALIZATION MAPPINGS
# ============================================================================

HARDSKILL_CANONICALIZATION: dict[str, str] = {
    # REST variants
    "restful": "rest",
    "restlets": "rest",
    "restlet": "rest",
    "rest api": "rest",
    # API variants
    "apis": "api",
    "web api": "api",
    "web apis": "api",
    # .NET variants
    ".net": "dotnet",
    ".net core": "dotnet",
    "asp.net": "dotnet",
    "asp.net core": "dotnet",
    "c#": "c#",
    "csharp": "c#",
    # Cloud providers
    "amazon web services": "aws",
    "google cloud": "gcp",
    "google cloud platform": "gcp",
    "azure": "azure",
    # Office tools
    "ms excel": "excel",
    "microsoft excel": "excel",
    "ms word": "word",
    "microsoft word": "word",
    "powerpoint": "powerpoint",
    # JS frameworks
    "vue": "vue.js",
    "vuejs": "vue.js",
    "vue.js": "vue.js",
    "react.js": "react",
    "reactjs": "react",
    "react": "react",
    "node": "node.js",
    "nodejs": "node.js",
    "node.js": "node.js",
    "angular": "angular",
    "angularjs": "angular",
    # Traffic/Infrastructure
    "nginx": "nginx",
    "apache": "apache",
    "kafka": "kafka",
    "rabbitmq": "rabbitmq",
    # Languages
    "golang": "go",
    "rust": "rust",
    "typescript": "typescript",
    "ts": "typescript",
    "javascript": "javascript",
    "js": "javascript",
    "python": "python",
    "py": "python",
    "java": "java",
    "kotlin": "kotlin",
    "swift": "swift",
    "objective-c": "objective-c",
    "ruby": "ruby",
    "ruby on rails": "ruby on rails",
    # AI/ML
    "artificial intelligence": "ai",
    "machine learning": "ml",
    "deep learning": "dl",
    "natural language processing": "nlp",
    "computer vision": "cv",
    "pytorch": "pytorch",
    "tensorflow": "tensorflow",
    "scikit-learn": "scikit-learn",
    "sklearn": "scikit-learn",
    "keras": "keras",
    "pandas": "pandas",
    "numpy": "numpy",
    # Containers & Orchestration
    "k8s": "kubernetes",
    "kubernetes": "kubernetes",
    "docker": "docker",
    "containerization": "containers",
    # Databases
    "postgres": "postgresql",
    "postgresql": "postgresql",
    "mongo": "mongodb",
    "mongodb": "mongodb",
    "mssql": "sql server",
    "microsoft sql server": "sql server",
    "mysql": "mysql",
    "redis": "redis",
    "dynamodb": "dynamodb",
    "cassandra": "cassandra",
    # Testing
    "tdd": "test-driven development",
    "unit testing": "testing",
    "automated testing": "testing",
    "selenium": "selenium",
    "jest": "jest",
    "cypress": "cypress",
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
    "collaboration": "collaboration",
    "cross-functional collaboration": "collaboration",
    # Problem solving
    "problem solving": "problem-solving",
    "analytical thinking": "analytical skills",
    "critical thinking": "analytical skills",
    # Leadership
    "leadership": "leadership",
    "mentoring": "mentoring",
    # Work Ethic/Style
    "detail-oriented": "attention to detail",
    "attention to detail": "attention to detail",
    "self-starter": "initiative",
    "proactive": "initiative",
    "adaptable": "adaptability",
    "flexibility": "adaptability",
    "autonomous": "independence",
    "independent": "independence",
    "time management": "time management",
}


class SemanticSkillNormalizer:
    """Uses vector embeddings to fuzzy-match skills to a canonical list."""
    
    _instance = None
    
    def __init__(self) -> None:
        self.embedding_service = EmbeddingService()
        self.canonical_hardskills = sorted(list(set(HARDSKILL_CANONICALIZATION.values())))
        self.canonical_softskills = sorted(list(set(SOFTSKILL_CANONICALIZATION.values())))
        
        # Cache embeddings
        logger.info("Pre-computing canonical skill embeddings...")
        self.hardskill_embeddings = self.embedding_service.embed_batch(self.canonical_hardskills)
        self.softskill_embeddings = self.embedding_service.embed_batch(self.canonical_softskills)
        logger.info("Canonical skill embeddings cached.")

    @classmethod
    def get_instance(cls) -> "SemanticSkillNormalizer":
        if cls._instance is None:
            cls._instance = SemanticSkillNormalizer()
        return cls._instance

    def find_matches(self, raw_skills: List[str], skill_type: str = "hard", threshold: float = 0.6) -> List[str]:
        """Find semantic matches for a list of raw skills."""
        if not raw_skills:
            return []
            
        canonical_list = self.canonical_hardskills if skill_type == "hard" else self.canonical_softskills
        canonical_embeddings = self.hardskill_embeddings if skill_type == "hard" else self.softskill_embeddings
        
        # Embed raw skills
        raw_embeddings = self.embedding_service.embed_batch(raw_skills)
        
        matches = []
        for i, raw_emb in enumerate(raw_embeddings):
            if raw_emb is None or len(raw_emb) == 0:
                matches.append(raw_skills[i]) # Keep original if embedding failed
                continue
                
            # Compute cosine similarities
            # (Dot product of normalized vectors) -> here we assume vectors are unnormalized?
            # SentenceTransformers usually output normalized vectors if requested, but let's do cos sim manually
            # sim = (A . B) / (|A|*|B|)
            # For simplicity with numpy:
            
            raw_vec = np.array(raw_emb)
            norm_raw = np.linalg.norm(raw_vec)
            
            best_score = -1.0
            best_match = None
            
            for j, cand_emb in enumerate(canonical_embeddings):
                cand_vec = np.array(cand_emb)
                norm_cand = np.linalg.norm(cand_vec)
                
                score = np.dot(raw_vec, cand_vec) / (norm_raw * norm_cand)
                
                if score > best_score:
                    best_score = score
                    best_match = canonical_list[j]
            
            if best_score >= threshold and best_match:
                matches.append(best_match)
                # logger.debug(f"Semantic Match: '{raw_skills[i]}' -> '{best_match}' (score: {best_score:.2f})")
            else:
                matches.append(raw_skills[i]) # Keep original if no good match
        
        return matches

# Singleton instance
_SEMANTIC_NORMALIZER: Optional[SemanticSkillNormalizer] = None

def get_semantic_normalizer() -> SemanticSkillNormalizer:
    global _SEMANTIC_NORMALIZER
    if _SEMANTIC_NORMALIZER is None:
        _SEMANTIC_NORMALIZER = SemanticSkillNormalizer()
    return _SEMANTIC_NORMALIZER


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

    # 2. Schema Matching (Optional)
    if use_semantic:
        try:
            normalizer = get_semantic_normalizer()
            # Feed current unique skills to see if they map to canonicals
            unique_list = list(final_skills)
            matched = normalizer.find_matches(unique_list, skill_type="hard", threshold=0.75)
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
            matched = normalizer.find_matches(unique_list, skill_type="soft", threshold=0.75)
            final_skills = set(matched)
        except Exception as e:
            logger.warning(f"Semantic normalization failed: {e}")

    return ", ".join(sorted([s for s in final_skills if s]))
