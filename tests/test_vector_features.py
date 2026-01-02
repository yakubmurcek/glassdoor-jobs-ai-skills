# -*- coding: utf-8 -*-
"""Verification script for vector embedding features."""

import logging
import sys
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.append(str(PROJECT_ROOT))

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_semantic_normalization():
    logger.info("=== Testing Semantic Skill Normalization ===")
    from ai_skills.skill_normalizer import get_semantic_normalizer, normalize_hardskills
    
    # 1. Initialize (loads model)
    normalizer = get_semantic_normalizer()
    logger.info("Model loaded.")
    
    # 2. Test Cases
    test_cases = [
        ("ReactJS", "react"), # Should match regex or dict, but good to check
        ("Postgres 14", "postgresql"), # Semantic match
        ("Amazn Web Srvices", "aws"), # Typo / Semantic
        ("Vue 3", "vue.js"), # Semantic match
        ("NotASkillAtAll12345", None), # Should NOT match (keep original)
    ]
    
    for raw, expected in test_cases:
        # Note: normalize_hardskills returns a comma-separated string of canonical skills
        # If no match found, it returns the cleaned raw skill (or whatever normalize logic does)
        # Here we just want to see if the semantic matcher mapped it
        
        # We use the internal find_matches for direct testing of the semantic logic
        matches = normalizer.find_matches([raw], skill_type="hard", threshold=0.75)
        matched_val = matches[0] if matches else raw
        
        logger.info(f"Input: '{raw}' -> Match: '{matched_val}' (Expected: '{expected}')")
        
        # Note: If expected is None, we expect the output to be the input (or close to it), 
        # but NOT one of the canonical skills if it's nonsense.
        if expected:
            if matched_val == expected:
                logger.info("  [PASS]")
            else:
                logger.warning(f"  [FAIL] Expected '{expected}', got '{matched_val}'")
        else:
            if matched_val not in normalizer.canonical_hardskills:
                 logger.info("  [PASS] Correctly did not match to canonical")
            else:
                 logger.warning(f"  [FAIL] Should not have matched, but got '{matched_val}'")


def test_vector_store():
    logger.info("\n=== Testing Vector Store ===")
    from ai_skills.vector_store import VectorStoreManager
    import shutil
    
    # Use a temp dir for testing
    test_db_dir = PROJECT_ROOT / "data" / "test_vectordb"
    if test_db_dir.exists():
        shutil.rmtree(test_db_dir)
        
    vs = VectorStoreManager(persist_dir=test_db_dir)
    
    # Add dummy jobs
    jobs = [
        ("job_1", "We are looking for a Python developer with Django experience."),
        ("job_2", "Need a frontend engineer proficient in React and CSS."),
        ("job_3", "Data Scientist needed to build ML models using PyTorch."),
    ]
    
    vs.add_jobs(
        ids=[j[0] for j in jobs], 
        documents=[j[1] for j in jobs],
        metadatas=[{"title": "Dev"}, {"title": "Frontend"}, {"title": "DS"}]
    )
    
    logger.info(f"Added {len(jobs)} jobs. Collection count: {vs.count()}")
    
    # Query
    query = "machine learning expert"
    results = vs.query_similar_jobs(query, n_results=1)
    
    if results and results['ids']:
        top_match_id = results['ids'][0][0]
        top_match_doc = results['documents'][0][0]
        logger.info(f"Query: '{query}' -> Top ID: {top_match_id}")
        logger.info(f"Doc: {top_match_doc}")
        
        if top_match_id == "job_3":
             logger.info("  [PASS] Correctly retrieved the Data Science job")
        else:
             logger.warning(f"  [FAIL] Expected job_3, got {top_match_id}")
    else:
        logger.error("  [FAIL] No results returned")

    # Cleanup
    if test_db_dir.exists():
        shutil.rmtree(test_db_dir)


if __name__ == "__main__":
    test_semantic_normalization()
    test_vector_store()
