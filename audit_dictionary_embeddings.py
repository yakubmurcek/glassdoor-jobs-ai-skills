
import numpy as np
import sys
import os

# Ensure we can import from the current directory
sys.path.append(os.getcwd())

from ai_skills.skill_normalizer import get_semantic_normalizer

def audit_semantic_duplicates(threshold=0.85):
    print("Initializing Normalizer (loading model)...")
    normalizer = get_semantic_normalizer()
    
    skills = normalizer.canonical_hardskills
    matrix = normalizer.hard_matrix
    
    if len(skills) == 0:
        print("No hard skills found in dictionary.")
        return

    print(f"Auditing {len(skills)} canonical hard skills for semantic duplicates (threshold={threshold})...")
    
    # Compute Cosine Similarity Matrix
    # matrix is already normalized, so dot product is cosine similarity
    similarity_matrix = np.dot(matrix, matrix.T)
    
    # Find pairs
    duplicates_found = 0
    # Iterate upper triangle only
    num_skills = len(skills)
    for i in range(num_skills):
        for j in range(i + 1, num_skills):
            score = similarity_matrix[i, j]
            if score >= threshold:
                skill_a = skills[i]
                skill_b = skills[j]
                # Filter out obvious substring matches which might be intentional (though redundant)
                print(f"[Similarity: {score:.4f}] '{skill_a}' <--> '{skill_b}'")
                duplicates_found += 1
                
    if duplicates_found == 0:
        print("No semantic duplicates found above threshold.")
    else:
        print(f"\nFound {duplicates_found} potential semantic duplicates.")

if __name__ == "__main__":
    audit_semantic_duplicates()
