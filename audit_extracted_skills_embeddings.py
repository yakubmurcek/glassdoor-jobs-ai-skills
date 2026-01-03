
import pandas as pd
import numpy as np
import sys
import os

# Ensure we can import from the current directory
sys.path.append(os.getcwd())

from ai_skills.skill_normalizer import get_semantic_normalizer

def audit_extracted_skills(input_csv: str, threshold: float = 0.85):
    print(f"Loading data from {input_csv}...")
    try:
        df = pd.read_csv(input_csv, sep=";")
    except Exception as e:
        print(f"Failed to load CSV: {e}")
        return

    # Extract all unique skills found in the data
    all_skills = set()
    
    # Check hardskills column
    if 'hardskills' in df.columns:
        for val in df['hardskills'].fillna("").astype(str):
            if not val: continue
            # Assume comma separated as per our previous work
            parts = [s.strip() for s in val.split(',')]
            for p in parts:
                if p: all_skills.add(p)
                
    # Check softskills column
    if 'softskills' in df.columns:
        for val in df['softskills'].fillna("").astype(str):
            if not val: continue
            parts = [s.strip() for s in val.split(',')]
            for p in parts:
                if p: all_skills.add(p)

    unique_skills = sorted(list(all_skills))
    print(f"Found {len(unique_skills)} unique skills in the dataset.")
    
    if len(unique_skills) < 2:
        print("Not enough skills to audit.")
        return

    # Initialize Embeddings
    print("Initializing Embedding Model...")
    normalizer = get_semantic_normalizer()
    service = normalizer.embedding_service
    
    print("Embedding extracted skills...")
    embeddings = service.embed_batch(unique_skills)
    
    if not embeddings:
        print("Failed to generate embeddings.")
        return

    # Normalize vectors
    matrix = np.array(embeddings)
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    norm_matrix = matrix / norms
    
    # Compute similarity
    print("Computing similarities...")
    sim_matrix = np.dot(norm_matrix, norm_matrix.T)
    
    # Find overlapping pairs
    print(f"\n=== POTENTIAL DUPLICATES (Similarity > {threshold}) ===")
    print("These are terms appearing in your data that mean nearly the same thing.")
    print("Add these to 'skills_dictionary.py' to fix them everywhere.\n")
    
    found_duplicates = False
    count = 0
    # Iterate upper triangle
    for i in range(len(unique_skills)):
        for j in range(i + 1, len(unique_skills)):
            score = sim_matrix[i, j]
            if score >= threshold:
                s1 = unique_skills[i]
                s2 = unique_skills[j]
                
                # Ignore exact substrings which are often just variations like "React" vs "React.js"
                # Actually, show them, that's what the user wants to see
                print(f"[Similarity: {score:.4f}] '{s1}' <--> '{s2}'")
                found_duplicates = True
                count += 1

    if not found_duplicates:
        print("No duplicates found above threshold.")
    else:
        print(f"\nFound {count} candidate pairs for your dictionary.")

if __name__ == "__main__":
    # Default to the file we've been working on, or allow arg
    csv_file = "data/outputs/us_relevant_30_reprocessed.csv"
    threshold = 0.75  # Default lowered to catch HTML/HTML5 (0.77)
    
    if len(sys.argv) > 1:
        csv_file = sys.argv[1]
    
    if len(sys.argv) > 2:
        try:
            threshold = float(sys.argv[2])
        except ValueError:
            print("Invalid threshold provided, using default 0.75")
            
    audit_extracted_skills(csv_file, threshold=threshold)
