
import sys
import os
import numpy as np
sys.path.append(os.getcwd())

from ai_skills.skill_normalizer import get_semantic_normalizer

def check_pairs():
    normalizer = get_semantic_normalizer()
    service = normalizer.embedding_service
    
    pairs = [
        ("html", "html5"),
        ("sql", "sql server"),
        ("java", "javascript"),
        ("python", "python 3"),
        ("react", "react.js"),
        ("automated testing", "test automation"),
        ("frontend development", "backend development"),
        ("html", "hypertext markup language"),
        ("react", "reactjs"),
        ("node", "node.js"),
        ("c++", "cpp"),
        ("golang", "go")
    ]
    
    print("Checking specific pairs...")
    for s1, s2 in pairs:
        emb = service.embed_batch([s1, s2])
        v1 = np.array(emb[0])
        v2 = np.array(emb[1])
        
        # Cosine sim
        sim = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
        print(f"'{s1}' vs '{s2}': {sim:.4f}")

if __name__ == "__main__":
    check_pairs()
