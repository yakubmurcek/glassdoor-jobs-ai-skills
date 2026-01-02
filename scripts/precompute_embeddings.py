import sys
import os
import json
import logging
import numpy as np
from typing import Dict, List
from pathlib import Path

# Add project root to path
sys.path.append(os.getcwd())

from ai_skills.embeddings import EmbeddingService
from ai_skills.skill_normalizer import HARDSKILL_CANONICALIZATION, SOFTSKILL_CANONICALIZATION, get_unique_skills
from sklearn.manifold import TSNE

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def main():
    logger.info("Starting pre-computation of skill embeddings...")
    
    # Ensure data directory exists
    data_dir = Path("data")
    data_dir.mkdir(exist_ok=True)
    
    output_file = data_dir / "skills_embeddings.json"
    
    # Initialize embedding service
    embedding_service = EmbeddingService()
    
    # Get unique skills
    unique_skills_data = get_unique_skills()
    unique_skills_list = list(unique_skills_data.keys())
    logger.info(f"Found {len(unique_skills_list)} unique skills.")
    
    # Compute embeddings
    logger.info("Computing embeddings...")
    embeddings = embedding_service.embed_batch(unique_skills_list)
    
    # Compute t-SNE
    logger.info("Computing t-SNE coordinates...")
    if len(embeddings) > 5:
        n_samples = len(embeddings)
        perplexity = min(30, max(5, int(n_samples/4))) 
        tsne = TSNE(n_components=2, perplexity=perplexity, random_state=42, init='pca', learning_rate='auto')
        coords = tsne.fit_transform(np.array(embeddings))
        
        # Normalize to screen coordinates (-400 to 400)
        max_val = np.max(np.abs(coords))
        if max_val > 0:
            coords = (coords / max_val) * 400
    else:
        coords = np.zeros((len(embeddings), 2))
        
    # Prepare output data
    output_data = {}
    for i, skill in enumerate(unique_skills_list):
        data = unique_skills_data[skill]
        # Convert numpy arrays to lists for JSON serialization
        emb_list = embeddings[i]
        if hasattr(emb_list, 'tolist'):
            emb_list = emb_list.tolist()
            
        output_data[skill] = {
            "type": data["type"],
            "frequency": data["count"],
            "embedding": emb_list,
            "x": float(coords[i, 0]),
            "y": float(coords[i, 1])
        }
        
    # Save to JSON
    logger.info(f"Saving {len(output_data)} items to {output_file}...")
    with open(output_file, 'w') as f:
        json.dump(output_data, f)
        
    logger.info("Done!")

if __name__ == "__main__":
    main()
