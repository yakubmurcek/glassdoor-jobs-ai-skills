from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Dict, Any
import uvicorn
import logging

import json
import os

# from ai_skills.embeddings import EmbeddingService  <-- Moved inside startup_event to avoid heavy load
from ai_skills.skill_normalizer import HARDSKILL_CANONICALIZATION, SOFTSKILL_CANONICALIZATION, get_unique_skills

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="AI Skills Analysis API")

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Global services
embedding_service = None
skill_embeddings_cache = {}
skill_matrix = None
skill_index = []  # List of dicts matching matrix rows
SEARCH_MODE = "semantic"  # "semantic" or "simple"

class SearchRequest(BaseModel):
    query: str
    limit: int = 10

class SkillPoint(BaseModel):
    id: str
    label: str
    type: str  # "hard" or "soft"
    embedding: List[float]
    x: float
    y: float
    frequency: int

from sklearn.manifold import TSNE
import numpy as np

# ... (existing code) ...

@app.on_event("startup")
async def startup_event():
    """Initialize embedding service and cache skill embeddings on startup."""
    global embedding_service, skill_embeddings_cache, SEARCH_MODE, skill_matrix, skill_index
    
    # Check for pre-computed embeddings
    embeddings_file = "data/skills_embeddings.json"
    if os.path.exists(embeddings_file):
        logger.info(f"Found pre-computed embeddings at {embeddings_file}. Running in Lite Mode.")
        SEARCH_MODE = "simple"
        with open(embeddings_file, "r") as f:
            skill_embeddings_cache = json.load(f)
        logger.info(f"Loaded {len(skill_embeddings_cache)} skills from cache.")
        
        # Build Matrix for Vectorized Search (even in Lite Mode if embeddings exist)
        try:
            skill_index = []
            vectors = []
            for skill_name, data in skill_embeddings_cache.items():
                if "embedding" in data and data["embedding"]:
                    skill_index.append({"skill": skill_name, "data": data})
                    vectors.append(data["embedding"])
            
            if vectors:
                logger.info(f"Building vectorized index for {len(vectors)} skills...")
                mat = np.array(vectors)
                # Normalize matrix
                norms = np.linalg.norm(mat, axis=1, keepdims=True)
                norms[norms == 0] = 1.0
                skill_matrix = mat / norms
                SEARCH_MODE = "semantic_vectorized"
                logger.info("Vectorized search index ready.")
        except Exception as e:
            logger.error(f"Failed to build vector index: {e}")
            
        return

    # Fallback to full semantic mode
    SEARCH_MODE = "semantic"
    logger.info("No pre-computed embeddings found. Starting logic for Semantic Mode...")
    
    # DEBUG: List data directory to help troubleshoot
    if os.path.exists("data"):
        logger.info(f"Contents of 'data' directory: {os.listdir('data')}")
    else:
        logger.info("'data' directory not found in current working directory.")
    logger.info(f"Current working directory: {os.getcwd()}")
    
    # Lazy import to avoid loading torch/transformers if not needed
    from ai_skills.embeddings import EmbeddingService
    
    logger.info("Initializing EmbeddingService (this may take memory)...")
    embedding_service = EmbeddingService()
    
    # Pre-compute embeddings for all unique skills
    unique_skills_data = get_unique_skills()
    unique_skills_list = list(unique_skills_data.keys())
    
    logger.info(f"Computing embeddings for {len(unique_skills_list)} unique skills...")
    embeddings = embedding_service.embed_batch(unique_skills_list)
    
    # Compute 2D projection using t-SNE
    if len(embeddings) > 5:
        logger.info("Projecting embeddings to 2D using t-SNE...")
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
    
    skill_index = []
    vectors = []
    for i, skill in enumerate(unique_skills_list):
        data = unique_skills_data[skill]
        emb = embeddings[i]
        
        # Cache entry
        entry = {
            "type": data["type"],
            "frequency": data["count"],
            "embedding": emb,
            "x": float(coords[i, 0]),
            "y": float(coords[i, 1])
        }
        skill_embeddings_cache[skill] = entry
        
        # Matrix entry
        skill_index.append({"skill": skill, "data": entry})
        vectors.append(emb)
        
    if vectors:
        mat = np.array(vectors)
        norms = np.linalg.norm(mat, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        skill_matrix = mat / norms
        SEARCH_MODE = "semantic_vectorized"
        
    logger.info("Startup complete.")

@app.get("/api/health")
async def health_check():
    return {"status": "ok", "service": "ai-skills-backend", "mode": SEARCH_MODE}

@app.get("/api/skills", response_model=List[SkillPoint])
async def get_skills():
    """Return all cached skills with embeddings and coordinates."""
    results = []
    for skill_name, data in skill_embeddings_cache.items():
        # Ensure embedding is a list
        emb = data.get("embedding", [])
        if hasattr(emb, 'tolist'):
            emb = emb.tolist()
            
        results.append(SkillPoint(
            id=skill_name,
            label=skill_name.title(),
            type=data["type"],
            embedding=emb,
            x=data["x"],
            y=data["y"],
            frequency=data.get("frequency", 1)
        ))
    return results

@app.post("/api/search")
async def search_skills(request: SearchRequest):
    """Search for skills. Uses vectorized semantic search."""
    global embedding_service
    
    results = []
    
    # Ensure embedding service is loaded for query embedding
    if embedding_service is None:
        from ai_skills.embeddings import EmbeddingService
        embedding_service = EmbeddingService()
    
    if "semantic" in SEARCH_MODE or SEARCH_MODE == "semantic_vectorized":
        # Embed query
        query_emb = embedding_service.embed_text(request.query)
        if query_emb is None or len(query_emb) == 0:
            return {"results": [], "mode": SEARCH_MODE}
            
        q_vec = np.array(query_emb)
        q_norm = np.linalg.norm(q_vec)
        if q_norm > 0:
            q_vec = q_vec / q_norm
            
        # Vectorized Search
        if skill_matrix is not None:
            # Dot product (Cosine Similarity since both normalized)
            scores = np.dot(skill_matrix, q_vec)
            
            # Top K
            top_k_indices = np.argsort(scores)[::-1][:request.limit]
            
            for idx in top_k_indices:
                score = scores[idx]
                if score < 0.1: break # filtering
                
                item = skill_index[idx]
                results.append({
                    "skill": item["skill"],
                    "type": item["data"]["type"],
                    "score": float(score)
                })
        else:
            # Fallback (should not happen if initialized correctly)
            logger.warning("Skill matrix not initialized, returning empty.")
            
    else:
        # Simple string matching
        query = request.query.lower()
        for skill_name, data in skill_embeddings_cache.items():
            if query in skill_name.lower():
                if query == skill_name.lower():
                    score = 1.0
                elif skill_name.lower().startswith(query):
                    score = 0.8
                else:
                    score = 0.5
                
                results.append({
                    "skill": skill_name,
                    "type": data["type"],
                    "score": score
                })
        results.sort(key=lambda x: (x["score"], -len(x["skill"])), reverse=True)
        results = results[:request.limit]

    return {"results": results, "mode": SEARCH_MODE}

def start():
    """Entry point for helping to run the app directly if needed"""
    uvicorn.run("ai_skills.server:app", host="0.0.0.0", port=8000, reload=True)

if __name__ == "__main__":
    start()
