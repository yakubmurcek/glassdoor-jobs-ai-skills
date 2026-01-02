from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Dict, Any
import uvicorn
import logging

from ai_skills.embeddings import EmbeddingService
from ai_skills.skill_normalizer import HARDSKILL_CANONICALIZATION, SOFTSKILL_CANONICALIZATION

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

from sklearn.manifold import TSNE
import numpy as np

# ... (existing code) ...

@app.on_event("startup")
async def startup_event():
    """Initialize embedding service and cache skill embeddings on startup."""
    global embedding_service, skill_embeddings_cache
    logger.info("Starting up... Initializing EmbeddingService")
    embedding_service = EmbeddingService()
    
    # Pre-compute embeddings for all unique skills
    unique_skills_data = get_unique_skills()
    unique_skills_list = list(unique_skills_data.keys())
    
    logger.info(f"Computing embeddings for {len(unique_skills_list)} unique skills...")
    embeddings = embedding_service.embed_batch(unique_skills_list)
    
    # Compute 2D projection using t-SNE
    # t-SNE preserves local structure better than PCA
    if len(embeddings) > 5:
        logger.info("Projecting embeddings to 2D using t-SNE...")
        # Perplexity must be less than n_samples
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
    
    for i, skill in enumerate(unique_skills_list):
        data = unique_skills_data[skill]
        skill_embeddings_cache[skill] = {
            "type": data["type"],
            "frequency": data["count"],
            "embedding": embeddings[i],
            "x": float(coords[i, 0]),
            "y": float(coords[i, 1])
        }
    logger.info("Startup complete.")

@app.get("/api/health")
async def health_check():
    return {"status": "ok", "service": "ai-skills-backend"}

@app.get("/api/skills", response_model=List[SkillPoint])
async def get_skills():
    """Return all cached skills with embeddings and coordinates."""
    results = []
    for skill_name, data in skill_embeddings_cache.items():
        results.append(SkillPoint(
            id=skill_name,
            label=skill_name.title(),
            type=data["type"],
            embedding=data["embedding"],
            x=data["x"],
            y=data["y"],
            frequency=data.get("frequency", 1)
        ))
    return results

@app.post("/api/search")
async def search_skills(request: SearchRequest):
    """Semantic search for skills."""
    if not embedding_service:
        raise HTTPException(status_code=503, detail="Service not initialized")
        
    query_emb = embedding_service.embed_text(request.query)
    # embed_text might return a numpy array, so use len check explicitly
    if query_emb is None or len(query_emb) == 0:
        return {"results": []}
        
    # Simple cosine similarity search against cache
    # In a real app with >10k items we'd use ChromaDB/FAISS, 
    # but for ~1000 items, numpy dot product is instant.
    import numpy as np
    
    q_vec = np.array(query_emb)
    q_norm = np.linalg.norm(q_vec)
    
    results = []
    for skill_name, data in skill_embeddings_cache.items():
        s_vec = np.array(data["embedding"])
        s_norm = np.linalg.norm(s_vec)
        
        if q_norm == 0 or s_norm == 0:
            score = 0
        else:
            score = np.dot(q_vec, s_vec) / (q_norm * s_norm)
            
        results.append({
            "skill": skill_name,
            "type": data["type"],
            "score": float(score)
        })
        
    # Sort by score descending
    results.sort(key=lambda x: x["score"], reverse=True)
    return {"results": results[:request.limit]}

def start():
    """Entry point for helping to run the app directly if needed"""
    uvicorn.run("ai_skills.server:app", host="0.0.0.0", port=8000, reload=True)

if __name__ == "__main__":
    start()
