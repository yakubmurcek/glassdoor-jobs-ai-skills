# -*- coding: utf-8 -*-
"""Skill clustering and discovery using vector embeddings."""

import logging
from typing import List, Dict, Any
from collections import defaultdict

import numpy as np
try:
    from sklearn.cluster import KMeans
    from sklearn.metrics import silhouette_score
except ImportError:
    KMeans = None

from .embeddings import EmbeddingService

logger = logging.getLogger(__name__)


class SkillClusterer:
    """Clusters skills to identify synonyms and groupings."""

    def __init__(self, embedding_service: EmbeddingService | None = None) -> None:
        self.embedding_service = embedding_service or EmbeddingService()

    def cluster_skills(self, skills: List[str], n_clusters: int = 10) -> Dict[int, List[str]]:
        """Cluster a list of skills into n_clusters groups."""
        if KMeans is None:
            logger.error("scikit-learn is required for clustering. Please run: uv add scikit-learn")
            return {}

        unique_skills = sorted(list(set(skills)))
        if not unique_skills:
            return {}

        logger.info(f"Embedding {len(unique_skills)} unique skills for clustering...")
        embeddings = self.embedding_service.embed_batch(unique_skills)
        
        if not embeddings:
            return {}

        X = np.array(embeddings)
        
        # Adjust n_clusters if we have fewer samples than requested clusters
        n_clusters = min(n_clusters, len(unique_skills))
        
        logger.info(f"Clustering into {n_clusters} clusters...")
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        labels = kmeans.fit_predict(X)
        
        clusters = defaultdict(list)
        for skill, label in zip(unique_skills, labels):
            clusters[int(label)].append(skill)
            
        return dict(clusters)

    def suggest_optimal_clusters(self, skills: List[str], max_k: int = 20) -> int:
        """Estimate optimal number of clusters using Silhouette Score."""
        if KMeans is None:
            return 5
            
        unique_skills = sorted(list(set(skills)))
        embeddings = self.embedding_service.embed_batch(unique_skills)
        X = np.array(embeddings)
        
        best_k = 2
        best_score = -1.0
        
        limit = min(max_k, len(unique_skills) - 1)
        if limit < 2:
            return 1
            
        for k in range(2, limit + 1):
            kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
            labels = kmeans.fit_predict(X)
            score = silhouette_score(X, labels)
            if score > best_score:
                best_score = score
                best_k = k
                
        return best_k
