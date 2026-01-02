# -*- coding: utf-8 -*-
"""Service for generating vector embeddings using local models.

This module encapsulates the embedding generation logic, using ChromaDB's 
built-in sentence-transformers (all-MiniLM-L6-v2) by default.
"""

import logging
from typing import List, Optional
import numpy as np

import chromadb.utils.embedding_functions as embedding_functions

logger = logging.getLogger(__name__)


class EmbeddingService:
    """Handles generation of embeddings for text strings."""

    def __init__(self, model_name: str = "all-MiniLM-L6-v2") -> None:
        """Initialize the embedding service.
        
        Args:
            model_name: The name of the sentence-transformer model to use.
                        Default is appropriate for general semantic similarity.
        """
        logger.info(f"Initializing EmbeddingService with local model '{model_name}'...")
        # ChromaDB's default embedding function uses SentenceTransformer
        self._embedding_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name=model_name
        )

    def embed_text(self, text: str) -> List[float]:
        """Generate embedding for a single string."""
        if not text:
            return []
        
        try:
            # Returns a list of embeddings (list of lists)
            embeddings = self._embedding_fn([text])
            return embeddings[0]
        except Exception as e:
            logger.error(f"Failed to embed text '{text[:50]}...': {e}")
            return []

    def embed_batch(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings for a list of strings."""
        valid_texts = [t for t in texts if t]
        if not valid_texts:
            return []

        try:
            return self._embedding_fn(valid_texts)
        except Exception as e:
            logger.error(f"Failed to embed batch of {len(texts)} texts: {e}")
            return []

    def classify_text(self, text: str, labels: List[str]) -> str:
        """Classify text into one of the provided labels based on semantic similarity.
        
        Args:
            text: The text to classify.
            labels: List of candidate labels.
            
        Returns:
            The label with the highest cosine similarity.
        """
        if not text or not labels:
            return ""
            
        # Embed text
        text_emb = self.embed_text(text)
        if text_emb is None or len(text_emb) == 0:
            return labels[0] # Fallback
            
        # Embed labels
        label_embs = self.embed_batch(labels)
        if label_embs is None or len(label_embs) == 0:
            return labels[0]
            
        # Compute similarities
        text_vec = np.array(text_emb)
        norm_text = np.linalg.norm(text_vec)
        
        best_score = -1.0
        best_label = labels[0]
        
        for i, label_emb in enumerate(label_embs):
            label_vec = np.array(label_emb)
            norm_label = np.linalg.norm(label_vec)
            
            if norm_text == 0 or norm_label == 0:
                continue
                
            score = np.dot(text_vec, label_vec) / (norm_text * norm_label)
            
            if score > best_score:
                best_score = score
                best_label = labels[i]
                
        return best_label
