# -*- coding: utf-8 -*-
"""Manager for the local ChromaDB vector store.

Handles storage and retrieval of job descriptions based on semantic similarity.
"""

import logging
from pathlib import Path
from typing import List, Dict, Any, Optional

import chromadb
from chromadb.config import Settings

from .config import DATA_DIR
from .embeddings import EmbeddingService

logger = logging.getLogger(__name__)


class VectorStoreManager:
    """Manages a local ChromaDB collection for job descriptions."""

    def __init__(
        self, 
        collection_name: str = "job_descriptions",
        persist_dir: Optional[Path] = None
    ) -> None:
        """Initialize the vector store.
        
        Args:
            collection_name: Name of the collection to distinct different data types.
            persist_dir: Directory to save the vector DB. Defaults to data/vectordb.
        """
        self.persist_dir = persist_dir or (DATA_DIR / "vectordb")
        self.persist_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Initializing ChromaDB at {self.persist_dir}")
        self.client = chromadb.PersistentClient(path=str(self.persist_dir))
        
        # We start with a default embedding function (can be customizing if needed)
        # Note: Chroma uses the default if none provided, which matches our EmbeddingService default.
        self.collection = self.client.get_or_create_collection(name=collection_name)

    def add_jobs(
        self, 
        ids: List[str], 
        documents: List[str], 
        metadatas: Optional[List[Dict[str, Any]]] = None
    ) -> None:
        """Add job descriptions to the vector store.
        
        Args:
            ids: Unique identifiers for the jobs (e.g., 'job_0').
            documents: The raw text content to embed and store.
            metadatas: Optional list of dictionaries with metadata (e.g., {'title': '...'})
        """
        if not ids or not documents:
            return

        try:
            logger.info(f"Adding {len(documents)} documents to vector store '{self.collection.name}'...")
            self.collection.upsert(
                ids=ids,
                documents=documents,
                metadatas=metadatas
            )
        except Exception as e:
            logger.error(f"Failed to add jobs to vector store: {e}")

    def query_similar_jobs(
        self, 
        query_text: str, 
        n_results: int = 5,
        where: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Search for jobs similar to the query text.
        
        Args:
            query_text: The search query (e.g., "jobs requiring python and aws").
            n_results: Number of results to return.
            where: Optional filtering criteria (e.g., {"tier": "core_ai"}).
            
        Returns:
            Dictionary with keys 'ids', 'distances', 'metadatas', 'documents'.
        """
        try:
            return self.collection.query(
                query_texts=[query_text],
                n_results=n_results,
                where=where
            )
        except Exception as e:
            logger.error(f"Vector query failed: {e}")
            return {}

    def count(self) -> int:
        """Return the number of documents in the collection."""
        return self.collection.count()
