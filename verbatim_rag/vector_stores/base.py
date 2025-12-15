"""
Base classes for vector stores.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, List, Optional


@dataclass
class SearchResult:
    """Result from vector search."""

    id: str
    score: float
    metadata: Dict[str, Any]
    text: str  # Original clean text for display
    enhanced_text: str = ""  # Enhanced text used for vectorization


class VectorStore(ABC):
    """Base vector store interface."""

    @abstractmethod
    def add_vectors(
        self,
        ids: List[str],
        dense_vectors: Optional[List[List[float]]],
        sparse_vectors: Optional[List[Dict[int, float]]],
        texts: List[str],
        enhanced_texts: List[str],
        metadatas: List[Dict[str, Any]],
    ):
        """Add vectors with metadata, original text, and enhanced text."""
        pass

    @abstractmethod
    def query(
        self,
        dense_query: Optional[List[float]] = None,
        sparse_query: Optional[Dict[int, float]] = None,
        text_query: Optional[str] = None,
        top_k: int = 5,
        search_type: str = "hybrid",
        filter: Optional[str] = None,
    ) -> List[SearchResult]:
        """Query for similar vectors using hybrid search."""
        pass

    @abstractmethod
    def delete(self, ids: List[str]):
        """Delete vectors by IDs."""
        pass
