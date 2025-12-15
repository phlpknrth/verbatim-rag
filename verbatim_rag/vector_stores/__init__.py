"""
Vector storage for the Verbatim RAG system.

This package provides vector store implementations for hybrid search
using dense and sparse embeddings.
"""

from .base import VectorStore, SearchResult
from .milvus_local import LocalMilvusStore
from .milvus_cloud import CloudMilvusStore

# Re-export internal functions with underscore prefix for backwards compatibility
from .hybrid_search import (
    sanitize_hybrid_weights as _sanitize_hybrid_weights,
    normalize_weights as _normalize_weights,
    merge_hybrid_results as _merge_hybrid_results,
    convert_hits_to_results as _convert_hits_to_results,
)

__all__ = [
    "VectorStore",
    "SearchResult",
    "LocalMilvusStore",
    "CloudMilvusStore",
]
