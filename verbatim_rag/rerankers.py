"""
Reranker interfaces and provider adapters for VerbatimRAG.
"""

from __future__ import annotations

import asyncio
from abc import ABC, abstractmethod
from typing import List

from verbatim_rag.vector_stores.base import SearchResult


class Reranker(ABC):
    """Abstract base class for rerankers."""

    @abstractmethod
    def rerank(self, question: str, results: List[SearchResult]) -> List[SearchResult]:
        raise NotImplementedError

    async def rerank_async(
        self, question: str, results: List[SearchResult]
    ) -> List[SearchResult]:
        return await asyncio.to_thread(self.rerank, question, results)


class BaseReranker(Reranker):
    """Base helper for rerankers with shared options."""

    def __init__(self, rerank_k: int = 50, text_field: str = "text"):
        self.rerank_k = rerank_k
        self.text_field = text_field

    def _split_results(self, results: List[SearchResult]):
        head = results[: self.rerank_k]
        tail = results[self.rerank_k :]
        return head, tail

    def _get_texts(self, results: List[SearchResult]) -> List[str]:
        if self.text_field == "enhanced_text":
            return [r.enhanced_text or r.text for r in results]
        return [r.text for r in results]


class CohereReranker(BaseReranker):
    """Cohere reranker adapter."""

    def __init__(
        self,
        api_key: str,
        model: str = "rerank-english-v3.0",
        rerank_k: int = 50,
        text_field: str = "text",
    ):
        super().__init__(rerank_k=rerank_k, text_field=text_field)
        try:
            import cohere
        except ImportError as exc:
            raise ImportError("pip install cohere") from exc
        self.client = cohere.Client(api_key)
        self.model = model

    def rerank(self, question: str, results: List[SearchResult]) -> List[SearchResult]:
        head, tail = self._split_results(results)
        if not head:
            return results
        texts = self._get_texts(head)
        response = self.client.rerank(model=self.model, query=question, documents=texts)
        ranked = [head[item.index] for item in response.results]
        return ranked + tail


class JinaReranker(BaseReranker):
    """Jina reranker adapter (HTTP API)."""

    def __init__(
        self,
        api_key: str,
        model: str = "jina-reranker-v1-base-en",
        base_url: str = "https://api.jina.ai/v1/rerank",
        rerank_k: int = 50,
        text_field: str = "text",
    ):
        super().__init__(rerank_k=rerank_k, text_field=text_field)
        self.api_key = api_key
        self.model = model
        self.base_url = base_url

    def rerank(self, question: str, results: List[SearchResult]) -> List[SearchResult]:
        head, tail = self._split_results(results)
        if not head:
            return results
        try:
            import requests
        except ImportError as exc:
            raise ImportError("pip install requests") from exc

        payload = {
            "model": self.model,
            "query": question,
            "documents": self._get_texts(head),
        }
        headers = {"Authorization": f"Bearer {self.api_key}"}
        resp = requests.post(self.base_url, json=payload, headers=headers, timeout=60)
        resp.raise_for_status()
        data = resp.json()
        ranked = [head[item["index"]] for item in data.get("results", [])]
        return ranked + tail


class SentenceTransformersReranker(BaseReranker):
    """Local cross-encoder reranker using sentence-transformers."""

    def __init__(
        self,
        model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
        device: str = "cpu",
        rerank_k: int = 50,
        text_field: str = "text",
    ):
        super().__init__(rerank_k=rerank_k, text_field=text_field)
        try:
            from sentence_transformers import CrossEncoder
        except ImportError as exc:
            raise ImportError("pip install sentence-transformers") from exc
        self.model = CrossEncoder(model, device=device, trust_remote_code=True)

    def rerank(self, question: str, results: List[SearchResult]) -> List[SearchResult]:
        head, tail = self._split_results(results)
        if not head:
            return results
        texts = self._get_texts(head)
        pairs = [(question, text) for text in texts]
        scores = self.model.predict(pairs)
        ranked = [r for _, r in sorted(zip(scores, head), reverse=True)]
        return ranked + tail


class JinaV3Reranker(BaseReranker):
    """Jina V3 reranker using transformers."""

    def __init__(
        self,
        model: str = "jinaai/jina-reranker-v3",
        rerank_k: int = 50,
        text_field: str = "text",
    ):
        super().__init__(rerank_k=rerank_k, text_field=text_field)
        try:
            from transformers import AutoModel
        except ImportError as exc:
            raise ImportError("pip install transformers") from exc
        self.model = AutoModel.from_pretrained(
            model,
            dtype="auto",
            trust_remote_code=True,
        )
        self.model.eval()

    def rerank(self, question: str, results: List[SearchResult]) -> List[SearchResult]:
        head, tail = self._split_results(results)
        if not head:
            return results
        texts = self._get_texts(head)
        results = self.model.rerank(question, texts, top_n=self.rerank_k)
        return [head[item["index"]] for item in results]
