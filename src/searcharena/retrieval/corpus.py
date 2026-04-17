"""High-level corpus abstraction over chunk retrieval."""

from __future__ import annotations

from collections import defaultdict
from typing import Any

from ..models import Chunk, ChunkSummary
from .base import ChunkRetriever, TokenCounter
from .bm25 import BM25Index, HeuristicTokenCounter
from .rerank import ChunkReranker, NoOpReranker


class DocumentCorpus(ChunkRetriever):
    """Document corpus manager with pluggable retrieval stages."""

    def __init__(
        self,
        config: dict[str, Any] | None = None,
        *,
        index: BM25Index | None = None,
        reranker: ChunkReranker | None = None,
        token_counter: TokenCounter | None = None,
    ) -> None:
        self.config = config or {}
        self.token_counter = token_counter or HeuristicTokenCounter(
            self.config.get("token_estimation_chars_per_token", 4)
        )
        self.index = index or BM25Index(
            k1=self.config.get("bm25_k1", 1.5),
            b=self.config.get("bm25_b", 0.75),
            token_counter=self.token_counter,
        )
        self.reranker = reranker or NoOpReranker()
        self.documents: dict[str, list[str]] = defaultdict(list)

    def add_document(
        self,
        doc_id: str,
        content: str,
        metadata: dict[str, Any] | None = None,
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
    ) -> list[str]:
        metadata = metadata or {}
        chunks = self._chunk_text(content, chunk_size, chunk_overlap)
        chunk_ids: list[str] = []

        for index, chunk_text in enumerate(chunks):
            chunk_id = f"{doc_id}_chunk_{index}"
            chunk = Chunk(
                chunk_id=chunk_id,
                document_id=doc_id,
                content=chunk_text,
                metadata={**metadata, "chunk_index": index, "total_chunks": len(chunks)},
            )
            self.index.add_chunk(chunk)
            self.documents[doc_id].append(chunk_id)
            chunk_ids.append(chunk_id)

        return chunk_ids

    def add_chunk(self, chunk: Chunk) -> None:
        self.index.add_chunk(chunk)
        self.documents[chunk.document_id].append(chunk.chunk_id)

    def _chunk_text(self, text: str, chunk_size: int, overlap: int) -> list[str]:
        if len(text) <= chunk_size:
            return [text]

        chunks: list[str] = []
        start = 0
        while start < len(text):
            end = start + chunk_size
            if end < len(text):
                search_start = max(start + chunk_size - 100, start)
                search_end = min(start + chunk_size + 100, len(text))
                search_region = text[search_start:search_end]
                for separator in [". ", ".\n", "! ", "? "]:
                    last_sep = search_region.rfind(separator)
                    if last_sep != -1:
                        end = search_start + last_sep + len(separator)
                        break

            chunk_text = text[start:end].strip()
            if chunk_text:
                chunks.append(chunk_text)

            start = end - overlap
            if start >= len(text) - overlap:
                break

        return chunks

    def search(
        self,
        query: str,
        top_k: int = 10,
        exclude_ids: set[str] | None = None,
        snippet_length: int = 200,
    ) -> list[ChunkSummary]:
        scored_results = self.index.search(query, top_k, exclude_ids)
        summaries: list[ChunkSummary] = []
        for chunk_id, score in scored_results:
            summary = self.index.create_summary(chunk_id, score, snippet_length)
            if summary is not None:
                summaries.append(summary)

        reranked = self.reranker.rerank(query, summaries)
        rerank_top_k = int(self.config.get("rerank_top_k", 0) or 0)
        if rerank_top_k > 0:
            reranked = reranked[: min(len(reranked), rerank_top_k)]

        return reranked

    def get_chunk(self, chunk_id: str) -> Chunk | None:
        return self.index.get_chunk(chunk_id)

    def get_chunks(self, chunk_ids: list[str]) -> list[Chunk]:
        chunks: list[Chunk] = []
        for chunk_id in chunk_ids:
            chunk = self.get_chunk(chunk_id)
            if chunk is not None:
                chunks.append(chunk)
        return chunks

    def get_document_chunks(self, doc_id: str) -> list[Chunk]:
        return self.get_chunks(self.documents.get(doc_id, []))

    @property
    def num_chunks(self) -> int:
        return self.index.size

    @property
    def num_documents(self) -> int:
        return len(self.documents)
