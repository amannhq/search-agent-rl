"""BM25 retrieval primitives."""

from __future__ import annotations

import math
import re
from collections import defaultdict

from ..models import Chunk, ChunkSummary
from .base import TokenCounter


class HeuristicTokenCounter:
    """Fallback token counter based on character count."""

    def __init__(self, chars_per_token: int = 4) -> None:
        self.chars_per_token = max(1, chars_per_token)

    def estimate_tokens(self, text: str) -> int:
        return max(1, len(text) // self.chars_per_token)


class BM25Index:
    """Simple Okapi BM25 index for chunk retrieval."""

    def __init__(
        self,
        k1: float = 1.5,
        b: float = 0.75,
        epsilon: float = 0.25,
        token_counter: TokenCounter | None = None,
    ) -> None:
        self.k1 = k1
        self.b = b
        self.epsilon = epsilon
        self.token_counter = token_counter or HeuristicTokenCounter()

        self.chunks: dict[str, Chunk] = {}
        self.doc_freqs: dict[str, int] = defaultdict(int)
        self.doc_lens: dict[str, int] = {}
        self.avg_doc_len = 0.0
        self.corpus_size = 0
        self._total_doc_len = 0
        self.inverted_index: dict[str, dict[str, int]] = defaultdict(dict)
        self._token_counts: dict[str, int] = {}

    def _tokenize(self, text: str) -> list[str]:
        return re.findall(r"\b\w+\b", text.lower())

    def add_chunk(self, chunk: Chunk) -> None:
        if chunk.chunk_id in self.chunks:
            return

        self.chunks[chunk.chunk_id] = chunk
        tokens = self._tokenize(chunk.content)
        doc_len = len(tokens)
        self.doc_lens[chunk.chunk_id] = doc_len

        if chunk.token_count == 0:
            chunk.token_count = self.token_counter.estimate_tokens(chunk.content)
        self._token_counts[chunk.chunk_id] = chunk.token_count

        term_freqs: dict[str, int] = defaultdict(int)
        for token in tokens:
            term_freqs[token] += 1

        for term, freq in term_freqs.items():
            if chunk.chunk_id not in self.inverted_index[term]:
                self.doc_freqs[term] += 1
            self.inverted_index[term][chunk.chunk_id] = freq

        self.corpus_size += 1
        self._total_doc_len += doc_len
        self.avg_doc_len = self._total_doc_len / self.corpus_size

    def add_chunks(self, chunks: list[Chunk]) -> None:
        for chunk in chunks:
            self.add_chunk(chunk)

    def _compute_idf(self, term: str) -> float:
        doc_freq = self.doc_freqs.get(term, 0)
        if doc_freq == 0:
            return 0.0
        idf = math.log((self.corpus_size - doc_freq + 0.5) / (doc_freq + 0.5) + 1)
        return max(idf, self.epsilon)

    def _score_document(self, chunk_id: str, query_terms: list[str]) -> float:
        score = 0.0
        doc_len = self.doc_lens.get(chunk_id, 0)

        for term in query_terms:
            if term not in self.inverted_index or chunk_id not in self.inverted_index[term]:
                continue

            tf = self.inverted_index[term][chunk_id]
            idf = self._compute_idf(term)
            numerator = tf * (self.k1 + 1)
            denominator = tf + self.k1 * (1 - self.b + self.b * doc_len / self.avg_doc_len)
            score += idf * (numerator / denominator)

        return score

    def search(
        self,
        query: str,
        top_k: int = 10,
        exclude_ids: set[str] | None = None,
    ) -> list[tuple[str, float]]:
        exclude_ids = exclude_ids or set()
        query_terms = self._tokenize(query)
        if not query_terms:
            return []

        candidates: set[str] = set()
        for term in query_terms:
            candidates.update(self.inverted_index.get(term, {}).keys())

        candidates -= exclude_ids

        scores: list[tuple[str, float]] = []
        for chunk_id in candidates:
            score = self._score_document(chunk_id, query_terms)
            if score > 0:
                scores.append((chunk_id, score))

        scores.sort(key=lambda item: item[1], reverse=True)
        return scores[:top_k]

    def get_chunk(self, chunk_id: str) -> Chunk | None:
        return self.chunks.get(chunk_id)

    def get_chunks(self, chunk_ids: list[str]) -> list[Chunk]:
        return [self.chunks[cid] for cid in chunk_ids if cid in self.chunks]

    def create_summary(
        self,
        chunk_id: str,
        score: float = 0.0,
        snippet_length: int = 200,
    ) -> ChunkSummary | None:
        chunk = self.chunks.get(chunk_id)
        if chunk is None:
            return None

        snippet = chunk.content[:snippet_length]
        if len(chunk.content) > snippet_length:
            snippet = snippet.rsplit(" ", 1)[0] + "..."

        return ChunkSummary(
            chunk_id=chunk_id,
            document_id=chunk.document_id,
            title=chunk.metadata.get("title", chunk.document_id),
            snippet=snippet,
            score=score,
            token_count=chunk.token_count,
        )

    @property
    def size(self) -> int:
        return self.corpus_size
