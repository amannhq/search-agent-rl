"""
Retrieval system for the Search RL Environment.
Implements BM25 search for the MVP, with hooks for future hybrid search.
"""

import math
import re
from collections import defaultdict
from typing import Any, Dict, List, Optional, Set, Tuple

try:
    from ..models import Chunk, ChunkSummary
except ImportError:
    from models import Chunk, ChunkSummary


class BM25Index:
    """
    BM25 search index for document retrieval.

    A simple but effective implementation of Okapi BM25 for the MVP.
    Can be replaced with more sophisticated hybrid search later.
    """

    def __init__(
        self,
        k1: float = 1.5,
        b: float = 0.75,
        epsilon: float = 0.25,
    ):
        """
        Initialize BM25 index.

        Args:
            k1: Term frequency saturation parameter
            b: Length normalization parameter
            epsilon: Floor for IDF to handle terms in all docs
        """
        self.k1 = k1
        self.b = b
        self.epsilon = epsilon

        # Document storage
        self.chunks: Dict[str, Chunk] = {}
        self.doc_freqs: Dict[str, int] = defaultdict(int)  # term -> doc count
        self.doc_lens: Dict[str, int] = {}  # chunk_id -> doc length
        self.avg_doc_len: float = 0.0
        self.corpus_size: int = 0

        # Inverted index: term -> {chunk_id: term_freq}
        self.inverted_index: Dict[str, Dict[str, int]] = defaultdict(dict)

        # Token count cache
        self._token_counts: Dict[str, int] = {}

    def _tokenize(self, text: str) -> List[str]:
        """Simple tokenization: lowercase, split on non-alphanumeric."""
        text = text.lower()
        tokens = re.findall(r"\b\w+\b", text)
        return tokens

    def _estimate_tokens(self, text: str) -> int:
        """Estimate token count (roughly 4 chars per token for English)."""
        return max(1, len(text) // 4)

    def add_chunk(self, chunk: Chunk) -> None:
        """Add a chunk to the index."""
        if chunk.chunk_id in self.chunks:
            return  # Already indexed

        self.chunks[chunk.chunk_id] = chunk

        # Tokenize content
        tokens = self._tokenize(chunk.content)
        doc_len = len(tokens)
        self.doc_lens[chunk.chunk_id] = doc_len

        # Update token count
        if chunk.token_count == 0:
            chunk.token_count = self._estimate_tokens(chunk.content)
        self._token_counts[chunk.chunk_id] = chunk.token_count

        # Count term frequencies
        term_freqs: Dict[str, int] = defaultdict(int)
        for token in tokens:
            term_freqs[token] += 1

        # Update inverted index and document frequencies
        for term, freq in term_freqs.items():
            if chunk.chunk_id not in self.inverted_index[term]:
                self.doc_freqs[term] += 1
            self.inverted_index[term][chunk.chunk_id] = freq

        # Update corpus stats
        self.corpus_size += 1
        total_len = sum(self.doc_lens.values())
        self.avg_doc_len = total_len / self.corpus_size if self.corpus_size > 0 else 0

    def add_chunks(self, chunks: List[Chunk]) -> None:
        """Add multiple chunks to the index."""
        for chunk in chunks:
            self.add_chunk(chunk)

    def _compute_idf(self, term: str) -> float:
        """Compute IDF for a term."""
        doc_freq = self.doc_freqs.get(term, 0)
        if doc_freq == 0:
            return 0.0

        idf = math.log((self.corpus_size - doc_freq + 0.5) / (doc_freq + 0.5) + 1)
        return max(idf, self.epsilon)

    def _score_document(self, chunk_id: str, query_terms: List[str]) -> float:
        """Compute BM25 score for a document given query terms."""
        score = 0.0
        doc_len = self.doc_lens.get(chunk_id, 0)

        for term in query_terms:
            if term not in self.inverted_index:
                continue
            if chunk_id not in self.inverted_index[term]:
                continue

            tf = self.inverted_index[term][chunk_id]
            idf = self._compute_idf(term)

            # BM25 formula
            numerator = tf * (self.k1 + 1)
            denominator = tf + self.k1 * (
                1 - self.b + self.b * doc_len / self.avg_doc_len
            )
            score += idf * (numerator / denominator)

        return score

    def search(
        self,
        query: str,
        top_k: int = 10,
        exclude_ids: Optional[Set[str]] = None,
    ) -> List[Tuple[str, float]]:
        """
        Search the index.

        Args:
            query: Search query string
            top_k: Number of results to return
            exclude_ids: Chunk IDs to exclude from results

        Returns:
            List of (chunk_id, score) tuples, sorted by score descending
        """
        exclude_ids = exclude_ids or set()
        query_terms = self._tokenize(query)

        if not query_terms:
            return []

        # Find candidate documents (those containing at least one query term)
        candidates: Set[str] = set()
        for term in query_terms:
            if term in self.inverted_index:
                candidates.update(self.inverted_index[term].keys())

        # Remove excluded IDs
        candidates -= exclude_ids

        # Score candidates
        scores: List[Tuple[str, float]] = []
        for chunk_id in candidates:
            score = self._score_document(chunk_id, query_terms)
            if score > 0:
                scores.append((chunk_id, score))

        # Sort by score descending
        scores.sort(key=lambda x: x[1], reverse=True)

        return scores[:top_k]

    def get_chunk(self, chunk_id: str) -> Optional[Chunk]:
        """Get a chunk by ID."""
        return self.chunks.get(chunk_id)

    def get_chunks(self, chunk_ids: List[str]) -> List[Chunk]:
        """Get multiple chunks by ID."""
        return [self.chunks[cid] for cid in chunk_ids if cid in self.chunks]

    def get_token_count(self, chunk_id: str) -> int:
        """Get token count for a chunk."""
        return self._token_counts.get(chunk_id, 0)

    def create_summary(
        self, chunk_id: str, score: float = 0.0, snippet_length: int = 200
    ) -> Optional[ChunkSummary]:
        """Create a ChunkSummary from a chunk ID."""
        chunk = self.chunks.get(chunk_id)
        if not chunk:
            return None

        # Create snippet (first N characters)
        snippet = chunk.content[:snippet_length]
        if len(chunk.content) > snippet_length:
            snippet = snippet.rsplit(" ", 1)[0] + "..."

        # Get title from metadata or use doc ID
        title = chunk.metadata.get("title", chunk.document_id)

        return ChunkSummary(
            chunk_id=chunk_id,
            document_id=chunk.document_id,
            title=title,
            snippet=snippet,
            score=score,
            token_count=chunk.token_count,
        )

    @property
    def size(self) -> int:
        """Number of chunks in the index."""
        return self.corpus_size


class DocumentCorpus:
    """
    Document corpus manager.

    Wraps the search index and provides higher-level operations.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize corpus with optional config."""
        self.config = config or {}
        self.index = BM25Index(
            k1=self.config.get("bm25_k1", 1.5),
            b=self.config.get("bm25_b", 0.75),
        )

        # Track documents (groups of chunks)
        self.documents: Dict[str, List[str]] = defaultdict(
            list
        )  # doc_id -> [chunk_ids]

    def add_document(
        self,
        doc_id: str,
        content: str,
        metadata: Optional[Dict[str, Any]] = None,
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
    ) -> List[str]:
        """
        Add a document to the corpus, chunking if necessary.

        Args:
            doc_id: Document identifier
            content: Full document content
            metadata: Document metadata
            chunk_size: Target chunk size in characters
            chunk_overlap: Overlap between chunks

        Returns:
            List of chunk IDs created
        """
        metadata = metadata or {}
        chunks = self._chunk_text(content, chunk_size, chunk_overlap)
        chunk_ids = []

        for i, chunk_text in enumerate(chunks):
            chunk_id = f"{doc_id}_chunk_{i}"
            chunk = Chunk(
                chunk_id=chunk_id,
                document_id=doc_id,
                content=chunk_text,
                metadata={**metadata, "chunk_index": i, "total_chunks": len(chunks)},
            )
            self.index.add_chunk(chunk)
            self.documents[doc_id].append(chunk_id)
            chunk_ids.append(chunk_id)

        return chunk_ids

    def add_chunk(self, chunk: Chunk) -> None:
        """Add a pre-created chunk to the corpus."""
        self.index.add_chunk(chunk)
        self.documents[chunk.document_id].append(chunk.chunk_id)

    def _chunk_text(self, text: str, chunk_size: int, overlap: int) -> List[str]:
        """Split text into overlapping chunks."""
        if len(text) <= chunk_size:
            return [text]

        chunks = []
        start = 0

        while start < len(text):
            end = start + chunk_size

            # Try to break at sentence boundary
            if end < len(text):
                # Look for sentence end near the chunk boundary
                search_start = max(start + chunk_size - 100, start)
                search_end = min(start + chunk_size + 100, len(text))
                search_region = text[search_start:search_end]

                # Find last sentence boundary in search region
                for sep in [". ", ".\n", "! ", "? "]:
                    last_sep = search_region.rfind(sep)
                    if last_sep != -1:
                        end = search_start + last_sep + len(sep)
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
        exclude_ids: Optional[Set[str]] = None,
        snippet_length: int = 200,
    ) -> List[ChunkSummary]:
        """
        Search the corpus.

        Args:
            query: Search query
            top_k: Number of results
            exclude_ids: Chunk IDs to exclude
            snippet_length: Length of snippets in results

        Returns:
            List of ChunkSummary objects
        """
        results = self.index.search(query, top_k, exclude_ids)
        summaries = []

        for chunk_id, score in results:
            summary = self.index.create_summary(chunk_id, score, snippet_length)
            if summary:
                summaries.append(summary)

        return summaries

    def get_chunk(self, chunk_id: str) -> Optional[Chunk]:
        """Get a chunk by ID."""
        return self.index.get_chunk(chunk_id)

    def get_chunks(self, chunk_ids: List[str]) -> List[Chunk]:
        """Get multiple chunks."""
        return self.index.get_chunks(chunk_ids)

    def get_document_chunks(self, doc_id: str) -> List[Chunk]:
        """Get all chunks for a document."""
        chunk_ids = self.documents.get(doc_id, [])
        return self.get_chunks(chunk_ids)

    @property
    def num_chunks(self) -> int:
        """Total number of chunks."""
        return self.index.size

    @property
    def num_documents(self) -> int:
        """Total number of documents."""
        return len(self.documents)
