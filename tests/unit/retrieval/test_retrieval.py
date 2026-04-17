"""Tests for BM25 search index and document corpus."""

from searcharena import BM25Index, Chunk, DocumentCorpus


class TestBM25Index:
    """Tests for the BM25 search index."""

    def test_add_and_search_chunk(self) -> None:
        """Should be able to add chunks and search them."""
        index = BM25Index()
        chunk = Chunk(
            chunk_id="test_1",
            document_id="doc_1",
            content="Facebook was founded by Mark Zuckerberg in 2004.",
        )
        index.add_chunk(chunk)

        results = index.search("Facebook Zuckerberg")

        assert len(results) == 1
        assert results[0][0] == "test_1"
        assert results[0][1] > 0

    def test_search_excludes_ids(self) -> None:
        """Search should exclude specified IDs."""
        index = BM25Index()
        index.add_chunk(
            Chunk(chunk_id="c1", document_id="d1", content="Facebook social network")
        )
        index.add_chunk(
            Chunk(chunk_id="c2", document_id="d2", content="Facebook company")
        )

        results = index.search("Facebook", exclude_ids={"c1"})

        assert len(results) == 1
        assert results[0][0] == "c2"

    def test_search_returns_top_k(self) -> None:
        """Search should return at most top_k results."""
        index = BM25Index()
        for i in range(10):
            index.add_chunk(
                Chunk(
                    chunk_id=f"c{i}", document_id=f"d{i}", content=f"document {i} test"
                )
            )

        results = index.search("document test", top_k=3)

        assert len(results) == 3

    def test_search_empty_query(self) -> None:
        """Search with empty query should not crash."""
        index = BM25Index()
        index.add_chunk(Chunk(chunk_id="c1", document_id="d1", content="test content"))

        results = index.search("")
        assert isinstance(results, list)

    def test_search_no_results(self) -> None:
        """Search with no matches should return empty list."""
        index = BM25Index()
        index.add_chunk(
            Chunk(chunk_id="c1", document_id="d1", content="apple banana cherry")
        )

        results = index.search("xyznonexistent")
        assert results == [] or all(score == 0 for _, score in results)


class TestDocumentCorpus:
    """Tests for the DocumentCorpus class."""

    def test_add_document_creates_chunks(self, empty_corpus: DocumentCorpus) -> None:
        """Adding a document should create searchable chunks."""
        chunk_ids = empty_corpus.add_document(
            doc_id="doc_1",
            content="This is a test document about technology.",
            metadata={"title": "Test Doc"},
        )

        assert len(chunk_ids) >= 1
        assert empty_corpus.num_chunks >= 1
        assert empty_corpus.num_documents == 1

    def test_chunking_long_documents(self, empty_corpus: DocumentCorpus) -> None:
        """Long documents should be split into multiple chunks."""
        long_content = "This is a sentence. " * 100  # ~2000 characters

        chunk_ids = empty_corpus.add_document(
            doc_id="long_doc", content=long_content, chunk_size=500, chunk_overlap=100
        )

        assert len(chunk_ids) > 1

    def test_search_returns_summaries(self, empty_corpus: DocumentCorpus) -> None:
        """Search should return ChunkSummary objects."""
        empty_corpus.add_document(
            doc_id="doc_1",
            content="Facebook was founded by Mark Zuckerberg.",
            metadata={"title": "Facebook History"},
        )

        results = empty_corpus.search("Facebook Zuckerberg")

        assert len(results) >= 1
        assert hasattr(results[0], "chunk_id")
        assert hasattr(results[0], "snippet")
        assert hasattr(results[0], "score")

    def test_get_chunk_by_id(self, empty_corpus: DocumentCorpus) -> None:
        """Should retrieve a chunk by its ID."""
        chunk_ids = empty_corpus.add_document(
            doc_id="doc_1",
            content="Test content for retrieval.",
            metadata={"title": "Test"},
        )

        chunk = empty_corpus.get_chunk(chunk_ids[0])

        assert chunk is not None
        assert chunk.chunk_id == chunk_ids[0]
        assert "Test content" in chunk.content

    def test_get_nonexistent_chunk(self, empty_corpus: DocumentCorpus) -> None:
        """Getting non-existent chunk should return None."""
        chunk = empty_corpus.get_chunk("nonexistent_id")
        assert chunk is None

    def test_multiple_documents(self, empty_corpus: DocumentCorpus) -> None:
        """Should handle multiple documents correctly."""
        empty_corpus.add_document(doc_id="doc_1", content="Apple Inc technology")
        empty_corpus.add_document(doc_id="doc_2", content="Google search engine")
        empty_corpus.add_document(doc_id="doc_3", content="Microsoft Windows")

        assert empty_corpus.num_documents == 3
        assert empty_corpus.num_chunks >= 3

        # Search should find relevant docs
        results = empty_corpus.search("technology")
        assert len(results) >= 1
