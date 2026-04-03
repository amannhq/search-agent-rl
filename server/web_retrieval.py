"""
Live web retrieval helpers for the Search RL Environment.

This module keeps the agent-facing tool contract unchanged:
- search() returns ChunkSummary handles
- read() resolves those handles into Chunk objects

The current implementation uses Serper for Google SERP results and fetches
pages directly over HTTP when a result is read. Content extraction uses
trafilatura for robust main-content extraction.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Set, Tuple
from urllib import error, request

import trafilatura  # type: ignore[import-untyped]

try:
    from ..log_utils import append_log_record
    from ..models import Chunk, ChunkSummary
except ImportError:
    from log_utils import append_log_record
    from models import Chunk, ChunkSummary


class ContentExtractor:
    """
    HTML content extractor using trafilatura.
    Extracts main article content, removing navigation, ads, boilerplate.
    """

    def __init__(self, max_chars: int = 8000):
        self.max_chars = max_chars

    def extract(self, html: str, url: str) -> Tuple[str, str]:
        """
        Extract main content from HTML using trafilatura.

        Args:
            html: Raw HTML content
            url: Source URL (for context)

        Returns:
            Tuple of (title, content)
        """
        # Extract main text content - use simple defaults
        # (favor_precision + deduplicate + custom config can cause empty results)
        content = trafilatura.extract(
            html,
            url=url,
            include_comments=False,
            include_tables=True,
        )

        # Extract metadata for title
        title = ""
        metadata = trafilatura.extract_metadata(html)
        if metadata and metadata.title:
            title = metadata.title

        # Clean and truncate
        content = self._clean_text(content or "")
        content = self._truncate(content)
        title = self._clean_text(title)[:200]

        return title, content

    def _clean_text(self, text: str) -> str:
        """Clean and normalize text."""
        if not text:
            return ""
        # Collapse whitespace
        text = re.sub(r"\s+", " ", text)
        # Remove excessive punctuation
        text = re.sub(r"[.]{3,}", "...", text)
        text = re.sub(r"[-]{3,}", "---", text)
        return text.strip()

    def _truncate(self, text: str) -> str:
        """Truncate text to max chars, preserving word boundaries."""
        if len(text) <= self.max_chars:
            return text
        truncated = text[: self.max_chars]
        # Try to break at word boundary
        if " " in truncated:
            truncated = truncated.rsplit(" ", 1)[0]
        return truncated.strip()


class SerperWebSearchBackend:
    """Serper-backed live web search plus lazy page fetching."""

    def __init__(
        self,
        api_key: Optional[str],
        api_url: str,
        gl: str,
        hl: str,
        timeout_s: float,
        max_read_chars: int,
        user_agent: str,
    ) -> None:
        self.api_key = api_key
        self.api_url = api_url
        self.gl = gl
        self.hl = hl
        self.timeout_s = timeout_s
        self.max_read_chars = max_read_chars
        self.user_agent = user_agent
        self._summary_cache: Dict[str, Dict[str, Any]] = {}
        self._chunk_cache: Dict[str, Chunk] = {}
        self._extractor = ContentExtractor(max_chars=max_read_chars)
        self.log_file = os.getenv("LOG_FILE", "").strip()

    def _log_retrieval_event(self, event: str, payload: Dict[str, Any]) -> None:
        append_log_record(
            self.log_file,
            {
                "ts": datetime.now(timezone.utc).isoformat(),
                "event": event,
                "backend": "serper",
                **payload,
            },
        )

    @staticmethod
    def _estimate_tokens(text: str) -> int:
        return max(1, len(text) // 4)

    @staticmethod
    def _stable_document_id(url: str) -> str:
        digest = hashlib.sha256(url.encode("utf-8")).hexdigest()[:16]
        return f"webdoc_{digest}"

    @classmethod
    def _stable_chunk_id(cls, url: str) -> str:
        return f"{cls._stable_document_id(url)}_chunk_0"

    @staticmethod
    def _collapse_text(text: str) -> str:
        text = re.sub(r"\s+", " ", text)
        return text.strip()

    def _search_request(self, query: str, top_k: int) -> Dict[str, Any]:
        if not self.api_key:
            raise RuntimeError(
                "SEARCH_BACKEND=serper requires SERPER_API_KEY to be set"
            )

        payload = {
            "q": query,
            "num": top_k,
            "gl": self.gl,
            "hl": self.hl,
        }
        data = json.dumps(payload).encode("utf-8")
        req = request.Request(
            self.api_url,
            data=data,
            headers={
                "Content-Type": "application/json",
                "X-API-KEY": self.api_key,
                "User-Agent": self.user_agent,
            },
            method="POST",
        )

        try:
            with request.urlopen(req, timeout=self.timeout_s) as response:
                charset = response.headers.get_content_charset() or "utf-8"
                body = response.read().decode(charset, errors="replace")
        except error.HTTPError as exc:
            body = exc.read().decode("utf-8", errors="replace")
            raise RuntimeError(
                f"Serper request failed with HTTP {exc.code}: {body}"
            ) from exc
        except error.URLError as exc:
            raise RuntimeError(f"Serper request failed: {exc.reason}") from exc

        try:
            payload = json.loads(body)
        except json.JSONDecodeError as exc:
            raise RuntimeError("Serper response was not valid JSON") from exc

        self._log_retrieval_event(
            "serper_response",
            {
                "query": query,
                "top_k": top_k,
                "request": {
                    "q": query,
                    "num": top_k,
                    "gl": self.gl,
                    "hl": self.hl,
                },
                "response": payload,
            },
        )
        return payload

    def search(
        self,
        query: str,
        top_k: int,
        exclude_ids: Optional[Set[str]],
        snippet_length: int,
    ) -> List[ChunkSummary]:
        exclude_ids = exclude_ids or set()
        payload = self._search_request(query, top_k)

        summaries: List[ChunkSummary] = []
        for result in payload.get("organic", []):
            url = str(result.get("link") or "").strip()
            if not url:
                continue

            chunk_id = self._stable_chunk_id(url)
            if chunk_id in exclude_ids:
                continue

            title = self._collapse_text(str(result.get("title") or url))
            snippet = self._collapse_text(str(result.get("snippet") or title))
            if len(snippet) > snippet_length:
                snippet = snippet[:snippet_length].rsplit(" ", 1)[0].strip() + "..."

            position = int(result.get("position") or (len(summaries) + 1))
            score = 1.0 / max(position, 1)
            document_id = self._stable_document_id(url)

            self._summary_cache[chunk_id] = {
                "chunk_id": chunk_id,
                "document_id": document_id,
                "url": url,
                "title": title,
                "snippet": snippet,
                "query": query,
                "position": position,
                "score": score,
                "source": result.get("source"),
                "date": result.get("date"),
            }

            # Estimate token count from snippet (actual count computed on read)
            estimated_tokens = self._estimate_tokens(snippet)
            summaries.append(
                ChunkSummary(
                    chunk_id=chunk_id,
                    document_id=document_id,
                    title=title,
                    snippet=snippet,
                    score=score,
                    token_count=estimated_tokens,
                )
            )
            if len(summaries) >= top_k:
                break

        self._log_retrieval_event(
            "serper_normalized_results",
            {
                "query": query,
                "top_k": top_k,
                "exclude_ids": sorted(exclude_ids),
                "results": [summary.model_dump() for summary in summaries],
            },
        )
        return summaries

    def _fetch_page(self, url: str) -> str:
        """Fetch raw HTML from URL."""
        req = request.Request(
            url,
            headers={
                "User-Agent": self.user_agent,
                "Accept": "text/html,application/xhtml+xml,text/plain;q=0.9,*/*;q=0.8",
                "Accept-Language": "en-US,en;q=0.9",
            },
        )

        try:
            with request.urlopen(req, timeout=self.timeout_s) as response:
                charset = response.headers.get_content_charset() or "utf-8"
                # Read enough HTML for trafilatura to find main content
                # (Wikipedia pages need ~100KB+ to include article body)
                raw = response.read(150000)
                return raw.decode(charset, errors="replace")
        except error.HTTPError as exc:
            raise RuntimeError(
                f"Page fetch failed with HTTP {exc.code}: {url}"
            ) from exc
        except error.URLError as exc:
            raise RuntimeError(f"Page fetch failed for {url}: {exc.reason}") from exc

    def get_chunk(self, chunk_id: str) -> Optional[Chunk]:
        cached = self._chunk_cache.get(chunk_id)
        if cached is not None:
            return cached

        summary = self._summary_cache.get(chunk_id)
        if summary is None:
            return None

        url = summary["url"]

        # Fetch and extract full page content
        html = self._fetch_page(url)
        title, content = self._extractor.extract(html, url)

        # Use summary title if extraction didn't get one
        if not title:
            title = summary["title"]

        chunk = Chunk(
            chunk_id=chunk_id,
            document_id=summary["document_id"],
            content=content,
            token_count=self._estimate_tokens(content),
            metadata={
                "title": title,
                "url": url,
                "source": summary.get("source"),
                "date": summary.get("date"),
                "query": summary["query"],
                "position": summary["position"],
                "backend": "serper",
            },
            retrieval_score=summary["score"],
        )
        self._chunk_cache[chunk_id] = chunk

        self._log_retrieval_event(
            "serper_read_materialized",
            {
                "chunk_id": chunk_id,
                "document_id": summary["document_id"],
                "url": url,
                "query": summary["query"],
                "position": summary["position"],
                "title": title,
                "content": content[:2000] if content else "",  # Truncate for log
                "content_length": len(content) if content else 0,
                "metadata": chunk.metadata,
            },
        )
        return chunk
