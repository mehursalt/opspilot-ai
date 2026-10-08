"""Local keyword retrieval for 运维 knowledge documents.

The vector store is good at semantic similarity, while operations questions often
contain exact signals such as alert names, error codes, service names and file
names. This service adds a lightweight BM25-like lexical channel without
introducing Elasticsearch as a new dependency.
"""

from __future__ import annotations

import math
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from loguru import logger

from app.config import config


@dataclass
class KeywordSearchHit:
    query: str
    content: str
    source: str
    score: float
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class KeywordSearchService:
    """File-system backed keyword search over local knowledge documents."""

    SUPPORTED_SUFFIXES = {".md", ".markdown", ".txt"}

    def search(self, query: str, top_k: int | None = None) -> list[KeywordSearchHit]:
        if not config.enable_keyword_retrieval:
            return []

        top_k = top_k or config.keyword_search_top_k
        query_terms = self._terms(query)
        if not query_terms:
            return []

        hits: list[KeywordSearchHit] = []
        for file_path in self._iter_files():
            try:
                text = file_path.read_text(encoding="utf-8", errors="ignore")
            except Exception as e:
                logger.warning("关键词检索读取文件失败 {}: {}", file_path, e)
                continue

            for idx, chunk in enumerate(self._split_text(text)):
                score = self._score(query, query_terms, chunk, file_path)
                if score <= 0:
                    continue
                hits.append(
                    KeywordSearchHit(
                        query=query,
                        content=chunk.strip(),
                        source=str(file_path),
                        score=score,
                        metadata={
                            "_source": str(file_path),
                            "_file_name": file_path.name,
                            "chunk_index": idx,
                            "retrieval_score": score,
                        },
                    )
                )

        hits.sort(key=lambda hit: hit.score, reverse=True)
        return hits[:top_k]

    def _iter_files(self) -> list[Path]:
        files: list[Path] = []
        for raw_dir in config.keyword_search_dirs.split(","):
            raw_dir = raw_dir.strip()
            if not raw_dir:
                continue
            root = Path(raw_dir)
            if not root.exists():
                continue
            for path in root.rglob("*"):
                if path.is_file() and path.suffix.lower() in self.SUPPORTED_SUFFIXES:
                    files.append(path)
        return files

    def _split_text(self, text: str) -> list[str]:
        blocks = re.split(r"(?m)(?=^#{1,3}\s+)|\n\s*\n", text)
        chunks: list[str] = []
        buffer = ""
        max_chars = max(300, config.chunk_max_size * 2)
        for block in blocks:
            block = block.strip()
            if not block:
                continue
            if len(buffer) + len(block) <= max_chars:
                buffer = f"{buffer}\n\n{block}".strip()
            else:
                if buffer:
                    chunks.append(buffer)
                buffer = block
        if buffer:
            chunks.append(buffer)
        return chunks

    def _score(self, query: str, query_terms: list[str], chunk: str, file_path: Path) -> float:
        haystack = f"{file_path.name}\n{chunk}".lower()
        chunk_terms = self._terms(chunk)
        if not chunk_terms:
            return 0.0

        term_counts: dict[str, int] = {}
        for term in chunk_terms:
            term_counts[term] = term_counts.get(term, 0) + 1

        score = 0.0
        for term in query_terms:
            count = term_counts.get(term, 0)
            if count:
                score += 1.0 + math.log1p(count)
            if len(term) >= 2 and term in haystack:
                score += 1.5

        normalized_query = query.lower().strip()
        if len(normalized_query) >= 4 and normalized_query in haystack:
            score += 4.0

        if any(term in file_path.name.lower() for term in query_terms if len(term) >= 2):
            score += 2.0

        return round(score, 4)

    def _terms(self, text: str) -> list[str]:
        lowered = text.lower()
        ascii_terms = re.findall(r"[a-zA-Z][a-zA-Z0-9_\-]{1,}|[0-9]{2,}", lowered)
        chinese_terms = re.findall(r"[\u4e00-\u9fff]{2,}", lowered)
        char_bigrams: list[str] = []
        for phrase in chinese_terms:
            if len(phrase) <= 4:
                char_bigrams.append(phrase)
            for idx in range(len(phrase) - 1):
                char_bigrams.append(phrase[idx : idx + 2])

        terms = ascii_terms + chinese_terms + char_bigrams
        seen: set[str] = set()
        result: list[str] = []
        for term in terms:
            term = term.strip()
            if not term or term in seen:
                continue
            seen.add(term)
            result.append(term)
        return result


keyword_search_service = KeywordSearchService()
