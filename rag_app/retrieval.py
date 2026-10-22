from __future__ import annotations

import json
import math
import re
from collections import Counter, defaultdict
from pathlib import Path

from .chunking import tokenize
from .models import DocumentChunk, RetrievedChunk


class HybridRetriever:
    """BM25 + TF-IDF retrieval with reciprocal-rank fusion."""

    def __init__(self, chunks: list[DocumentChunk] | None = None) -> None:
        self.chunks = chunks or []
        self._terms: list[list[str]] = []
        self._idf: dict[str, float] = {}
        self._avg_len = 0.0
        self._df: Counter[str] = Counter()
        self._vectors: list[dict[str, float]] = []
        if chunks:
            self.build(chunks)

    def build(self, chunks: list[DocumentChunk]) -> None:
        self.chunks = list(chunks)
        self._terms = [tokenize(chunk.text) for chunk in self.chunks]
        self._df = Counter(term for terms in self._terms for term in set(terms))
        n = max(1, len(self.chunks))
        self._idf = {term: math.log((n + 1) / (count + 1)) + 1 for term, count in self._df.items()}
        self._avg_len = sum(len(terms) for terms in self._terms) / n
        self._vectors = []
        for terms in self._terms:
            counts = Counter(terms)
            weights = {term: (1 + math.log(count)) * self._idf.get(term, 1.0) for term, count in counts.items()}
            norm = math.sqrt(sum(value * value for value in weights.values())) or 1.0
            self._vectors.append({term: value / norm for term, value in weights.items()})

    def _bm25(self, query_terms: list[str], index: int, k1: float = 1.2, b: float = 0.75) -> float:
        terms = self._terms[index]
        counts = Counter(terms)
        length = len(terms) or 1
        score = 0.0
        for term in query_terms:
            if term not in counts:
                continue
            idf = math.log(1 + (len(self.chunks) - self._df.get(term, 0) + 0.5) / (self._df.get(term, 0) + 0.5))
            tf = counts[term]
            score += idf * (tf * (k1 + 1)) / (tf + k1 * (1 - b + b * length / (self._avg_len or 1)))
        return score

    def _tfidf(self, query_terms: list[str], index: int) -> float:
        counts = Counter(query_terms)
        weights = {term: (1 + math.log(count)) * self._idf.get(term, 0.0) for term, count in counts.items()}
        norm = math.sqrt(sum(value * value for value in weights.values())) or 1.0
        query = {term: value / norm for term, value in weights.items()}
        return sum(query.get(term, 0.0) * self._vectors[index].get(term, 0.0) for term in query)

    @staticmethod
    def _rank(scores: list[float]) -> dict[int, int]:
        ordered = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
        return {index: rank for rank, index in enumerate(ordered, 1) if scores[index] > 0}

    def search(self, query: str, top_k: int = 5, strategy: str = "hybrid") -> list[RetrievedChunk]:
        """Search with an explicit strategy for reproducible ablation tests."""
        if not self.chunks:
            return []
        if strategy not in {"hybrid", "bm25", "tfidf"}:
            raise ValueError("strategy must be hybrid, bm25, or tfidf")
        query_terms = tokenize(query)
        bm25_scores = [self._bm25(query_terms, i) for i in range(len(self.chunks))]
        tfidf_scores = [self._tfidf(query_terms, i) for i in range(len(self.chunks))]
        bm25_rank = self._rank(bm25_scores)
        tfidf_rank = self._rank(tfidf_scores)
        limit = max(1, top_k)
        if strategy == "bm25":
            ordered = sorted(range(len(self.chunks)), key=lambda i: bm25_scores[i], reverse=True)
            return [RetrievedChunk(self.chunks[i], bm25_scores[i], bm25_scores[i], tfidf_scores[i], bm25_rank.get(i), tfidf_rank.get(i)) for i in ordered[:limit] if bm25_scores[i] > 0]
        if strategy == "tfidf":
            ordered = sorted(range(len(self.chunks)), key=lambda i: tfidf_scores[i], reverse=True)
            return [RetrievedChunk(self.chunks[i], tfidf_scores[i], bm25_scores[i], tfidf_scores[i], bm25_rank.get(i), tfidf_rank.get(i)) for i in ordered[:limit] if tfidf_scores[i] > 0]
        candidates = set(bm25_rank) | set(tfidf_rank)
        fused: list[RetrievedChunk] = []
        for index in candidates:
            r1 = bm25_rank.get(index)
            r2 = tfidf_rank.get(index)
            # RRF keeps scores comparable even when BM25 and cosine have different scales.
            score = (0.65 / (60 + r1) if r1 else 0.0) + (0.35 / (60 + r2) if r2 else 0.0)
            fused.append(RetrievedChunk(self.chunks[index], score, bm25_scores[index], tfidf_scores[index], r1, r2))
        fused.sort(key=lambda item: item.score, reverse=True)
        return fused[:limit]

    def save(self, path: str | Path) -> None:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps({"version": 1, "chunks": [chunk.to_dict() for chunk in self.chunks]}, ensure_ascii=False, indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> "HybridRetriever":
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls([DocumentChunk.from_dict(value) for value in payload.get("chunks", [])])
