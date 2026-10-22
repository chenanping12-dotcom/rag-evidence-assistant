from __future__ import annotations

import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .chunking import chunk_documents
from .generation import Generator, build_generator
from .models import RetrievedChunk
from .parsers import parse_directory
from .retrieval import HybridRetriever


class RAGPipeline:
    def __init__(self, index_path: str | Path, generator: Generator | None = None) -> None:
        self.index_path = Path(index_path)
        self.generator = generator or build_generator()
        self.retriever: HybridRetriever | None = None

    def load(self) -> None:
        if not self.index_path.is_file():
            raise FileNotFoundError(f"Index not found: {self.index_path}. Run ingest first.")
        self.retriever = HybridRetriever.load(self.index_path)

    def ingest(self, source_dir: str | Path, max_tokens: int = 180, overlap: int = 35) -> dict[str, Any]:
        documents = parse_directory(source_dir)
        chunks = chunk_documents(documents, max_tokens=max_tokens, overlap=overlap)
        self.retriever = HybridRetriever(chunks)
        self.retriever.save(self.index_path)
        return {
            "documents": len(documents),
            "chunks": len(chunks),
            "source_dir": str(Path(source_dir).resolve()),
            "index": str(self.index_path.resolve()),
            "chunk_tokens": max_tokens,
            "overlap": overlap,
        }

    def _ensure_loaded(self) -> HybridRetriever:
        if self.retriever is None:
            self.load()
        assert self.retriever is not None
        return self.retriever

    @staticmethod
    def _context(results: list[RetrievedChunk]) -> tuple[str, list[dict[str, Any]]]:
        citations: list[dict[str, Any]] = []
        blocks: list[str] = []
        for index, result in enumerate(results, 1):
            marker = f"[C{index}]"
            chunk = result.chunk
            citations.append({
                "id": marker,
                "title": chunk.title,
                "source": chunk.source_path,
                "page": chunk.page,
                "section": chunk.section,
                "lines": [chunk.start_line, chunk.end_line],
                "snippet": chunk.text[:320],
                "retrieval_score": round(result.score, 6),
            })
            blocks.append(f"{marker} {chunk.title} / {chunk.section}\n{chunk.text}")
        return "\n\n".join(blocks), citations

    def query(self, query: str, top_k: int = 5) -> dict[str, Any]:
        if not query.strip():
            raise ValueError("query must not be empty")
        retriever = self._ensure_loaded()
        results = retriever.search(query, top_k=top_k)
        context, citations = self._context(results)
        markers = [item["id"] for item in citations]
        answer = self.generator.generate(query, context, markers)
        mentioned = sorted(set(re.findall(r"\[C\d+\]", answer)))
        valid_markers = set(markers)
        invalid_markers = [marker for marker in mentioned if marker not in valid_markers]
        if invalid_markers:
            invalid_set = set(invalid_markers)
            answer = re.sub(
                r"\[C\d+\]",
                lambda match: "[未验证引用]" if match.group(0) in invalid_set else match.group(0),
                answer,
            )
        return {
            "query": query,
            "answer": answer,
            "citations": citations,
            "used_citations": [marker for marker in mentioned if marker in valid_markers],
            "citation_valid": not invalid_markers,
            "invalid_citations": invalid_markers,
            "retrieval": [result.to_dict() for result in results],
            "generator": self.generator.__class__.__name__,
        }

    def evaluate(self, questions_path: str | Path, strategy: str = "hybrid") -> dict[str, Any]:
        retriever = self._ensure_loaded()
        questions = json.loads(Path(questions_path).read_text(encoding="utf-8"))
        rows: list[dict[str, Any]] = []
        for item in questions:
            results = retriever.search(item["question"], top_k=max(5, int(item.get("k", 5))), strategy=strategy)
            expected = set(item["expected_sources"])
            sources = [Path(result.chunk.source_path).name for result in results]
            relevant = [source in expected for source in sources]
            hit_rank = next((index + 1 for index, value in enumerate(relevant) if value), None)
            k = int(item.get("k", 5))
            rows.append({
                "id": item["id"],
                "question": item["question"],
                "hit_at_k": bool(hit_rank and hit_rank <= k),
                "reciprocal_rank": round(1 / hit_rank, 6) if hit_rank and hit_rank <= k else 0.0,
                "precision_at_k": round(sum(relevant[:k]) / k, 6),
                "expected_sources": sorted(expected),
                "retrieved_sources": sources[:k],
                "rank": hit_rank,
            })
        count = len(rows) or 1
        report = {
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "index": str(self.index_path.resolve()),
            "strategy": strategy,
            "questions": len(rows),
            "metrics": {
                "recall_at_5": round(sum(row["hit_at_k"] for row in rows) / count, 6),
                "mrr_at_5": round(sum(row["reciprocal_rank"] for row in rows) / count, 6),
                "precision_at_5": round(sum(row["precision_at_k"] for row in rows) / count, 6),
            },
            "note": "Demo corpus evaluation; metrics measure source retrieval, not general answer quality.",
            "cases": rows,
        }
        return report
