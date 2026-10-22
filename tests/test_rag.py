from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from rag_app.pipeline import RAGPipeline


class HallucinatingGenerator:
    def generate(self, query: str, context: str, citations: list[str]) -> str:
        return "结论 [C999]"


class RAGPipelineTests(unittest.TestCase):
    def test_ingest_retrieve_and_emit_citation(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            docs = root / "docs"
            docs.mkdir()
            (docs / "guide.md").write_text("# Retrieval\n\nBM25 matches exact terms and numbers.\n", encoding="utf-8")
            index = root / "index.json"
            pipeline = RAGPipeline(index)
            stats = pipeline.ingest(docs)
            self.assertEqual(stats["documents"], 1)
            self.assertEqual(len({chunk.chunk_id for chunk in pipeline.retriever.chunks}), stats["chunks"])
            result = pipeline.query("How does BM25 match exact terms?")
            self.assertIn("[C1]", result["answer"])
            self.assertEqual(result["citations"][0]["title"], "Retrieval")

    def test_metrics_are_written_by_caller(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            docs = root / "docs"
            docs.mkdir()
            (docs / "guide.md").write_text("# Guide\n\nCitations need a source.\n", encoding="utf-8")
            questions = root / "questions.json"
            questions.write_text(json.dumps([{"id": "q", "question": "Where are citations?", "expected_sources": ["guide.md"], "k": 5}], ensure_ascii=False), encoding="utf-8")
            pipeline = RAGPipeline(root / "index.json")
            pipeline.ingest(docs)
            report = pipeline.evaluate(questions)
            self.assertEqual(report["questions"], 1)
            self.assertEqual(report["metrics"]["recall_at_5"], 1.0)

    def test_invalid_citation_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            docs = root / "docs"
            docs.mkdir()
            (docs / "guide.md").write_text("# Guide\n\nCitations need a source.\n", encoding="utf-8")
            pipeline = RAGPipeline(root / "index.json", generator=HallucinatingGenerator())
            pipeline.ingest(docs)
            result = pipeline.query("Where are citations?")
            self.assertFalse(result["citation_valid"])
            self.assertEqual(result["invalid_citations"], ["[C999]"])
            self.assertIn("[未验证引用]", result["answer"])


if __name__ == "__main__":
    unittest.main()
