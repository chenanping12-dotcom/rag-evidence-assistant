from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any


@dataclass(frozen=True)
class Page:
    number: int
    text: str


@dataclass(frozen=True)
class ParsedDocument:
    doc_id: str
    title: str
    source_path: str
    file_type: str
    pages: list[Page]


@dataclass(frozen=True)
class DocumentChunk:
    chunk_id: str
    doc_id: str
    title: str
    source_path: str
    file_type: str
    page: int
    section: str
    text: str
    start_line: int
    end_line: int

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "DocumentChunk":
        return cls(**value)


@dataclass(frozen=True)
class RetrievedChunk:
    chunk: DocumentChunk
    score: float
    bm25_score: float
    tfidf_score: float
    rank_bm25: int | None
    rank_tfidf: int | None

    def to_dict(self) -> dict[str, Any]:
        return {
            "chunk": self.chunk.to_dict(),
            "score": round(self.score, 6),
            "bm25_score": round(self.bm25_score, 6),
            "tfidf_score": round(self.tfidf_score, 6),
            "rank_bm25": self.rank_bm25,
            "rank_tfidf": self.rank_tfidf,
        }
