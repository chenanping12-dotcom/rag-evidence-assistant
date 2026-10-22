from __future__ import annotations

import argparse
import json
from pathlib import Path

from .pipeline import RAGPipeline


def _json(value: object) -> None:
    print(json.dumps(value, ensure_ascii=False, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(description="Evidence-first hybrid RAG assistant")
    parser.add_argument("--index", default="storage/index.json")
    sub = parser.add_subparsers(dest="command", required=True)

    ingest = sub.add_parser("ingest", help="parse documents and build the index")
    ingest.add_argument("source_dir")
    ingest.add_argument("--max-tokens", type=int, default=180)
    ingest.add_argument("--overlap", type=int, default=35)

    query = sub.add_parser("query", help="retrieve evidence and answer a question")
    query.add_argument("question")
    query.add_argument("--top-k", type=int, default=5)

    evaluate = sub.add_parser("evaluate", help="run retrieval evaluation")
    evaluate.add_argument("questions", default="data/eval/questions.json", nargs="?")
    evaluate.add_argument("--out", default="reports/evaluation.json")
    evaluate.add_argument("--strategy", choices=["hybrid", "bm25", "tfidf"], default="hybrid")

    args = parser.parse_args()
    pipeline = RAGPipeline(args.index)
    if args.command == "ingest":
        _json(pipeline.ingest(args.source_dir, args.max_tokens, args.overlap))
    elif args.command == "query":
        _json(pipeline.query(args.question, args.top_k))
    elif args.command == "evaluate":
        report = pipeline.evaluate(args.questions, strategy=args.strategy)
        target = Path(args.out)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        _json(report)


if __name__ == "__main__":
    main()
