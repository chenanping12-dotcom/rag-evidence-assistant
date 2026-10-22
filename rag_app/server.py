from __future__ import annotations

import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlparse

from .pipeline import RAGPipeline


class Handler(BaseHTTPRequestHandler):
    pipeline: RAGPipeline

    def _write(self, status: int, payload: object) -> None:
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:
        path = urlparse(self.path).path
        if path == "/health":
            self._write(200, {"status": "ok", "index": str(self.pipeline.index_path), "generator": self.pipeline.generator.__class__.__name__})
            return
        self._write(404, {"error": "not_found"})

    def do_POST(self) -> None:
        if urlparse(self.path).path != "/query":
            self._write(404, {"error": "not_found"})
            return
        try:
            length = int(self.headers.get("Content-Length", "0"))
            payload = json.loads(self.rfile.read(length).decode("utf-8"))
            result = self.pipeline.query(str(payload.get("query", "")), int(payload.get("top_k", 5)))
        except (ValueError, KeyError, json.JSONDecodeError, FileNotFoundError) as exc:
            self._write(400, {"error": str(exc)})
            return
        self._write(200, result)


def serve(index: str, host: str = "127.0.0.1", port: int = 8090) -> None:
    pipeline = RAGPipeline(index)
    Handler.pipeline = pipeline
    server = ThreadingHTTPServer((host, port), Handler)
    print(f"RAG API listening on http://{host}:{port}")
    server.serve_forever()
