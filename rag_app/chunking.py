from __future__ import annotations

import re

from .models import DocumentChunk, ParsedDocument


HEADING_RE = re.compile(r"^(#{1,6})\s+(.+?)\s*$")
WORD_RE = re.compile(r"[A-Za-z0-9_]+|[\u4e00-\u9fff]|[^\s]", re.UNICODE)


def tokenize(text: str) -> list[str]:
    return [item.lower() for item in WORD_RE.findall(text) if item.strip()]


def _sections(text: str) -> list[tuple[str, str, int, int]]:
    lines = text.splitlines()
    current = "正文"
    body: list[str] = []
    start_line = 1
    result: list[tuple[str, str, int, int]] = []

    def flush(end_line: int) -> None:
        nonlocal body
        value = "\n".join(body).strip()
        if value:
            result.append((current, value, start_line, end_line))
        body = []

    for number, line in enumerate(lines, 1):
        match = HEADING_RE.match(line.strip())
        if match:
            flush(number - 1)
            current = match.group(2)
            start_line = number
        else:
            if not body:
                start_line = number
            body.append(line)
    flush(len(lines))
    return result


def chunk_document(document: ParsedDocument, max_tokens: int = 180, overlap: int = 35) -> list[DocumentChunk]:
    if max_tokens <= overlap:
        raise ValueError("max_tokens must be greater than overlap")
    chunks: list[DocumentChunk] = []
    for page in document.pages:
        page_ordinal = 0
        for section, text, start_line, end_line in _sections(page.text):
            matches = [match for match in WORD_RE.finditer(text) if match.group().strip()]
            if not matches:
                continue
            start = 0
            while start < len(matches):
                stop = min(len(matches), start + max_tokens)
                # Use token spans only to choose boundaries; preserve the original text inside the chunk.
                piece = text[matches[start].start() : matches[stop - 1].end()].strip()
                chunks.append(
                    DocumentChunk(
                        chunk_id=f"{document.doc_id}-p{page.number}-c{page_ordinal}",
                        doc_id=document.doc_id,
                        title=document.title,
                        source_path=document.source_path,
                        file_type=document.file_type,
                        page=page.number,
                        section=section,
                        text=piece,
                        start_line=start_line,
                        end_line=end_line,
                    )
                )
                page_ordinal += 1
                if stop == len(matches):
                    break
                start = stop - overlap
    return chunks


def chunk_documents(documents: list[ParsedDocument], max_tokens: int = 180, overlap: int = 35) -> list[DocumentChunk]:
    result: list[DocumentChunk] = []
    for document in documents:
        result.extend(chunk_document(document, max_tokens=max_tokens, overlap=overlap))
    if not result:
        raise ValueError("Documents contain no extractable text")
    return result
