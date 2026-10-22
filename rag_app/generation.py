from __future__ import annotations

import json
import os
import re
import urllib.error
import urllib.request
from typing import Sequence

from .models import RetrievedChunk


class Generator:
    def generate(self, query: str, context: str, citations: Sequence[str]) -> str:
        raise NotImplementedError


class ExtractiveGenerator(Generator):
    """Offline fallback: returns evidence sentences instead of inventing facts."""

    def generate(self, query: str, context: str, citations: Sequence[str]) -> str:
        blocks = [block.strip() for block in context.split("\n\n") if block.strip()]
        selected = blocks[:3]
        if not selected:
            return "未检索到可引用的资料。"
        lines = []
        for index, block in enumerate(selected):
            marker = citations[index] if index < len(citations) else "[未标注来源]"
            clean = re.sub(r"\s+", " ", block)
            if len(clean) > 220:
                clean = clean[:220].rstrip() + "..."
            lines.append(f"- {clean} {marker}")
        return "基于检索到的文档片段：\n" + "\n".join(lines)


class OpenAICompatibleGenerator(Generator):
    def __init__(self, base_url: str, api_key: str, model: str, timeout: float = 45.0) -> None:
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.model = model
        self.timeout = timeout

    def generate(self, query: str, context: str, citations: Sequence[str]) -> str:
        citation_rule = ", ".join(citations) or "无"
        prompt = (
            "你是企业知识库助手。只能根据 CONTEXT 回答，不确定时明确说资料不足。"
            "每个可验证结论后必须带 [C#] 引用标记，不能编造引用。\n\n"
            f"QUESTION:\n{query}\n\nCONTEXT:\n{context}\n\nVALID CITATIONS: {citation_rule}"
        )
        body = json.dumps({"model": self.model, "temperature": 0, "messages": [
            {"role": "system", "content": "回答简洁，优先列出结论和限制。"},
            {"role": "user", "content": prompt},
        ]}).encode("utf-8")
        request = urllib.request.Request(
            f"{self.base_url}/chat/completions",
            data=body,
            headers={"Content-Type": "application/json", "Authorization": f"Bearer {self.api_key}"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                payload = json.loads(response.read().decode("utf-8"))
        except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as exc:
            raise RuntimeError(f"LLM request failed: {exc}") from exc
        return str(payload["choices"][0]["message"]["content"]).strip()


def build_generator() -> Generator:
    base_url = os.getenv("RAG_LLM_BASE_URL", "").strip()
    api_key = os.getenv("RAG_LLM_API_KEY", "").strip()
    model = os.getenv("RAG_LLM_MODEL", "").strip()
    if base_url and api_key and model:
        return OpenAICompatibleGenerator(base_url, api_key, model)
    return ExtractiveGenerator()
