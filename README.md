# Evidence RAG Assistant

一个面向企业知识库问答的可运行 RAG 项目。项目参考了 GitHub 上的 LlamaIndex 文档处理与检索分层思路，但没有复制上游代码，核心链路由本项目独立实现。

## 能力范围

- 文档解析：Markdown、TXT、PDF、DOCX；保留标题、页码、行号和源文件。
- 分块：标题感知分块，默认 180 词元、35 词元重叠。
- 检索：BM25 + TF-IDF 余弦相似度，再用 RRF 融合排序。
- 回答：有 OpenAI 兼容接口配置时调用 LLM；没有密钥时使用抽取式回答，不编造事实。
- 引用溯源：答案使用 `[C1]` 标记，返回来源文件、章节、页码/行号、片段和检索分数。
- 评测：固定问题集输出 Recall@5、MRR@5、Precision@5，并保留逐题命中排名。
- 服务接口：标准库 HTTP 服务提供 `/health` 和 `/query`，不依赖 Docker。

## 为什么选这个方向

参考项目 `run-llama/llama_index` 在本次查阅时约 5.25 万 Star，官方定位覆盖文档解析、数据结构化、检索和 LLM 应用。RAGFlow 关注度更高但更偏完整平台，通常需要 Docker 和多服务依赖；本项目选择轻量、可解释、能在 Windows 本地复现的实现作为实训版本。

## 快速运行

在项目根目录执行：

```powershell
$env:PYTHONPATH = (Get-Location).Path
py -3.13 -m rag_app.cli --index storage/index.json ingest data/docs
py -3.13 -m rag_app.cli --index storage/index.json query "BM25 和 TF-IDF 如何融合？"
py -3.13 -m rag_app.cli --index storage/index.json evaluate data/eval/questions.json --out reports/evaluation.json
```

如果需要接入 OpenAI 兼容服务：

```powershell
$env:RAG_LLM_BASE_URL = "https://your-endpoint/v1"
$env:RAG_LLM_API_KEY = "your-key"
$env:RAG_LLM_MODEL = "your-model"
```

没有配置时不会请求外部网络，回答模式会显示为 `ExtractiveGenerator`。

## REST 服务

```powershell
$env:PYTHONPATH = (Get-Location).Path
py -3.13 -c "from rag_app.server import serve; serve('storage/index.json')"
```

```powershell
Invoke-RestMethod http://127.0.0.1:8090/health
Invoke-RestMethod -Method Post -Uri http://127.0.0.1:8090/query -ContentType 'application/json' -Body '{"query":"预算超限后怎么办？","top_k":5}'
```

## 真实评测边界

仓库中的评测集是 12 份演示知识文档（40 个分块）和 8 个固定问题，用来验证检索、引用和评测代码，不代表生产准确率。

三臂消融（`evaluate --strategy {hybrid,bm25,tfidf}`，`reports/` 下报告均可由当前代码原地复现）：

| 检索策略 | Recall@5 | MRR@5 | Precision@5 |
| --- | --- | --- | --- |
| BM25 | 1.000 | 0.875 | 0.375 |
| TF-IDF | 1.000 | 0.875 | 0.375 |
| RRF 混合（权重 0.65/0.35） | 1.000 | 0.875 | 0.350 |

**必须如实说明**：在这个 8 题演示集上，混合检索的 MRR 并没有超过任何单路召回，Precision@5 还略低于单路（0.350 vs 0.375）。原因是语料只有 40 个分块、8 道题在单路召回里都已命中，RRF 融合没有发挥空间，差异落在 Precision 的噪声上。因此本项目不主张"混合检索带来提升"；要验证融合是否有效，需要先扩到有区分度的语料和人工标注集。

接入真实企业文档后，应补充权限过滤、文档版本、中文 embedding、reranker、人工标注的引用正确性和端到端延迟。

## 目录

```text
rag_app/                 解析、分块、混合检索、生成、评测和 HTTP API
data/docs/               可公开演示的知识库文档
data/eval/               固定评测问题
storage/                 本地持久化索引
reports/                 评测报告
tests/                   unittest 行为测试
```
