$ErrorActionPreference = "Stop"
$projectRoot = Split-Path -Parent $PSScriptRoot
Set-Location $projectRoot
$env:PYTHONPATH = $projectRoot

py -3.13 -m rag_app.cli --index storage/index.json ingest data/docs
py -3.13 -m rag_app.cli --index storage/index.json query "哪些 Provider 错误可以重试？"
py -3.13 -m rag_app.cli --index storage/index.json evaluate data/eval/questions.json --out reports/evaluation.json
