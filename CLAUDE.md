# RAG Agent — 知识库问答系统

## 技术栈
- **Python** 3.10+
- **LlamaIndex** — RAG 全流程（加载/分块/索引/检索/合成）
- **ChromaDB** — 向量存储（本地嵌入）
- **sentence-transformers** — 本地 embedding 模型
- **langchain-openai / openai SDK** — LLM 调用（兼容 DeepSeek API）
- **FastAPI** — HTTP 服务
- **SQLite** — 日志与反馈存储

## 目录结构
```
rag-agent/
├── main.py               # FastAPI 入口，路由定义
├── ingest.py             # 文档加载、分块、入库
├── retriever.py          # 检索逻辑（向量检索 + 可选重排）
├── generator.py          # 答案生成（prompt 模板 + LLM 调用）
├── data/documents/       # 原始文档（MD/TXT/PDF）
├── eval/                 # 评测脚本与评测集
├── docs/                 # 项目文档
├── logs/                 # 运行日志
└── CLAUDE.md             # 本文件
```

## 代码风格
- 中文注释，描述"为什么"而非"是什么"
- 函数签名使用类型标注
- 一次只改一个文件，改动前先读代码
- 命名清晰优先，避免过度抽象
- 错误处理从第一版就做，不留 TODO
- 遵循 PEP 8，行宽 100

## 工作流
1. 文档预置 → `ingest.py` 加载/分块/向量化/入库
2. 用户提问 → `retriever.py` 检索 Top-K 相关片段
3. 片段 + 问题 → `generator.py` 合成带依据的回答
4. 日志写入 SQLite（问题/检索结果/回答/耗时）

## 环境变量
- `DEEPSEEK_API_KEY` — DeepSeek API 密钥
- `DEEPSEEK_BASE_URL` — API 地址（默认 https://api.deepseek.com）
- `EMBED_MODEL` — embedding 模型名（默认 BAAI/bge-small-zh-v1.5）
- `LLM_MODEL` — LLM 模型名（默认 deepseek-v4-pro）
