# RAG Agent — 知识库问答系统

一个基于 RAG（检索增强生成）的企业知识库问答 Agent。预置文档后，用户以自然语言提问，系统检索最相关的内容片段，由 LLM 生成带来源引用的回答。

## 架构

```
离线段（ingest）                      在线段（query）
┌──────────┐    ┌──────┐    ┌────────┐   ┌──────────┐    ┌──────┐    ┌──────────┐
│ Markdown │ → │ 分块  │ → │ 向量化 │   │ 用户问题 │ → │ 检索 │ → │ LLM 生成 │
│ 文档     │    │300字  │    │ChromaDB│   └──────────┘    │Top-5 │    │ 带来源    │
└──────────┘    └──────┘    └────────┘                   └──────┘    └──────────┘
```

- **离线段**：`ingest.py` — SimpleDirectoryReader 加载 → SentenceSplitter 分块 → HuggingFaceEmbedding 向量化 → ChromaDB 入库
- **在线段**：`retriever.py` + `generator.py` — 问题向量化 → ChromaDB Top-5 检索 → DeepSeek LLM 生成回答
- **服务**：`main.py` — FastAPI，`/query` API + `/` Web 界面，SQLite 查询日志
- **评估**：`eval/` — 24 条测试用例 + 自动评测脚本

## 目录结构

```
rag-agent/
├── main.py               # FastAPI 入口（/query + /）
├── ingest.py             # 文档入库管线
├── retriever.py          # 向量检索
├── generator.py          # LLM 生成（openai SDK → DeepSeek）
├── query_pipeline.py     # 命令行问答调试
├── .env                  # API Key 配置（不提交）
├── .gitignore
├── data/
│   ├── documents/        # 预置 Markdown 文档（5 份）
│   ├── chroma_db/        # ChromaDB 持久化向量库
│   └── queries.db        # SQLite 查询日志
├── eval/
│   ├── test_cases.json   # 24 条测试用例
│   ├── run_eval.py       # 自动评测脚本
│   └── results.json      # 评测结果
├── docs/
│   ├── requirements.md   # 需求文档
│   └── tech_selection.md # 技术选型说明
└── CLAUDE.md             # 项目约定
```

## 快速开始

### 1. 安装依赖

```bash
pip install llama-index chromadb sentence-transformers \
    llama-index-vector-stores-chroma llama-index-embeddings-huggingface \
    openai fastapi uvicorn python-dotenv
```

### 2. 配置 API Key

在项目根目录创建 `.env`：

```env
DEEPSEEK_API_KEY=你的DeepSeek_API_Key
DEEPSEEK_BASE_URL=https://api.deepseek.com
LLM_MODEL=deepseek-v4-pro
```

### 3. 文档入库

```bash
python ingest.py
```

首次运行会自动下载 embedding 模型（BAAI/bge-small-zh-v1.5，约 400MB）。

### 4. 启动服务

```bash
uvicorn main:app --host 127.0.0.1 --port 8002
```

打开浏览器访问 http://127.0.0.1:8002，输入问题即可。

### 5. 命令行测试

```bash
python query_pipeline.py
```

运行 3 个预设问题，查看检索和回答详情。

## 评估结果

24 条测试用例，两轮优化迭代：

| 版本 | 检索命中率 | 回答准确率 | 来源正确率 | 说明 |
|------|-----------|-----------|-----------|------|
| v1 | 95.8% | 79.2% | 87.5% | chunk_size=400，基础 prompt |
| v2 | **100%** | **91.7%** | 87.5% | chunk_size=300，增强 prompt |

运行评测：

```bash
python eval/run_eval.py
```

## 技术栈

| 组件 | 选型 | 说明 |
|------|------|------|
| RAG 框架 | LlamaIndex | 加载/分块/索引全流程 |
| Embedding | BAAI/bge-small-zh-v1.5 | 512 维，本地 CPU，零成本 |
| 向量库 | ChromaDB | Python 原生，零部署 |
| LLM | DeepSeek (openai SDK) | 兼容接口，中文能力强 |
| Web | FastAPI | 异步 + 自动文档 |
| 日志 | SQLite | 零部署，查询记录持久化 |

详细选型理由见 [docs/tech_selection.md](docs/tech_selection.md)。

## 文档

- [需求文档](docs/requirements.md) — 业务背景、目标、成功指标、范围
- [技术选型](docs/tech_selection.md) — 各组件选型理由与备选对比

## License

MIT
