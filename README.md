# RAG Agent — 客服政策知识库问答

## 功能

基于 LlamaIndex + ChromaDB 的 RAG 管道：加载客服政策文档 → 分块 → 向量化 → 入库。用户自然语言提问，检索 Top-5 片段，DeepSeek 生成带来源引用的回答。

**当前知识库**：退货政策、退款政策、换货政策、物流配送政策（4 份文档，8 chunks）

## 技术栈

- LlamaIndex（文档加载、分块、索引）
- ChromaDB（向量存储，余弦相似度）
- BAAI/bge-small-zh-v1.5（Embedding，本地 CPU）
- DeepSeek（LLM 生成）
- FastAPI（HTTP 接口）

## 启动

```bash
pip install -r requirements.txt
python ingest.py                    # 入库文档
uvicorn main:app --port 8002        # 启动服务
```

或 Docker：
```bash
docker build -t rag-agent .
docker run -p 8002:8002 --env-file .env rag-agent
```

## API

POST /query
```json
{"question": "退货需要什么条件"}
→ {"answer": "...", "sources": ["退货政策.md"], "elapsed_ms": 1234}
```

## 评估

| 指标 | 全量（20条） | 留出集（4条） |
|------|-------------|--------------|
| 检索命中率 | 100% | 100% |
| 回答准确率 | 90% | 100% |
| 来源正确率 | 100% | 100% |

测试集分层抽样覆盖 4 份文档，8:2 留出集切分（seed=42）。

## 已知限制

- 知识库仅覆盖客服政策领域，超出范围回答"文档中没有相关内容"
- 单轮问答，不支持多轮对话
- 无用户认证
- Embedding 模型需首次下载（~400MB）

## 健康检查

GET /health → `{"status":"ok","service":"rag-agent"}`

## 项目结构

```
rag-agent/
├── ingest.py          # 文档入库
├── retriever.py       # ChromaDB 检索
├── generator.py       # LLM 生成（含重试）
├── main.py            # FastAPI 入口
├── eval/
│   ├── test_cases.json
│   ├── train_cases.json
│   ├── holdout_cases.json
│   └── run_eval.py
└── data/
    ├── documents/     # 4 份客服政策 Markdown
    └── chroma_db/     # 向量库持久化
```
