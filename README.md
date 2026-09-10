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

**测试集**：20 条用例 = 16 条开发集 + 4 条留出集（按 8:2 分层切分，seed=42，4 份政策文档各 5 条、每份至少 1 条进留出集）。
**数字来源**：`eval/results.json`（由 `eval/run_eval.py` 自动生成，README 只引用它）。

| 指标 | 全量（20 条） | 开发集（16 条） | 留出集（4 条） |
|------|-------------|---------------|--------------|
| 检索命中率 | **100%**（20/20） | 100%（16/16） | **100%**（4/4） |
| 回答准确率 | 95.0%（19/20） | 93.8%（15/16） | **100%**（4/4） |
| 来源正确率 | 90.0%（18/20） | **100%**（16/16） | 50%（2/4） |

**3 条偏差**：

- **R09「退货退款和差价退款有什么区别」、R14「换货和退货有什么区别」**：跨文档对比类问题只标注了单一来源（回答内容正确、检索也命中）——来源正确率目前是薄弱项
- **R20「能否指定快递公司配送」**：政策文档本身没写这条规则，模型如实回答"文档中没有相关内容"——属于**知识库覆盖度**问题，不是生成环节的编造

> 检索与回答是核心指标；来源标注在多文档对比问题上偏弱，改进方向是让生成端在跨文档问题时输出多个来源（接口的 `sources` 已是数组，但生成端目前只给一个）。

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
