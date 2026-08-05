# 技术选型说明

> 撰写角度：技术评审会上向团队说明"为什么选这个、备选是什么、为什么不选备选"。

---

## 1. RAG 框架：LlamaIndex

**选型结论**：LlamaIndex（llama_index 库）

**理由**：
- RAG 全流程覆盖——文档加载（SimpleDirectoryReader）、分块（SentenceSplitter）、索引（VectorStoreIndex）、检索（RetrieverQueryEngine）、合成（ResponseSynthesizer），全部集成在一个框架里，不需要自己拼
- 检索是 RAG 的核心，LlamaIndex 内置多种检索策略（向量检索、关键词混合、递归检索），切换成本低
- 和 ChromaDB 有官方集成（`llama-index-vector-stores-chroma`），一行代码接上
- 社区活跃，文档质量好，学习曲线比 LangChain 平缓

### 备选：LangChain

LangChain 的强项是编排——把 LLM、工具、记忆串成 agent 工作流。但 RAG 场景下它的抽象层太厚，同样的功能比 LlamaIndex 多写 30%~50% 代码。选 LangChain 意味着为编排放了一个框架，却只用它最弱的部分。

### 备选：手写 RAG

从头写分块逻辑、向量检索、重排、合成——所有轮子自己造。灵活度最高，但求职作品集场景下，面试官想看的是你选型决策和集成能力，不是造轮子。而且分块策略的调优成本（中英文、文档类型、chunk size）远比想象中大。

### 备选：LangGraph

面向多 agent 编排的图执行框架。本项目是纯检索+生成管线，没有条件路由、没有多步骤决策、没有工具调用循环。用 LangGraph 等于拿手术刀劈柴。

---

## 2. Embedding 模型：sentence-transformers（本地）

**选型结论**：`BAAI/bge-small-zh-v1.5`（512 维），通过 `HuggingFaceEmbedding` 加载

**理由**：
- 零 API 成本——本地 CPU 推理，不调外部服务
- 中文检索效果在 C-MTEB 榜单上同尺寸排名靠前
- 512 维向量体积小，ChromaDB 存几千条无压力
- 数据不出本机，隐私无风险（求职作品集不需要但面试官会认可这个意识）

### 备选：OpenAI text-embedding-ada-002

每条向量都要调 API，有成本、有延迟、有网络依赖。RAG 管道的 embedding 调用是批量的（文档分块后逐条入库），本地模型跑一次的事，调 API 要排队等。而且中文效果不如专门的中文模型。

### 备选：text2vec-large-chinese

效果接近 bge-small-zh，但 1024 维向量是 bge 的 2 倍。向量维度翻倍意味着 ChromaDB 存储和检索的耗时都增加，在 512 维足够好时没必要。

---

## 3. 向量数据库：ChromaDB

**选型结论**：ChromaDB（Python 原生，持久化模式）

**理由**：
- 安装即用——`pip install chromadb`，不需要 Docker、不需要服务进程
- Python 原生 API，和 LlamaIndex 的集成方式就是改一行 `vector_store` 参数
- 轻量但够用——几千条文档片段下检索延迟 < 50ms
- 持久化到本地文件，项目无需额外部署步骤

### 备选：FAISS

检索速度极快（十亿级向量），但这是大厂场景。几百条文档的 MVP 阶段，FAISS 的性能优势完全体现不出来。而且 FAISS 的序列化/反序列化、索引重建都比 ChromaDB 多写不少样板代码。

### 备选：Qdrant / Milvus

功能完整的向量数据库，但不适合求职作品集。需要 Docker 部署，面试官 clone 项目后要 `docker-compose up` 才能跑起来——每多一步环境配置就多一个放弃率。ChromaDB 零部署，`pip install` 搞定。

### 备选：PostgreSQL + pgvector

太重。本项目的元数据量极少（基本只有文档名和 chunk 序号），不需要关系型数据库的查询能力。SQLite 记录日志就够了。

---

## 4. LLM 接口：openai SDK（DeepSeek 兼容模式）

**选型结论**：用 `openai` SDK，设置 `base_url` 指向 DeepSeek API

**理由**：
- DeepSeek 兼容 OpenAI 接口格式，SDK 可以直接用
- DeepSeek API 成本极低（约 ￥1/百万 token），中文能力强
- 不锁死 DeepSeek——以后换成任何兼容 OpenAI 接口的服务（通义千问、智谱 GLM、本地 vLLM）都只需改 `base_url` 和 `api_key`
- 面试官看到的是标准接口调用，不是某个厂商的专有 SDK

### 备选：langchain-openai

功能一样，多了一层抽象。在本项目的调用场景下（一个 prompt 模板 + 一次 LLM 调用），多引入一个依赖不值得。

### 备选：直接调 HTTP API（requests）

需要自己处理流式、重试、超时、错误码映射——openai SDK 都做了，且是大部分 Python 开发者的共识选择。

---

## 5. Web 框架：FastAPI

**选型结论**：FastAPI

**理由**：
- 异步支持——RAG 查询是 IO 密集型（LLM 调用 2~3 秒），异步不阻塞其他请求
- 自动生成 OpenAPI 文档（`/docs`），面试官可以直接在 Swagger UI 里试接口
- Pydantic 类型校验——请求/响应 schema 定义即文档

### 备选：Flask

同步模型下每个 LLM 调用阻塞整个 worker。单用户演示无所谓，但技术选型文档里要写异步方案。

---

## 总结

| 组件 | 选择 | 核心逻辑 |
|------|------|----------|
| RAG 框架 | LlamaIndex | 检索是核心，LlamaIndex 专注做这件事 |
| Embedding | bge-small-zh-v1.5 (本地) | 零成本、中文好、512 维够用 |
| 向量存储 | ChromaDB | 零部署、Python 原生、轻量 |
| LLM | openai SDK → DeepSeek | 便宜、中文好、接口标准 |
| Web | FastAPI | 异步、自动文档 |
