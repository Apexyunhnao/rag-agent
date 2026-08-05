# P0/P1 修复指令存档（外部评审，2026-08-05）

> 场景：项目二收到外部评审（桌面 审批.txt），有条件通过。以下是指令原文，
> 贴给 Claude Code 执行。跑完把改动清单发羔丸验证，验证后提交 GitHub。

## P0 必修 4 条

【P0-1】创建 requirements.txt：锁定项目实际依赖（llama-index、llama-index-vector-stores-chroma、llama-index-embeddings-huggingface、chromadb、sentence-transformers、openai、fastapi、uvicorn、python-dotenv），写主要版本约束，从当前已安装版本生成。

【P0-2】修复向量距离语义：ingest.py 的 create_collection 加 metadata={'hnsw:space': 'cosine'}，明确使用余弦距离（当前默认 L2，retriever 却按 cosine 换算 score，名不副实）。改完重新运行 python ingest.py 重新入库。

【P0-3】评估和生成加重试：参考项目一的做法，eval/run_eval.py 的 LLM 调用和 generator.py 的生成调用都包重试（指数退避 2/4/8 秒，最多 3 次，只重试网络/超时/5xx 错误），防止 API 抖动污染评估数字。

【P0-4】文件类型名实相符：ingest.py 的 required_exts 改为支持 md/txt/pdf（['.md', '.txt', '.pdf']，SimpleDirectoryReader 原生支持），CLAUDE.md 保持 MD/TXT/PDF 说明不再矛盾。

## P1 建议 3 条

【P1-1】README 注明评估口径：回答准确率=至少命中 1 个关键词；来源正确率=匹配回答中第一个 [来源:xxx] 且等于目标文档。

【P1-2】requirements.md 修正：'选 10 个问题评测'改为'全部 24 条评测'。

【P1-3】eval/results.json 从 .gitignore 移除（评估结果应进仓库，和项目一一致），保留当前结果文件。

## 验证清单（羔丸验收用）

- [ ] requirements.txt 存在且版本合理
- [ ] ingest.py 有 hnsw:space=cosine，重跑后检索正常
- [ ] eval/generator 有重试逻辑
- [ ] ingest 支持 md/txt/pdf
- [ ] README/requirements.md 口径注明、文档一致
- [ ] eval/results.json 在仓库里（.gitignore 已移除）
- [ ] 重跑 eval/run_eval.py，指标不低于 100%/91.7%
