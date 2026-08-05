"""
文档入库管线：加载 data/documents/ 下文档 → 分块 → 向量化 → 存入 ChromaDB。

一次一个文件，每个 chunk 记录来源文档和序号。
"""

import os
import sys
from pathlib import Path

# 优先加载 .env，确保 HF_ENDPOINT 在导入 huggingface 前生效
from dotenv import load_dotenv

load_dotenv()

# 如需国内镜像，在 .env 中设置 HF_ENDPOINT=https://hf-mirror.com 即可

from llama_index.core import SimpleDirectoryReader, VectorStoreIndex, StorageContext
from llama_index.core.node_parser import SentenceSplitter
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.vector_stores.chroma import ChromaVectorStore
import chromadb


def ingest() -> None:
    # 1. 加载文档
    docs_dir = Path("data/documents")
    if not docs_dir.exists():
        print(f"错误：文档目录不存在 — {docs_dir}")
        sys.exit(1)

    reader = SimpleDirectoryReader(input_dir=str(docs_dir), required_exts=[".md"])
    documents = reader.load_data()
    print(f"读取到 {len(documents)} 份文档\n")

    # 2. 分块（中文按句切，chunk 300~500 字，重叠 50 字）
    splitter = SentenceSplitter(
        chunk_size=300,
        chunk_overlap=50,
        separator="。",  # 中文句号优先切分
    )
    nodes = splitter.get_nodes_from_documents(documents)

    # 统计每份文档的 chunk 数
    doc_stats: dict[str, int] = {}
    for node in nodes:
        source = node.metadata.get("file_name", "unknown")
        doc_stats[source] = doc_stats.get(source, 0) + 1

    print("各文档分块统计：")
    for doc_name, count in sorted(doc_stats.items()):
        print(f"  {doc_name}: {count} 块")
    print(f"  总计: {len(nodes)} 块\n")

    # 3. 向量化模型（本地 CPU）
    embed_model = HuggingFaceEmbedding(
        model_name=os.environ.get("EMBED_MODEL", "BAAI/bge-small-zh-v1.5"),
        device="cpu",
    )

    # 4. ChromaDB 持久化存储
    chroma_dir = Path("data/chroma_db")
    chroma_dir.mkdir(parents=True, exist_ok=True)

    chroma_client = chromadb.PersistentClient(path=str(chroma_dir))
    # 每次入库重建 collection，保证幂等
    try:
        chroma_client.delete_collection("doc_chunks")
    except Exception:
        pass
    chroma_collection = chroma_client.create_collection("doc_chunks")
    vector_store = ChromaVectorStore(chroma_collection=chroma_collection)

    storage_context = StorageContext.from_defaults(vector_store=vector_store)

    # 5. 索引入库
    print("正在向量化并入库（首次运行需下载 embedding 模型，约 400MB）...")
    index = VectorStoreIndex(
        nodes,
        storage_context=storage_context,
        embed_model=embed_model,
    )
    print(f"入库完成！collection 名称: doc_chunks, 持久化目录: {chroma_dir.resolve()}\n")

    # 6. 输出 chunk 级别 metadata（前 5 条示例）
    print("Chunk metadata 示例（前 5 条）：")
    for i, node in enumerate(nodes[:5], 1):
        print(f"  [{i}] source={node.metadata.get('file_name')}, "
              f"chunk_size={len(node.text)}字")


if __name__ == "__main__":
    ingest()
