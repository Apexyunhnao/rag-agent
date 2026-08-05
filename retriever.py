"""
检索模块：将用户问题向量化，在 ChromaDB 中查找最相关的文档片段。

ChromaDB 直接查询，不绕 LlamaIndex，简单可靠。
"""

import os
from pathlib import Path
from dataclasses import dataclass
from dotenv import load_dotenv

load_dotenv()

from llama_index.embeddings.huggingface import HuggingFaceEmbedding
import chromadb


@dataclass
class Chunk:
    """检索结果"""
    text: str
    source_doc: str
    chunk_index: int
    score: float


# 全局加载一次，避免每次查询都加载模型和 collection
_embed_model: HuggingFaceEmbedding | None = None
_collection: chromadb.Collection | None = None


def _get_embed_model() -> HuggingFaceEmbedding:
    """延迟加载 embedding 模型（单例）。"""
    global _embed_model
    if _embed_model is None:
        _embed_model = HuggingFaceEmbedding(
            model_name=os.environ.get("EMBED_MODEL", "BAAI/bge-small-zh-v1.5"),
            device="cpu",
        )
    return _embed_model


def _get_collection() -> chromadb.Collection:
    """延迟加载 ChromaDB collection（单例）。"""
    global _collection
    if _collection is None:
        chroma_dir = Path("data/chroma_db")
        if not chroma_dir.exists():
            raise FileNotFoundError(f"ChromaDB 目录不存在: {chroma_dir}，请先运行 ingest.py")
        client = chromadb.PersistentClient(path=str(chroma_dir))
        _collection = client.get_collection("doc_chunks")
    return _collection


def retrieve(query: str, top_k: int = 5) -> list[Chunk]:
    """
    检索与 query 最相关的文档片段。

    Args:
        query: 用户自然语言问题
        top_k: 返回的片段数量，默认 5

    Returns:
        按相似度降序排列的 Chunk 列表
    """
    embed_model = _get_embed_model()
    col = _get_collection()

    # 问题向量化
    query_embedding = embed_model.get_query_embedding(query)

    # ChromaDB 相似度检索
    results = col.query(
        query_embeddings=[query_embedding],
        n_results=top_k,
        include=["documents", "metadatas", "distances"],
    )

    chunks: list[Chunk] = []
    for i in range(len(results["ids"][0])):
        meta = results["metadatas"][0][i]
        text = results["documents"][0][i]
        distance = results["distances"][0][i]
        # ChromaDB cosine distance → 转为 0~1 相似度分数
        score = 1.0 - distance

        chunks.append(Chunk(
            text=text,
            source_doc=meta.get("file_name", meta.get("source_doc", "unknown")),
            chunk_index=meta.get("chunk_index", i),
            score=round(score, 4),
        ))

    return chunks


if __name__ == "__main__":
    # 快速自测
    chunks = retrieve("年假能休几天", top_k=3)
    for c in chunks:
        print(f"[{c.source_doc}] score={c.score:.4f}")
        print(f"  {c.text[:120]}...")
        print()
