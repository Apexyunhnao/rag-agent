"""
端到端查询管线：问题 → 检索 → 生成 → 打印结果。

用法: python query_pipeline.py
"""

from retriever import retrieve, Chunk
from generator import generate


def ask(question: str, top_k: int = 5) -> dict:
    """
    单轮问答，返回中间结果供调试。

    Returns:
        {"question": ..., "chunks": [...], "answer": ...}
    """
    chunks = retrieve(question, top_k=top_k)
    contexts = [(c.text, c.source_doc) for c in chunks]
    answer = generate(question, contexts)
    return {"question": question, "chunks": chunks, "answer": answer}


def print_result(result: dict) -> None:
    """格式化打印一次查询的完整结果。"""
    print(f"{'=' * 70}")
    print(f"[问题] {result['question']}")
    print(f"{'=' * 70}")

    print(f"\n[检索命中片段 Top-2]:")
    for i, c in enumerate(result["chunks"][:2], 1):
        preview = c.text[:150].replace("\n", " ")
        print(f"  [{i}] {c.source_doc}  (score={c.score:.4f})")
        print(f"      \"{preview}...\"")

    print(f"\n[回答]:")
    print(f"   {result['answer']}")
    print()


def main() -> None:
    test_questions = [
        "年假能休几天",
        "VPN连不上怎么办",
        "报销需要什么材料",
    ]

    for q in test_questions:
        result = ask(q)
        print_result(result)


if __name__ == "__main__":
    main()
