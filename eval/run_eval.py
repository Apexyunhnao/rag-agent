"""
评测脚本：加载 test_cases.json，逐条跑检索+生成，计算三项指标并输出报表。

用法: python eval/run_eval.py
"""

import json
import re
import sys
import time
from pathlib import Path
from collections import defaultdict

# 确保项目根目录在 sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from retriever import retrieve
from generator import generate


def check_retrieval_hit(chunks: list, source_doc: str) -> bool:
    """Top-5 片段中是否包含目标文档。"""
    return any(c.source_doc == source_doc for c in chunks)


def _norm(s: str) -> str:
    """归一化：去所有空格（含全角）、转小写，避免'5 天'与'5天'匹配失败。"""
    return s.replace(" ", "").replace("\u3000", "").lower()


def check_answer_accuracy(answer: str, keywords: list[str]) -> bool:
    """回答中是否包含至少 1 个关键词（忽略空格、大小写不敏感）。"""
    answer_norm = _norm(answer)
    return any(_norm(kw) in answer_norm for kw in keywords)


def check_source_correct(answer: str, source_doc: str) -> bool:
    """回答末尾的 [来源: xxx] 是否与 source_doc 一致。"""
    # 匹配末尾的 [来源: ...] 模式
    match = re.search(r"\[来源[：:]\s*([^\]]+)\]", answer)
    if not match:
        return False
    cited = match.group(1).strip()
    return cited == source_doc


def evaluate_one(case: dict) -> dict:
    """评测单条用例，返回完整结果。"""
    qid = case["id"]
    question = case["question"]
    source_doc = case["source_doc"]
    keywords = case["answer_keywords"]

    # 检索
    chunks = retrieve(question, top_k=5)

    # 生成
    contexts = [(c.text, c.source_doc) for c in chunks]
    t0 = time.time()
    answer = generate(question, contexts)
    elapsed = round(time.time() - t0, 2)

    # 三项判定
    retrieval_hit = check_retrieval_hit(chunks, source_doc)
    answer_accurate = check_answer_accuracy(answer, keywords)
    source_correct = check_source_correct(answer, source_doc)

    # 提取命中的关键词（调试用）
    answer_norm = _norm(answer)
    matched_kw = [kw for kw in keywords if _norm(kw) in answer_norm]

    return {
        "id": qid,
        "question": question,
        "source_doc": source_doc,
        "keywords_expected": keywords,
        "keywords_matched": matched_kw,
        "retrieval_hit": retrieval_hit,
        "answer_accurate": answer_accurate,
        "source_correct": source_correct,
        "elapsed_sec": elapsed,
        "top_chunks": [
            {"source": c.source_doc, "score": c.score, "text_preview": c.text[:80]}
            for c in chunks[:3]
        ],
        "answer": answer,
    }


def print_report(results: list[dict]) -> None:
    """输出评测报表。"""
    total = len(results)
    hits = sum(1 for r in results if r["retrieval_hit"])
    acc = sum(1 for r in results if r["answer_accurate"])
    src_ok = sum(1 for r in results if r["source_correct"])

    print("=" * 70)
    print("                RAG Agent 评测报告")
    print("=" * 70)

    # 概览
    print(f"\n[概览]")
    print(f"  测试用例总数: {total}")
    print(f"  检索命中率:   {hits}/{total} = {hits/total*100:.1f}%")
    print(f"  回答准确率:   {acc}/{total} = {acc/total*100:.1f}%")
    print(f"  来源正确率:   {src_ok}/{total} = {src_ok/total*100:.1f}%")
    avg_elapsed = sum(r["elapsed_sec"] for r in results) / total
    print(f"  平均耗时:     {avg_elapsed:.1f}s")

    # 按文档统计
    print(f"\n[按文档统计]")
    doc_groups = defaultdict(list)
    for r in results:
        doc_groups[r["source_doc"]].append(r)
    for doc in sorted(doc_groups):
        grp = doc_groups[doc]
        g_total = len(grp)
        g_hits = sum(1 for r in grp if r["retrieval_hit"])
        g_acc = sum(1 for r in grp if r["answer_accurate"])
        g_src = sum(1 for r in grp if r["source_correct"])
        print(f"  {doc}: {g_total} 条, 检索 {g_hits}/{g_total}, "
              f"准确 {g_acc}/{g_total}, 来源 {g_src}/{g_total}")

    # 失败用例
    failures = [r for r in results
                if not (r["retrieval_hit"] and r["answer_accurate"] and r["source_correct"])]
    if failures:
        print(f"\n[失败用例清单] ({len(failures)} 条)")
        for r in failures:
            reasons = []
            if not r["retrieval_hit"]:
                reasons.append("检索未命中目标文档")
            if not r["answer_accurate"]:
                reasons.append(f"关键词未匹配(期望: {r['keywords_expected'][:3]}...)")
            if not r["source_correct"]:
                reasons.append("来源标注错误或缺失")
            print(f"\n  [{r['id']}] {r['question']}")
            print(f"    目标文档: {r['source_doc']}")
            print(f"    失败原因: {'; '.join(reasons)}")
            print(f"    实际 Top-1: {r['top_chunks'][0]['source'] if r['top_chunks'] else 'N/A'}")
            if r["keywords_matched"]:
                print(f"    已匹配关键词: {r['keywords_matched']}")
    else:
        print(f"\n[全部通过！]")

    print(f"\n{'=' * 70}")

    # 耗时分布
    print(f"\n[耗时分布]")
    times = sorted(r["elapsed_sec"] for r in results)
    print(f"  最快: {times[0]:.1f}s, 最慢: {times[-1]:.1f}s, "
          f"P50: {times[len(times)//2]:.1f}s")


def main() -> None:
    # 加载测试用例
    cases_path = Path(__file__).resolve().parent / "test_cases.json"
    with open(cases_path, "r", encoding="utf-8") as f:
        cases = json.load(f)

    print(f"开始评测 {len(cases)} 条用例，预计 3~5 分钟...\n")

    results = []
    for i, case in enumerate(cases, 1):
        print(f"  [{i}/{len(cases)}] {case['id']}: {case['question']}")
        result = evaluate_one(case)
        results.append(result)

    print(f"\n评测完成。\n")

    # 输出报表
    print_report(results)

    # 保存详细结果
    output_path = Path(__file__).resolve().parent / "results.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"\n详细结果已保存: {output_path}")


if __name__ == "__main__":
    main()
