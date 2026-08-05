"""
生成模块：用检索到的文档片段 + 用户问题，调用 LLM 生成带来源引用的回答。

使用 openai SDK 调用 DeepSeek 兼容接口，不依赖 langchain。
API 调用内置指数退避重试（网络/超时/5xx，最多 3 次）。
"""

import os
import time
from dotenv import load_dotenv

load_dotenv()

from openai import OpenAI, APIError, APIConnectionError, APITimeoutError, RateLimitError

_client: OpenAI | None = None

PROMPT_TEMPLATE = """你是一个企业知识库助手。请根据以下文档片段回答用户问题。

## 要求
1. 只能根据片段中的内容回答，不要编造信息。
2. 如果片段中有明确的步骤、条款或要求，必须完整逐条列出，不能遗漏或省略任何一条。
3. 如果片段中有明确答案，直接给出并标注来源。
4. 如果片段中没有相关内容，回答"文档中没有相关内容"。
5. 回答末尾用 [来源: 文档名] 标注所引用的文档。

## 文档片段
{contexts}

## 用户问题
{question}

## 回答"""

# 可重试的错误类型
RETRYABLE_ERRORS = (APIConnectionError, APITimeoutError, RateLimitError)

MAX_RETRIES = 3
RETRY_BASE_SEC = 2  # 指数退避: 2/4/8 秒


def _get_client() -> OpenAI:
    """延迟初始化 DeepSeek 客户端（兼容 openai SDK）。"""
    global _client
    if _client is None:
        _client = OpenAI(
            api_key=os.environ["DEEPSEEK_API_KEY"],
            base_url=os.environ.get("DEEPSEEK_BASE_URL", "https://api.deepseek.com"),
        )
    return _client


def generate(question: str, contexts: list[tuple[str, str]]) -> str:
    """
    基于检索到的文档片段生成回答。

    Args:
        question: 用户原始问题
        contexts: (文本, 来源文档名) 列表

    Returns:
        LLM 生成的回答（含来源标注）
    """
    client = _get_client()

    formatted = "\n\n---\n\n".join(
        f"[片段 {i + 1}]（来源：{source}）\n{text}"
        for i, (text, source) in enumerate(contexts)
    )

    prompt = PROMPT_TEMPLATE.format(contexts=formatted, question=question)
    model = os.environ.get("LLM_MODEL", "deepseek-v4-pro")
    last_error: Exception | None = None

    for attempt in range(MAX_RETRIES + 1):
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": "你是一个严谨的企业知识库助手。"},
                    {"role": "user", "content": prompt},
                ],
                temperature=0.3,
                max_tokens=1024,
            )
            return resp.choices[0].message.content or ""
        except RETRYABLE_ERRORS as e:
            last_error = e
            if attempt >= MAX_RETRIES:
                raise
            wait = RETRY_BASE_SEC * (2 ** attempt)
            print(f"  [LLM 重试 {attempt + 1}/{MAX_RETRIES}] {e.__class__.__name__}: {e}, {wait}s 后重试...")
            time.sleep(wait)
        except APIError as e:
            # 5xx 服务端错误重试，4xx 不重试
            if e.http_status and e.http_status >= 500:
                last_error = e
                if attempt >= MAX_RETRIES:
                    raise
                wait = RETRY_BASE_SEC * (2 ** attempt)
                print(f"  [LLM 重试 {attempt + 1}/{MAX_RETRIES}] HTTP {e.http_status}, {wait}s 后重试...")
                time.sleep(wait)
            else:
                raise
    # 理论上不会走到这里，但兜底
    raise last_error  # type: ignore[misc]


if __name__ == "__main__":
    answer = generate(
        "年假能休几天？",
        [
            ("入职满1年的员工享有带薪年假5天。工龄5~10年者年假10天。", "员工手册.md"),
            ("年假以自然年为单位计算，未休完可顺延至次年3月31日。", "员工手册.md"),
        ],
    )
    print(answer)
