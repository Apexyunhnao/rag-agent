"""
FastAPI 服务入口：提供 /query API 和 Web 问答界面。

用法: uvicorn main:app --host 127.0.0.1 --port 8002
"""

import os
import sqlite3
import time
from contextlib import asynccontextmanager

from dotenv import load_dotenv

load_dotenv()

from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from pydantic import BaseModel

from retriever import retrieve
from generator import generate

DB_PATH = "data/queries.db"


# ---- SQLite 初始化 ----
def init_db() -> None:
    db_dir = os.path.dirname(DB_PATH)
    if db_dir:
        os.makedirs(db_dir, exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS queries (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            question TEXT NOT NULL,
            sources TEXT NOT NULL,
            answer_summary TEXT NOT NULL,
            elapsed_ms INTEGER NOT NULL,
            created_at TEXT NOT NULL DEFAULT (datetime('now', 'localtime'))
        )
    """)
    conn.commit()
    conn.close()


@asynccontextmanager
async def lifespan(app: FastAPI):
    init_db()
    yield


app = FastAPI(title="RAG Agent - 知识库问答", lifespan=lifespan)


# ---- 请求/响应模型 ----
class QueryRequest(BaseModel):
    question: str


class QueryResponse(BaseModel):
    question: str
    answer: str
    sources: list[str]
    elapsed_ms: int


# ---- API ----
@app.post("/query", response_model=QueryResponse)
def query(req: QueryRequest) -> QueryResponse:
    t0 = time.time()

    # 检索
    chunks = retrieve(req.question, top_k=5)
    sources = list({c.source_doc for c in chunks})

    # 生成
    contexts = [(c.text, c.source_doc) for c in chunks]
    answer = generate(req.question, contexts)

    elapsed_ms = int((time.time() - t0) * 1000)

    # 写日志
    summary = answer[:200] if answer else ""
    conn = sqlite3.connect(DB_PATH)
    conn.execute(
        "INSERT INTO queries (question, sources, answer_summary, elapsed_ms) VALUES (?, ?, ?, ?)",
        (req.question, ", ".join(sources), summary, elapsed_ms),
    )
    conn.commit()
    conn.close()

    return QueryResponse(
        question=req.question,
        answer=answer,
        sources=sources,
        elapsed_ms=elapsed_ms,
    )


# ---- Web 界面 ----
@app.get("/", response_class=HTMLResponse)
def index() -> str:
    return """<!DOCTYPE html>
<html lang="zh-CN">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>RAG Agent - 知识库问答</title>
    <style>
        * { box-sizing: border-box; margin: 0; padding: 0; }
        body {
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
            background: #f5f7fa; color: #333; min-height: 100vh;
        }
        .container { max-width: 720px; margin: 0 auto; padding: 2rem 1rem; }
        h1 { font-size: 1.6rem; margin-bottom: 0.3rem; color: #1a1a2e; }
        .subtitle { color: #888; font-size: 0.9rem; margin-bottom: 1.5rem; }
        .input-group { display: flex; gap: 0.5rem; margin-bottom: 1.5rem; }
        .input-group input {
            flex: 1; padding: 0.75rem 1rem; font-size: 1rem;
            border: 1px solid #d0d5dd; border-radius: 8px; outline: none;
        }
        .input-group input:focus { border-color: #4f46e5; box-shadow: 0 0 0 3px rgba(79,70,229,0.1); }
        .input-group button {
            padding: 0.75rem 1.5rem; font-size: 1rem;
            background: #4f46e5; color: #fff; border: none; border-radius: 8px; cursor: pointer;
            white-space: nowrap;
        }
        .input-group button:hover { background: #4338ca; }
        .input-group button:disabled { background: #a5b4fc; cursor: not-allowed; }
        .result {
            background: #fff; border-radius: 12px; padding: 1.5rem;
            box-shadow: 0 1px 3px rgba(0,0,0,0.08); display: none;
        }
        .result-header { font-size: 0.85rem; color: #888; margin-bottom: 0.75rem; }
        .result-header span { margin-right: 1rem; }
        .answer { font-size: 1rem; line-height: 1.75; white-space: pre-wrap; }
        .sources { margin-top: 1rem; padding-top: 0.75rem; border-top: 1px solid #eee; font-size: 0.85rem; color: #666; }
        .sources strong { color: #333; }
        .spinner { display: none; text-align: center; padding: 2rem; color: #888; }
        .error { color: #dc2626; padding: 1rem; background: #fef2f2; border-radius: 8px; display: none; }
        .preset { margin-bottom: 1rem; display: flex; gap: 0.5rem; flex-wrap: wrap; }
        .preset button {
            padding: 0.35rem 0.75rem; font-size: 0.85rem;
            background: #eef2ff; color: #4f46e5; border: 1px solid #c7d2fe; border-radius: 20px; cursor: pointer;
        }
        .preset button:hover { background: #e0e7ff; }
    </style>
</head>
<body>
<div class="container">
    <h1>RAG Agent</h1>
    <p class="subtitle">企业知识库问答 — 覆盖员工手册、IT FAQ、信息安全、入职指南、绩效考核</p>

    <div class="preset">
        <button onclick="ask('年假能休几天')">年假能休几天</button>
        <button onclick="ask('VPN连不上怎么办')">VPN连不上怎么办</button>
        <button onclick="ask('报销需要什么材料')">报销需要什么材料</button>
        <button onclick="ask('试用期多久')">试用期多久</button>
        <button onclick="ask('绩效考核怎么申诉')">绩效考核怎么申诉</button>
    </div>

    <div class="input-group">
        <input id="question" type="text" placeholder="输入你的问题，如：年假能休几天..." autofocus
               onkeydown="if(event.key==='Enter')ask()">
        <button id="askBtn" onclick="ask()">提问</button>
    </div>

    <div class="error" id="error"></div>
    <div class="spinner" id="spinner">思考中，请稍候...</div>
    <div class="result" id="result">
        <div class="result-header">
            <span id="elapsed"></span>
        </div>
        <div class="answer" id="answer"></div>
        <div class="sources" id="sources"></div>
    </div>
</div>

<script>
async function ask(q) {
    const question = q || document.getElementById('question').value.trim();
    if (!question) return;
    if (!q) document.getElementById('question').value = question;

    const btn = document.getElementById('askBtn');
    const spinner = document.getElementById('spinner');
    const result = document.getElementById('result');
    const error = document.getElementById('error');

    btn.disabled = true;
    spinner.style.display = 'block';
    result.style.display = 'none';
    error.style.display = 'none';

    try {
        const resp = await fetch('/query', {
            method: 'POST',
            headers: {'Content-Type': 'application/json'},
            body: JSON.stringify({question})
        });
        const data = await resp.json();
        if (!resp.ok) throw new Error(data.detail || '请求失败');

        document.getElementById('answer').textContent = data.answer;
        document.getElementById('sources').innerHTML = '<strong>来源文档：</strong>' + data.sources.join('、');
        document.getElementById('elapsed').textContent = '耗时 ' + (data.elapsed_ms / 1000).toFixed(1) + 's';
        result.style.display = 'block';
    } catch (e) {
        error.textContent = '错误：' + e.message;
        error.style.display = 'block';
    } finally {
        btn.disabled = false;
        spinner.style.display = 'none';
    }
}
</script>
</body>
</html>"""
