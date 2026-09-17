"""
FastAPI 服务入口：提供 /query API 和 Web 问答界面。

用法: uvicorn main:app --host 127.0.0.1 --port 8002
"""

import json
import os
import sqlite3
import time
import uuid
from contextlib import asynccontextmanager

from dotenv import load_dotenv

load_dotenv()

from fastapi import Depends, FastAPI, Request
from fastapi.responses import HTMLResponse, JSONResponse
from pydantic import BaseModel

from auth import current_user, require_roles
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
    # 轻量迁移：老库补 steps 列（存检索命中与生成详情，供观测页回放）
    # 注意 CREATE TABLE IF NOT EXISTS 不会给已存在的表加列，必须显式迁移
    cols = [r[1] for r in conn.execute("PRAGMA table_info(queries)")]
    if "steps" not in cols:
        conn.execute("ALTER TABLE queries ADD COLUMN steps TEXT NOT NULL DEFAULT '[]'")
    # 轻量迁移：补 trace_id 列（跨服务关联标识，一次用户请求在三个服务里共用一个）
    if "trace_id" not in cols:
        conn.execute("ALTER TABLE queries ADD COLUMN trace_id TEXT NOT NULL DEFAULT ''")
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
    steps: list[dict] = []   # 执行链路：检索命中（含相似度分数）→ 生成答案
    trace_id: str = ""       # 跨服务关联标识（调用方传入或本服务生成）


# ---- API ----
@app.post("/query", response_model=QueryResponse)
def query(req: QueryRequest, request: Request,
          user: dict = Depends(current_user)) -> QueryResponse:
    # 身份由本服务自己验签（Bearer 优先，其次 Cookie）—— 不信任上游传来的角色字符串
    # 关联标识：优先沿用调用方传入的 trace_id（跨服务链路），没有则自己生成
    trace_id = request.headers.get("X-Trace-Id") or uuid.uuid4().hex[:16]

    # 输入校验：空问题直接 400
    if not req.question or not req.question.strip():
        return JSONResponse(
            status_code=400,
            content={"error": "问题不能为空", "trace_id": trace_id},
        )

    t0 = time.time()

    # 检索
    try:
        chunks = retrieve(req.question, top_k=5)
    except FileNotFoundError as e:
        # 知识库未入库：明确告诉调用方去跑 ingest
        return JSONResponse(
            status_code=503,
            content={"error": f"知识库未初始化: {e}", "trace_id": trace_id},
        )
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"error": f"检索失败: {e}", "trace_id": trace_id},
        )

    sources = list({c.source_doc for c in chunks})
    t_retrieved = time.time()

    # 生成
    contexts = [(c.text, c.source_doc) for c in chunks]
    try:
        answer = generate(req.question, contexts)
    except Exception as e:
        # LLM 重试已耗尽（网络/超时/5xx 重试 3 次后仍失败）
        return JSONResponse(
            status_code=502,
            content={"error": f"LLM 生成失败（重试已耗尽）: {e}", "trace_id": trace_id},
        )

    elapsed_ms = int((time.time() - t0) * 1000)

    # 执行链路：检索命中（含相似度分数）→ 生成答案
    steps = [
        {
            "node": "retrieve",
            "step_ms": int((t_retrieved - t0) * 1000),
            "total_ms": int((t_retrieved - t0) * 1000),
            "items": [
                {
                    "type": "RetrievalHit",
                    "name": c.source_doc,
                    "score": c.score,
                    "rank": i + 1,
                    "content": c.text[:600],
                }
                for i, c in enumerate(chunks)
            ],
        },
        {
            "node": "generate",
            "step_ms": int((time.time() - t_retrieved) * 1000),
            "total_ms": elapsed_ms,
            "items": [{"type": "StateField", "name": "生成答案", "content": answer[:800]}],
        },
    ]

    # 写日志（含链路与关联标识）
    summary = answer[:200] if answer else ""
    conn = sqlite3.connect(DB_PATH)
    conn.execute(
        "INSERT INTO queries (question, sources, answer_summary, elapsed_ms, steps, trace_id)"
        " VALUES (?, ?, ?, ?, ?, ?)",
        (req.question, ", ".join(sources), summary, elapsed_ms,
         json.dumps(steps, ensure_ascii=False), trace_id),
    )
    conn.commit()
    conn.close()

    return QueryResponse(
        question=req.question,
        answer=answer,
        sources=sources,
        elapsed_ms=elapsed_ms,
        steps=steps,
        trace_id=trace_id,
    )


@app.get("/health")
def health() -> dict:
    """健康检查。"""
    return {"healthy": True, "service": "rag-agent"}


@app.get("/stats")
def stats(_user: dict = Depends(require_roles("agent", "admin"))) -> dict:
    """问答统计 + 最近记录（含检索链路），供观测页使用。仅客服/管理员可看。"""
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row

    total = conn.execute("SELECT COUNT(*) FROM queries").fetchone()[0]
    avg_ms = conn.execute("SELECT AVG(elapsed_ms) FROM queries").fetchone()[0] or 0

    recent: list[dict] = []
    for row in conn.execute(
        "SELECT id, question, sources, answer_summary, elapsed_ms, created_at, steps"
        " FROM queries ORDER BY id DESC LIMIT 30"
    ).fetchall():
        item = dict(row)
        try:
            item["steps"] = json.loads(item.get("steps") or "[]")
        except (json.JSONDecodeError, TypeError):
            item["steps"] = []
        item["sources"] = [s.strip() for s in (item.get("sources") or "").split(",") if s.strip()]
        recent.append(item)

    # 来源文档命中分布
    doc_hits: dict[str, int] = {}
    for r in recent:
        for d in r["sources"]:
            doc_hits[d] = doc_hits.get(d, 0) + 1

    conn.close()
    return {
        "total": total,
        "avg_elapsed_ms": int(avg_ms),
        "doc_hits": doc_hits,
        "recent": recent,
    }


# ---- Web 界面 ----
# 演示页面禁用缓存，保证改动即时可见
_NO_CACHE = {
    "Cache-Control": "no-store, no-cache, must-revalidate, max-age=0",
    "Pragma": "no-cache",
    "Expires": "0",
}


@app.get("/", response_class=HTMLResponse)
def index() -> HTMLResponse:
    """客户端：政策问答（面向使用者，不暴露内部实现细节）。"""
    return HTMLResponse(_USER_PAGE, headers=_NO_CACHE)


@app.get("/ops", response_class=HTMLResponse)
def ops_console(_user: dict = Depends(require_roles("agent", "admin"))) -> HTMLResponse:
    """服务端观测台：检索命中片段与相似度分数。仅客服/管理员可看（客户不得访问运营数据）。"""
    return HTMLResponse(_OPS_PAGE, headers=_NO_CACHE)


# ---- 页面模板（置于文件末尾，避免遮挡主逻辑）----
_USER_PAGE = r"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>政策问答 · RAG 政策问答</title>
<style>
  :root {
    --bg:#F5F7FC; --surface:#FFFFFF; --sunken:#F7F9FD;
    --border:#E3E7F2; --border-strong:#CBD3E4; --hair:#EFF2F8;
    --text:#0F1520; --text-2:#3A4560; --text-3:#7A8BA0;
    --accent:#2563EB; --accent-hover:#1D4ED8; --accent-soft:#EEF3FF;
    --ok:#15803D; --ok-bg:#E7F6EC; --warn:#9A6700; --warn-bg:#FFF8E1;
    --err:#C1272D; --err-bg:#FDEBEC;
    --mono: ui-monospace, SFMono-Regular, "Cascadia Mono", Consolas, "Courier New", monospace;
    --s1:8px; --s2:12px; --s3:16px; --s4:24px; --s5:32px; --s6:48px;
    --radius:12px; --radius-sm:8px;
    --shadow-sm: 0 1px 2px rgba(20,45,110,.06), 0 0 0 1px rgba(20,45,110,.05);
    --shadow-md: 0 2px 4px rgba(20,45,110,.05), 0 8px 24px rgba(20,45,110,.10), 0 0 0 1px rgba(20,45,110,.06);
    --ease: cubic-bezier(.22,1,.36,1);
    --t-fast:140ms; --t-base:220ms; --t-slow:380ms;
  }
  * { box-sizing:border-box; margin:0; padding:0; }
  html, body { height:100%; }
  body {
    display:flex; overflow:hidden;
    background:var(--bg); color:var(--text);
    font-family:-apple-system,BlinkMacSystemFont,"Segoe UI Variable Display","Segoe UI","Microsoft YaHei","PingFang SC",sans-serif;
    font-size:14.5px; line-height:1.6; -webkit-font-smoothing:antialiased;
  }
  .sidebar {
    width:248px; flex:none; background:var(--surface);
    border-right:1px solid var(--border);
    display:flex; flex-direction:column; padding:var(--s4) var(--s3); gap:var(--s5);
  }
  .brand { display:flex; align-items:center; gap:10px; padding:0 6px; }
  .brand-logo {
    width:32px; height:32px; flex:none; border-radius:9px;
    background:linear-gradient(135deg,#2563EB,#60A5FA); color:#fff;
    display:flex; align-items:center; justify-content:center;
    font-weight:800; font-size:14px; box-shadow:0 2px 8px rgba(37,99,235,.32);
  }
  .brand-name { font-size:13.5px; font-weight:700; letter-spacing:-.01em; line-height:1.3; }
  .brand-sub { font-size:11px; color:var(--text-3); }
  .nav { display:flex; flex-direction:column; gap:2px; }
  .nav-label {
    font-size:10.5px; font-weight:700; letter-spacing:.09em; text-transform:uppercase;
    color:var(--text-3); padding:0 8px; margin-bottom:6px;
  }
  .nav-item {
    display:flex; align-items:center; gap:10px;
    padding:9px 10px; border-radius:var(--radius-sm);
    color:var(--text-2); text-decoration:none; font-size:13.5px; font-weight:500;
    transition:background var(--t-fast) var(--ease), color var(--t-fast) var(--ease);
  }
  .nav-item:hover { background:var(--sunken); color:var(--text); }
  .nav-item.active { background:var(--accent-soft); color:var(--accent); font-weight:650; }
  .nav-item .ico { width:17px; text-align:center; font-size:13.5px; flex:none; opacity:.9; }
  .sidebar-foot { margin-top:auto; padding:0 8px; font-size:11px; color:var(--text-3); line-height:1.9; }
  .dot-ok { display:inline-block; width:6px; height:6px; border-radius:50%; background:#22C55E; margin-right:5px; vertical-align:1px; }

  .main { flex:1; min-width:0; display:flex; flex-direction:column; }
  .topbar {
    height:58px; flex:none; border-bottom:1px solid var(--border);
    background:rgba(255,255,255,.78); backdrop-filter:blur(10px);
    display:flex; align-items:center; justify-content:space-between; gap:var(--s3);
    padding:0 var(--s5);
  }
  .topbar h1 { font-size:15.5px; font-weight:700; letter-spacing:-.01em; }
  .topbar .sub { font-size:12.5px; color:var(--text-3); }
  .content { flex:1; overflow-y:auto; padding:var(--s5); }

  .btn-primary {
    padding:11px 26px; font-size:14.5px; font-family:inherit; font-weight:600;
    color:#fff; border:0; border-radius:var(--radius-sm); cursor:pointer; white-space:nowrap;
    background:linear-gradient(180deg,#3B76F0,#2563EB);
    box-shadow:0 1px 2px rgba(37,99,235,.30), inset 0 1px 0 rgba(255,255,255,.22);
    transition:filter var(--t-fast) var(--ease), transform var(--t-fast) var(--ease), box-shadow var(--t-fast) var(--ease);
  }
  .btn-primary:hover:not(:disabled) { filter:brightness(1.07); }
  .btn-primary:active:not(:disabled) { transform:translateY(1px) scale(.995); box-shadow:inset 0 1px 3px rgba(0,0,0,.16); }
  .btn-primary:disabled { opacity:.45; cursor:not-allowed; }
  .chip {
    font-size:12.5px; font-family:inherit; color:var(--accent);
    background:var(--accent-soft); border:0; box-shadow:inset 0 0 0 1px rgba(37,99,235,.15);
    border-radius:999px; padding:5px 13px; cursor:pointer; text-align:left;
    transition:background var(--t-fast) var(--ease), box-shadow var(--t-fast) var(--ease), transform var(--t-fast) var(--ease);
  }
  .chip:hover { background:#E3ECFF; box-shadow:inset 0 0 0 1px rgba(37,99,235,.28); }
  .chip:active { transform:scale(.97); }
  .badge { display:inline-flex; align-items:center; gap:4px; font-size:11.5px; font-weight:650; padding:2px 9px; border-radius:999px; }
  .badge-ok { color:var(--ok); background:var(--ok-bg); box-shadow:inset 0 0 0 1px rgba(21,128,61,.16); }
  .badge-warn { color:var(--warn); background:var(--warn-bg); box-shadow:inset 0 0 0 1px rgba(154,103,0,.16); }
  .badge-plain { color:var(--text-2); background:#F1F4FA; box-shadow:inset 0 0 0 1px var(--border); }
  .tag-tool {
    font-family:var(--mono); font-size:12px; background:#F1F4FA;
    box-shadow:inset 0 0 0 1px var(--border); border-radius:6px; padding:2px 8px; color:var(--text-2);
  }
  .arrow { color:var(--text-3); margin:0 5px; }
  .panel { background:var(--surface); border-radius:var(--radius); box-shadow:var(--shadow-sm); overflow:hidden; }
  .panel-head {
    padding:12px var(--s3); border-bottom:1px solid var(--hair);
    font-size:11px; font-weight:700; letter-spacing:.07em; text-transform:uppercase; color:var(--text-3);
  }
  .empty { color:var(--text-3); font-size:13px; padding:var(--s4); text-align:center; }

  /* ══ 客户端：全宽对话（行业做法：不用左右气泡）══ */
  .thread { max-width:768px; margin:0 auto; }
  .turn { padding:var(--s4) 0; border-top:1px solid var(--hair); animation:rise var(--t-slow) var(--ease) both; }
  .turn:first-child { border-top:0; padding-top:0; }
  .turn-head { display:flex; align-items:center; gap:8px; margin-bottom:10px; }
  .turn-badge {
    width:24px; height:24px; flex:none; border-radius:7px;
    display:flex; align-items:center; justify-content:center; font-size:11px; font-weight:700;
  }
  .turn.bot .turn-badge { background:linear-gradient(135deg,#2563EB,#60A5FA); color:#fff; box-shadow:0 2px 6px rgba(37,99,235,.26); }
  .turn.user .turn-badge { background:#E6ECF7; color:var(--text-2); }
  .turn-name { font-size:13px; font-weight:650; }
  .turn-time { font-size:11.5px; color:var(--text-3); font-variant-numeric:tabular-nums; }
  .turn-body { font-size:14.5px; line-height:1.72; white-space:pre-wrap; word-break:break-word; }
  .turn.user .turn-body { color:var(--text-2); }
  .turn.bot .turn-body.typing { color:var(--text-3); }
  .turn-meta { margin-top:10px; font-size:11.5px; color:var(--text-3); }
  .turn-meta a { color:var(--accent); text-decoration:none; font-weight:600; }
  .turn-meta a:hover { text-decoration:underline; }
  .composer {
    flex:none; border-top:1px solid var(--border);
    background:rgba(255,255,255,.86); backdrop-filter:blur(10px);
    padding:var(--s3) var(--s5) var(--s4);
  }
  .composer-inner { max-width:768px; margin:0 auto; }
  .composer-row { display:flex; gap:var(--s2); }
  .composer input[type=text] {
    flex:1; min-width:0; padding:12px 15px; font-size:14.5px; font-family:inherit;
    color:var(--text); background:#fff; border:0; border-radius:var(--radius-sm);
    box-shadow:inset 0 0 0 1px var(--border-strong); outline:none;
    transition:box-shadow var(--t-fast) var(--ease);
  }
  .composer input[type=text]:focus { box-shadow:inset 0 0 0 1px var(--accent), 0 0 0 4px rgba(37,99,235,.13); }
  .composer-hint { display:flex; flex-wrap:wrap; gap:var(--s1); align-items:center; margin-top:var(--s2); }
  .composer-hint .lbl { font-size:12px; color:var(--text-3); }

  /* ══ 服务端：观测台 ══ */
  .metrics { display:grid; grid-template-columns:repeat(auto-fit,minmax(170px,1fr)); gap:var(--s3); margin-bottom:var(--s3); }
  .metric { background:var(--surface); border-radius:var(--radius); padding:var(--s3) var(--s4); box-shadow:var(--shadow-sm); }
  .metric .lbl { font-size:11.5px; color:var(--text-3); margin-bottom:5px; font-weight:600; }
  .metric .val { font-size:25px; font-weight:750; font-variant-numeric:tabular-nums; letter-spacing:-.02em; }
  .metric .val.small { font-size:15px; font-weight:650; padding-top:6px; }
  .ops { display:grid; grid-template-columns:minmax(260px,340px) minmax(0,1fr); gap:var(--s3); align-items:start; }
  .list { max-height:calc(100vh - 300px); overflow-y:auto; }
  .row {
    padding:11px var(--s3); border-bottom:1px solid var(--hair); cursor:pointer;
    transition:background var(--t-fast) var(--ease);
  }
  .row:last-child { border-bottom:0; }
  .row:hover { background:var(--sunken); }
  .row.sel { background:var(--accent-soft); }
  .row-title { font-size:13px; color:var(--text); overflow:hidden; text-overflow:ellipsis; white-space:nowrap; }
  .row-meta { font-size:11px; color:var(--text-3); margin-top:3px; font-variant-numeric:tabular-nums; }
  .turn-card { border-bottom:1px solid var(--hair); }
  .turn-card:last-child { border-bottom:0; }
  .turn-card > summary {
    list-style:none; cursor:pointer; padding:var(--s3); display:block;
    transition:background var(--t-fast) var(--ease);
  }
  .turn-card > summary::-webkit-details-marker { display:none; }
  .turn-card > summary:hover { background:var(--sunken); }
  .turn-q { font-size:13.5px; font-weight:600; margin-bottom:5px; }
  .turn-sub { font-size:11.5px; color:var(--text-3); font-variant-numeric:tabular-nums; }
  .steps { padding:0 var(--s3) var(--s3); }
  .step { position:relative; padding-left:30px; padding-bottom:var(--s3); }
  .step:last-child { padding-bottom:2px; }
  .step::before { content:""; position:absolute; left:9px; top:24px; bottom:-2px; width:1px; background:var(--border); }
  .step:last-child::before { display:none; }
  .step-no {
    position:absolute; left:0; top:3px; width:20px; height:20px; border-radius:50%;
    background:var(--accent-soft); color:var(--accent); box-shadow:inset 0 0 0 1px rgba(37,99,235,.18);
    font-size:11px; font-weight:700; line-height:20px; text-align:center;
  }
  .step-head { display:flex; align-items:baseline; gap:var(--s2); flex-wrap:wrap; }
  .step-node { font-size:13.5px; font-weight:650; }
  .step-ms { font-size:12px; color:var(--accent); font-variant-numeric:tabular-nums; }
  .step-total { font-size:11.5px; color:var(--text-3); font-variant-numeric:tabular-nums; }
  .step-body { margin-top:var(--s2); display:flex; flex-direction:column; gap:var(--s2); }
  .item {
    background:var(--sunken); border-radius:var(--radius-sm);
    box-shadow:inset 0 0 0 1px var(--border); padding:var(--s2) var(--s3); font-size:12.5px;
  }
  .item-role { display:inline-block; font-size:11px; font-weight:700; color:var(--text-3); letter-spacing:.04em; margin-bottom:5px; }
  .call { display:flex; flex-direction:column; gap:4px; margin:4px 0; }
  .call-name { font-family:var(--mono); font-size:12.5px; font-weight:650; color:var(--accent); }
  .call-args {
    font-family:var(--mono); font-size:11.5px; color:var(--text-2); background:#fff;
    border-radius:6px; box-shadow:inset 0 0 0 1px var(--border);
    padding:6px 9px; white-space:pre-wrap; word-break:break-all;
  }
  .item-text {
    font-size:12.5px; color:var(--text-2); line-height:1.65; white-space:pre-wrap; word-break:break-word;
    max-height:160px; overflow-y:auto;
  }

  @keyframes rise { from { opacity:0; transform:translateY(7px); } to { opacity:1; transform:none; } }
  @media (prefers-reduced-motion: reduce) {
    *, *::before, *::after { animation:none !important; transition:none !important; }
  }
  @media (max-width: 900px) {
    .sidebar { width:64px; padding:var(--s3) 10px; }
    .brand-text, .nav-item span:not(.ico), .nav-label, .sidebar-foot, .topbar .sub { display:none; }
    .ops { grid-template-columns:1fr; }
    .content { padding:var(--s3); }
    .topbar { padding:0 var(--s3); }
    .composer { padding:var(--s3); }
  }
  @media (max-width: 560px) {
    body { font-size:14px; }
    .composer-row { flex-direction:column; }
    .btn-primary { width:100%; }
    .composer-hint { flex-direction:column; align-items:stretch; }
  }

  /* ══ rag 专用：问答卡 + 相似度条 ══ */
  .qa { max-width:768px; margin:0 auto; }
  .qa-input { display:flex; gap:var(--s2); }
  .qa-input input {
    flex:1; min-width:0; padding:12px 15px; font-size:14.5px; font-family:inherit;
    color:var(--text); background:#fff; border:0; border-radius:var(--radius-sm);
    box-shadow:inset 0 0 0 1px var(--border-strong); outline:none;
    transition:box-shadow var(--t-fast) var(--ease);
  }
  .qa-input input:focus { box-shadow:inset 0 0 0 1px var(--accent), 0 0 0 4px rgba(37,99,235,.13); }
  .presets { display:flex; flex-wrap:wrap; gap:var(--s1); align-items:center; margin-top:var(--s2); }
  .presets .lbl { font-size:12px; color:var(--text-3); }
  .hit { padding:var(--s2) var(--s3); border-bottom:1px solid var(--hair); }
  .hit:last-child { border-bottom:0; }
  .hit-head { display:flex; align-items:center; justify-content:space-between; gap:var(--s2); margin-bottom:6px; }
  .hit-doc { font-size:13px; font-weight:650; display:flex; align-items:center; gap:7px; }
  .hit-rank {
    width:19px; height:19px; border-radius:50%; background:var(--accent-soft); color:var(--accent);
    box-shadow:inset 0 0 0 1px rgba(37,99,235,.18);
    font-size:11px; font-weight:700; line-height:19px; text-align:center; flex:none;
  }
  .hit-score { font-family:var(--mono); font-size:12px; color:var(--accent); font-variant-numeric:tabular-nums; }
  .score-bar { height:5px; border-radius:3px; background:#E9EEF7; overflow:hidden; margin:7px 0 8px; }
  .score-fill { height:100%; border-radius:3px; background:linear-gradient(90deg,#60A5FA,#2563EB); }
  .hit-text { font-size:12.5px; color:var(--text-2); line-height:1.6; white-space:pre-wrap; word-break:break-word; max-height:120px; overflow-y:auto; }
  .src-tag { font-size:11.5px; background:#F1F4FA; box-shadow:inset 0 0 0 1px var(--border); border-radius:6px; padding:2px 8px; color:var(--text-2); }
</style>
</head>
<body>
<aside class="sidebar">
  <div class="brand">
    <div class="brand-logo">R</div>
    <div class="brand-text">
      <div class="brand-name">RAG 政策问答</div>
      <div class="brand-sub">Retrieval-Augmented</div>
    </div>
  </div>
  <nav class="nav">
    <div class="nav-label">工作台</div>
    <a class="nav-item active" href="/"><span class="ico">💬</span><span>政策问答</span></a>
    <a class="nav-item" href="/ops"><span class="ico">📊</span><span>运行观测</span></a>
  </nav>
  <div class="sidebar-foot">
    <div><span class="dot-ok"></span>服务正常</div>
    <div>build v2 · 应用壳</div>
  </div>
</aside>
<div class="main">
  <div class="topbar">
    <h1>政策问答</h1>
    <div class="sub">BGE-small 向量检索 + Top-5 片段 + 来源引用</div>
  </div>
  <div class="content" id="scroll">
    <div class="qa">
      <div class="qa-input">
        <input id="q" type="text" placeholder="输入售后政策问题，例如：退货需要什么条件" autocomplete="off" />
        <button id="askBtn" class="btn-primary" onclick="ask()">提问</button>
      </div>
      <div class="presets" id="presets"><span class="lbl">常问：</span></div>
    </div>
    <div class="qa" id="resultWrap" style="margin-top:var(--s4);display:none">
      <div class="panel" style="margin-bottom:var(--s3)">
        <div class="panel-head">回答</div>
        <div style="padding:var(--s3)">
          <div id="ans" style="font-size:14.5px;line-height:1.75;white-space:pre-wrap"></div>
          <div style="margin-top:var(--s3);padding-top:var(--s3);border-top:1px solid var(--hair);display:flex;gap:var(--s2);align-items:center;flex-wrap:wrap">
            <span style="font-size:12.5px;color:var(--text-3)">来源</span>
            <span id="srcs"></span>
          </div>
        </div>
      </div>
      <div class="panel">
        <div class="panel-head">检索命中（含相似度）</div>
        <div id="hits"></div>
      </div>
    </div>
  </div>
</div>
<script>
const $ = id => document.getElementById(id);
const PRESETS = ["退货需要什么条件", "退款多久到账", "换货的运费谁承担", "下单后多久能发货", "偏远地区要多久送到"];
function esc(s) {
  return String(s == null ? "" : s).replace(/&/g, "&amp;").replace(/</g, "&lt;")
    .replace(/>/g, "&gt;").replace(/"/g, "&quot;");
}
PRESETS.forEach(q => {
  const b = document.createElement("button");
  b.type = "button"; b.className = "chip"; b.textContent = q;
  b.onclick = () => { $("q").value = q; ask(); };
  $("presets").appendChild(b);
});
function renderHits(steps) {
  const ret = (steps || []).find(s => s.node === "retrieve");
  const items = (ret && ret.items) || [];
  if (!items.length) { $("hits").innerHTML = '<div class="empty">没有命中记录</div>'; return; }
  $("hits").innerHTML = items.map(it => {
    const pct = Math.max(0, Math.min(1, it.score || 0));
    return '<div class="hit">' +
      '<div class="hit-head">' +
        '<span class="hit-doc"><span class="hit-rank">' + (it.rank || "") + '</span>' + esc(it.name) + '</span>' +
        '<span class="hit-score">' + (it.score != null ? it.score.toFixed(4) : "-") + '</span>' +
      '</div>' +
      '<div class="score-bar"><div class="score-fill" style="width:' + (pct * 100).toFixed(1) + '%"></div></div>' +
      '<div class="hit-text">' + esc(it.content) + '</div>' +
    '</div>';
  }).join("");
}
async function ask() {
  const q = $("q").value.trim();
  if (!q) { $("q").focus(); return; }
  $("askBtn").disabled = true;
  $("askBtn").textContent = "检索中…";
  try {
    const resp = await fetch("/query", {
      method: "POST", headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ question: q })
    });
    const d = await resp.json();
    if (!resp.ok) { alert("失败：" + ((d && (d.error || d.detail)) || ("HTTP " + resp.status))); return; }
    $("resultWrap").style.display = "block";
    $("ans").textContent = d.answer || "（无回答）";
    $("srcs").innerHTML = (d.sources || []).map(s => '<span class="src-tag">' + esc(s) + "</span>").join("　") ||
      '<span class="badge badge-warn">未命中来源</span>';
    renderHits(d.steps);
    $("askBtn").textContent = "提问（" + (d.elapsed_ms / 1000).toFixed(1) + "s）";
    setTimeout(() => { $("askBtn").textContent = "提问"; }, 2500);
  } catch (e) {
    alert("网络或服务异常：" + (e && e.message ? e.message : e));
  } finally {
    $("askBtn").disabled = false;
    if ($("askBtn").textContent === "检索中…") $("askBtn").textContent = "提问";
  }
}
$("q").addEventListener("keydown", e => { if (e.key === "Enter") ask(); });
</script>
</body>
</html>"""


_OPS_PAGE = r"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>运行观测 · RAG 政策问答</title>
<style>
  :root {
    --bg:#F5F7FC; --surface:#FFFFFF; --sunken:#F7F9FD;
    --border:#E3E7F2; --border-strong:#CBD3E4; --hair:#EFF2F8;
    --text:#0F1520; --text-2:#3A4560; --text-3:#7A8BA0;
    --accent:#2563EB; --accent-hover:#1D4ED8; --accent-soft:#EEF3FF;
    --ok:#15803D; --ok-bg:#E7F6EC; --warn:#9A6700; --warn-bg:#FFF8E1;
    --err:#C1272D; --err-bg:#FDEBEC;
    --mono: ui-monospace, SFMono-Regular, "Cascadia Mono", Consolas, "Courier New", monospace;
    --s1:8px; --s2:12px; --s3:16px; --s4:24px; --s5:32px; --s6:48px;
    --radius:12px; --radius-sm:8px;
    --shadow-sm: 0 1px 2px rgba(20,45,110,.06), 0 0 0 1px rgba(20,45,110,.05);
    --shadow-md: 0 2px 4px rgba(20,45,110,.05), 0 8px 24px rgba(20,45,110,.10), 0 0 0 1px rgba(20,45,110,.06);
    --ease: cubic-bezier(.22,1,.36,1);
    --t-fast:140ms; --t-base:220ms; --t-slow:380ms;
  }
  * { box-sizing:border-box; margin:0; padding:0; }
  html, body { height:100%; }
  body {
    display:flex; overflow:hidden;
    background:var(--bg); color:var(--text);
    font-family:-apple-system,BlinkMacSystemFont,"Segoe UI Variable Display","Segoe UI","Microsoft YaHei","PingFang SC",sans-serif;
    font-size:14.5px; line-height:1.6; -webkit-font-smoothing:antialiased;
  }
  .sidebar {
    width:248px; flex:none; background:var(--surface);
    border-right:1px solid var(--border);
    display:flex; flex-direction:column; padding:var(--s4) var(--s3); gap:var(--s5);
  }
  .brand { display:flex; align-items:center; gap:10px; padding:0 6px; }
  .brand-logo {
    width:32px; height:32px; flex:none; border-radius:9px;
    background:linear-gradient(135deg,#2563EB,#60A5FA); color:#fff;
    display:flex; align-items:center; justify-content:center;
    font-weight:800; font-size:14px; box-shadow:0 2px 8px rgba(37,99,235,.32);
  }
  .brand-name { font-size:13.5px; font-weight:700; letter-spacing:-.01em; line-height:1.3; }
  .brand-sub { font-size:11px; color:var(--text-3); }
  .nav { display:flex; flex-direction:column; gap:2px; }
  .nav-label {
    font-size:10.5px; font-weight:700; letter-spacing:.09em; text-transform:uppercase;
    color:var(--text-3); padding:0 8px; margin-bottom:6px;
  }
  .nav-item {
    display:flex; align-items:center; gap:10px;
    padding:9px 10px; border-radius:var(--radius-sm);
    color:var(--text-2); text-decoration:none; font-size:13.5px; font-weight:500;
    transition:background var(--t-fast) var(--ease), color var(--t-fast) var(--ease);
  }
  .nav-item:hover { background:var(--sunken); color:var(--text); }
  .nav-item.active { background:var(--accent-soft); color:var(--accent); font-weight:650; }
  .nav-item .ico { width:17px; text-align:center; font-size:13.5px; flex:none; opacity:.9; }
  .sidebar-foot { margin-top:auto; padding:0 8px; font-size:11px; color:var(--text-3); line-height:1.9; }
  .dot-ok { display:inline-block; width:6px; height:6px; border-radius:50%; background:#22C55E; margin-right:5px; vertical-align:1px; }

  .main { flex:1; min-width:0; display:flex; flex-direction:column; }
  .topbar {
    height:58px; flex:none; border-bottom:1px solid var(--border);
    background:rgba(255,255,255,.78); backdrop-filter:blur(10px);
    display:flex; align-items:center; justify-content:space-between; gap:var(--s3);
    padding:0 var(--s5);
  }
  .topbar h1 { font-size:15.5px; font-weight:700; letter-spacing:-.01em; }
  .topbar .sub { font-size:12.5px; color:var(--text-3); }
  .content { flex:1; overflow-y:auto; padding:var(--s5); }

  .btn-primary {
    padding:11px 26px; font-size:14.5px; font-family:inherit; font-weight:600;
    color:#fff; border:0; border-radius:var(--radius-sm); cursor:pointer; white-space:nowrap;
    background:linear-gradient(180deg,#3B76F0,#2563EB);
    box-shadow:0 1px 2px rgba(37,99,235,.30), inset 0 1px 0 rgba(255,255,255,.22);
    transition:filter var(--t-fast) var(--ease), transform var(--t-fast) var(--ease), box-shadow var(--t-fast) var(--ease);
  }
  .btn-primary:hover:not(:disabled) { filter:brightness(1.07); }
  .btn-primary:active:not(:disabled) { transform:translateY(1px) scale(.995); box-shadow:inset 0 1px 3px rgba(0,0,0,.16); }
  .btn-primary:disabled { opacity:.45; cursor:not-allowed; }
  .chip {
    font-size:12.5px; font-family:inherit; color:var(--accent);
    background:var(--accent-soft); border:0; box-shadow:inset 0 0 0 1px rgba(37,99,235,.15);
    border-radius:999px; padding:5px 13px; cursor:pointer; text-align:left;
    transition:background var(--t-fast) var(--ease), box-shadow var(--t-fast) var(--ease), transform var(--t-fast) var(--ease);
  }
  .chip:hover { background:#E3ECFF; box-shadow:inset 0 0 0 1px rgba(37,99,235,.28); }
  .chip:active { transform:scale(.97); }
  .badge { display:inline-flex; align-items:center; gap:4px; font-size:11.5px; font-weight:650; padding:2px 9px; border-radius:999px; }
  .badge-ok { color:var(--ok); background:var(--ok-bg); box-shadow:inset 0 0 0 1px rgba(21,128,61,.16); }
  .badge-warn { color:var(--warn); background:var(--warn-bg); box-shadow:inset 0 0 0 1px rgba(154,103,0,.16); }
  .badge-plain { color:var(--text-2); background:#F1F4FA; box-shadow:inset 0 0 0 1px var(--border); }
  .tag-tool {
    font-family:var(--mono); font-size:12px; background:#F1F4FA;
    box-shadow:inset 0 0 0 1px var(--border); border-radius:6px; padding:2px 8px; color:var(--text-2);
  }
  .arrow { color:var(--text-3); margin:0 5px; }
  .panel { background:var(--surface); border-radius:var(--radius); box-shadow:var(--shadow-sm); overflow:hidden; }
  .panel-head {
    padding:12px var(--s3); border-bottom:1px solid var(--hair);
    font-size:11px; font-weight:700; letter-spacing:.07em; text-transform:uppercase; color:var(--text-3);
  }
  .empty { color:var(--text-3); font-size:13px; padding:var(--s4); text-align:center; }

  /* ══ 客户端：全宽对话（行业做法：不用左右气泡）══ */
  .thread { max-width:768px; margin:0 auto; }
  .turn { padding:var(--s4) 0; border-top:1px solid var(--hair); animation:rise var(--t-slow) var(--ease) both; }
  .turn:first-child { border-top:0; padding-top:0; }
  .turn-head { display:flex; align-items:center; gap:8px; margin-bottom:10px; }
  .turn-badge {
    width:24px; height:24px; flex:none; border-radius:7px;
    display:flex; align-items:center; justify-content:center; font-size:11px; font-weight:700;
  }
  .turn.bot .turn-badge { background:linear-gradient(135deg,#2563EB,#60A5FA); color:#fff; box-shadow:0 2px 6px rgba(37,99,235,.26); }
  .turn.user .turn-badge { background:#E6ECF7; color:var(--text-2); }
  .turn-name { font-size:13px; font-weight:650; }
  .turn-time { font-size:11.5px; color:var(--text-3); font-variant-numeric:tabular-nums; }
  .turn-body { font-size:14.5px; line-height:1.72; white-space:pre-wrap; word-break:break-word; }
  .turn.user .turn-body { color:var(--text-2); }
  .turn.bot .turn-body.typing { color:var(--text-3); }
  .turn-meta { margin-top:10px; font-size:11.5px; color:var(--text-3); }
  .turn-meta a { color:var(--accent); text-decoration:none; font-weight:600; }
  .turn-meta a:hover { text-decoration:underline; }
  .composer {
    flex:none; border-top:1px solid var(--border);
    background:rgba(255,255,255,.86); backdrop-filter:blur(10px);
    padding:var(--s3) var(--s5) var(--s4);
  }
  .composer-inner { max-width:768px; margin:0 auto; }
  .composer-row { display:flex; gap:var(--s2); }
  .composer input[type=text] {
    flex:1; min-width:0; padding:12px 15px; font-size:14.5px; font-family:inherit;
    color:var(--text); background:#fff; border:0; border-radius:var(--radius-sm);
    box-shadow:inset 0 0 0 1px var(--border-strong); outline:none;
    transition:box-shadow var(--t-fast) var(--ease);
  }
  .composer input[type=text]:focus { box-shadow:inset 0 0 0 1px var(--accent), 0 0 0 4px rgba(37,99,235,.13); }
  .composer-hint { display:flex; flex-wrap:wrap; gap:var(--s1); align-items:center; margin-top:var(--s2); }
  .composer-hint .lbl { font-size:12px; color:var(--text-3); }

  /* ══ 服务端：观测台 ══ */
  .metrics { display:grid; grid-template-columns:repeat(auto-fit,minmax(170px,1fr)); gap:var(--s3); margin-bottom:var(--s3); }
  .metric { background:var(--surface); border-radius:var(--radius); padding:var(--s3) var(--s4); box-shadow:var(--shadow-sm); }
  .metric .lbl { font-size:11.5px; color:var(--text-3); margin-bottom:5px; font-weight:600; }
  .metric .val { font-size:25px; font-weight:750; font-variant-numeric:tabular-nums; letter-spacing:-.02em; }
  .metric .val.small { font-size:15px; font-weight:650; padding-top:6px; }
  .ops { display:grid; grid-template-columns:minmax(260px,340px) minmax(0,1fr); gap:var(--s3); align-items:start; }
  .list { max-height:calc(100vh - 300px); overflow-y:auto; }
  .row {
    padding:11px var(--s3); border-bottom:1px solid var(--hair); cursor:pointer;
    transition:background var(--t-fast) var(--ease);
  }
  .row:last-child { border-bottom:0; }
  .row:hover { background:var(--sunken); }
  .row.sel { background:var(--accent-soft); }
  .row-title { font-size:13px; color:var(--text); overflow:hidden; text-overflow:ellipsis; white-space:nowrap; }
  .row-meta { font-size:11px; color:var(--text-3); margin-top:3px; font-variant-numeric:tabular-nums; }
  .turn-card { border-bottom:1px solid var(--hair); }
  .turn-card:last-child { border-bottom:0; }
  .turn-card > summary {
    list-style:none; cursor:pointer; padding:var(--s3); display:block;
    transition:background var(--t-fast) var(--ease);
  }
  .turn-card > summary::-webkit-details-marker { display:none; }
  .turn-card > summary:hover { background:var(--sunken); }
  .turn-q { font-size:13.5px; font-weight:600; margin-bottom:5px; }
  .turn-sub { font-size:11.5px; color:var(--text-3); font-variant-numeric:tabular-nums; }
  .steps { padding:0 var(--s3) var(--s3); }
  .step { position:relative; padding-left:30px; padding-bottom:var(--s3); }
  .step:last-child { padding-bottom:2px; }
  .step::before { content:""; position:absolute; left:9px; top:24px; bottom:-2px; width:1px; background:var(--border); }
  .step:last-child::before { display:none; }
  .step-no {
    position:absolute; left:0; top:3px; width:20px; height:20px; border-radius:50%;
    background:var(--accent-soft); color:var(--accent); box-shadow:inset 0 0 0 1px rgba(37,99,235,.18);
    font-size:11px; font-weight:700; line-height:20px; text-align:center;
  }
  .step-head { display:flex; align-items:baseline; gap:var(--s2); flex-wrap:wrap; }
  .step-node { font-size:13.5px; font-weight:650; }
  .step-ms { font-size:12px; color:var(--accent); font-variant-numeric:tabular-nums; }
  .step-total { font-size:11.5px; color:var(--text-3); font-variant-numeric:tabular-nums; }
  .step-body { margin-top:var(--s2); display:flex; flex-direction:column; gap:var(--s2); }
  .item {
    background:var(--sunken); border-radius:var(--radius-sm);
    box-shadow:inset 0 0 0 1px var(--border); padding:var(--s2) var(--s3); font-size:12.5px;
  }
  .item-role { display:inline-block; font-size:11px; font-weight:700; color:var(--text-3); letter-spacing:.04em; margin-bottom:5px; }
  .call { display:flex; flex-direction:column; gap:4px; margin:4px 0; }
  .call-name { font-family:var(--mono); font-size:12.5px; font-weight:650; color:var(--accent); }
  .call-args {
    font-family:var(--mono); font-size:11.5px; color:var(--text-2); background:#fff;
    border-radius:6px; box-shadow:inset 0 0 0 1px var(--border);
    padding:6px 9px; white-space:pre-wrap; word-break:break-all;
  }
  .item-text {
    font-size:12.5px; color:var(--text-2); line-height:1.65; white-space:pre-wrap; word-break:break-word;
    max-height:160px; overflow-y:auto;
  }

  @keyframes rise { from { opacity:0; transform:translateY(7px); } to { opacity:1; transform:none; } }
  @media (prefers-reduced-motion: reduce) {
    *, *::before, *::after { animation:none !important; transition:none !important; }
  }
  @media (max-width: 900px) {
    .sidebar { width:64px; padding:var(--s3) 10px; }
    .brand-text, .nav-item span:not(.ico), .nav-label, .sidebar-foot, .topbar .sub { display:none; }
    .ops { grid-template-columns:1fr; }
    .content { padding:var(--s3); }
    .topbar { padding:0 var(--s3); }
    .composer { padding:var(--s3); }
  }
  @media (max-width: 560px) {
    body { font-size:14px; }
    .composer-row { flex-direction:column; }
    .btn-primary { width:100%; }
    .composer-hint { flex-direction:column; align-items:stretch; }
  }

  /* ══ rag 专用：问答卡 + 相似度条 ══ */
  .qa { max-width:768px; margin:0 auto; }
  .qa-input { display:flex; gap:var(--s2); }
  .qa-input input {
    flex:1; min-width:0; padding:12px 15px; font-size:14.5px; font-family:inherit;
    color:var(--text); background:#fff; border:0; border-radius:var(--radius-sm);
    box-shadow:inset 0 0 0 1px var(--border-strong); outline:none;
    transition:box-shadow var(--t-fast) var(--ease);
  }
  .qa-input input:focus { box-shadow:inset 0 0 0 1px var(--accent), 0 0 0 4px rgba(37,99,235,.13); }
  .presets { display:flex; flex-wrap:wrap; gap:var(--s1); align-items:center; margin-top:var(--s2); }
  .presets .lbl { font-size:12px; color:var(--text-3); }
  .hit { padding:var(--s2) var(--s3); border-bottom:1px solid var(--hair); }
  .hit:last-child { border-bottom:0; }
  .hit-head { display:flex; align-items:center; justify-content:space-between; gap:var(--s2); margin-bottom:6px; }
  .hit-doc { font-size:13px; font-weight:650; display:flex; align-items:center; gap:7px; }
  .hit-rank {
    width:19px; height:19px; border-radius:50%; background:var(--accent-soft); color:var(--accent);
    box-shadow:inset 0 0 0 1px rgba(37,99,235,.18);
    font-size:11px; font-weight:700; line-height:19px; text-align:center; flex:none;
  }
  .hit-score { font-family:var(--mono); font-size:12px; color:var(--accent); font-variant-numeric:tabular-nums; }
  .score-bar { height:5px; border-radius:3px; background:#E9EEF7; overflow:hidden; margin:7px 0 8px; }
  .score-fill { height:100%; border-radius:3px; background:linear-gradient(90deg,#60A5FA,#2563EB); }
  .hit-text { font-size:12.5px; color:var(--text-2); line-height:1.6; white-space:pre-wrap; word-break:break-word; max-height:120px; overflow-y:auto; }
  .src-tag { font-size:11.5px; background:#F1F4FA; box-shadow:inset 0 0 0 1px var(--border); border-radius:6px; padding:2px 8px; color:var(--text-2); }
</style>
</head>
<body>
<aside class="sidebar">
  <div class="brand">
    <div class="brand-logo">R</div>
    <div class="brand-text">
      <div class="brand-name">RAG 政策问答</div>
      <div class="brand-sub">Retrieval-Augmented</div>
    </div>
  </div>
  <nav class="nav">
    <div class="nav-label">工作台</div>
    <a class="nav-item" href="/"><span class="ico">💬</span><span>政策问答</span></a>
    <a class="nav-item active" href="/ops"><span class="ico">📊</span><span>运行观测</span></a>
  </nav>
  <div class="sidebar-foot">
    <div><span class="dot-ok"></span>服务正常</div>
    <div>build v2 · 应用壳</div>
  </div>
</aside>
<div class="main">
  <div class="topbar">
    <h1>运行观测</h1>
    <div class="sub">检索命中 · 相似度分数 · 来源分布</div>
  </div>
  <div class="content">
    <div class="metrics" id="metrics"></div>
    <div class="ops">
      <div class="panel">
        <div class="panel-head">问答记录</div>
        <div class="list" id="list"><div class="empty">加载中…</div></div>
      </div>
      <div class="panel">
        <div class="panel-head">检索详情 · 命中片段与相似度</div>
        <div id="detail"><div class="empty">从左侧选择一条提问，查看它的检索命中情况</div></div>
      </div>
    </div>
  </div>
</div>
<script>
const $ = id => document.getElementById(id);
function esc(s) {
  return String(s == null ? "" : s).replace(/&/g, "&amp;").replace(/</g, "&lt;")
    .replace(/>/g, "&gt;").replace(/"/g, "&quot;");
}
let STATS = null, SEL = null;
let METRIC_SIG = "", LIST_SIG = "", DETAIL_SIG = "";

function sigList(d) { return JSON.stringify((d.recent || []).map(r => [r.id, r.elapsed_ms])); }
function sigDetail(d) {
  const r = (d.recent || []).find(x => x.id === SEL);
  return String(SEL) + "|" + (r ? r.elapsed_ms + "|" + ((r.steps || []).length) : "none");
}
function avgHits() {
  const recs = (STATS && STATS.recent) || [];
  if (!recs.length) return "—";
  let n = 0, c = 0;
  recs.forEach(r => {
    const ret = (r.steps || []).find(s => s.node === "retrieve");
    if (ret) { n += (ret.items || []).length; c += 1; }
  });
  return c ? (n / c).toFixed(1) + " 片" : "—";
}

function renderMetrics() {
  const s = STATS || {};
  const hits = s.doc_hits || {};
  const hitStr = Object.keys(hits).length
    ? Object.entries(hits).map(([k, v]) => k + " ×" + v).join("　") : "—";
  const items = [
    { lbl: "累计提问", val: (s.total || 0) },
    { lbl: "平均耗时", val: s.avg_elapsed_ms ? (s.avg_elapsed_ms / 1000).toFixed(1) + "s" : "—" },
    { lbl: "平均命中片段", val: avgHits() },
    { lbl: "来源文档分布", val: hitStr, small: true }
  ];
  $("metrics").innerHTML = items.map(i =>
    '<div class="metric"><div class="lbl">' + esc(i.lbl) + '</div>' +
    '<div class="val' + (i.small ? " small" : "") + '">' + esc(i.val) + '</div></div>').join("");
}

function renderList() {
  const list = (STATS && STATS.recent) || [];
  if (!list.length) {
    $("list").innerHTML = '<div class="empty">还没有提问记录<br/>去「政策问答」问一条就会有</div>';
    return;
  }
  $("list").innerHTML = list.map(r =>
    '<div class="row' + (r.id === SEL ? " sel" : "") + '" data-id="' + r.id + '">' +
      '<div class="row-title">' + esc(r.question) + '</div>' +
      '<div class="row-meta">' + (r.elapsed_ms / 1000).toFixed(1) + 's　·　' +
        ((r.sources || []).length) + ' 个来源　·　' + esc(r.created_at) + '</div>' +
    '</div>').join("");
  Array.from($("list").children).forEach(el => {
    el.onclick = () => {
      SEL = parseInt(el.dataset.id, 10);
      DETAIL_SIG = sigDetail(STATS);
      renderList(); renderDetail();
    };
  });
}

function renderDetail() {
  const rec = ((STATS && STATS.recent) || []).find(r => r.id === SEL);
  if (!rec) { $("detail").innerHTML = '<div class="empty">从左侧选择一条提问，查看它的检索命中情况</div>'; return; }
  const ret = (rec.steps || []).find(s => s.node === "retrieve");
  const gen = (rec.steps || []).find(s => s.node === "generate");
  const items = (ret && ret.items) || [];

  const head =
    '<div style="padding:var(--s3);border-bottom:1px solid var(--hair)">' +
      '<div style="font-size:13.5px;font-weight:600;margin-bottom:8px">' + esc(rec.question) + '</div>' +
      '<div style="font-size:11.5px;color:var(--text-3)">' + esc(rec.created_at) + '　·　' +
        (rec.elapsed_ms / 1000).toFixed(1) + 's' +
        (ret ? '　·　检索 ' + (ret.step_ms / 1000).toFixed(1) + 's' : '') +
        (gen ? '　·　生成 ' + (gen.step_ms / 1000).toFixed(1) + 's' : '') +
        '　·　命中 ' + items.length + ' 片</div>' +
    '</div>';

  const body = items.length
    ? items.map(it => {
        const pct = Math.max(0, Math.min(1, it.score || 0));
        return '<div class="hit">' +
          '<div class="hit-head">' +
            '<span class="hit-doc"><span class="hit-rank">' + (it.rank || "") + '</span>' + esc(it.name) + '</span>' +
            '<span class="hit-score">相似度 ' + (it.score != null ? it.score.toFixed(4) : "-") + '</span>' +
          '</div>' +
          '<div class="score-bar"><div class="score-fill" style="width:' + (pct * 100).toFixed(1) + '%"></div></div>' +
          '<div class="hit-text">' + esc(it.content) + '</div>' +
        '</div>';
      }).join("")
    : '<div class="empty">这条记录没有检索详情（本次改造前的历史记录）</div>';

  const genBlock = gen
    ? '<div style="padding:var(--s3);border-top:1px solid var(--hair)">' +
        '<div class="panel-head" style="padding:0 0 8px;border:0">生成答案</div>' +
        '<div style="font-size:13px;color:var(--text-2);line-height:1.7;white-space:pre-wrap">' +
          esc((gen.items && gen.items[0] && gen.items[0].content) || "") + '</div>' +
      '</div>'
    : "";

  $("detail").innerHTML = head + body + genBlock;
}

async function load() {
  try {
    const r = await fetch("/stats");
    const data = await r.json();
    STATS = data;
    if (!SEL && data.recent && data.recent.length) SEL = data.recent[0].id;

    const mSig = JSON.stringify([data.total, data.avg_elapsed_ms, data.doc_hits]);
    if (mSig !== METRIC_SIG) { METRIC_SIG = mSig; renderMetrics(); }

    const lSig = sigList(data);
    if (lSig !== LIST_SIG) { LIST_SIG = lSig; renderList(); }

    const dSig = sigDetail(data);
    if (dSig !== DETAIL_SIG) { DETAIL_SIG = dSig; renderDetail(); }
  } catch (e) {
    $("list").innerHTML = '<div class="empty">加载失败：' + esc(e.message) + '</div>';
  }
}
load();
setInterval(load, 5000);
</script>
</body>
</html>"""
