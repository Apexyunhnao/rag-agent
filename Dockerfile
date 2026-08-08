FROM python:3.10-slim

WORKDIR /app

# 系统依赖（ChromaDB + sentence-transformers 需要）
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential curl \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# 预下载嵌入模型（避免首次查询时下载）
RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('BAAI/bge-small-zh-v1.5')" 2>/dev/null; true

# 入库（如果 chroma_db 不存在则创建）
RUN if [ ! -d "data/chroma_db" ] || [ ! -f "data/chroma_db/chroma.sqlite3" ]; then python ingest.py; fi

EXPOSE 8002
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8002"]
