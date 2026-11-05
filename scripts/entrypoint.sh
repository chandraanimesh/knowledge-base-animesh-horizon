#!/bin/sh
set -e

echo "[RAG Engine] Starting backend initialization..."

# 1. Run database migrations
echo "[RAG Engine] Running Alembic database migrations..."
alembic upgrade head

# 2. Seed default administrator and canonical departments
echo "[RAG Engine] Seeding administrator and canonical departments..."
python scripts/seed_admin.py || echo "[RAG Engine] Admin seeding completed or already initialized."

# 3. Start production ASGI server
echo "[RAG Engine] Starting Uvicorn on 0.0.0.0:8000..."
exec uvicorn rag_app.main:app --host 0.0.0.0 --port 8000

