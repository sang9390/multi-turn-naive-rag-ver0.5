#!/usr/bin/env bash
set -e

# .env를 안전하게 로드
if [ -f .env ]; then
    set -a
    source <(grep -v '^#' .env | grep -v '^$' | sed 's/\r$//')
    set +a
fi

# 기본값 설정
HOST=${HOST:-0.0.0.0}
PORT=${PORT:-8000}
WORKERS=${WORKERS:-1}

export PYTHONUNBUFFERED=1

echo "Starting uvicorn on ${HOST}:${PORT} (workers=${WORKERS})"
uvicorn main:app --host ${HOST} --port ${PORT} --workers ${WORKERS}
