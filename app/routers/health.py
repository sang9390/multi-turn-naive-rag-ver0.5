# app/routers/health.py
from __future__ import annotations

from fastapi import APIRouter, HTTPException
from ..core.runtime import runtime
from ..core.config import global_settings as C

router = APIRouter()

@router.get("/health")
async def health():
    try:
        models = {
            "generator": {"provider": C.GENERATOR_PROVIDER, "model": C.LLM_MODEL},
            "embed": {"provider": C.EMBED_PROVIDER, "model": C.EMBED_MODEL},
        }
        index_info = {
            "loaded": runtime.has_index,
            "vector_store_dir": runtime.persist_dir or (C.FAISS_LOAD_DIR or C.VECTOR_STORE_DIR),
            "bm25": True,  # BM25 노드 로드 여부를 더 정확히 원하면 런타임에 공개 프로퍼티 추가 가능
        }
        return {
            "status": "ok",
            "version": "20250929_runtime_lazyload",
            "models": models,
            "index": index_info,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
