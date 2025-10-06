# app/routers/document.py
from __future__ import annotations

import asyncio
from pathlib import Path
from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse
from ..core.models import DocumentRequest, DocumentResponse
from ..services.indexing import index_path
from ..core.runtime import runtime

router = APIRouter()


@router.post("/document")
async def upsert_document(req: DocumentRequest):
    """
    Build/Load index and inject it into runtime.
    """
    try:
        index, bm25_nodes, nfiles, nnodes, persist_dir = await asyncio.to_thread(
            index_path, req.path, req.recursive, req.rebuild, req.file_types
        )
        # Inject to runtime so /query can use it.
        runtime.set_index(index=index, bm25_nodes=bm25_nodes, persist_dir=persist_dir)

        return DocumentResponse(
            status="ok",
            indexed_files=nfiles,
            nodes=nnodes,
            persist_dir=persist_dir or "",
            hybrid_enabled=bool(bm25_nodes),
        ).model_dump()
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/document/view")
async def view_document(path: str):
    """
    문서 뷰어: 파일 경로 → 바이너리/텍스트 전송

    Example: GET /document/view?path=/data/railway_md/file.md
    """
    try:
        file_path = Path(path).expanduser().resolve()
        if not file_path.exists() or not file_path.is_file():
            raise HTTPException(status_code=404, detail="File not found")

        return FileResponse(file_path)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
