import asyncio
import os
import logging
from fastapi import APIRouter, HTTPException
from llama_index.core import VectorStoreIndex, StorageContext, Document
from llama_index.vector_stores.faiss import FaissVectorStore
from llama_index.core.schema import TextNode
from llama_index.core.node_parser import MarkdownNodeParser
from llama_index.core.storage.docstore import SimpleDocumentStore
from llama_index.core.storage.index_store import SimpleIndexStore

from ..core.models import SessionInitRequest, SessionSwitchRequest, SessionResponse
from ..core.session_manager import session_manager
from ..services.summarizer import build_summaries
from ..core.config import global_settings as C

router = APIRouter(prefix="/session", tags=["session"])
logger = logging.getLogger(__name__)


@router.post("/init")
async def init_session(req: SessionInitRequest):
    """세션 초기화 (캐시 디렉토리만 준비)"""
    try:
        cache = session_manager.init_session(req.session_id)
        return SessionResponse(status="ok", session_id=req.session_id)
    except Exception as e:
        logger.exception("session_init failed")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/switch")
async def switch_session(req: SessionSwitchRequest):
    """세션 스위치/동기화 (메시지 이력 기반 요약/인덱스 재구성)"""
    try:
        # 1. 요약 생성
        summary_all, summary_recent5 = await asyncio.to_thread(
            build_summaries, req.messages
        )

        logger.info(f"Session {req.session_id} summaries generated")

        # 2. QA 청킹 및 임베딩
        nodes = []
        parser = MarkdownNodeParser()
        for msg in req.messages:
            qa_text = f"Q: {msg.user_query}\nA: {msg.rag_answer}"
            doc = Document(
                text=qa_text,
                metadata={"turn_id": msg.id, "created_at": msg.created_at}
            )
            nodes.extend(parser.get_nodes_from_documents([doc]))

        # 3. FAISS 인덱스 구성
        index = None
        bm25_nodes = []
        if nodes:
            try:
                import faiss
                from llama_index.core import Settings

                # 차원 프로브
                probe_vec = Settings.embed_model.get_text_embedding("probe")
                dim = len(probe_vec)

                faiss_index = faiss.IndexFlatL2(dim)
                vs = FaissVectorStore(faiss_index=faiss_index)
                docstore = SimpleDocumentStore()
                index_store = SimpleIndexStore()
                sc = StorageContext.from_defaults(
                    vector_store=vs,
                    docstore=docstore,
                    index_store=index_store
                )

                index = VectorStoreIndex(nodes, storage_context=sc, show_progress=False)

                # 세션 캐시에 저장
                cache = session_manager.get_or_create(req.session_id)
                cache_dir = cache.cache_dir
                os.makedirs(cache_dir, exist_ok=True)
                sc.persist(persist_dir=cache_dir)

                bm25_nodes = nodes  # BM25용
                logger.info(f"Session {req.session_id} indexed {len(nodes)} nodes")
            except Exception as e:
                logger.warning(f"Session index build failed: {e}")

        # 4. 세션 업데이트
        session_manager.update_session(
            session_id=req.session_id,
            summary_all=summary_all,
            summary_recent5=summary_recent5,
            index=index,
            bm25_nodes=bm25_nodes
        )

        return SessionResponse(status="ok", session_id=req.session_id)

    except Exception as e:
        logger.exception("session_switch failed")
        raise HTTPException(status_code=500, detail=str(e))