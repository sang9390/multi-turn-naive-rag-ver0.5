from __future__ import annotations

import asyncio
import os
import time
import logging
from typing import Any, Dict, Iterable, List, Optional, Tuple

from llama_index.core import (
    Settings,
    VectorStoreIndex,
    StorageContext,
    load_index_from_storage,
)
from llama_index.core.schema import NodeWithScore, TextNode
from llama_index.vector_stores.faiss import FaissVectorStore
from llama_index.core.storage.docstore import SimpleDocumentStore
from llama_index.core.storage.index_store import SimpleIndexStore

from ..core.config import global_settings as C
from ..core.session_manager import session_manager
from ..services.generation import (
    init_generation_models,
    sglang_stream,
    build_context,
    QA_TMPL,
    _apply_reveal_from_filter,
)
from ..services.retrieval import build_retriever, postprocess_nodes
from ..services.indexing import load_nodes_jsonl
from ..services.query_repair import repair_query

logger = logging.getLogger(__name__)


def _norm_event(ev: str) -> str:
    e = (ev or "").lower()
    if e in {"delta", "text", "token", "message"}:
        return "content"
    if e in {"thought", "chain_of_thought", "cot", "reasoning"}:
        return "reasoning"
    if e in {"first_token", "ttft"}:
        return "ttft"
    if e in {"end", "complete", "completion", "done"}:
        return "done"
    return e or "content"


class Runtime:
    def __init__(self) -> None:
        self._llm = None
        self._embed = None
        self._index: Optional[VectorStoreIndex] = None
        self._bm25_nodes: List[TextNode] = []
        self._retriever = None
        self._persist_dir: Optional[str] = None
        self._sem = asyncio.Semaphore(C.MAX_CONCURRENCY)
        self._lazyload_attempted = False

    async def init(self) -> None:
        self._llm, self._embed = init_generation_models()
        Settings.embed_model = self._embed
        try:
            Settings.llm = self._llm
        except Exception:
            pass

    def set_index(
            self,
            index: VectorStoreIndex,
            bm25_nodes: List[TextNode],
            persist_dir: Optional[str] = None,
    ) -> None:
        self._index = index
        self._bm25_nodes = bm25_nodes or []
        self._persist_dir = persist_dir
        try:
            self._retriever = build_retriever(self._index, self._bm25_nodes)
        except Exception as e:
            logger.warning("build_retriever failed, fallback to as_retriever(): %s", e)
            try:
                self._retriever = self._index.as_retriever(similarity_top_k=C.SIMILARITY_TOP_K)
            except Exception as e2:
                self._retriever = None
                raise RuntimeError(f"Failed to initialize retriever: {e2}") from e

    @property
    def has_index(self) -> bool:
        return self._index is not None and self._retriever is not None

    @property
    def persist_dir(self) -> Optional[str]:
        return self._persist_dir

    def _try_lazy_load_index(self) -> None:
        if self._index is not None and self._retriever is not None:
            return
        if self._lazyload_attempted:
            return
        self._lazyload_attempted = True

        candidates: List[str] = []
        if C.FAISS_LOAD_DIR:
            candidates.append(C.FAISS_LOAD_DIR)
        if C.VECTOR_STORE_DIR:
            candidates.append(C.VECTOR_STORE_DIR)

        last_err: Optional[Exception] = None

        for d in candidates:
            if not d:
                continue
            abs_dir = os.path.abspath(os.path.expanduser(d))
            try:
                if not os.path.isdir(abs_dir):
                    logger.info("[lazyload] skip non-dir: %s", abs_dir)
                    continue
                logger.info("[lazyload] trying persist_dir: %s", abs_dir)

                docstore = SimpleDocumentStore.from_persist_dir(abs_dir)
                index_store = SimpleIndexStore.from_persist_dir(abs_dir)
                vs = FaissVectorStore.from_persist_dir(abs_dir)
                sc = StorageContext.from_defaults(docstore=docstore, index_store=index_store, vector_store=vs)
                index = load_index_from_storage(sc)

                nodes_path = os.path.join(abs_dir, "nodes.jsonl")
                bm25_nodes = load_nodes_jsonl(nodes_path) if os.path.exists(nodes_path) else []
                self.set_index(index=index, bm25_nodes=bm25_nodes, persist_dir=abs_dir)
                logger.info("[lazyload] success: %s (bm25_nodes=%d)", abs_dir, len(bm25_nodes))
                return
            except Exception as e:
                last_err = e
                logger.exception("[lazyload] failed at %s", abs_dir)

        if self._index is None or self._retriever is None:
            raise RuntimeError(
                f"Lazy-load failed. Checked: {[os.path.abspath(os.path.expanduser(x)) for x in candidates if x]}. "
                f"Last error: {last_err}"
            )

    def _ensure_ready(self) -> None:
        if self._index is None or self._retriever is None:
            self._try_lazy_load_index()
        if self._index is None or self._retriever is None:
            raise RuntimeError("Index not loaded. POST /document first or set FAISS_LOAD_DIR")

    async def _retrieve(self, query: str, top_k: int) -> List[NodeWithScore]:
        def _run() -> List[NodeWithScore]:
            nodes = self._retriever.retrieve(query)  # type: ignore
            nodes = postprocess_nodes(nodes)
            return nodes[: max(top_k, 1)]

        return await asyncio.to_thread(_run)

    async def _retrieve_from_session(
            self,
            session_id: int,
            query: str,
            top_k: int
    ) -> List[Tuple[int, str]]:
        """세션 QA 이력에서 검색"""
        cache = session_manager.get_or_create(session_id)
        if not cache.index:
            return []

        try:
            retriever = build_retriever(cache.index, cache.bm25_nodes or [])
            nodes = await asyncio.to_thread(
                lambda: postprocess_nodes(retriever.retrieve(query))[:top_k]
            )
            return [
                (node.metadata.get("turn_id", -1), node.get_content())
                for node in nodes
            ]
        except Exception as e:
            logger.warning("Session retrieval failed: %s", e)
            return []

    async def handle_query(self, req) -> Dict[str, Any]:
        async with self._sem:
            self._ensure_ready()
            t0 = time.monotonic()

            # ✨ 멀티홉 쿼리 증강 (세션 있을 때만)
            repair_context = None
            final_query = req.query

            if req.session_id and not req.eval_mode and C.ENABLE_QUERY_REPAIR:
                cache = session_manager.get_or_create(req.session_id)

                # 이전 QA 검색
                topk_chunks = await self._retrieve_from_session(
                    req.session_id, req.query, top_k=3
                )

                # 쿼리 리페어
                repair_context = await asyncio.to_thread(
                    repair_query,
                    user_query=req.query,
                    summary_all=cache.summary_all,
                    summary_recent5=cache.summary_recent5,
                    topk_chunks=topk_chunks
                )

                # 개선된 쿼리 사용
                if repair_context.improved_query:
                    final_query = repair_context.improved_query
                    logger.info(f"Query repaired: {req.query} → {final_query}")

            # 문서 검색
            nodes = await self._retrieve(final_query, req.top_k or C.SIMILARITY_TOP_K)
            context_str, files, contexts = build_context(nodes)
            prompt = QA_TMPL.format(context_str=context_str, query_str=req.query)

            enable_thinking = (req.think_mode or "off").lower() == "on"
            separate_reasoning = bool(req.include_reasoning and enable_thinking)

            answer_parts: List[str] = []
            reasoning_parts: List[str] = []
            ttft_sec: Optional[float] = None
            text_gen_start: Optional[float] = None
            text_gen_end: Optional[float] = None
            think_first: Optional[float] = None
            think_last: Optional[float] = None

            for event, data in sglang_stream(
                    prompt=prompt,
                    model=C.LLM_MODEL,
                    enable_thinking=enable_thinking,
                    separate_reasoning=separate_reasoning,
                    temperature=C.GEN_TEMPERATURE,
                    max_tokens=C.GEN_MAX_OUTPUT_TOKENS,
                    timeout_sec=C.GEN_TIMEOUT_SEC,
            ):
                ev = _norm_event(event)
                if ev == "ttft":
                    try:
                        ttft_sec = float(data)
                    except Exception:
                        ttft_sec = None
                    text_gen_start = time.monotonic()
                elif ev == "content":
                    answer_parts.append(str(data))
                    text_gen_end = time.monotonic()
                elif ev == "reasoning":
                    reasoning_parts.append(str(data))
                    if think_first is None:
                        think_first = time.monotonic()
                    think_last = time.monotonic()

            answer = "".join(answer_parts).strip()
            reasoning = "".join(reasoning_parts).strip() if separate_reasoning else None

            # REVEAL_FROM 필터 적용 (논스트리밍)
            if C.REVEAL_FROM_ENABLED and not enable_thinking:
                answer = _apply_reveal_from_filter(answer, C.REVEAL_FROM_TOKEN, C.REVEAL_FALLBACK)

            timing = {
                "retrieval_sec": time.monotonic() - t0,
                "ttft_sec": ttft_sec,
                "text_gen_sec": (text_gen_end - text_gen_start) if (text_gen_start and text_gen_end) else None,
                "think_total_sec": (think_last - think_first) if (think_first and think_last) else None,
            }

            return {
                "answer": answer,
                "reasoning": reasoning,
                "contexts": contexts,
                "timing": timing,
                "files": files,
                "repair_context": repair_context.model_dump() if repair_context else None,
                "used_query": final_query if final_query != req.query else None,
            }

    async def handle_query_stream(self, req) -> Iterable[Tuple[str, Any]]:
        self._ensure_ready()

        # ✨ 멀티홉 쿼리 증강
        final_query = req.query
        if req.session_id and not req.eval_mode and C.ENABLE_QUERY_REPAIR:
            cache = session_manager.get_or_create(req.session_id)
            topk_chunks = await self._retrieve_from_session(req.session_id, req.query, top_k=3)
            repair_context = await asyncio.to_thread(
                repair_query,
                user_query=req.query,
                summary_all=cache.summary_all,
                summary_recent5=cache.summary_recent5,
                topk_chunks=topk_chunks
            )
            if repair_context.improved_query:
                final_query = repair_context.improved_query

        nodes = await self._retrieve(final_query, req.top_k or C.SIMILARITY_TOP_K)
        context_str, _files, _contexts = build_context(nodes)
        prompt = QA_TMPL.format(context_str=context_str, query_str=req.query)

        enable_thinking = (req.think_mode or "off").lower() == "on"
        separate_reasoning = bool(req.include_reasoning and enable_thinking)

        queue: asyncio.Queue[Optional[Tuple[str, Any]]] = asyncio.Queue(maxsize=100)
        loop = asyncio.get_running_loop()

        def _producer(_loop: asyncio.AbstractEventLoop) -> None:
            ttft_sec: Optional[float] = None
            text_gen_start: Optional[float] = None
            text_gen_end: Optional[float] = None
            think_first: Optional[float] = None
            think_last: Optional[float] = None
            sent_content = False

            try:
                for raw_event, data in sglang_stream(
                        prompt=prompt,
                        model=C.LLM_MODEL,
                        enable_thinking=enable_thinking,
                        separate_reasoning=separate_reasoning,
                        temperature=C.GEN_TEMPERATURE,
                        max_tokens=C.GEN_MAX_OUTPUT_TOKENS,
                        timeout_sec=C.GEN_TIMEOUT_SEC,
                ):
                    ev = _norm_event(raw_event)

                    if ev == "ttft":
                        try:
                            ttft_sec = float(data)
                        except Exception:
                            ttft_sec = None
                        text_gen_start = time.monotonic()

                    if ev == "content":
                        sent_content = True
                        text_gen_end = time.monotonic()

                    if ev == "reasoning":
                        now = time.monotonic()
                        if think_first is None:
                            think_first = now
                        think_last = now
                        if C.STREAM_COMPAT_DUP_CONTENT:
                            asyncio.run_coroutine_threadsafe(queue.put(("content", data)), _loop)
                            sent_content = True

                    asyncio.run_coroutine_threadsafe(queue.put((ev, data)), _loop)

                if not sent_content and C.STREAM_COMPAT_DUP_CONTENT:
                    asyncio.run_coroutine_threadsafe(queue.put(("content", "")), _loop)

            except Exception as e:
                asyncio.run_coroutine_threadsafe(queue.put(("error", {"message": str(e)})), _loop)
            finally:
                timing = {
                    "retrieval_sec": None,
                    "ttft_sec": ttft_sec,
                    "text_gen_sec": (text_gen_end - text_gen_start) if (text_gen_start and text_gen_end) else None,
                    "think_total_sec": (think_last - think_first) if (think_first and think_last) else None,
                }
                asyncio.run_coroutine_threadsafe(queue.put(("done", {"timing": timing})), _loop)
                asyncio.run_coroutine_threadsafe(queue.put(None), _loop)

        loop.run_in_executor(None, _producer, loop)

        while True:
            item = await queue.get()
            if item is None:
                break
            yield item


runtime = Runtime()