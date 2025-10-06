import os
import json
import shutil
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta
from collections import OrderedDict

from llama_index.core import VectorStoreIndex, StorageContext
from llama_index.vector_stores.faiss import FaissVectorStore
from llama_index.core.schema import TextNode
from llama_index.core.storage.docstore import SimpleDocumentStore
from llama_index.core.storage.index_store import SimpleIndexStore

from ..core.config import global_settings as C


class SessionCache:
    """세션별 캐시 데이터"""

    def __init__(self, session_id: int, cache_dir: str):
        self.session_id = session_id
        self.cache_dir = cache_dir
        self.meta_path = os.path.join(cache_dir, "session_meta.json")
        self.summary_all: str = ""
        self.summary_recent5: str = ""
        self.updated_at: Optional[datetime] = None
        self.index: Optional[VectorStoreIndex] = None
        self.bm25_nodes: List[TextNode] = []

    def load_meta(self):
        """메타데이터 로드"""
        if os.path.exists(self.meta_path):
            with open(self.meta_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                self.summary_all = data.get("summary_all", "")
                self.summary_recent5 = data.get("recent5", "")
                ts = data.get("updated_at")
                if ts:
                    self.updated_at = datetime.fromisoformat(ts)

    def save_meta(self):
        """메타데이터 저장"""
        os.makedirs(self.cache_dir, exist_ok=True)
        with open(self.meta_path, 'w', encoding='utf-8') as f:
            json.dump({
                "summary_all": self.summary_all,
                "recent5": self.summary_recent5,
                "updated_at": datetime.now().isoformat()
            }, f, ensure_ascii=False, indent=2)
        self.updated_at = datetime.now()


class SessionManager:
    """LRU 세션 캐시 관리자"""

    def __init__(self):
        self.base_dir = C.SESSION_CACHE_DIR
        self.max_sessions = C.SESSION_MAX_COUNT
        self.ttl_hours = C.SESSION_TTL_HOURS
        self._cache: OrderedDict[int, SessionCache] = OrderedDict()
        os.makedirs(self.base_dir, exist_ok=True)

    def _evict_lru(self):
        """LRU 제거"""
        while len(self._cache) > self.max_sessions:
            old_id, old_cache = self._cache.popitem(last=False)
            if os.path.exists(old_cache.cache_dir):
                shutil.rmtree(old_cache.cache_dir)

    def _evict_expired(self):
        """TTL 초과 세션 제거"""
        if self.ttl_hours <= 0:
            return
        now = datetime.now()
        expired = []
        for sid, cache in self._cache.items():
            if cache.updated_at and (now - cache.updated_at) > timedelta(hours=self.ttl_hours):
                expired.append(sid)
        for sid in expired:
            cache = self._cache.pop(sid)
            if os.path.exists(cache.cache_dir):
                shutil.rmtree(cache.cache_dir)

    def init_session(self, session_id: int) -> SessionCache:
        """세션 초기화"""
        cache_dir = os.path.join(self.base_dir, str(session_id))
        os.makedirs(cache_dir, exist_ok=True)

        cache = SessionCache(session_id, cache_dir)
        cache.save_meta()

        self._cache[session_id] = cache
        self._cache.move_to_end(session_id)
        self._evict_lru()

        return cache

    def get_or_create(self, session_id: int) -> SessionCache:
        """세션 가져오기 (없으면 생성)"""
        if session_id in self._cache:
            self._cache.move_to_end(session_id)
            return self._cache[session_id]

        cache_dir = os.path.join(self.base_dir, str(session_id))
        if os.path.exists(cache_dir):
            cache = SessionCache(session_id, cache_dir)
            cache.load_meta()
            self._cache[session_id] = cache
            self._cache.move_to_end(session_id)
            return cache

        return self.init_session(session_id)

    def update_session(
            self,
            session_id: int,
            summary_all: str,
            summary_recent5: str,
            index: Optional[VectorStoreIndex] = None,
            bm25_nodes: Optional[List[TextNode]] = None
    ):
        """세션 업데이트"""
        cache = self.get_or_create(session_id)
        cache.summary_all = summary_all
        cache.summary_recent5 = summary_recent5
        if index:
            cache.index = index
        if bm25_nodes:
            cache.bm25_nodes = bm25_nodes
        cache.save_meta()

        self._cache.move_to_end(session_id)
        self._evict_expired()


session_manager = SessionManager()