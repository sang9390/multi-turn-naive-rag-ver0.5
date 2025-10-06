from pydantic import BaseModel, Field
from typing import List, Optional, Any, Dict
from datetime import datetime

# ========================================
# 기존 모델 (Document 관련)
# ========================================
class DocumentRequest(BaseModel):
    path: str
    recursive: bool = True
    rebuild: bool = False
    file_types: List[str] = ["md", "txt", "pdf"]

class DocumentResponse(BaseModel):
    status: str = "ok"
    indexed_files: int
    nodes: int
    persist_dir: str
    hybrid_enabled: bool

# ========================================
# 기존 모델 (Query 관련)
# ========================================
class ContextItem(BaseModel):
    text: str
    score: Optional[float] = None
    metadata: Dict[str, Any] = {}

class Timing(BaseModel):
    retrieval_sec: Optional[float] = None
    ttft_sec: Optional[float] = None
    text_gen_sec: Optional[float] = None
    think_total_sec: Optional[float] = None

# ========================================
# 세션 모델 ✨ 추가
# ========================================
class MessageHistory(BaseModel):
    id: int
    user_query: str
    rag_answer: str
    created_at: str  # ISO format

class SessionInitRequest(BaseModel):
    session_id: int
    new_session: bool = True

class SessionSwitchRequest(BaseModel):
    session_id: int
    new_session: bool = False
    messages: List[MessageHistory] = []

class SessionResponse(BaseModel):
    status: str = "ok"
    session_id: Optional[int] = None

# ========================================
# 쿼리 리페어 모델 ✨ 추가
# ========================================
class RepairContext(BaseModel):
    """쿼리 리페어 결과"""
    corrections: List[str] = []
    questions: List[str] = []
    improved_query: str
    assumptions: List[str] = []

# ========================================
# Query 모델 (확장)
# ========================================
class QueryRequest(BaseModel):
    query: str
    session_id: Optional[int] = None  # ✨ 추가
    eval_mode: bool = Field(default=False)  # ✨ 추가
    top_k: int = 5
    think_mode: str = Field(default="off", description="off|on")
    include_reasoning: bool = False
    stream: Optional[bool] = None

class QueryResponse(BaseModel):
    answer: str
    reasoning: Optional[str] = None
    contexts: List[ContextItem]
    timing: Timing
    files: List[str] = []
    repair_context: Optional[RepairContext] = None  # ✨ 추가
    used_query: Optional[str] = None  # ✨ 추가