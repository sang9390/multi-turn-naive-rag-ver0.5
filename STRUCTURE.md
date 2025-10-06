# 프로젝트 구조

## 📂 디렉토리 구조

```
rag_multihop/
├── .env                          # 환경 설정
├── main.py                       # FastAPI 엔트리포인트
├── requirements.txt              # Python 의존성
│
├── app/
│   ├── core/
│   │   ├── config.py            # 설정 (Pydantic Settings)
│   │   ├── models.py            # API 요청/응답 모델
│   │   ├── runtime.py           # 런타임 (인덱스/세션 관리)
│   │   └── session_manager.py  # ✨ 세션 캐시 LRU 관리자
│   │
│   ├── services/
│   │   ├── generation.py        # LLM 생성 (SGLang)
│   │   ├── indexing.py          # 문서 인덱싱 (FAISS/BM25)
│   │   ├── retrieval.py         # 하이브리드 검색
│   │   ├── query_repair.py      # ✨ 멀티홉 쿼리 증강
│   │   └── summarizer.py        # ✨ 세션 요약 생성
│   │
│   └── routers/
│       ├── document.py          # POST /document (인덱싱)
│       ├── health.py            # GET /health
│       ├── query.py             # POST /query (일반/스트림)
│       └── session.py           # ✨ POST /session/init|switch
│
├── clients/
│   ├── chatbot.py               # ✨ Gradio UI (멀티홉 지원)
│   ├── test_multihop.py         # ✨ 멀티홉 테스트 클라이언트
│   ├── index_documents.py       # 인덱싱 전용 클라이언트
│   ├── python_client.py         # Python 예제
│   ├── js_client_sse.js         # Node.js 예제
│   └── smoke_all_endpoints.py   # 통합 스모크 테스트
│
├── scripts/
│   ├── uvicorn.sh               # 서버 실행 스크립트
│   └── smoke_load.py            # 부하 테스트
│
├── cache/                        # ✨ 세션 캐시 (런타임 생성)
│   └── session/
│       └── {session_id}/
│           ├── session_meta.json # a: 전체요약, b: 최근5요약
│           ├── faiss.index
│           ├── bm25/
│           └── chunks/*.jsonl
│
├── data/                         # 문서 인덱스 (영구 저장)
│   ├── vector_store/
│   └── faiss/
│
└── docs/                         # 문서
    ├── README.md
    ├── STRUCTURE.md             # (이 파일)
    └── API.md
```

---

## 🔑 핵심 컴포넌트

### 1. **Core (app/core/)**

#### `config.py`
- **역할**: 환경 변수 로드 및 검증
- **주요 설정**:
  - `SESSION_*`: 세션 캐시 정책
  - `ENABLE_QUERY_REPAIR`: 멀티홉 활성화
  - `LLM_MODEL`: 생성/요약/리페어 공통 모델

#### `runtime.py`
- **역할**: 전역 런타임 인스턴스 (인덱스/세션 관리)
- **기능**:
  - `handle_query()`: 비스트림 쿼리 처리
  - `handle_query_stream()`: SSE 스트림 처리
  - `_retrieve_from_session()`: 세션 QA 이력 검색
  - Lazy-load: `FAISS_LOAD_DIR` 자동 로드

#### `session_manager.py` ✨
- **역할**: 세션 캐시 LRU 관리
- **정책**:
  - LRU 100개 유지 (`SESSION_MAX_COUNT`)
  - TTL 72시간 (`SESSION_TTL_HOURS`)
  - 캐시 구조: `./cache/session/{session_id}/`

#### `models.py`
- **역할**: Pydantic 모델 (API 계약)
- **주요 모델**:
  - `SessionInitRequest`/`SessionSwitchRequest`
  - `QueryRequest` (session_id, eval_mode 추가)
  - `RepairContext` (corrections, questions, improved_query, assumptions)

---

### 2. **Services (app/services/)**

#### `query_repair.py` ✨
- **역할**: 멀티홉 쿼리 증강 (Self-RAG + CRAG)
- **프로세스**:
  1. 이전 QA 검색 (topk chunks)
  2. Self-Reflection (불확실성 감지)
  3. Query Decomposition (하위 질문)
  4. Contextual Rewrite (독립적 질의)
  5. Assumption Tracking
- **출력**: `RepairContext` (정정/질문/개선질의/가정)

#### `summarizer.py` ✨
- **역할**: 세션 요약 생성
- **기능**:
  - `summarize_all()`: 전체 대화 요약 (300자)
  - `summarize_recent5()`: 최근 5개 QA 요약 (200자)
- **사용**: `/session/switch` 호출 시 자동 실행

#### `generation.py`
- **역할**: LLM 생성 (SGLang OpenAI 호환)
- **기능**:
  - `sglang_stream()`: SSE 스트리밍
  - `build_context()`: 검색 결과 → 프롬프트 변환

#### `indexing.py`
- **역할**: 문서 로드/청킹/인덱싱
- **지원 형식**: `.md`, `.txt`, `.pdf`
- **청킹**: Semantic Splitter (LlamaIndex)

#### `retrieval.py`
- **역할**: 하이브리드 검색 (FAISS + BM25)
- **후처리**: LongContextReorder, SimilarityPostprocessor

---

### 3. **Routers (app/routers/)**

#### `session.py` ✨
- **엔드포인트**:
  - `POST /session/init`: 세션 초기화
  - `POST /session/switch`: 이력 동기화 + 요약/인덱스 생성
- **플로우**:
  1. 메시지 이력 수신 (백엔드 → RAG)
  2. 전체/최근5 요약 생성
  3. QA 청킹 + FAISS 인덱스 구축
  4. 세션 캐시 저장

#### `query.py`
- **엔드포인트**: `POST /query`
- **모드**:
  - `eval_mode=false` + `session_id` → 멀티홉 (쿼리 증강)
  - `eval_mode=true` → Naive RAG (증강 비활성화)
  - `stream=true` → SSE 스트리밍
- **응답**:
  - `repair_context`: 리페어 결과
  - `used_query`: 개선된 질의

#### `document.py`
- **엔드포인트**: `POST /document`
- **기능**: 문서 인덱싱 (FAISS/BM25)

#### `health.py`
- **엔드포인트**: `GET /health`
- **응답**: 모델 정보, 인덱스 상태

---

### 4. **Clients (clients/)**

#### `chatbot.py` ✨
- **기술**: Gradio
- **기능**:
  - 세션 관리 UI
  - 멀티홉 스트리밍
  - 쿼리 리페어 결과 실시간 표시

#### `test_multihop.py` ✨
- **역할**: 멀티홉 기능 통합 테스트
- **시나리오**:
  1. 세션 초기화
  2. 이력 동기화 (3개 QA)
  3. Naive RAG (세션 없음)
  4. Multihop RAG (모호한 질의)
  5. 평가 모드
  6. 스트리밍

---

## 🔄 데이터 플로우

### 멀티홉 쿼리 플로우

```
[User Query] "그거 주기는?"
    ↓
[Runtime] session_id=1001 감지
    ↓
[SessionManager] 캐시 로드
    └→ summary_all: "정비 절차 논의 중"
    └→ summary_recent5: "주기점검 3/6/12개월"
    └→ QA 인덱스 (FAISS/BM25)
    ↓
[Retrieval] 이전 QA 검색
    └→ topk=3: [(turn_id=1, "Q: 주기점검 주기는? A: 3/6/12개월"), ...]
    ↓
[QueryRepair] 리페어 실행
    └→ LLM 호출 (REPAIR_PROMPT_V2)
    └→ 출력:
        - corrections: ["'그거'가 주기점검/특별점검 불명확"]
        - questions: ["어느 점검을 말씀하시나요?"]
        - improved_query: "주기점검(3/6/12개월)의 구체적 절차"
        - assumptions: ["주기점검 지칭 가정"]
    ↓
[Retrieval] 개선된 질의로 문서 재검색
    └→ 문서 인덱스 (FAISS/BM25)
    └→ topk=5 contexts
    ↓
[Generation] 최종 답변 생성
    └→ prompt = contexts + improved_query
    └→ LLM 호출
    ↓
[Response]
    - answer: "주기점검은 3/6/12개월..."
    - repair_context: {...}
    - used_query: "주기점검(3/6/12개월)..."
    - timing: {...}
```

---

## 🗄️ 캐시 구조

### 세션 캐시 (`./cache/session/{session_id}/`)

```
1001/
├── session_meta.json          # 메타데이터
│   {
│     "summary_all": "전체 대화 요약",
│     "recent5": "최근 5개 QA 요약",
│     "updated_at": "2025-10-04T10:30:00"
│   }
│
├── faiss.index               # QA 임베딩 (FAISS IndexFlatL2)
├── docstore.json             # 문서 저장소
├── index_store.json          # 인덱스 메타데이터
├── nodes.jsonl               # BM25용 노드 (JSONL)
└── tmp/                      # 임시 파일
```

### 문서 인덱스 (`./data/faiss/`)

```
faiss/
├── faiss.index               # 전체 문서 임베딩
├── docstore.json
├── index_store.json
└── nodes.jsonl
```

---

## ⚙️ 설정 우선순위

1. **환경 변수** (`.env`)
2. **기본값** (`config.py` Field default)
3. **런타임 오버라이드** (API 파라미터)

### 예시: LLM 모델 선택
```python
# .env
LLM_MODEL=Qwen/Qwen3-8B-Instruct

# 쿼리 리페어/요약은 동일 모델 재사용
# query_repair.py, summarizer.py → C.LLM_MODEL 참조
```

---

## 🧪 테스트 전략

### 1. **단위 테스트**
```bash
pytest tests/unit/
```

### 2. **통합 테스트**
```bash
python clients/test_multihop.py
```

### 3. **부하 테스트**
```bash
python scripts/smoke_load.py
```

### 4. **수동 테스트** (Gradio)
```bash
python clients/chatbot.py
```

---

## 📈 확장 가능성

### 1. **다중 세션 동시 처리**
- 현재: 싱글 워커 (`uvicorn --workers 1`)
- 확장: Redis 세션 스토어 + 멀티 워커

### 2. **백엔드 통합**
```python
# 예: Django/Flask 백엔드
from rag_multihop import RAGClient

client = RAGClient("http://rag-server:8000")
await client.session_switch(session_id=1001, messages=history)
result = await client.query(session_id=1001, query="...")
```

### 3. **커스텀 리페어 로직**
```python
# app/services/query_repair.py
def domain_specific_repair(query, context):
    # 도메인 특화 규칙 추가
    if "안전" in query:
        return f"[긴급] {query}"
    return query
```

---

## 🔐 보안 고려사항

1. **세션 격리**: 각 `session_id`는 독립된 캐시
2. **TTL 정책**: 72시간 후 자동 삭제
3. **입력 검증**: Pydantic 모델 자동 검증
4. **LLM API 키**: `.env`에서 관리 (Git 제외)

---

**Last Updated**: 2025-10-04