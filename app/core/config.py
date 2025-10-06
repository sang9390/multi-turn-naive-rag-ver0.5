from pydantic_settings import BaseSettings
from pydantic import Field, field_validator
from typing import Optional


class Settings(BaseSettings):
    # Server
    HOST: str = Field(default="0.0.0.0")
    PORT: int = Field(default=8000)
    MAX_CONCURRENCY: int = Field(default=32)
    REQUEST_TIMEOUT_SEC: int = Field(default=120)

    # Generation
    GENERATOR_PROVIDER: str = Field(default="sglang")  # sglang|google_genai
    LLM_MODEL: str = Field(default="Qwen/Qwen3-8B-Instruct")
    SGLANG_LLM_BASE_URL: str = Field(default="http://localhost:30000/v1")
    SGLANG_LLM_API_KEY: str = Field(default="sk-test")
    GEN_TIMEOUT_SEC: int = Field(default=600)
    GEN_NUM_CTX: int = Field(default=8192)
    GEN_TEMPERATURE: float = Field(default=0.2)
    GEN_MAX_OUTPUT_TOKENS: int = Field(default=600)
    IS_CHAT_MODEL: bool = Field(default=True)
    ENABLE_THINKING: str = Field(default="off")  # off|on
    SEPARATE_REASONING: bool = Field(default=True)
    STREAM_COMPAT_DUP_CONTENT: bool = True
    STREAM_DEFAULT: bool = Field(default=True)

    # Embedding
    EMBED_PROVIDER: str = Field(default="sglang")  # sglang|google_genai
    EMBED_MODEL: str = Field(default="Qwen/Qwen3-Embedding-0.6B")
    SGLANG_EMBED_BASE_URL: str = Field(default="http://localhost:30001/v1")
    SGLANG_EMBED_API_KEY: str = Field(default="sk-test")
    EMBED_BATCH_SIZE: int = Field(default=8)
    EMBED_INPUT_TOKEN_LIMIT: int = Field(default=8192)
    # NEW: 임베딩 차원 수를 .env로 지정(미지정 시 자동 프로브)
    EMBED_DIM: Optional[int] = Field(default=None)

    # Retrieval / Index
    SIMILARITY_TOP_K: int = Field(default=5)
    BM25_TOP_K: int = Field(default=8)
    SIMILARITY_CUTOFF: Optional[float] = Field(default=None)
    VECTOR_STORE_DIR: str = Field(default="./data/vector_store")
    FAISS_SAVE_DIR: Optional[str] = Field(default=None)
    FAISS_LOAD_DIR: Optional[str] = Field(default=None)

    # Chunking
    MD_KEEP_CODEBLOCKS: bool = Field(default=True)
    SEMANTIC_SPLIT_ALL: bool = Field(default=True)
    SEMANTIC_TARGET_CHARS: int = Field(default=1200)
    SEMANTIC_BREAKPOINT_PERCENTILE: int = Field(default=95)

    # Context building limits for /query
    MAX_NODE_CHARS: int = Field(default=2800)      # per-node hard cap (character-based)
    CTX_CHARS_PER_NODE: int = Field(default=900)
    CTX_MAX_TOTAL_CHARS: int = Field(default=6000)

    # token-based hard cap (optional)
    MAX_NODE_TOKENS: Optional[int] = Field(default=None)
    TOKENIZER_NAME: Optional[str] = Field(default="cl100k_base")

    # Page Split (md/txt only)
    PAGE_SPLIT_ENABLE: bool = Field(default=False)
    PAGE_SPLIT_APPLIES_TO: str = Field(default="md,txt")
    PAGE_SPLIT_MODE: str = Field(default="symbol")  # symbol|regex
    PAGE_SPLIT_SYMBOL_TEMPLATE: str = Field(default="- {n} -")
    PAGE_SPLIT_MIN_N: int = Field(default=1)
    PAGE_SPLIT_MAX_N: int = Field(default=1000000)
    PAGE_SPLIT_TRIM_WHITESPACE: bool = Field(default=True)
    PAGE_SPLIT_REGEX: Optional[str] = Field(default=None)
    PAGE_SPLIT_REQUIRE_MONOTONIC: bool = Field(default=True)
    PAGE_SPLIT_RESET_ON_FILE: bool = Field(default=True)
    PAGE_SPLIT_INCLUDE_MARKER: bool = Field(default=False)
    PAGE_SPLIT_MAX_PAGES_PER_FILE: int = Field(default=100000)

    # Optional external keys
    GOOGLE_API_KEY: Optional[str] = Field(default=None)

    # Session Management
    SESSION_CACHE_DIR: str = Field(default="./cache/session")
    SESSION_MAX_COUNT: int = Field(default=100)
    SESSION_TTL_HOURS: int = Field(default=72)
    RECENT_QA_WINDOW: int = Field(default=5)

    # Query Repair (Multihop) - 기존 LLM_MODEL 재사용
    ENABLE_QUERY_REPAIR: bool = Field(default=True)
    REPAIR_MAX_TOKENS: int = Field(default=800)
    REPAIR_TEMPERATURE: float = Field(default=0.3)

    # Summarization - 기존 LLM_MODEL 재사용
    SUMMARY_MAX_TOKENS: int = Field(default=600)

    # REVEAL_FROM Feature
    REVEAL_FROM_ENABLED: bool = Field(default=False)
    REVEAL_FROM_TOKEN: str = Field(default="<<<FINAL>>>")
    REVEAL_FALLBACK: str = Field(default="keep_all")  # keep_all | empty | after_think_tag

    # --- Validators ---
    @field_validator("SIMILARITY_CUTOFF", mode="before")
    @classmethod
    def _cutoff_empty_to_none(cls, v):
        if v is None:
            return None
        if isinstance(v, str) and v.strip().lower() in ("", "none", "null"):
            return None
        return v

    @field_validator("MAX_NODE_TOKENS", mode="before")
    @classmethod
    def _empty_to_none_tokens(cls, v):
        if v is None:
            return None
        if isinstance(v, str) and v.strip() == "":
            return None
        return v

    @field_validator("EMBED_DIM", mode="before")
    @classmethod
    def _empty_to_none_embed_dim(cls, v):
        if v is None:
            return None
        if isinstance(v, str) and v.strip() in ("", "none", "null"):
            return None
        return v

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"


global_settings = Settings()
