import os
import re
from pathlib import Path
from typing import List, Optional, Tuple

from llama_index.core import (
    Document,
    VectorStoreIndex,
    Settings,
    StorageContext,
    load_index_from_storage,
)
from llama_index.vector_stores.faiss import FaissVectorStore
from llama_index.core.schema import TextNode
from llama_index.core.storage.docstore import SimpleDocumentStore
from llama_index.core.storage.index_store import SimpleIndexStore

from ..core.config import global_settings as C

# ===============================
# Page split helpers
# ===============================

def split_symbol_template_to_regex(template: str) -> re.Pattern:
    esc = re.escape(template).replace(re.escape("{n}"), r"(\\d+)")
    if C.PAGE_SPLIT_TRIM_WHITESPACE:
        esc = rf"^\s*{esc}\s*$"
    return re.compile(esc)


def _iter_pages_symbol(text: str):
    pat = split_symbol_template_to_regex(C.PAGE_SPLIT_SYMBOL_TEMPLATE)
    lines = text.splitlines()
    pages = []
    current_start = 0
    last_no = 0
    for i, line in enumerate(lines):
        m = pat.match(line)
        if m:
            try:
                no = int(m.group(1))
            except Exception:
                continue
            if C.PAGE_SPLIT_REQUIRE_MONOTONIC and (no <= last_no):
                continue
            if last_no > 0:
                pages.append((last_no, current_start, i, lines[i - 1] if i > 0 else ""))
            current_start = i + (0 if C.PAGE_SPLIT_INCLUDE_MARKER else 1)
            last_no = no
    if last_no > 0:
        pages.append((last_no, current_start, len(lines), lines[-1] if lines else ""))
    return pages


def build_pages_from_markers(text: str):
    if not C.PAGE_SPLIT_ENABLE:
        return []
    mode = (C.PAGE_SPLIT_MODE or "symbol").lower()
    if mode == "regex" and C.PAGE_SPLIT_REGEX:
        pat = re.compile(C.PAGE_SPLIT_REGEX)
        lines = text.splitlines()
        out = []
        buf = []
        cur_no: Optional[int] = None
        for line in lines:
            m = pat.match(line)
            if m:
                no = int(m.group(1))
                if cur_no is not None:
                    out.append((cur_no, "\n".join(buf)))
                buf = []
                cur_no = no
                if C.PAGE_SPLIT_INCLUDE_MARKER:
                    buf.append(line)
            else:
                buf.append(line)
        if cur_no is not None:
            out.append((cur_no, "\n".join(buf)))
        return out
    pages = _iter_pages_symbol(text)
    if not pages:
        return []
    lines = text.splitlines()
    out = []
    for no, s, e, _ in pages:
        body = "\n".join(lines[s:e])
        out.append((no, body))
    return out

# ===============================
# Length caps (env-controlled)
# ===============================

def _truncate_by_chars(text: str, max_chars: int) -> str:
    return text if len(text) <= max_chars else text[:max_chars]


def _truncate_by_tokens(text: str, max_tokens: int, tokenizer_name: Optional[str]) -> str:
    try:
        import tiktoken
        enc = tiktoken.get_encoding(tokenizer_name or "cl100k_base")
        toks = enc.encode(text)
        if len(toks) <= max_tokens:
            return text
        return enc.decode(toks[:max_tokens])
    except Exception:
        approx_chars = max_tokens * 4
        return _truncate_by_chars(text, approx_chars)


def _cap_text(text: str) -> str:
    if C.MAX_NODE_TOKENS:
        text = _truncate_by_tokens(text, int(C.MAX_NODE_TOKENS), C.TOKENIZER_NAME)
    max_chars = int(getattr(C, "MAX_NODE_CHARS", 2800) or 2800)
    return _truncate_by_chars(text, max_chars)


def enforce_node_caps(nodes: List[TextNode]) -> List[TextNode]:
    out: List[TextNode] = []
    for n in nodes:
        txt = n.get_content() or ""
        capped = _cap_text(txt)
        if capped is not txt:
            out.append(TextNode(text=capped, metadata=(n.metadata or {})))
        else:
            out.append(n)
    return out

# ===============================
# IO helpers
# ===============================

def scan_files(root: str, exts: List[str]) -> List[Path]:
    want = {x.lower().strip() for x in exts}
    return sorted([p for p in Path(root).rglob("*") if p.is_file() and p.suffix.lower().lstrip(".") in want])


def load_text(fp: Path) -> str:
    try:
        return fp.read_text(encoding="utf-8")
    except Exception:
        return fp.read_text(encoding="cp949", errors="ignore")

# ===============================
# Node building
# ===============================

def build_nodes_from_paths(paths: List[Path]) -> List[TextNode]:
    from llama_index.core.node_parser import MarkdownNodeParser
    nodes: List[TextNode] = []
    md_parser = MarkdownNodeParser()
    for fp in paths:
        meta = {"file": fp.name, "path": str(fp)}
        if fp.suffix.lower() in (".md", ".txt"):
            txt = load_text(fp)
            if C.PAGE_SPLIT_ENABLE and fp.suffix.lower().lstrip(".") in C.PAGE_SPLIT_APPLIES_TO.split(","):
                pages = build_pages_from_markers(txt)
                if pages:
                    for no, body in pages:
                        d = Document(text=body, metadata={**meta, "page": no})
                        nodes.extend(md_parser.get_nodes_from_documents([d]))
                    continue
            d = Document(text=txt, metadata=meta)
            nodes.extend(md_parser.get_nodes_from_documents([d]))
        elif fp.suffix.lower() == ".pdf":
            try:
                from llama_index.readers.file import PyMuPDFReader
                r = PyMuPDFReader()
                docs = r.load(file_path=str(fp))
                for d in docs:
                    nodes.extend(md_parser.get_nodes_from_documents([d]))
            except Exception:
                pass

    if C.SEMANTIC_SPLIT_ALL:
        try:
            from llama_index.core.node_parser import SemanticSplitterNodeParser
            splitter = SemanticSplitterNodeParser(
                buffer_size=1,
                breakpoint_percentile_threshold=C.SEMANTIC_BREAKPOINT_PERCENTILE,
                embed_model=Settings.embed_model,
                chunk_size=C.SEMANTIC_TARGET_CHARS,
            )
            new_nodes: List[TextNode] = []
            for n in nodes:
                text, meta = n.get_content(), n.metadata or {}
                new_nodes.extend(splitter.get_nodes_from_documents([Document(text=text, metadata=meta)]))
            nodes = new_nodes
        except Exception:
            pass

    nodes = enforce_node_caps(nodes)
    return nodes

# ===============================
# Persist / Load helpers
# ===============================

def save_nodes_jsonl(nodes: List[TextNode], path: str):
    import json
    with open(path, "w", encoding="utf-8") as f:
        for n in nodes:
            obj = {"text": n.get_content(), "metadata": n.metadata}
            f.write(json.dumps(obj, ensure_ascii=False) + "\n")


def load_nodes_jsonl(path: str) -> List[TextNode]:
    import json
    out: List[TextNode] = []
    if not os.path.exists(path):
        return out
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            obj = json.loads(line)
            out.append(TextNode(text=obj.get("text", ""), metadata=obj.get("metadata") or {}))
    return out

# ===============================
# Build or load index
# ===============================

def _abs(p: Optional[str]) -> Optional[str]:
    if not p:
        return None
    return os.path.abspath(os.path.expanduser(p))


def build_or_load_index(nodes: List[TextNode]) -> Tuple[VectorStoreIndex, List[TextNode], str]:
    """
    LOAD: 명시적 로더 사용 (FaissVectorStore + SimpleDocumentStore + SimpleIndexStore)
    BUILD: 신규 구축 후 sc.persist(...)
    """
    load_dir = _abs(C.FAISS_LOAD_DIR) or _abs(C.VECTOR_STORE_DIR)
    if load_dir and os.path.isdir(load_dir):
        # ★ 명시적 컴포넌트 로드 (SimpleVectorStore 자동 감지 방지)
        docstore = SimpleDocumentStore.from_persist_dir(load_dir)
        index_store = SimpleIndexStore.from_persist_dir(load_dir)
        vs = FaissVectorStore.from_persist_dir(load_dir)
        sc = StorageContext.from_defaults(docstore=docstore, index_store=index_store, vector_store=vs)
        index = load_index_from_storage(sc)
        bm25_nodes = load_nodes_jsonl(os.path.join(load_dir, "nodes.jsonl"))
        return index, bm25_nodes, load_dir

    # 신규 구축
    try:
        if C.EMBED_DIM is not None:
            dim = int(C.EMBED_DIM)
        else:
            probe_vec = Settings.embed_model.get_text_embedding("dim-probe")
            dim = len(probe_vec)
    except Exception as e:
        raise RuntimeError(f"Failed to determine embedding dimension. Set EMBED_DIM or check EMBED_* env & endpoint. Original: {e}")

    try:
        import faiss
    except Exception as e:
        raise RuntimeError("faiss not installed. `pip install faiss-cpu`") from e

    faiss_index = faiss.IndexFlatL2(dim)
    vs = FaissVectorStore(faiss_index=faiss_index)
    docstore = SimpleDocumentStore()
    index_store = SimpleIndexStore()
    sc = StorageContext.from_defaults(vector_store=vs, docstore=docstore, index_store=index_store)

    index = VectorStoreIndex(nodes, storage_context=sc, show_progress=False)

    persist_dir = _abs(C.FAISS_SAVE_DIR) or _abs(C.VECTOR_STORE_DIR) or "./data/vector_store"
    os.makedirs(persist_dir, exist_ok=True)
    sc.persist(persist_dir=persist_dir)
    save_nodes_jsonl(nodes, os.path.join(persist_dir, "nodes.jsonl"))
    return index, nodes, persist_dir

# ===============================
# Public API
# ===============================

def index_path(path: str, recursive: bool, rebuild: bool, file_types: List[str]):
    p = Path(path)
    if p.is_file():
        paths = [p]
    else:
        paths = scan_files(path, file_types) if recursive else [q for q in p.iterdir() if q.is_file()]
    nodes = build_nodes_from_paths(paths)
    index, bm25_nodes, persist_dir = build_or_load_index(nodes)
    return index, bm25_nodes, len(paths), len(nodes), persist_dir
