from typing import List

from llama_index.core import VectorStoreIndex
from llama_index.core.retrievers import QueryFusionRetriever
from llama_index.core.postprocessor import LongContextReorder, SimilarityPostprocessor
from llama_index.core.schema import TextNode

# ✅ 모듈형 BM25 우선 임포트, 실패 시 구버전 네임스페이스로 폴백
try:
    from llama_index.retrievers.bm25 import BM25Retriever
except Exception:  # pragma: no cover
    from llama_index.core.retrievers import BM25Retriever

from ..core.config import global_settings as C


def build_retriever(index: VectorStoreIndex, bm25_nodes: List[TextNode]):
    vec = index.as_retriever(similarity_top_k=C.SIMILARITY_TOP_K)
    retriever = vec
    if C.BM25_TOP_K > 0 and bm25_nodes:
        try:
            bm25 = BM25Retriever.from_nodes(bm25_nodes, similarity_top_k=C.BM25_TOP_K)
            retriever = QueryFusionRetriever(
                retrievers=[vec, bm25],
                num_queries=1,
                similarity_top_k=max(C.SIMILARITY_TOP_K, C.BM25_TOP_K),
                mode="reciprocal_rerank",
            )
        except Exception:
            pass
    return retriever


def postprocess_nodes(nodes_with_scores):
    out = nodes_with_scores
    try:
        out = LongContextReorder().postprocess_nodes(out)
    except Exception:
        pass
    if C.SIMILARITY_CUTOFF is not None:
        try:
            out = SimilarityPostprocessor(similarity_cutoff=C.SIMILARITY_CUTOFF).postprocess_nodes(out)
        except Exception:
            pass
    return out
