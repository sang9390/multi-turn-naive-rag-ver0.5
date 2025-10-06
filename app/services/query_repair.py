import re
import logging
from typing import List, Tuple
from openai import OpenAI
from ..core.config import global_settings as C
from ..core.models import RepairContext

logger = logging.getLogger(__name__)

# ✨ 최신 논문 기법 적용: Self-RAG + CRAG 스타일 쿼리 리페어
REPAIR_PROMPT_V2 = """[역할] 너는 "멀티홉 쿼리 수선기(Multihop Query Repairer)"다. 
이전 대화의 불확실성을 감지하고, 다음 검색이 더 정확해지도록 쿼리를 개선한다.

[방법론 (Self-RAG + CRAG + Iter-RetGen 참조)]
1. **Self-Reflection**: 이전 QA에서 [incorrect_suspected | ambiguous | missing_context] 항목 감지
2. **Query Decomposition**: 복잡한 질문을 하위 질문으로 분해
3. **Contextual Rewrite**: 대화 맥락을 반영한 독립적 질의로 재작성
4. **Assumption Tracking**: 불확실한 전제를 명시적으로 추적

[입력]
사용자 질문: {user_query}
전체 요약: {summary_all}
최근5 요약: {summary_recent5}
이전 QA 이력 (topk chunk):
{topk_chunks}

[지시]
A) **정정 대상** (Reflection): 이전 답변 중 불확실/오류 의심 항목을 나열
   형식: - (turn_id=X) 요약 — [incorrect_suspected|ambiguous|missing_context]

B) **확인 질문** (Clarification): 사용자에게 확인할 핵심 질문 1-3개 (닫힌 질문 선호)

C) **개선된 질의** (Rewrite): 
   - 모호한 대명사 제거 (이것→구체적 명사)
   - 시간/버전/환경 명시
   - 하위 질문 포함 (필요시)
   - 이전 QA 표현 재사용 금지 (중립적 재구성)

D) **가정** (Assumptions): 불확실한 전제 명시

[출력 형식]
정정_대상:
- (turn_id=X) ... — [태그]

확인_질문:
1) ...
2) ...

개선된_질의:
... (2-3문장, 독립적으로 이해 가능하도록)

가정:
- ...

[제약]
- topk chunk는 비권위 힌트. 사실로 단정 금지
- 외부 지식 추가 금지 (대화 컨텍스트만 사용)
"""


def _parse_repair_output(text: str) -> RepairContext:
    """리페어 출력 파싱"""
    corrections = []
    questions = []
    improved_query = ""
    assumptions = []

    # 정정_대상 파싱
    corr_match = re.search(r'정정_대상:(.*?)(?=확인_질문:|개선된_질의:|가정:|$)', text, re.DOTALL)
    if corr_match:
        for line in corr_match.group(1).strip().split('\n'):
            if line.strip().startswith('-'):
                corrections.append(line.strip()[1:].strip())

    # 확인_질문 파싱
    q_match = re.search(r'확인_질문:(.*?)(?=개선된_질의:|가정:|$)', text, re.DOTALL)
    if q_match:
        for line in q_match.group(1).strip().split('\n'):
            if re.match(r'^\d+\)', line.strip()):
                questions.append(re.sub(r'^\d+\)\s*', '', line.strip()))

    # 개선된_질의 파싱
    iq_match = re.search(r'개선된_질의:(.*?)(?=가정:|$)', text, re.DOTALL)
    if iq_match:
        improved_query = iq_match.group(1).strip()

    # 가정 파싱
    ass_match = re.search(r'가정:(.*?)', text, re.DOTALL)
    if ass_match:
        for line in ass_match.group(1).strip().split('\n'):
            if line.strip().startswith('-'):
                assumptions.append(line.strip()[1:].strip())

    return RepairContext(
        corrections=corrections,
        questions=questions,
        improved_query=improved_query or "",
        assumptions=assumptions
    )


def repair_query(
        user_query: str,
        summary_all: str,
        summary_recent5: str,
        topk_chunks: List[Tuple[int, str]]  # [(turn_id, text), ...]
) -> RepairContext:
    """멀티홉 쿼리 리페어 실행 (기존 LLM_MODEL 사용)"""
    if not C.ENABLE_QUERY_REPAIR:
        return RepairContext(improved_query=user_query)

    try:
        # topk chunk 포맷팅
        chunks_text = "\n\n".join([
            f"[turn_id={tid}]\n{text[:800]}"
            for tid, text in topk_chunks
        ]) if topk_chunks else "(이전 QA 없음)"

        prompt = REPAIR_PROMPT_V2.format(
            user_query=user_query,
            summary_all=summary_all or "(없음)",
            summary_recent5=summary_recent5 or "(없음)",
            topk_chunks=chunks_text
        )

        client = OpenAI(api_key=C.SGLANG_LLM_API_KEY, base_url=C.SGLANG_LLM_BASE_URL)
        resp = client.chat.completions.create(
            model=C.LLM_MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=C.REPAIR_TEMPERATURE,
            max_tokens=C.REPAIR_MAX_TOKENS,
            timeout=60,
        )

        # ✨ None 체크
        output = resp.choices[0].message.content
        if not output:
            logger.warning("repair_query: LLM returned None")
            return RepairContext(improved_query=user_query)

        return _parse_repair_output(output.strip())

    except Exception as e:
        logger.error(f"repair_query failed: {e}")
        return RepairContext(improved_query=user_query)