from typing import List, Tuple
from openai import OpenAI
from ..core.config import global_settings as C
from ..core.models import MessageHistory

SUMMARY_ALL_PROMPT = """다음은 사용자와 RAG 시스템의 전체 대화 이력입니다.
핵심 주제, 반복 질문, 주요 답변 내용을 300자 이내로 요약하세요.

{messages}

요약:"""

SUMMARY_RECENT5_PROMPT = """다음은 최근 5개 QA입니다.
현재 논의 중인 핵심 주제와 미해결 질문을 200자 이내로 요약하세요.

{messages}

요약:"""


def _format_messages(messages: List[MessageHistory]) -> str:
    lines = []
    for m in messages:
        lines.append(f"Q{m.id}: {m.user_query}")
        lines.append(f"A{m.id}: {m.rag_answer[:500]}")  # 답변 500자 제한
    return "\n".join(lines)


def summarize_all(messages: List[MessageHistory]) -> str:
    """전체 대화 요약 (기존 LLM_MODEL 사용)"""
    if not messages:
        return ""

    try:
        client = OpenAI(api_key=C.SGLANG_LLM_API_KEY, base_url=C.SGLANG_LLM_BASE_URL)
        prompt = SUMMARY_ALL_PROMPT.format(messages=_format_messages(messages))

        resp = client.chat.completions.create(
            model=C.LLM_MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3,
            max_tokens=C.SUMMARY_MAX_TOKENS,
            timeout=30,
        )

        # ✨ None 체크
        content = resp.choices[0].message.content
        return content.strip() if content else ""
    except Exception as e:
        # ✨ 에러 처리
        import logging
        logging.error(f"summarize_all failed: {e}")
        return ""


def summarize_recent5(messages: List[MessageHistory]) -> str:
    """최근 5개 QA 요약 (기존 LLM_MODEL 사용)"""
    if not messages:
        return ""

    try:
        recent = messages[-C.RECENT_QA_WINDOW:]
        client = OpenAI(api_key=C.SGLANG_LLM_API_KEY, base_url=C.SGLANG_LLM_BASE_URL)
        prompt = SUMMARY_RECENT5_PROMPT.format(messages=_format_messages(recent))

        resp = client.chat.completions.create(
            model=C.LLM_MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3,
            max_tokens=C.SUMMARY_MAX_TOKENS,
            timeout=30,
        )

        # ✨ None 체크
        content = resp.choices[0].message.content
        return content.strip() if content else ""
    except Exception as e:
        # ✨ 에러 처리
        import logging
        logging.error(f"summarize_recent5 failed: {e}")
        return ""


def build_summaries(messages: List[MessageHistory]) -> Tuple[str, str]:
    """전체 요약 + 최근5 요약 생성"""
    summary_all = summarize_all(messages)
    summary_recent5 = summarize_recent5(messages)
    return summary_all, summary_recent5