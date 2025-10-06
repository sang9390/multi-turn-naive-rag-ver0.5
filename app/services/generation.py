import time, re
from typing import Dict, Any, List, Tuple
from openai import OpenAI
from llama_index.core import Settings
from ..core.config import global_settings as C

# System prompt (샘플)
QA_TMPL = (
    "당신은 한국 열차 정비 도메인의 전문 어시스턴트입니다.\n"
    "원칙: 1) 컨텍스트 내 근거만, 없으면 '문서에 근거 없음' 명시. 2) 안전 경고 우선. 3) 절차는 단계별 서술. 4) 사용자의 질문이 모호하다면, context 를 기반으로 정확히 어떤 부분에 대한 질문인지 재확인하시오.\n"
    "[컨텍스트]\n{context_str}\n[질문]\n{query_str}\n[답변]\n"
)


def init_generation_models():
    # ✅ OpenAI-like embedding (SGLang 등 OpenAI 호환 엔드포인트)
    from llama_index.embeddings.openai_like import OpenAILikeEmbedding

    emb = OpenAILikeEmbedding(
        model_name=C.EMBED_MODEL,
        api_base=C.SGLANG_EMBED_BASE_URL,
        api_key=C.SGLANG_EMBED_API_KEY,
        embed_batch_size=C.EMBED_BATCH_SIZE,
    )
    Settings.embed_model = emb

    # LLM도 OpenAI 호환 API 클라이언트 사용
    llm = OpenAI(api_key=C.SGLANG_LLM_API_KEY, base_url=C.SGLANG_LLM_BASE_URL)
    return llm, emb


def _strip_think(s: str) -> str:
    import re
    if not s:
        return s
    return re.sub(r"\s*<think>.*?</think>\s*", "", s, flags=re.DOTALL | re.IGNORECASE).strip()


def _apply_reveal_from_filter(text: str, token: str, fallback: str) -> str:
    """논스트리밍용 REVEAL_FROM 필터"""
    if not C.REVEAL_FROM_ENABLED:
        return text

    # 토큰 마지막 등장 이후만 반환
    if token in text:
        parts = text.split(token)
        return parts[-1].strip()

    # 토큰 미등장 시 Fallback 정책
    if fallback == "empty":
        return ""
    elif fallback == "after_think_tag":
        # <think>...</think> 이후 텍스트 반환
        match = re.search(r"</think>\s*(.*)", text, re.DOTALL | re.IGNORECASE)
        return match.group(1).strip() if match else text
    else:  # keep_all
        return text


class RevealFromBuffer:
    """스트리밍용 REVEAL_FROM 버퍼"""
    def __init__(self, token: str):
        self.token = token
        self.buffer = ""
        self.revealed = False

    def add(self, chunk: str) -> str:
        """토큰 검출 전까지 버퍼링, 검출 후 즉시 출력"""
        if self.revealed:
            return chunk

        self.buffer += chunk
        if self.token in self.buffer:
            # 토큰 이후 텍스트 추출
            parts = self.buffer.split(self.token, 1)
            self.revealed = True
            self.buffer = ""
            return parts[1] if len(parts) > 1 else ""

        # 아직 미검출
        return ""


def sglang_stream(
    prompt: str,
    model: str,
    enable_thinking: bool,
    separate_reasoning: bool,
    temperature: float,
    max_tokens: int,
    timeout_sec: int,
):
    reveal_buffer = RevealFromBuffer(C.REVEAL_FROM_TOKEN) if C.REVEAL_FROM_ENABLED else None
    client = OpenAI(api_key=C.SGLANG_LLM_API_KEY, base_url=C.SGLANG_LLM_BASE_URL)
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": prompt},
    ]
    extra_body = {
        "enable_thinking": bool(enable_thinking),
        "separate_reasoning": bool(separate_reasoning if enable_thinking else False),
        "chat_template_kwargs": {"enable_thinking": bool(enable_thinking)},
    }
    t_req = time.monotonic()
    stream = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=temperature,
        max_tokens=max_tokens,
        stream=True,
        extra_body=extra_body,
        timeout=timeout_sec,
    )
    first_content = None
    last_content = None
    first_reason = None
    last_reason = None
    answer_parts: List[str] = []
    reason_parts: List[str] = []
    for chunk in stream:
        delta = chunk.choices[0].delta
        now = time.monotonic()
        rcontent = getattr(delta, "reasoning_content", None)
        content = getattr(delta, "content", None)
        if rcontent:
            if first_reason is None:
                first_reason = now
            last_reason = now
            yield ("reasoning", rcontent)
            reason_parts.append(rcontent)
        if content:
            # REVEAL_FROM 필터 적용
            if reveal_buffer:
                filtered = reveal_buffer.add(content)
                if not filtered:
                    continue  # 토큰 전까지 미출력
                content = filtered

            if first_content is None:
                first_content = now
                yield ("ttft", first_content - t_req)
            last_content = now
            yield ("content", content)
            answer_parts.append(content)

    # Fallback 처리 (토큰 미등장 시)
    if reveal_buffer and not reveal_buffer.revealed and C.REVEAL_FALLBACK == "empty":
        answer_parts = []

    timing = {
        "ttft_sec": (first_content - t_req) if first_content else None,
        "text_gen_sec": (last_content - first_content) if (first_content and last_content) else None,
        "think_total_sec": (last_reason - first_reason) if (first_reason and last_reason) else None,
    }
    answer = "".join(answer_parts)
    reasoning = "".join(reason_parts)
    if not enable_thinking:
        answer = _strip_think(answer)
        reasoning = ""
    yield ("done", {"timing": timing, "answer": answer, "reasoning": reasoning})


def build_context(nodes_with_scores) -> Tuple[str, List[str], List[Dict[str, Any]]]:
    blocks: List[str] = []
    files: List[str] = []
    contexts: List[Dict[str, Any]] = []
    total = 0
    for i, sn in enumerate(nodes_with_scores, start=1):
        node = sn.node
        txt = node.get_content()
        meta = node.metadata or {}
        file_meta = meta.get("file") or meta.get("file_name") or meta.get("path")
        files.append(file_meta)
        chunk = txt[: C.CTX_CHARS_PER_NODE]
        if total + len(chunk) > C.CTX_MAX_TOTAL_CHARS:
            break
        blocks.append(f"[{i}] 파일:{file_meta}\n{chunk}")
        total += len(chunk)
        contexts.append(
            {
                "text": txt,
                "score": float(getattr(sn, "score", 0.0) or 0.0),
                "metadata": meta,
            }
        )
    return "\n\n---\n\n".join(blocks), files, contexts
