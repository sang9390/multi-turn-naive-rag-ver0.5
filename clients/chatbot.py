#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
RAG 통합 테스트 챗봇 (Gradio)
- 세션 관리 (초기화/전환)
- 멀티홉 쿼리 (REVEAL_FROM 포함)
- Context 확인
- 상세 로그 탭
"""
import os
import json
import time
import uuid
from datetime import datetime
from typing import List, Dict, Any, Optional, Tuple

import requests
import gradio as gr

# ==========================================
# Config
# ==========================================
RAG_SERVER = os.environ.get("RAG_SERVER", "http://localhost:8001")
DEFAULT_TOP_K = 5


# ==========================================
# Global State (세션별 메시지 이력)
# ==========================================
class SessionState:
    def __init__(self):
        self.sessions: Dict[int, List[Dict[str, Any]]] = {}
        self.current_session_id: Optional[int] = None
        self.logs: List[str] = []

    def new_session(self) -> int:
        """새 세션 ID 생성"""
        session_id = int(time.time() * 1000) % 100000000
        self.sessions[session_id] = []
        self.current_session_id = session_id
        self.add_log(f"✨ 새 세션 생성: {session_id}")
        return session_id

    def add_message(self, session_id: int, user_query: str, rag_answer: str):
        """메시지 추가"""
        if session_id not in self.sessions:
            self.sessions[session_id] = []

        msg_id = len(self.sessions[session_id])
        self.sessions[session_id].append({
            "id": msg_id,
            "user_query": user_query,
            "rag_answer": rag_answer,
            "created_at": datetime.utcnow().isoformat() + "Z"
        })

    def get_messages(self, session_id: int) -> List[Dict[str, Any]]:
        """세션 메시지 조회"""
        return self.sessions.get(session_id, [])

    def add_log(self, msg: str):
        """로그 추가"""
        timestamp = datetime.now().strftime("%H:%M:%S.%f")[:-3]
        self.logs.append(f"[{timestamp}] {msg}")

    def get_logs(self) -> str:
        """로그 전체 조회"""
        return "\n".join(self.logs[-100:])  # 최근 100개


state = SessionState()


# ==========================================
# API 호출 함수
# ==========================================
def call_session_init(session_id: int) -> Dict[str, Any]:
    """POST /session/init"""
    try:
        resp = requests.post(
            f"{RAG_SERVER}/session/init",
            json={"session_id": session_id, "new_session": True},
            timeout=30
        )
        state.add_log(f"POST /session/init: {resp.status_code}")
        resp.raise_for_status()
        return resp.json()
    except Exception as e:
        state.add_log(f"❌ /session/init 실패: {e}")
        raise


def call_session_switch(session_id: int, messages: List[Dict[str, Any]]) -> Dict[str, Any]:
    """POST /session/switch"""
    try:
        resp = requests.post(
            f"{RAG_SERVER}/session/switch",
            json={
                "session_id": session_id,
                "new_session": False,
                "messages": messages
            },
            timeout=60
        )
        state.add_log(f"POST /session/switch: {resp.status_code} (messages={len(messages)})")
        resp.raise_for_status()
        return resp.json()
    except Exception as e:
        state.add_log(f"❌ /session/switch 실패: {e}")
        raise


def call_query(
        session_id: Optional[int],
        query: str,
        top_k: int,
        think_mode: str,
        eval_mode: bool,
        include_reasoning: bool
) -> Dict[str, Any]:
    """POST /query (non-stream)"""
    try:
        payload = {
            "query": query,
            "session_id": session_id if session_id and session_id > 0 else None,
            "top_k": top_k,
            "think_mode": think_mode,
            "eval_mode": eval_mode,
            "include_reasoning": include_reasoning,
            "stream": False
        }

        state.add_log(f"POST /query: session_id={session_id}, eval_mode={eval_mode}")

        resp = requests.post(
            f"{RAG_SERVER}/query",
            json=payload,
            timeout=120
        )
        resp.raise_for_status()
        data = resp.json()

        state.add_log(f"✅ 응답: ttft={data.get('timing', {}).get('ttft_sec')}s")
        return data
    except Exception as e:
        state.add_log(f"❌ /query 실패: {e}")
        raise


def call_health() -> Dict[str, Any]:
    """GET /health"""
    try:
        resp = requests.get(f"{RAG_SERVER}/health", timeout=10)
        resp.raise_for_status()
        return resp.json()
    except Exception as e:
        state.add_log(f"❌ /health 실패: {e}")
        raise


# ==========================================
# UI 핸들러
# ==========================================
def handle_new_session() -> Tuple[int, str, str]:
    """새 세션 버튼"""
    try:
        session_id = state.new_session()
        call_session_init(session_id)

        return (
            session_id,
            "✅ 새 세션 생성 완료",
            state.get_logs()
        )
    except Exception as e:
        return (
            state.current_session_id or 0,
            f"❌ 세션 생성 실패: {e}",
            state.get_logs()
        )


def handle_send_message(
        session_id: int,
        query: str,
        chat_history: List[Dict[str, str]],
        top_k: int,
        think_mode: str,
        eval_mode: bool,
        include_reasoning: bool
) -> Tuple[List[Dict[str, str]], str, str, str]:
    """메시지 전송"""
    if not query.strip():
        return chat_history, "", "", state.get_logs()

    try:
        # 세션 스위치 (이력 동기화)
        if session_id > 0 and not eval_mode:
            messages = state.get_messages(session_id)
            if messages:
                state.add_log(f"🔄 세션 스위치: {len(messages)}개 이력 동기화")
                call_session_switch(session_id, messages)

        # 쿼리 실행
        state.add_log(f"💬 질문: {query[:50]}...")
        result = call_query(
            session_id if session_id > 0 else None,
            query,
            top_k,
            think_mode,
            eval_mode,
            include_reasoning
        )

        answer = result.get("answer", "")
        contexts = result.get("contexts", [])
        repair_context = result.get("repair_context")
        used_query = result.get("used_query")
        timing = result.get("timing", {})

        # 로그 추가
        state.add_log(f"📊 Timing: {json.dumps(timing, indent=2)}")
        if repair_context:
            state.add_log(f"🔧 Repair Context: {json.dumps(repair_context, ensure_ascii=False, indent=2)}")

        # 세션에 메시지 추가
        if session_id > 0:
            state.add_message(session_id, query, answer)

        # 채팅 히스토리 업데이트
        history = list(chat_history or [])
        history.append({"role": "user", "content": query})
        history.append({"role": "assistant", "content": answer})

        # Context 포맷팅
        context_text = "### 📚 Retrieved Contexts\n\n"
        for i, ctx in enumerate(contexts[:5], 1):
            score = ctx.get("score", 0.0)
            text = ctx.get("text", "")[:300]
            meta = ctx.get("metadata", {})
            file = meta.get("file", meta.get("path", "unknown"))

            context_text += f"**[{i}] {file}** (score: {score:.3f})\n```\n{text}...\n```\n\n"

        # 메타 정보 포맷팅
        meta_text = "### 🔍 메타 정보\n\n"
        if used_query:
            meta_text += f"**개선된 질의:** {used_query}\n\n"

        if repair_context:
            meta_text += "**🔧 쿼리 리페어:**\n"
            if repair_context.get("corrections"):
                meta_text += f"- 정정 대상: {', '.join(repair_context['corrections'][:2])}\n"
            if repair_context.get("questions"):
                meta_text += f"- 확인 질문: {', '.join(repair_context['questions'][:2])}\n"
            if repair_context.get("assumptions"):
                meta_text += f"- 가정: {', '.join(repair_context['assumptions'][:2])}\n"
            meta_text += "\n"

        meta_text += f"**⏱️ Timing:**\n"
        meta_text += f"- TTFT: {timing.get('ttft_sec', 0):.3f}s\n"
        meta_text += f"- Text Gen: {timing.get('text_gen_sec', 0):.3f}s\n"
        meta_text += f"- Retrieval: {timing.get('retrieval_sec', 0):.3f}s\n"
        if timing.get('think_total_sec'):
            meta_text += f"- Think Total: {timing['think_total_sec']:.3f}s\n"

        return (
            history,
            context_text,
            meta_text,
            state.get_logs()
        )

    except Exception as e:
        state.add_log(f"❌ 메시지 처리 실패: {e}")
        import traceback
        state.add_log(traceback.format_exc())

        history = list(chat_history or [])
        history.append({"role": "user", "content": query})
        history.append({"role": "assistant", "content": f"❌ 오류: {e}"})

        return (
            history,
            "",
            "",
            state.get_logs()
        )


def handle_clear() -> Tuple[List, str, str, str]:
    """채팅 초기화"""
    state.add_log("🗑️ 채팅 초기화")
    return ([], "", "", state.get_logs())


def handle_health_check() -> str:
    """헬스체크"""
    try:
        data = call_health()
        state.add_log("✅ Health check 성공")
        return f"```json\n{json.dumps(data, ensure_ascii=False, indent=2)}\n```"
    except Exception as e:
        return f"❌ Health check 실패: {e}"


# ==========================================
# Gradio UI
# ==========================================
def build_ui():
    with gr.Blocks(
            title="RAG 통합 테스트",
            theme=gr.themes.Soft(),
            css="""
        .context-box { max-height: 400px; overflow-y: auto; }
        .log-box { font-family: monospace; font-size: 12px; }
        """
    ) as demo:
        gr.Markdown(
            """
# 🚀 RAG 통합 테스트 챗봇

**기능:**
- ✅ 세션 관리 (초기화/전환)
- ✅ 멀티홉 쿼리 (쿼리 리페어)
- ✅ REVEAL_FROM 필터
- ✅ Context 확인
- ✅ 상세 로그
            """
        )

        with gr.Tabs():
            # ==========================================
            # 탭 1: 채팅
            # ==========================================
            with gr.Tab("💬 채팅"):
                with gr.Row():
                    # 좌측: 채팅 영역
                    with gr.Column(scale=6):
                        chat = gr.Chatbot(
                            label="대화",
                            height=500,
                            type="messages",
                            show_copy_button=True
                        )

                        with gr.Row():
                            session_id_display = gr.Number(
                                label="현재 세션 ID",
                                value=0,
                                interactive=False,
                                precision=0
                            )
                            new_session_btn = gr.Button("🆕 새 세션", variant="primary")
                            clear_btn = gr.Button("🗑️ 초기화", variant="secondary")

                        query_input = gr.Textbox(
                            label="질문",
                            placeholder="질문을 입력하세요 (예: 그거 주기가 어떻게 돼?)",
                            lines=2
                        )

                        with gr.Row():
                            send_btn = gr.Button("📤 전송", variant="primary")

                        with gr.Accordion("⚙️ 설정", open=False):
                            with gr.Row():
                                top_k_slider = gr.Slider(
                                    1, 20,
                                    value=DEFAULT_TOP_K,
                                    step=1,
                                    label="top_k"
                                )
                                think_radio = gr.Radio(
                                    choices=["off", "on"],
                                    value="off",
                                    label="think_mode"
                                )

                            with gr.Row():
                                eval_checkbox = gr.Checkbox(
                                    label="평가 모드 (증강 비활성화)",
                                    value=False
                                )
                                reasoning_checkbox = gr.Checkbox(
                                    label="Reasoning 포함",
                                    value=False
                                )

                    # 우측: Context & 메타
                    with gr.Column(scale=4):
                        context_md = gr.Markdown(
                            "### 📚 Context\n\n질문을 전송하면 여기에 표시됩니다.",
                            elem_classes=["context-box"]
                        )

                        meta_md = gr.Markdown(
                            "### 🔍 메타 정보\n\n타이밍, 리페어 결과 등",
                        )

                # 상태 표시
                status_md = gr.Markdown("준비 완료")

            # ==========================================
            # 탭 2: 로그
            # ==========================================
            with gr.Tab("📋 로그"):
                gr.Markdown("### 🔍 상세 로그 (실시간)")

                log_output = gr.Textbox(
                    label="",
                    lines=30,
                    max_lines=30,
                    interactive=False,
                    elem_classes=["log-box"]
                )

                with gr.Row():
                    refresh_log_btn = gr.Button("🔄 로그 새로고침")
                    clear_log_btn = gr.Button("🗑️ 로그 초기화")

            # ==========================================
            # 탭 3: 헬스체크
            # ==========================================
            with gr.Tab("🏥 Health"):
                gr.Markdown("### 서버 상태 확인")

                health_output = gr.Markdown("버튼을 클릭하세요.")

                health_btn = gr.Button("🔍 Health Check", variant="primary")

        # ==========================================
        # 이벤트 핸들러
        # ==========================================

        # 새 세션
        new_session_btn.click(
            handle_new_session,
            outputs=[session_id_display, status_md, log_output]
        )

        # 메시지 전송
        def send_wrapper(*args):
            return handle_send_message(*args)

        send_btn.click(
            send_wrapper,
            inputs=[
                session_id_display,
                query_input,
                chat,
                top_k_slider,
                think_radio,
                eval_checkbox,
                reasoning_checkbox
            ],
            outputs=[chat, context_md, meta_md, log_output]
        ).then(
            lambda: "",  # 입력창 초기화
            outputs=[query_input]
        )

        query_input.submit(
            send_wrapper,
            inputs=[
                session_id_display,
                query_input,
                chat,
                top_k_slider,
                think_radio,
                eval_checkbox,
                reasoning_checkbox
            ],
            outputs=[chat, context_md, meta_md, log_output]
        ).then(
            lambda: "",
            outputs=[query_input]
        )

        # 초기화
        clear_btn.click(
            handle_clear,
            outputs=[chat, context_md, meta_md, log_output]
        )

        # 로그 새로고침
        refresh_log_btn.click(
            lambda: state.get_logs(),
            outputs=[log_output]
        )

        # 로그 초기화
        clear_log_btn.click(
            lambda: (state.logs.clear(), ""),
            outputs=[log_output]
        )

        # Health check
        health_btn.click(
            handle_health_check,
            outputs=[health_output]
        )

    return demo


# ==========================================
# 메인
# ==========================================
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--server", default=RAG_SERVER, help="RAG 서버 URL")
    parser.add_argument("--port", type=int, default=7866, help="Gradio 포트")
    parser.add_argument("--share", default=True, help="Public URL 생성")
    args = parser.parse_args()

    RAG_SERVER = args.server

    print(f"""
╔═══════════════════════════════════════╗
║   RAG 통합 테스트 챗봇 시작          ║
╠═══════════════════════════════════════╣
║  서버: {RAG_SERVER:30s} ║
║  포트: {args.port:<30d} ║
╚═══════════════════════════════════════╝
    """)

    demo = build_ui()
    demo.queue()
    demo.launch(
        server_name="0.0.0.0",
        server_port=args.port,
        share=args.share,
        show_error=True
    )