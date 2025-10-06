#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
REVEAL_FROM 기능 단위 테스트
"""
import pytest
from app.services.generation import _apply_reveal_from_filter, RevealFromBuffer
from app.core.config import global_settings as C

# Mock 설정
C.REVEAL_FROM_ENABLED = True
C.REVEAL_FROM_TOKEN = "<<<FINAL>>>"
C.REVEAL_FALLBACK = "keep_all"


class TestRevealFromNonStreaming:
    """논스트리밍 필터 테스트"""

    def test_token_found(self):
        """토큰 발견 시 이후 텍스트만 반환"""
        text = "prefix <<<FINAL>>> suffix"
        result = _apply_reveal_from_filter(text, "<<<FINAL>>>", "keep_all")
        assert result == "suffix"

    def test_token_not_found_keep_all(self):
        """토큰 미발견 + keep_all → 전체 유지"""
        text = "no token here"
        result = _apply_reveal_from_filter(text, "<<<FINAL>>>", "keep_all")
        assert result == "no token here"

    def test_token_not_found_empty(self):
        """토큰 미발견 + empty → 빈 문자열"""
        text = "no token here"
        result = _apply_reveal_from_filter(text, "<<<FINAL>>>", "empty")
        assert result == ""

    def test_token_not_found_after_think_tag(self):
        """토큰 미발견 + after_think_tag → think 태그 이후"""
        text = "<think>reasoning</think> answer text"
        result = _apply_reveal_from_filter(text, "<<<FINAL>>>", "after_think_tag")
        assert result == "answer text"


class TestRevealFromStreaming:
    """스트리밍 버퍼 테스트"""

    def test_buffer_before_token(self):
        """토큰 전까지 미출력"""
        buf = RevealFromBuffer("<<<FINAL>>>")
        assert buf.add("chunk1 ") == ""
        assert buf.add("chunk2 ") == ""
        assert not buf.revealed

    def test_buffer_token_detected(self):
        """토큰 검출 즉시 이후 토큰 출력"""
        buf = RevealFromBuffer("<<<FINAL>>>")
        buf.add("prefix ")
        result = buf.add("<<<FINAL>>> suffix")
        assert result == " suffix"
        assert buf.revealed

    def test_buffer_after_reveal(self):
        """토큰 이후엔 모든 청크 통과"""
        buf = RevealFromBuffer("<<<FINAL>>>")
        buf.add("<<<FINAL>>> ")
        buf.revealed = True
        assert buf.add("chunk3") == "chunk3"
        assert buf.add("chunk4") == "chunk4"


class TestIntegration:
    """통합 시나리오"""

    @pytest.mark.asyncio
    async def test_streaming_with_reveal_from(self):
        """스트리밍 경로에서 REVEAL_FROM 동작 확인"""
        from app.services.generation import sglang_stream
        from app.core.config import global_settings as C

        # Mock LLM 응답 (실제론 OpenAI 클라이언트 호출)
        # 실제 통합 테스트는 SGLang 서버 필요
        pass  # TODO: httpx-mock 사용한 통합 테스트 추가

    def test_nonstreaming_with_reveal_from(self):
        """논스트리밍 경로에서 REVEAL_FROM 동작 확인"""
        # Mock runtime.handle_query 결과
        text = "preamble <<<FINAL>>> final answer"
        filtered = _apply_reveal_from_filter(text, "<<<FINAL>>>", "keep_all")
        assert filtered == "final answer"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])