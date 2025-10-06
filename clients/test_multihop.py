#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
멀티홉 RAG 테스트 클라이언트
- 세션 생성/전환
- 멀티턴 쿼리 증강 검증
"""
import asyncio
import json
import httpx
from datetime import datetime

BASE = "http://localhost:8001"


async def test_session_init():
    """세션 초기화 테스트"""
    print("\n=== 1. 세션 초기화 ===")
    async with httpx.AsyncClient(timeout=30) as client:
        resp = await client.post(
            f"{BASE}/session/init",
            json={"session_id": 1001, "new_session": True}
        )
        print(f"Status: {resp.status_code}")
        print(f"Response: {resp.json()}")
        assert resp.status_code == 200


async def test_session_switch():
    """세션 스위치 테스트 (메시지 이력 동기화)"""
    print("\n=== 2. 세션 스위치 (이력 동기화) ===")
    messages = [
        {
            "id": 0,
            "user_query": "열차 정비 절차가 뭐야?",
            "rag_answer": "열차 정비는 일상점검, 주기점검, 특별점검으로 나뉩니다.",
            "created_at": "2025-10-03T09:00:00Z"
        },
        {
            "id": 1,
            "user_query": "주기점검 주기는?",
            "rag_answer": "주기점검은 3개월, 6개월, 12개월 주기로 실시됩니다.",
            "created_at": "2025-10-03T09:02:00Z"
        },
        {
            "id": 2,
            "user_query": "특별점검은 언제 해?",
            "rag_answer": "특별점검은 사고 후나 이상 징후 발견 시 즉시 실시합니다.",
            "created_at": "2025-10-03T09:05:00Z"
        }
    ]

    async with httpx.AsyncClient(timeout=60) as client:
        resp = await client.post(
            f"{BASE}/session/switch",
            json={
                "session_id": 1001,
                "new_session": False,
                "messages": messages
            }
        )
        print(f"Status: {resp.status_code}")
        print(f"Response: {resp.json()}")
        assert resp.status_code == 200


async def test_query_naive(session_id=None):
    """일반 쿼리 (세션 없음, Naive RAG)"""
    print("\n=== 3. Naive RAG (세션 없음) ===")
    async with httpx.AsyncClient(timeout=60) as client:
        resp = await client.post(
            f"{BASE}/query",
            json={
                "query": "주기점검의 구체적 절차는?",
                "top_k": 5,
                "eval_mode": False,
                "stream": False
            }
        )
        print(f"Status: {resp.status_code}")
        data = resp.json()
        print(f"Answer: {data.get('answer', '')[:200]}...")
        print(f"Used Query: {data.get('used_query')}")
        print(f"Repair Context: {data.get('repair_context')}")


async def test_query_multihop():
    """멀티홉 쿼리 (세션 기반 증강)"""
    print("\n=== 4. Multihop RAG (세션 1001) ===")
    async with httpx.AsyncClient(timeout=60) as client:
        resp = await client.post(
            f"{BASE}/query",
            json={
                "session_id": 1001,  # ✨ 세션 활성화
                "query": "그거 주기가 정확히 어떻게 돼?",  # 모호한 대명사
                "top_k": 5,
                "eval_mode": False,
                "stream": False
            }
        )
        print(f"Status: {resp.status_code}")
        data = resp.json()
        print(f"\n[Original Query] 그거 주기가 정확히 어떻게 돼?")
        print(f"[Improved Query] {data.get('used_query')}")
        print(f"\n[Repair Context]")
        rc = data.get('repair_context')
        if rc:
            print(f"  - 정정 대상: {rc.get('corrections')}")
            print(f"  - 확인 질문: {rc.get('questions')}")
            print(f"  - 가정: {rc.get('assumptions')}")
        print(f"\n[Answer] {data.get('answer', '')[:300]}...")


async def test_query_eval_mode():
    """평가 모드 (증강 사용 안 함)"""
    print("\n=== 5. 평가 모드 (eval_mode=true) ===")
    async with httpx.AsyncClient(timeout=60) as client:
        resp = await client.post(
            f"{BASE}/query",
            json={
                "session_id": 1001,
                "query": "주기점검 절차 설명해줘",
                "top_k": 5,
                "eval_mode": True,  # ✨ 순수 Naive RAG
                "stream": False
            }
        )
        print(f"Status: {resp.status_code}")
        data = resp.json()
        print(f"Used Query: {data.get('used_query')} (None 예상)")
        print(f"Repair Context: {data.get('repair_context')} (None 예상)")
        print(f"Answer: {data.get('answer', '')[:200]}...")


async def test_streaming():
    """스트리밍 쿼리 (멀티홉)"""
    print("\n=== 6. 스트리밍 (세션 1001) ===")
    async with httpx.AsyncClient(timeout=60) as client:
        async with client.stream(
                "POST",
                f"{BASE}/query?stream=true",
                json={
                    "session_id": 1001,
                    "query": "그 절차 자세히 알려줘",
                    "top_k": 5,
                    "think_mode": "on",
                    "include_reasoning": True
                }
        ) as resp:
            print(f"Status: {resp.status_code}")
            event = None
            async for line in resp.aiter_lines():
                if line.startswith("event:"):
                    event = line.split(":", 1)[1].strip()
                elif line.startswith("data:"):
                    data = line.split(":", 1)[1].strip()
                    if event == "content":
                        print(data, end="", flush=True)
                    elif event == "done":
                        print(f"\n[Timing] {data}")


async def main():
    """전체 테스트 시퀀스"""
    try:
        # 1. 세션 초기화
        await test_session_init()

        # 2. 세션 스위치 (이력 주입)
        await test_session_switch()

        # 3. Naive RAG (세션 없음)
        await test_query_naive()

        # 4. Multihop RAG (세션 기반)
        await test_query_multihop()

        # 5. 평가 모드
        await test_query_eval_mode()

        # 6. 스트리밍
        await test_streaming()

        print("\n\n✅ 모든 테스트 완료!")

    except Exception as e:
        print(f"\n❌ 테스트 실패: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(main())