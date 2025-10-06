#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
전체 엔드포인트 스모크 테스트 (멀티홉 통합)
- GET /health
- POST /session/init
- POST /session/switch
- POST /query (non-stream, Naive RAG)
- POST /query (non-stream, Multihop RAG)
- POST /query (eval_mode=true)
- POST /query (SSE stream)
"""
import asyncio
import httpx
import json
import time
import argparse
from datetime import datetime

DEFAULT_BASE = "http://localhost:8001"

GREEN = "\033[32m"
RED = "\033[31m"
YELLOW = "\033[33m"
BOLD = "\033[1m"
RESET = "\033[0m"


def ok(msg): print(f"{GREEN}✔{RESET} {msg}")


def fail(msg): print(f"{RED}✘{RESET} {msg}")


def info(msg): print(f"{YELLOW}…{RESET} {msg}")


def bold(msg): print(f"{BOLD}{msg}{RESET}")


async def check_health(client: httpx.AsyncClient, base: str):
    """1. GET /health"""
    url = f"{base}/health"
    info(f"GET {url}")
    r = await client.get(url)
    if r.status_code != 200:
        fail(f"/health status={r.status_code}")
        return False, {}
    data = r.json()
    if data.get("status") != "ok":
        fail("/health missing status=ok")
        return False, data
    ok(f"/health → {data.get('version', 'unknown')}")
    return True, data


async def test_session_init(client: httpx.AsyncClient, base: str, session_id: int = 9001):
    """2. POST /session/init"""
    url = f"{base}/session/init"
    payload = {"session_id": session_id, "new_session": True}
    info(f"POST {url} {payload}")

    r = await client.post(url, json=payload)
    if r.status_code != 200:
        fail(f"/session/init status={r.status_code} body={r.text[:200]}")
        return False, {}

    data = r.json()
    if data.get("status") != "ok":
        fail(f"/session/init non-ok: {data}")
        return False, data

    ok(f"/session/init → session_id={data.get('session_id')}")
    return True, data


async def test_session_switch(client: httpx.AsyncClient, base: str, session_id: int = 9001):
    """3. POST /session/switch (이력 동기화)"""
    url = f"{base}/session/switch"

    # 샘플 메시지 이력
    messages = [
        {
            "id": 0,
            "user_query": "열차 정비 절차가 뭐야?",
            "rag_answer": "열차 정비는 일상점검, 주기점검, 특별점검으로 나뉩니다. 일상점검은 매일, 주기점검은 3/6/12개월, 특별점검은 사고 발생 시 실시합니다.",
            "created_at": "2025-10-04T09:00:00Z"
        },
        {
            "id": 1,
            "user_query": "주기점검 주기는?",
            "rag_answer": "주기점검은 3개월, 6개월, 12개월 주기로 실시됩니다. 각 주기별로 점검 항목이 상이합니다.",
            "created_at": "2025-10-04T09:02:00Z"
        },
        {
            "id": 2,
            "user_query": "특별점검은 언제?",
            "rag_answer": "특별점검은 사고 발생 시, 이상 징후 발견 시, 또는 정비 책임자 지시 시 즉시 실시합니다.",
            "created_at": "2025-10-04T09:05:00Z"
        }
    ]

    payload = {
        "session_id": session_id,
        "new_session": False,
        "messages": messages
    }

    info(f"POST {url} (messages={len(messages)})")
    t0 = time.perf_counter()
    r = await client.post(url, json=payload, timeout=60)
    dt = time.perf_counter() - t0

    if r.status_code != 200:
        fail(f"/session/switch status={r.status_code} body={r.text[:200]}")
        return False, {}

    data = r.json()
    if data.get("status") != "ok":
        fail(f"/session/switch non-ok: {data}")
        return False, data

    ok(f"/session/switch → {dt:.2f}s (요약+인덱싱 완료)")
    return True, data


async def query_naive(client: httpx.AsyncClient, base: str, q: str):
    """4. POST /query (Naive RAG, 세션 없음)"""
    url = f"{base}/query"
    payload = {
        "query": q,
        "top_k": 5,
        "think_mode": "off",
        "stream": False,
        "eval_mode": False
    }
    info(f"POST {url} (Naive RAG) '{q}'")
    t0 = time.perf_counter()
    r = await client.post(url, json=payload)
    dt = time.perf_counter() - t0

    if r.status_code != 200:
        fail(f"/query Naive status={r.status_code} body={r.text[:200]}")
        return False, {}

    data = r.json()
    if "answer" not in data:
        fail("/query Naive missing answer")
        return False, data

    answer_preview = data["answer"][:80] + "..." if len(data["answer"]) > 80 else data["answer"]
    used_query = data.get("used_query")
    repair = data.get("repair_context")

    ok(f"/query Naive → {dt:.2f}s, ttft={data.get('timing', {}).get('ttft_sec')}")
    print(f"   Answer: {answer_preview}")
    if used_query:
        print(f"   Used Query: {used_query}")
    if repair:
        print(f"   Repair: {repair}")
    return True, data


async def query_multihop(client: httpx.AsyncClient, base: str, session_id: int, q: str):
    """5. POST /query (Multihop RAG, 세션 기반)"""
    url = f"{base}/query"
    payload = {
        "session_id": session_id,
        "query": q,
        "top_k": 5,
        "think_mode": "off",
        "stream": False,
        "eval_mode": False
    }
    info(f"POST {url} (Multihop RAG) session={session_id} '{q}'")
    t0 = time.perf_counter()
    r = await client.post(url, json=payload, timeout=60)
    dt = time.perf_counter() - t0

    if r.status_code != 200:
        fail(f"/query Multihop status={r.status_code} body={r.text[:200]}")
        return False, {}

    data = r.json()
    if "answer" not in data:
        fail("/query Multihop missing answer")
        return False, data

    answer_preview = data["answer"][:80] + "..." if len(data["answer"]) > 80 else data["answer"]
    used_query = data.get("used_query")
    repair = data.get("repair_context")

    ok(f"/query Multihop → {dt:.2f}s, ttft={data.get('timing', {}).get('ttft_sec')}")
    print(f"   Original: {q}")
    if used_query:
        print(f"   Improved: {used_query}")
    if repair:
        rc = repair
        if rc.get("corrections"):
            print(f"   정정: {rc['corrections'][:2]}")
        if rc.get("questions"):
            print(f"   질문: {rc['questions'][:2]}")
    print(f"   Answer: {answer_preview}")
    return True, data


async def query_eval_mode(client: httpx.AsyncClient, base: str, session_id: int, q: str):
    """6. POST /query (eval_mode=true, 순수 Naive RAG)"""
    url = f"{base}/query"
    payload = {
        "session_id": session_id,
        "query": q,
        "top_k": 5,
        "eval_mode": True,  # 증강 비활성화
        "stream": False
    }
    info(f"POST {url} (eval_mode=true) '{q}'")
    t0 = time.perf_counter()
    r = await client.post(url, json=payload)
    dt = time.perf_counter() - t0

    if r.status_code != 200:
        fail(f"/query eval_mode status={r.status_code} body={r.text[:200]}")
        return False, {}

    data = r.json()
    if "answer" not in data:
        fail("/query eval_mode missing answer")
        return False, data

    # eval_mode에서는 repair_context가 None이어야 함
    if data.get("repair_context") is not None:
        fail("/query eval_mode should not have repair_context")
        return False, data

    ok(f"/query eval_mode → {dt:.2f}s (repair_context=None 확인)")
    return True, data


async def query_stream(client: httpx.AsyncClient, base: str, session_id: int, q: str, timeout: int = 120):
    """7. POST /query (SSE stream, Multihop)"""
    url = f"{base}/query?stream=true"
    payload = {
        "session_id": session_id,
        "query": q,
        "top_k": 5,
        "think_mode": "on",
        "include_reasoning": True
    }
    info(f"POST {url} (SSE) session={session_id} '{q}'")

    done_payload = {}
    got_content = False
    got_reasoning = False

    try:
        async with client.stream("POST", url, json=payload, timeout=timeout) as r:
            if r.status_code != 200:
                body = await r.aread()
                fail(f"/query stream status={r.status_code} body={body[:200]}")
                return False, {}

            cur_event = None
            cur_data_lines = []

            async for raw in r.aiter_lines():
                if not raw:  # event boundary
                    if cur_event and cur_data_lines:
                        data_str = "\n".join(
                            line[len("data: "):] if line.startswith("data: ") else line
                            for line in cur_data_lines
                        )
                        if cur_event == "content":
                            got_content = True
                        elif cur_event == "reasoning":
                            got_reasoning = True
                        elif cur_event == "done":
                            try:
                                done_payload = json.loads(data_str)
                            except Exception:
                                pass
                    cur_event = None
                    cur_data_lines = []
                    continue

                if raw.startswith("event:"):
                    cur_event = raw.split("event:", 1)[1].strip()
                elif raw.startswith("data:"):
                    cur_data_lines.append(raw)

    except Exception as e:
        fail(f"/query stream exception: {e}")
        return False, {}

    if not got_content:
        fail("/query stream: no content tokens")
        return False, done_payload

    if "timing" not in done_payload:
        fail("/query stream: missing final 'done' timing")
        return False, done_payload

    t = done_payload.get("timing", {})
    ok(f"/query stream → content={'✓' if got_content else '✗'}, reasoning={'✓' if got_reasoning else '✗'}, ttft={t.get('ttft_sec')}")
    return True, done_payload


async def main():
    ap = argparse.ArgumentParser(description="전체 엔드포인트 스모크 테스트 (멀티홉 통합)")
    ap.add_argument("--base", default=DEFAULT_BASE, help=f"Server base (default: {DEFAULT_BASE})")
    ap.add_argument("--timeout", type=int, default=120, help="HTTP timeout seconds")
    ap.add_argument("--session-id", type=int, default=9001, help="Test session ID")
    args = ap.parse_args()

    base = args.base.rstrip("/")
    session_id = args.session_id
    results = []

    bold("\n=== RAG 멀티홉 스모크 테스트 시작 ===\n")

    async with httpx.AsyncClient(timeout=args.timeout) as client:
        # 1. Health check
        ok1, data1 = await check_health(client, base)
        results.append(("GET /health", ok1, data1))

        if not ok1:
            fail("Health check 실패, 서버 확인 필요")
            return

        # 2. Session init
        ok2, data2 = await test_session_init(client, base, session_id)
        results.append(("POST /session/init", ok2, data2))

        # 3. Session switch (이력 동기화)
        ok3, data3 = await test_session_switch(client, base, session_id)
        results.append(("POST /session/switch", ok3, data3))

        # 4. Naive RAG (세션 없음)
        ok4, data4 = await query_naive(client, base, "열차 정비 절차 요약해줘")
        results.append(("POST /query (Naive)", ok4, data4))

        # 5. Multihop RAG (모호한 질의)
        ok5, data5 = await query_multihop(client, base, session_id, "그거 주기가 정확히 어떻게 돼?")
        results.append(("POST /query (Multihop)", ok5, data5))

        # 6. Eval mode (증강 비활성화)
        ok6, data6 = await query_eval_mode(client, base, session_id, "주기점검 절차 알려줘")
        results.append(("POST /query (eval_mode=true)", ok6, data6))

        # 7. SSE stream
        ok7, data7 = await query_stream(client, base, session_id, "특별점검 자세히 설명해줘", timeout=args.timeout)
        results.append(("POST /query?stream=true (SSE)", ok7, data7))

    bold("\n=== 테스트 결과 요약 ===")
    passed = sum(1 for _, okflag, _ in results if okflag)

    for name, okflag, _ in results:
        status = f"{GREEN}✔{RESET}" if okflag else f"{RED}✘{RESET}"
        print(f"{status} {name}")

    print(f"\n{BOLD}총 {passed}/{len(results)} 테스트 통과{RESET}")

    if passed == len(results):
        bold("\n🎉 모든 테스트 성공!")
    else:
        bold("\n⚠️  일부 테스트 실패")
        print("\n실패한 엔드포인트 확인:")
        for name, okflag, data in results:
            if not okflag:
                print(f"  - {name}: {data}")


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n중단됨.")