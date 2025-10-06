from __future__ import annotations

import asyncio
import json
from typing import AsyncIterator, Any

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import JSONResponse, StreamingResponse

from ..core.config import global_settings as C
from ..core.runtime import runtime
from ..core.models import QueryRequest

router = APIRouter(prefix="", tags=["query"])


def _jsonable(data: Any) -> str:
    """SSE data 직렬화: dict/list는 JSON, 그 외는 str."""
    if isinstance(data, (dict, list, tuple)):
        return json.dumps(data, ensure_ascii=False)
    if isinstance(data, (bytes, bytearray)):
        try:
            return data.decode("utf-8")
        except Exception:
            return str(data)
    return str(data)


def _format_sse(event: str, payload: str) -> bytes:
    """
    SSE 프레임:
      event: <name>
      data: <line1>
      data: <line2>
      \n
    """
    lines = payload.splitlines() or [""]
    out = [f"event: {event}"]
    for ln in lines:
        out.append(f"data: {ln}")
    out.append("")
    return ("\n".join(out) + "\n").encode("utf-8")


def _format_message_compat(event: str, text_payload: str) -> bytes:
    """
    이벤트명을 쓰지 않는 클라이언트 호환용 기본(message) 프레임.
    data: {"event":"content","text":"..."} 형태(JSON 한 줄)
    """
    obj = {"event": event, "text": text_payload}
    return f"data: {json.dumps(obj, ensure_ascii=False)}\n\n".encode("utf-8")


async def _sse_iter(req: QueryRequest) -> AsyncIterator[bytes]:
    """
    runtime.handle_query_stream(req) -> async generator of (event, data)
    를 양쪽 호환 프레임으로 변환.
    """
    # 워밍업: 초기 버퍼링 해제 유도
    yield b": warmup\n\n"

    try:
        async for event, data in _to_async_iter(runtime.handle_query_stream(req)):
            ev = str(event or "content")
            payload = _jsonable(data)

            # 1) 기본(message) 이벤트 호환 프레임
            yield _format_message_compat(ev, payload)

            # 2) 표준 event 프레임
            yield _format_sse(ev, payload)

    except Exception as e:
        msg = json.dumps({"message": str(e)}, ensure_ascii=False)
        yield _format_message_compat("error", msg)
        yield _format_sse("error", msg)


async def _to_async_iter(gen):
    """타입 힌트 혼선을 피하기 위한 래퍼."""
    async for item in gen:
        yield item


def _parse_bool(s: str) -> bool:
    return s.strip().lower() in {"1", "true", "t", "yes", "y", "on"}


@router.post("/query")
async def query(request: Request, body: QueryRequest):
    """
    POST /query
      - stream 우선순위: querystring ?stream= → body.stream → STREAM_DEFAULT
      - 비스트리밍: JSON
      - 스트리밍: SSE (content/reasoning/ttft/done/error)
    """
    # 1) 쿼리스트링 stream 우선
    qs_stream = request.query_params.get("stream")
    if qs_stream is not None:
        use_stream = _parse_bool(qs_stream)
    else:
        # 2) 바디 stream
        use_stream = body.stream if body.stream is not None else bool(C.STREAM_DEFAULT)

    async def _run_once():
        if use_stream:
            headers = {
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no",  # nginx 버퍼링 방지
            }
            return StreamingResponse(
                _sse_iter(body),
                media_type="text/event-stream; charset=utf-8",
                headers=headers,
            )
        else:
            try:
                result = await runtime.handle_query(body)
                return JSONResponse(result)
            except RuntimeError as e:
                # 인덱스 미로드 등
                raise HTTPException(status_code=409, detail=str(e))
            except Exception as e:
                raise HTTPException(status_code=500, detail=str(e))

    try:
        # 3.10 호환 타임아웃
        return await asyncio.wait_for(_run_once(), timeout=C.REQUEST_TIMEOUT_SEC)
    except asyncio.TimeoutError:
        raise HTTPException(status_code=504, detail="Request timed out")
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
