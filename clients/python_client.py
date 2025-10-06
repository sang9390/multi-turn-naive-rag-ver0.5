import sys, json, httpx, asyncio

BASE = "http://localhost:8000"

async def query_non_stream(q: str):
    async with httpx.AsyncClient(timeout=120) as h:
        r = await h.post(f"{BASE}/query", json={"query": q, "think_mode": "on", "include_reasoning": True, "stream": False})
        r.raise_for_status()
        print(json.dumps(r.json(), ensure_ascii=False, indent=2))

async def query_stream(q: str):
    async with httpx.AsyncClient(timeout=120) as h:
        async with h.stream("POST", f"{BASE}/query?stream=true", json={"query": q, "think_mode": "on", "include_reasoning": True}) as r:
            async for line in r.aiter_lines():
                if line.startswith("event:") or line.startswith("data:"):
                    print(line)

if __name__ == "__main__":
    q = sys.argv[1] if len(sys.argv) > 1 else "절차 요약 알려줘"
    asyncio.run(query_stream(q))