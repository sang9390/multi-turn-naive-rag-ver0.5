import asyncio, httpx, time

BASE = "http://localhost:8000"

async def one(i):
    q = f"테스트 질의 {i}"
    async with httpx.AsyncClient(timeout=60) as h:
        r = await h.post(f"{BASE}/query", json={"query": q, "think_mode": "off", "stream": False})
        return r.status_code

async def main(n=20):
    t0 = time.perf_counter()
    rs = await asyncio.gather(*[one(i) for i in range(n)], return_exceptions=True)
    dt = time.perf_counter() - t0
    ok = sum(1 for r in rs if isinstance(r, int) and r == 200)
    print(f"done: {ok}/{n} in {dt:.2f}s")

if __name__ == "__main__":
    asyncio.run(main())