import uvicorn
from contextlib import asynccontextmanager
from fastapi import FastAPI
from app.core.runtime import runtime
from app.routers import query, document, health, session
from app.core.config import global_settings as C


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    await runtime.init()
    yield
    # Shutdown (필요 시 정리 작업 추가)


app = FastAPI(title="RAG HTTP API - Multihop", lifespan=lifespan)
app.include_router(query.router)
app.include_router(document.router)
app.include_router(health.router)
app.include_router(session.router)


if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host=C.HOST,
        port=C.PORT,
        reload=False,
        workers=1
    )