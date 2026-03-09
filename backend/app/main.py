from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import logging
from app.core.config import settings
from app.api.routes import chat, causal, memory

logging.basicConfig(level=settings.LOG_LEVEL)

app = FastAPI(
    title="PRD-URI AI v4",
    description="Unified Relational Intelligence — SU(5) Causal AGI (Gemini Free)",
    version="4.0.0",
    docs_url="/api/docs",
)

app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])
app.include_router(chat.router,   prefix="/api/chat",   tags=["Chat"])
app.include_router(causal.router, prefix="/api/causal", tags=["Causal"])
app.include_router(memory.router, prefix="/api/memory", tags=["Memory"])

@app.get("/")
async def root():
    return {"app": "PRD-URI AI v4", "llm": "Google Gemini Free", "alpha": 1.274, "generators": 24}

@app.get("/api/health")
async def health():
    has_key = bool(settings.GEMINI_API_KEY)
    return {"status": "ok", "gemini_api": "configured" if has_key else "⚠️ missing GEMINI_API_KEY", "model": settings.GEMINI_MODEL}
