"""Chat Route v4 — Gemini Free API + PRD Causal Injection"""

from fastapi import APIRouter
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from typing import Optional, List, Dict
from datetime import datetime
import uuid, json, logging

from app.core.prd_engine import PRDCausalEngine
from app.core.counselor import CausalCounselor
from app.core.gemini_service import GeminiService
from app.core.config import settings

router = APIRouter()
logger = logging.getLogger(__name__)

_engine    = PRDCausalEngine()
_counselor = CausalCounselor(_engine)
_gemini    = GeminiService(api_key=settings.GEMINI_API_KEY)
_sessions: Dict[str, List[Dict]] = {}


class ChatRequest(BaseModel):
    message: str
    session_id: Optional[str] = None
    scale_L: float = 1e-10

class ChatResponse(BaseModel):
    response: str
    session_id: str
    timestamp: str
    upanissaya_score: float
    asevana_score: float
    alpha_correction: float
    dominant_paccaya: List


@router.post("/", response_model=ChatResponse)
async def chat(req: ChatRequest):
    sid = req.session_id or str(uuid.uuid4())
    if sid not in _sessions: _sessions[sid] = []

    ctx = _counselor.analyze(req.message, scale_L=req.scale_L)
    resp = await _gemini.chat(req.message, ctx["system_prompt"], _sessions[sid][-10:])

    _sessions[sid].append({"role": "user",      "content": req.message})
    _sessions[sid].append({"role": "assistant",  "content": resp})

    return ChatResponse(response=resp, session_id=sid,
        timestamp=datetime.utcnow().isoformat(),
        upanissaya_score=ctx["upanissaya_score"], asevana_score=ctx["asevana_score"],
        alpha_correction=ctx["alpha_correction"], dominant_paccaya=ctx["dominant_paccaya"])


@router.post("/stream")
async def chat_stream(req: ChatRequest):
    sid = req.session_id or str(uuid.uuid4())
    if sid not in _sessions: _sessions[sid] = []
    ctx = _counselor.analyze(req.message, scale_L=req.scale_L)

    async def generate():
        yield f"data: {json.dumps({'type':'meta','session_id':sid,'upanissaya_score':ctx['upanissaya_score'],'asevana_score':ctx['asevana_score'],'alpha_correction':ctx['alpha_correction'],'dominant_paccaya':ctx['dominant_paccaya']})}\n\n"
        full = ""
        async for token in _gemini.chat_stream(req.message, ctx["system_prompt"], _sessions[sid][-10:]):
            full += token
            yield f"data: {json.dumps({'type':'token','text':token})}\n\n"
        _sessions[sid].extend([{"role":"user","content":req.message},{"role":"assistant","content":full}])
        yield f"data: {json.dumps({'type':'done'})}\n\n"

    return StreamingResponse(generate(), media_type="text/event-stream")

@router.delete("/session/{session_id}")
async def clear_session(session_id: str):
    _sessions.pop(session_id, None)
    return {"cleared": session_id}
