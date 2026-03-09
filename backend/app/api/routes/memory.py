"""
Memory API endpoints
"""

from fastapi import APIRouter, HTTPException, Depends
from pydantic import BaseModel
from typing import Optional, Dict, List
import logging

from app.memory.world_memory import WorldMemory
from app.memory.experience_memory import ExperienceMemory
from app.memory.opinion_memory import OpinionMemory
from app.core.prd_engine import PRDCausalEngine

router = APIRouter()
logger = logging.getLogger(__name__)

world_memory = WorldMemory()
opinion_memory = OpinionMemory()
prd_engine = PRDCausalEngine()

class MemoryRequest(BaseModel):
    user_id: str
    text: str
    metadata: Optional[Dict] = {}

class MemoryResponse(BaseModel):
    id: int
    text: str
    causal_weights: Dict[str, float]
    similarity: float
    source: str
    timestamp: str

@router.post("/world/add")
async def add_to_world(request: MemoryRequest):
    causal_weights = prd_engine.compute_causal_weights(request.text)
    item_id = world_memory.add_fact(
        fact=request.text,
        causal_weights=causal_weights,
        metadata=request.metadata
    )
    return {"id": item_id, "status": "added"}

@router.post("/world/search")
async def search_world(request: MemoryRequest):
    results = world_memory.search_facts(request.text, k=10)
    return [MemoryResponse(
        id=r['id'], text=r['text'], causal_weights=r['causal_weights'],
        similarity=r.get('similarity', 0), source=r.get('source', 'unknown'),
        timestamp=r['timestamp']
    ) for r in results]

@router.get("/experience/{user_id}/recent")
async def get_recent_experiences(user_id: str, hours: int = 24):
    exp_memory = ExperienceMemory(user_id)
    return exp_memory.get_recent_interactions(hours)

@router.post("/opinion/add")
async def add_opinion(request: MemoryRequest):
    causal_weights = prd_engine.compute_causal_weights(request.text)
    item_id = opinion_memory.add_opinion(
        statement=request.text,
        causal_weights=causal_weights,
        confidence=request.metadata.get('confidence', 0.5),
        source=request.metadata.get('source', 'user')
    )
    return {"id": item_id, "status": "added"}

@router.get("/stats")
async def get_memory_stats():
    return {
        "world": world_memory.store.get_stats(),
        "opinion": opinion_memory.store.get_stats()
    }
