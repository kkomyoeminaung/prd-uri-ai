"""
Self-improvement API endpoints
"""

from fastapi import APIRouter, BackgroundTasks
from pydantic import BaseModel
from typing import List, Optional
import logging

from app.improvement.self_play import SelfPlayEngine

router = APIRouter()
logger = logging.getLogger(__name__)

self_play_engine = SelfPlayEngine()

class ImprovementRequest(BaseModel):
    documents: List[dict]
    iterations: int = 1
    background: bool = False

class ImprovementResponse(BaseModel):
    task_id: str
    status: str
    current_generation: int
    average_reward: float

@router.post("/start")
async def start_self_play(request: ImprovementRequest, background_tasks: BackgroundTasks):
    if request.background:
        task_id = f"selfplay_{id(request)}"
        async def run_iterations():
            for i in range(request.iterations):
                await self_play_engine.self_play_iteration(request.documents)
        background_tasks.add_task(run_iterations)
        return ImprovementResponse(
            task_id=task_id,
            status="started",
            current_generation=self_play_engine.generation_count,
            average_reward=self_play_engine.get_progress()['average_reward']
        )
    else:
        for i in range(request.iterations):
            await self_play_engine.self_play_iteration(request.documents)
        progress = self_play_engine.get_progress()
        return ImprovementResponse(
            task_id=f"completed_{id(request)}",
            status="completed",
            current_generation=progress['generations'],
            average_reward=progress['average_reward']
        )

@router.get("/status")
async def get_improvement_status():
    return self_play_engine.get_progress()

@router.post("/reset")
async def reset_self_play():
    global self_play_engine
    self_play_engine = SelfPlayEngine()
    return {"status": "reset"}
