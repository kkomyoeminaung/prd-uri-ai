"""
Learning API endpoints
"""

from fastapi import APIRouter, HTTPException, BackgroundTasks
from pydantic import BaseModel
from typing import List, Optional
import logging
import asyncio

from app.learning.web_agent import WebLearningAgent
from app.learning.curriculum import CurriculumPlanner

router = APIRouter()
logger = logging.getLogger(__name__)

web_agent = WebLearningAgent()
curriculum = CurriculumPlanner()

class LearnRequest(BaseModel):
    topic: str
    depth: int = 3
    background: bool = False

class LearnResponse(BaseModel):
    task_id: str
    status: str
    estimated_time: int

@router.post("/start")
async def start_learning(request: LearnRequest, background_tasks: BackgroundTasks):
    if request.background:
        task_id = f"learn_{request.topic.replace(' ', '_')}_{id(request)}"
        background_tasks.add_task(web_agent.learn_topic, request.topic, request.depth)
        return LearnResponse(task_id=task_id, status="started", estimated_time=request.depth*30)
    else:
        result = await web_agent.learn_topic(request.topic, request.depth)
        return result

@router.get("/curriculum/{topic}")
async def get_curriculum(topic: str, depth: int = 3):
    return curriculum.plan(topic, depth)

@router.post("/continuous")
async def start_continuous_learning(topics: List[str], interval_hours: int = 24):
    asyncio.create_task(web_agent.continuous_learning(topics, interval_hours))
    return {"status": "started", "topics": topics, "interval": interval_hours}
