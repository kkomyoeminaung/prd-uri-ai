from pydantic import BaseModel
from typing import Optional, List, Dict, Any

class ChatRequest(BaseModel):
    message: str
    session_id: Optional[str] = None
    use_memory: bool = True

class ChatResponse(BaseModel):
    response: str
    causal_weights: Dict[str, float]
    session_id: str
    sources: List[str] = []
    timestamp: str

class MemoryAddRequest(BaseModel):
    text: str
    memory_type: str = "world"
    causal_weights: Optional[Dict[str, float]] = None
    metadata: Optional[Dict[str, Any]] = None

class MemorySearchRequest(BaseModel):
    query: str
    memory_type: str = "world"
    top_k: int = 5

class LearnRequest(BaseModel):
    topic: str
    max_pages: int = 3

class ImproveRequest(BaseModel):
    num_rounds: int = 5
    topic: Optional[str] = None

class GraphQueryRequest(BaseModel):
    query: str
    query_type: str = "effects"
    start: Optional[str] = None
    end: Optional[str] = None
