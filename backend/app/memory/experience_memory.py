"""
Experience Memory - stores user interactions with PRD causal patterns
"""

from .vector_store import VectorStore
from typing import List, Dict, Optional
import numpy as np
from sentence_transformers import SentenceTransformer
import logging
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)

class ExperienceMemory:
    def __init__(self, user_id: str, model_name: str = "all-MiniLM-L6-v2"):
        self.user_id = user_id
        self.store = VectorStore("experience", user_id)
        self.encoder = SentenceTransformer(model_name)
        logger.info(f"ExperienceMemory for user {user_id} initialized")
    
    def add_interaction(self, query: str, response: str, causal_weights: Dict[str, float],
                        feedback: Optional[int] = None, session_id: Optional[str] = None) -> int:
        text = f"Q: {query}\nA: {response}"
        embedding = self.encoder.encode(text)
        metadata = {
            'query': query, 'response': response, 'feedback': feedback,
            'session_id': session_id, 'interaction_type': 'chat'
        }
        return self.store.add(text, embedding, causal_weights, metadata, source='user')
    
    def get_recent_interactions(self, hours: int = 24, limit: int = 50) -> List[Dict]:
        cutoff = datetime.utcnow() - timedelta(hours=hours)
        results = []
        for item in self.store.metadata['items']:
            item_time = datetime.fromisoformat(item['timestamp'])
            if item_time > cutoff:
                results.append(item)
        return results[:limit]
    
    def get_common_causal_patterns(self) -> Dict[str, float]:
        if not self.store.metadata['items']:
            return {}
        patterns = {}
        count = 0
        for item in self.store.metadata['items']:
            for cause, weight in item['causal_weights'].items():
                patterns[cause] = patterns.get(cause, 0) + weight
            count += 1
        for cause in patterns:
            patterns[cause] /= count
        return patterns
