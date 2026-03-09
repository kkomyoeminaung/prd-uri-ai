"""
World Memory - stores factual knowledge with PRD causal tagging
"""

from .vector_store import VectorStore
from typing import List, Dict, Optional
import numpy as np
from sentence_transformers import SentenceTransformer
import logging

logger = logging.getLogger(__name__)

class WorldMemory:
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self.store = VectorStore("world")
        self.encoder = SentenceTransformer(model_name)
        logger.info(f"WorldMemory initialized with {self.store.get_stats()['total_items']} items")
    
    def add_fact(self, fact: str, causal_weights: Dict[str, float], 
                 source: str = "web", confidence: float = 1.0, 
                 metadata: Optional[Dict] = None) -> int:
        embedding = self.encoder.encode(fact)
        meta = {'confidence': confidence, 'fact_type': 'fact', **(metadata or {})}
        return self.store.add(fact, embedding, causal_weights, meta, source)
    
    def search_facts(self, query: str, k: int = 10) -> List[Dict]:
        query_embedding = self.encoder.encode(query)
        return self.store.search(query_embedding, k)
    
    def get_facts_by_topic(self, topic: str, k: int = 20) -> List[Dict]:
        results = []
        for item in self.store.metadata['items']:
            if topic.lower() in item['text'].lower():
                results.append(item)
        return results[:k]
