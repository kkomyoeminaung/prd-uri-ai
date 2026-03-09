"""
Opinion Memory - stores subjective beliefs with confidence levels
"""

from .vector_store import VectorStore
from typing import List, Dict, Optional
import numpy as np
from sentence_transformers import SentenceTransformer
import logging

logger = logging.getLogger(__name__)

class OpinionMemory:
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self.store = VectorStore("opinion")
        self.encoder = SentenceTransformer(model_name)
        logger.info("OpinionMemory initialized")
    
    def add_opinion(self, statement: str, causal_weights: Dict[str, float],
                    confidence: float, source: str = "inference", 
                    supporting_facts: List[int] = None) -> int:
        embedding = self.encoder.encode(statement)
        metadata = {
            'confidence': confidence,
            'supporting_facts': supporting_facts or [],
            'opinion_type': 'belief'
        }
        return self.store.add(statement, embedding, causal_weights, metadata, source)
    
    def update_confidence(self, opinion_id: int, new_confidence: float) -> bool:
        for i, item in enumerate(self.store.metadata['items']):
            if item['id'] == opinion_id:
                self.store.metadata['items'][i]['metadata']['confidence'] = new_confidence
                self.store._save()
                return True
        return False
    
    def get_confident_opinions(self, threshold: float = 0.7) -> List[Dict]:
        return [item for item in self.store.metadata['items'] 
                if item['metadata'].get('confidence', 0) >= threshold]
    
    def resolve_conflict(self, opinion1_id: int, opinion2_id: int) -> Optional[Dict]:
        opinion1 = None
        opinion2 = None
        for item in self.store.metadata['items']:
            if item['id'] == opinion1_id:
                opinion1 = item
            if item['id'] == opinion2_id:
                opinion2 = item
        if not opinion1 or not opinion2:
            return None
        w1 = np.array(list(opinion1['causal_weights'].values()))
        w2 = np.array(list(opinion2['causal_weights'].values()))
        similarity = np.dot(w1, w2) / (np.linalg.norm(w1) * np.linalg.norm(w2) + 1e-8)
        if similarity < 0.5:
            conf1 = opinion1['metadata'].get('confidence', 0.5)
            conf2 = opinion2['metadata'].get('confidence', 0.5)
            return opinion1 if conf1 > conf2 else opinion2
        combined = opinion1.copy()
        combined['metadata']['confidence'] = max(
            opinion1['metadata'].get('confidence', 0.5),
            opinion2['metadata'].get('confidence', 0.5)
        )
        combined['metadata']['supporting_facts'] = list(set(
            opinion1['metadata'].get('supporting_facts', []) +
            opinion2['metadata'].get('supporting_facts', [])
        ))
        return combined
