"""
Vector database with FAISS for PRD memory storage
"""

import faiss
import numpy as np
import json
import os
from typing import List, Dict, Optional, Any
import logging
from datetime import datetime
import pickle

logger = logging.getLogger(__name__)

class VectorStore:
    """
    FAISS-based vector store with PRD causal tagging
    """
    
    def __init__(self, store_type: str, user_id: Optional[str] = None, dimension: int = 384):
        self.store_type = store_type
        self.user_id = user_id
        self.dimension = dimension
        
        if user_id:
            self.base_path = f"memory/vector_stores/{store_type}/{user_id}/"
        else:
            self.base_path = f"memory/vector_stores/{store_type}/"
        
        os.makedirs(self.base_path, exist_ok=True)
        
        self.index_path = os.path.join(self.base_path, "index.faiss")
        self.meta_path = os.path.join(self.base_path, "metadata.pkl")
        
        self._load_or_create_index()
    
    def _load_or_create_index(self):
        if os.path.exists(self.index_path) and os.path.exists(self.meta_path):
            self.index = faiss.read_index(self.index_path)
            with open(self.meta_path, 'rb') as f:
                self.metadata = pickle.load(f)
            logger.info(f"Loaded existing index with {self.index.ntotal} vectors")
        else:
            self.index = faiss.IndexFlatL2(self.dimension)
            self.metadata = {'items': [], 'id_counter': 0}
            logger.info(f"Created new index")
    
    def _save(self):
        faiss.write_index(self.index, self.index_path)
        with open(self.meta_path, 'wb') as f:
            pickle.dump(self.metadata, f)
    
    def add(self, text: str, embedding: np.ndarray, causal_weights: Dict[str, float], 
            metadata: Optional[Dict] = None, source: str = "user") -> int:
        item_id = self.metadata['id_counter']
        self.index.add(embedding.reshape(1, -1).astype(np.float32))
        
        item = {
            'id': item_id,
            'text': text,
            'causal_weights': causal_weights,
            'source': source,
            'timestamp': datetime.utcnow().isoformat(),
            'access_count': 0,
            'metadata': metadata or {}
        }
        self.metadata['items'].append(item)
        self.metadata['id_counter'] += 1
        self._save()
        logger.info(f"Added item {item_id} to {self.store_type} store")
        return item_id
    
    def search(self, query_embedding: np.ndarray, k: int = 10, threshold: float = 0.0) -> List[Dict]:
        if self.index.ntotal == 0:
            return []
        
        distances, indices = self.index.search(
            query_embedding.reshape(1, -1).astype(np.float32), 
            min(k, self.index.ntotal)
        )
        
        results = []
        for i, idx in enumerate(indices[0]):
            if idx >= 0 and idx < len(self.metadata['items']):
                item = self.metadata['items'][idx].copy()
                item['distance'] = float(distances[0][i])
                item['similarity'] = 1.0 / (1.0 + item['distance'])
                self.metadata['items'][idx]['access_count'] += 1
                results.append(item)
        
        self._save()
        return results
    
    def search_by_causal(self, causal_weights: Dict[str, float], k: int = 10) -> List[Dict]:
        if not self.metadata['items']:
            return []
        
        query_vector = np.array([causal_weights.get(name, 0) for name in 
                                 ['hetu','nissaya','indriya','avigata',
                                  'anantara','vigata','sahajata','annamanna']])
        
        scored = []
        for item in self.metadata['items']:
            item_vector = np.array([item['causal_weights'].get(name, 0) for name in 
                                   ['hetu','nissaya','indriya','avigata',
                                    'anantara','vigata','sahajata','annamanna']])
            similarity = np.dot(query_vector, item_vector) / (
                np.linalg.norm(query_vector) * np.linalg.norm(item_vector) + 1e-8)
            scored.append((similarity, item))
        
        scored.sort(key=lambda x: x[0], reverse=True)
        return [item for sim, item in scored[:k]]
    
    def get_stats(self) -> Dict:
        return {
            'total_items': len(self.metadata['items']),
            'store_type': self.store_type,
            'user_id': self.user_id,
            'dimension': self.dimension
        }
