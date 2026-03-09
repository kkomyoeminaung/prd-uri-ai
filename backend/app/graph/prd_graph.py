"""
PRD Graph Manager - integrates all graph components
"""

from typing import List, Dict, Optional
import logging
from datetime import datetime

from app.graph.causal_graph import PRDCausalGraph
from app.graph.inference import PRDInferenceEngine
from app.memory.world_memory import WorldMemory
from app.memory.opinion_memory import OpinionMemory
from app.core.prd_engine import PRDCausalEngine

logger = logging.getLogger(__name__)

class PRDGraphManager:
    def __init__(self):
        self.graph = PRDCausalGraph()
        self.inference = PRDInferenceEngine(self.graph)
        self.prd_engine = PRDCausalEngine()
        self.last_update = None
        self.update_count = 0
    
    def build_from_memory(self, world_memory: WorldMemory, opinion_memory: OpinionMemory):
        self.graph.build_from_memory(world_memory, opinion_memory)
        self.last_update = datetime.utcnow().isoformat()
        self.update_count += 1
        logger.info(f"Graph updated: {self.graph.graph.number_of_nodes()} nodes, {self.graph.graph.number_of_edges()} edges")
    
    def query(self, query_text: str, query_type: str = "causes", **kwargs) -> Dict:
        result = {
            'query': query_text,
            'type': query_type,
            'timestamp': datetime.utcnow().isoformat(),
            'results': []
        }
        if query_type == "causes":
            result['results'] = self.inference.infer_causes(query_text, **kwargs)
        elif query_type == "effects":
            result['results'] = self.inference.infer_effects(query_text, **kwargs)
        elif query_type == "chain":
            start = kwargs.get('start', '')
            end = kwargs.get('end', '')
            result['results'] = self.inference.causal_chain(start, end, **kwargs)
        elif query_type == "counterfactual":
            intervention = kwargs.get('intervention', '')
            outcome = kwargs.get('outcome', '')
            result['results'] = self.inference.counterfactual(query_text, intervention, outcome)
        else:
            result['error'] = f"Unknown query type: {query_type}"
        return result
    
    def add_knowledge(self, text: str, source: str = "user", metadata: Optional[Dict] = None):
        causal_weights = self.prd_engine.compute_causal_weights(text)
        node_id = str(hash(text + source + str(datetime.utcnow())))
        self.graph.add_node(
            node_id=node_id,
            text=text,
            causal_weights=causal_weights,
            node_type="knowledge",
            metadata={'source': source, **(metadata or {})}
        )
        self._connect_new_node(node_id)
        return node_id
    
    def _connect_new_node(self, node_id: str):
        node_data = self.graph.node_attributes[node_id]
        node_vector = np.array(list(node_data['causal_weights'].values()))
        for other_id, other_data in self.graph.node_attributes.items():
            if other_id == node_id:
                continue
            other_vector = np.array(list(other_data['causal_weights'].values()))
            similarity = np.dot(node_vector, other_vector) / (
                np.linalg.norm(node_vector) * np.linalg.norm(other_vector) + 1e-8)
            if similarity > 0.7:
                self.graph.add_edge(
                    from_id=node_id,
                    to_id=other_id,
                    relation_type='causal_similar',
                    weight=similarity,
                    causal_condition='sahajata'
                )
            elif similarity > 0.5:
                if node_data['causal_weights'].get('hetu', 0) > other_data['causal_weights'].get('hetu', 0):
                    self.graph.add_edge(
                        from_id=node_id,
                        to_id=other_id,
                        relation_type='causal_influence',
                        weight=similarity,
                        causal_condition='hetu'
                    )
                else:
                    self.graph.add_edge(
                        from_id=other_id,
                        to_id=node_id,
                        relation_type='causal_influence',
                        weight=similarity,
                        causal_condition='hetu'
                    )
    
    def get_stats(self) -> Dict:
        return {
            'nodes': self.graph.graph.number_of_nodes(),
            'edges': self.graph.graph.number_of_edges(),
            'last_update': self.last_update,
            'update_count': self.update_count
        }
