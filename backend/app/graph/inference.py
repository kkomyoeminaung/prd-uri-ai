"""
Causal Inference Engine using PRD graph
"""

from typing import List, Dict, Optional
import numpy as np
import logging

from app.graph.causal_graph import PRDCausalGraph
from app.core.prd_engine import PRDCausalEngine

logger = logging.getLogger(__name__)

class PRDInferenceEngine:
    def __init__(self, causal_graph: PRDCausalGraph):
        self.graph = causal_graph
        self.prd_engine = PRDCausalEngine()
    
    def infer_causes(self, effect_text: str, k: int = 10) -> List[Dict]:
        effect_nodes = self._find_matching_nodes(effect_text)
        if not effect_nodes:
            return []
        causes = []
        for node_id, node_data in effect_nodes:
            ancestors = self.graph.find_common_causes([node_id])
            causes.extend(ancestors)
        seen = set()
        unique_causes = []
        for cause in causes:
            if cause['node_id'] not in seen:
                seen.add(cause['node_id'])
                unique_causes.append(cause)
        return sorted(unique_causes, key=lambda x: -x['average_strength'])[:k]
    
    def infer_effects(self, cause_text: str, k: int = 10) -> List[Dict]:
        cause_nodes = self._find_matching_nodes(cause_text)
        if not cause_nodes:
            return []
        effects = []
        for node_id, node_data in cause_nodes:
            node_effects = self.graph.infer_effect(node_id)
            effects.extend(node_effects)
        seen = set()
        unique_effects = []
        for effect in effects:
            if effect['node_id'] not in seen:
                seen.add(effect['node_id'])
                unique_effects.append(effect)
        return sorted(unique_effects, key=lambda x: -x['path_strength'])[:k]
    
    def causal_chain(self, start_text: str, end_text: str, max_depth: int = 5) -> List[Dict]:
        start_nodes = self._find_matching_nodes(start_text)
        end_nodes = self._find_matching_nodes(end_text)
        if not start_nodes or not end_nodes:
            return []
        chains = []
        for s_id, s_data in start_nodes:
            for e_id, e_data in end_nodes:
                paths = self.graph.find_causal_path(s_id, e_id, max_depth)
                chains.extend(paths)
        return sorted(chains, key=lambda x: -x['total_weight'])
    
    def counterfactual(self, condition: str, intervention: str, outcome: str) -> Dict:
        condition_nodes = self._find_matching_nodes(condition)
        intervention_nodes = self._find_matching_nodes(intervention)
        outcome_nodes = self._find_matching_nodes(outcome)
        if not condition_nodes or not intervention_nodes or not outcome_nodes:
            return {'error': 'Nodes not found'}
        paths = []
        for c_id, _ in condition_nodes:
            for o_id, _ in outcome_nodes:
                paths.extend(self.graph.find_causal_path(c_id, o_id))
        if not paths:
            return {'result': 'No causal relationship found'}
        intervention_ids = [n[0] for n in intervention_nodes]
        affected = False
        for path in paths:
            if any(i in path['nodes'] for i in intervention_ids):
                affected = True
                break
        return {
            'result': 'affected' if affected else 'not_affected',
            'paths_analyzed': len(paths),
            'intervention_present': affected
        }
    
    def _find_matching_nodes(self, text: str, threshold: float = 0.5) -> List[Tuple[str, Dict]]:
        query_causal = self.prd_engine.compute_causal_weights(text)
        query_vector = np.array(list(query_causal.values()))
        matches = []
        for node_id, attrs in self.graph.node_attributes.items():
            node_vector = np.array(list(attrs['causal_weights'].values()))
            similarity = np.dot(query_vector, node_vector) / (
                np.linalg.norm(query_vector) * np.linalg.norm(node_vector) + 1e-8)
            if similarity > threshold:
                matches.append((node_id, attrs))
        return sorted(matches, key=lambda x: -self._text_similarity(text, x[1]['text']))[:20]
    
    def _text_similarity(self, text1: str, text2: str) -> float:
        words1 = set(text1.lower().split())
        words2 = set(text2.lower().split())
        intersection = words1 & words2
        union = words1 | words2
        return len(intersection) / max(len(union), 1)
