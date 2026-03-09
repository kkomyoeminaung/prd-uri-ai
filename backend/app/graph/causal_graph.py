"""
PRD Causal Graph - Network of causal relationships
"""

import networkx as nx
import numpy as np
from typing import List, Dict, Optional, Tuple
import logging
from collections import defaultdict
import json
import os

from app.core.prd_engine import PRDCausalEngine

logger = logging.getLogger(__name__)

class PRDCausalGraph:
    def __init__(self):
        self.graph = nx.DiGraph()
        self.prd_engine = PRDCausalEngine()
        self.node_attributes = {}
        self.edge_attributes = defaultdict(dict)
        self.graph_path = "memory/causal_graph.json"
        self._load()
    
    def add_node(self, node_id: str, text: str, causal_weights: Dict[str, float], 
                 node_type: str = "fact", metadata: Optional[Dict] = None):
        self.graph.add_node(node_id)
        self.node_attributes[node_id] = {
            'text': text,
            'causal_weights': causal_weights,
            'type': node_type,
            'metadata': metadata or {},
            'embedding': None
        }
    
    def add_edge(self, from_id: str, to_id: str, relation_type: str, 
                 weight: float = 1.0, causal_condition: str = None):
        self.graph.add_edge(from_id, to_id)
        self.edge_attributes[(from_id, to_id)] = {
            'relation_type': relation_type,
            'weight': weight,
            'causal_condition': causal_condition,
            'strength': weight
        }
    
    def find_causal_path(self, start_id: str, end_id: str, max_depth: int = 5) -> List[Dict]:
        if start_id not in self.graph or end_id not in self.graph:
            return []
        paths = []
        try:
            all_paths = nx.all_simple_paths(self.graph, start_id, end_id, cutoff=max_depth)
            for path in all_paths:
                path_info = {
                    'nodes': path,
                    'edges': [],
                    'total_weight': 1.0,
                    'causal_chain': []
                }
                for i in range(len(path)-1):
                    edge_data = self.edge_attributes.get((path[i], path[i+1]), {})
                    path_info['edges'].append({
                        'from': path[i], 'to': path[i+1],
                        'type': edge_data.get('relation_type', 'unknown'),
                        'causal_condition': edge_data.get('causal_condition')
                    })
                    path_info['total_weight'] *= edge_data.get('weight', 1.0)
                    if edge_data.get('causal_condition'):
                        path_info['causal_chain'].append(edge_data['causal_condition'])
                paths.append(path_info)
        except nx.NetworkXNoPath:
            pass
        return paths
    
    def find_common_causes(self, node_ids: List[str], min_strength: float = 0.1) -> List[Dict]:
        if not node_ids:
            return []
        ancestors_list = []
        for node_id in node_ids:
            if node_id in self.graph:
                ancestors = list(nx.ancestors(self.graph, node_id))
                ancestors_list.append(set(ancestors))
        if not ancestors_list:
            return []
        common = set.intersection(*ancestors_list) if ancestors_list else set()
        results = []
        for cause_id in common:
            if cause_id in self.node_attributes:
                strengths = []
                for target_id in node_ids:
                    paths = self.find_causal_path(cause_id, target_id)
                    if paths:
                        strengths.append(paths[0]['total_weight'])
                avg_strength = np.mean(strengths) if strengths else 0
                if avg_strength >= min_strength:
                    results.append({
                        'node_id': cause_id,
                        'text': self.node_attributes[cause_id]['text'][:100],
                        'average_strength': avg_strength,
                        'causal_weights': self.node_attributes[cause_id]['causal_weights']
                    })
        return sorted(results, key=lambda x: -x['average_strength'])
    
    def infer_effect(self, cause_id: str, max_depth: int = 3) -> List[Dict]:
        if cause_id not in self.graph:
            return []
        effects = []
        try:
            descendants = nx.descendants(self.graph, cause_id)
            for effect_id in descendants:
                if effect_id in self.node_attributes:
                    paths = self.find_causal_path(cause_id, effect_id, max_depth)
                    if paths:
                        effects.append({
                            'node_id': effect_id,
                            'text': self.node_attributes[effect_id]['text'][:100],
                            'path_strength': paths[0]['total_weight'],
                            'causal_chain': paths[0]['causal_chain']
                        })
        except:
            pass
        return sorted(effects, key=lambda x: -x['path_strength'])[:20]
    
    def build_from_memory(self, world_memory, opinion_memory, threshold: float = 0.5):
        logger.info("Building causal graph from memory...")
        world_items = world_memory.store.metadata['items']
        opinion_items = opinion_memory.store.metadata['items']
        all_items = world_items + opinion_items
        for item in all_items:
            self.add_node(
                node_id=str(item['id']),
                text=item['text'],
                causal_weights=item['causal_weights'],
                node_type=item.get('source', 'memory'),
                metadata=item.get('metadata', {})
            )
        for i, item1 in enumerate(all_items):
            for j, item2 in enumerate(all_items[i+1:], i+1):
                w1 = np.array(list(item1['causal_weights'].values()))
                w2 = np.array(list(item2['causal_weights'].values()))
                similarity = np.dot(w1, w2) / (np.linalg.norm(w1) * np.linalg.norm(w2) + 1e-8)
                if similarity > threshold:
                    if item1['causal_weights'].get('hetu', 0) > item2['causal_weights'].get('hetu', 0):
                        self.add_edge(
                            from_id=str(item1['id']),
                            to_id=str(item2['id']),
                            relation_type='causal_influence',
                            weight=similarity,
                            causal_condition='hetu'
                        )
                    else:
                        self.add_edge(
                            from_id=str(item2['id']),
                            to_id=str(item1['id']),
                            relation_type='causal_influence',
                            weight=similarity,
                            causal_condition='hetu'
                        )
        self._save()
        logger.info(f"Graph built: {self.graph.number_of_nodes()} nodes, {self.graph.number_of_edges()} edges")
    
    def _save(self):
        data = {
            'nodes': self.node_attributes,
            'edges': [(str(u), str(v), attrs) for (u, v), attrs in self.edge_attributes.items()]
        }
        with open(self.graph_path, 'w') as f:
            json.dump(data, f, indent=2)
    
    def _load(self):
        if os.path.exists(self.graph_path):
            with open(self.graph_path, 'r') as f:
                data = json.load(f)
            for node_id, attrs in data['nodes'].items():
                self.add_node(node_id, attrs['text'], attrs['causal_weights'],
                             attrs.get('type', 'fact'), attrs.get('metadata', {}))
            for u, v, attrs in data['edges']:
                self.add_edge(u, v, attrs['relation_type'], attrs['weight'],
                             attrs.get('causal_condition'))
            logger.info(f"Loaded graph with {self.graph.number_of_nodes()} nodes")
