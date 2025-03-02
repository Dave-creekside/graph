"""
Policy functions for the Hypergraph Evolution environment.
"""

import random
from hypergraph.core import Hypergraph

def choose_informed_action(hg, threshold=0.5, min_edges=5):
    """
    Choose an action based on the current state of the hypergraph.
    
    This policy removes edges with low semantic similarity if there are enough edges,
    otherwise it adds new edges.
    
    Args:
        hg: Hypergraph instance
        threshold: Semantic similarity threshold below which edges are considered for removal
        min_edges: Minimum number of edges to maintain in the hypergraph
        
    Returns:
        str: 'add' or 'remove'
    """
    low_sim_edges = [eid for eid, edge in hg.hyperedges.items() if edge.get("semantic_similarity", 1) < threshold]
    if low_sim_edges and len(hg.hyperedges) > min_edges:
        return 'remove'
    else:
        return 'add'

def generate_large_hypergraph(vocabulary, num_hyperedges=50, min_edge_size=2, max_edge_size=4):
    """
    Generate a large hypergraph with random hyperedges.
    
    Args:
        vocabulary: List of words to use as nodes
        num_hyperedges: Number of hyperedges to generate
        min_edge_size: Minimum number of nodes per hyperedge
        max_edge_size: Maximum number of nodes per hyperedge
        
    Returns:
        Hypergraph instance
    """
    hg = Hypergraph()
    random.seed(42)
    for _ in range(num_hyperedges):
        edge_size = random.randint(min_edge_size, max_edge_size)
        nodes = random.sample(vocabulary, edge_size)
        hg.add_hyperedge(nodes, label="Random Edge")
    return hg
