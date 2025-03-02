"""
Cellular Automata rules for evolving hypergraphs.
"""

from hypergraph.core import hypergraph_to_graph

def apply_ca_rule(hypergraph, common_neighbors_threshold=2):
    """
    Apply cellular automata rules to evolve the hypergraph.
    
    This rule looks for pairs of nodes that are not directly connected but share
    a certain number of common neighbors. When such pairs are found, a new hyperedge
    is created connecting them.
    
    Args:
        hypergraph: Hypergraph instance to evolve
        common_neighbors_threshold: Minimum number of common neighbors required to create a new edge
        
    Returns:
        tuple: (evolved_hypergraph, list_of_new_connections)
    """
    G = hypergraph_to_graph(hypergraph)
    nodes = list(hypergraph.get_nodes())
    candidate_pairs = []
    
    # Find pairs of nodes that are not directly connected but share enough common neighbors
    for i in range(len(nodes)):
        for j in range(i + 1, len(nodes)):
            if not G.has_edge(nodes[i], nodes[j]):
                neighbors1 = set(G.neighbors(nodes[i]))
                neighbors2 = set(G.neighbors(nodes[j]))
                if len(neighbors1.intersection(neighbors2)) >= common_neighbors_threshold:
                    candidate_pairs.append((nodes[i], nodes[j]))
    
    # Create new hyperedges for the candidate pairs
    for pair in candidate_pairs:
        hypergraph.add_hyperedge(list(pair), label="CA Rule Edge")
    
    return hypergraph, candidate_pairs
