"""
Core Hypergraph data structure and basic operations.
"""

class Hypergraph:
    """
    A hypergraph data structure that represents nodes and hyperedges.
    
    A hyperedge can connect any number of nodes, unlike a regular graph edge
    which connects exactly two nodes.
    """
    def __init__(self):
        """Initialize an empty hypergraph."""
        self.nodes = set()
        self.hyperedges = {}
        self.edge_counter = 0

    def add_node(self, node):
        """Add a node to the hypergraph."""
        self.nodes.add(node)
    
    def add_hyperedge(self, nodes, label=None):
        """
        Add a hyperedge connecting the given nodes.
        
        Args:
            nodes: List of nodes to connect with this hyperedge
            label: Optional label for the hyperedge
        """
        for node in nodes:
            self.add_node(node)
        self.hyperedges[self.edge_counter] = {"nodes": set(nodes), "label": label}
        self.edge_counter += 1

    def get_nodes(self):
        """Return the set of all nodes in the hypergraph."""
        return self.nodes

    def get_hyperedges(self):
        """Return the dictionary of all hyperedges in the hypergraph."""
        return self.hyperedges
    
    def to_dict(self):
        """
        Convert the hypergraph to a dictionary for serialization.
        Also includes the current reasoning parameters.
        """
        from config import REASONING_PARAMS
        
        return {
            "nodes": list(self.nodes),
            "hyperedges": {
                str(edge_id): {
                    "nodes": list(edge_data["nodes"]),
                    "label": edge_data.get("label"),
                    "semantic_similarity": edge_data.get("semantic_similarity", 0.0)
                }
                for edge_id, edge_data in self.hyperedges.items()
            },
            "edge_counter": self.edge_counter,
            "parameters": REASONING_PARAMS  # Include the current reasoning parameters
        }
    
    @classmethod
    def from_dict(cls, data):
        """
        Create a hypergraph from a dictionary representation.
        Also extracts and returns any saved reasoning parameters.
        
        Args:
            data: Dictionary representation of a hypergraph
            
        Returns:
            tuple: (hypergraph, saved_parameters)
        """
        hg = cls()
        hg.nodes = set(data["nodes"])
        hg.edge_counter = data["edge_counter"]
        
        for edge_id_str, edge_data in data["hyperedges"].items():
            edge_id = int(edge_id_str)
            nodes = set(edge_data["nodes"])
            label = edge_data.get("label")
            semantic_similarity = edge_data.get("semantic_similarity", 0.0)
            
            # Reconstruct the hyperedge
            hg.hyperedges[edge_id] = {
                "nodes": nodes,
                "label": label,
                "semantic_similarity": semantic_similarity
            }
        
        # Extract parameters if present
        saved_params = data.get("parameters", None)
        
        return hg, saved_params

def hypergraph_to_graph(hg):
    """
    Convert the hypergraph into a NetworkX graph via clique expansion.
    
    In clique expansion, each hyperedge is replaced by a clique (fully connected subgraph)
    of the nodes it contains.
    
    Args:
        hg: Hypergraph instance
        
    Returns:
        NetworkX graph
    """
    import networkx as nx
    
    G = nx.Graph()
    G.add_nodes_from(hg.get_nodes())
    for edge in hg.get_hyperedges().values():
        nodes = list(edge["nodes"])
        for i in range(len(nodes)):
            for j in range(i+1, len(nodes)):
                G.add_edge(nodes[i], nodes[j])
    return G
