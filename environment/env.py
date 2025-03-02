"""
Reinforcement Learning Environment for Hypergraph Evolution.
"""

import random
from hypergraph.core import Hypergraph

class HypergraphEvolutionEnv:
    """
    Reinforcement Learning Environment for evolving hypergraphs.
    
    This environment allows for adding and removing hyperedges based on
    reinforcement learning actions.
    """
    def __init__(self, vocabulary=None, max_steps=10):
        """
        Initialize the environment.
        
        Args:
            vocabulary: List of words to use as nodes in the hypergraph
            max_steps: Maximum number of steps to run the environment
        """
        if vocabulary is None:
            self.vocabulary = ["cat", "dog", "lion", "tiger", "mammal", "pet", "wild", "predator", "domestic"]
        else:
            self.vocabulary = vocabulary
        self.max_steps = max_steps
        self.current_step = 0
        self.hypergraph = None

    def reset(self, initial_hypergraph=None):
        """
        Reset the environment to an initial state.
        
        Args:
            initial_hypergraph: Optional hypergraph to start with
            
        Returns:
            The initial hypergraph
        """
        if initial_hypergraph is not None:
            self.hypergraph = initial_hypergraph
        else:
            self.hypergraph = Hypergraph()
            self.hypergraph.add_hyperedge(["cat", "mammal", "pet"], label="Initial Edge")
        self.current_step = 0
        return self.hypergraph

    def compute_structural_reward(self):
        """
        Compute a reward based on the structural properties of the hypergraph.
        
        Returns:
            Clustering coefficient of the graph
        """
        from hypergraph.core import hypergraph_to_graph
        import networkx as nx
        
        G = hypergraph_to_graph(self.hypergraph)
        if len(G.nodes()) > 2:
            return nx.average_clustering(G)
        else:
            return 0.0

    def step(self, action):
        """
        Execute an action ('add' or 'remove'):
          - 'add': add a hyperedge by randomly sampling 2 or 3 nodes.
          - 'remove': remove a random hyperedge (if available).
          
        Args:
            action: 'add' or 'remove'
            
        Returns:
            tuple: (hypergraph, reward, done, info)
        """
        info = {}
        if action == 'add':
            num_nodes = random.choice([2, 3])
            nodes = random.sample(self.vocabulary, num_nodes)
            self.hypergraph.add_hyperedge(nodes, label="Added Edge")
            info["action"] = f"Added hyperedge with nodes: {nodes}"
        elif action == 'remove':
            if self.hypergraph.hyperedges:
                edge_id = random.choice(list(self.hypergraph.hyperedges.keys()))
                removed_edge = self.hypergraph.hyperedges.pop(edge_id)
                info["action"] = f"Removed hyperedge {edge_id} with nodes: {removed_edge['nodes']}"
            else:
                info["action"] = "No hyperedge to remove."
        else:
            info["action"] = "Invalid action provided."
        self.current_step += 1
        done = self.current_step >= self.max_steps
        return self.hypergraph, None, done, info
