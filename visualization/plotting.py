"""
Visualization functions for hypergraphs.
"""

import matplotlib.pyplot as plt
import networkx as nx
from hypergraph.core import hypergraph_to_graph
from analysis.spectral import perform_spectral_analysis

def plot_and_save_graph(G, filename, title="Graph"):
    """
    Plot and save a graph visualization.
    
    Args:
        G: NetworkX graph
        filename: Path to save the plot
        title: Title for the plot
    """
    plt.figure(figsize=(10, 8))
    pos = nx.spring_layout(G, seed=42)
    nx.draw_networkx_nodes(G, pos, node_size=500, node_color='skyblue')
    nx.draw_networkx_edges(G, pos, width=1.5)
    nx.draw_networkx_labels(G, pos, font_size=10)
    plt.title(title)
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()

def plot_and_save_reward_progression(rewards, filename):
    """
    Plot and save the reward progression over RL steps.
    
    Args:
        rewards: List of reward values
        filename: Path to save the plot
    """
    plt.figure(figsize=(8, 4))
    plt.plot(range(1, len(rewards)+1), rewards, marker='o', linestyle='-')
    plt.title("Reward Progression over RL Steps")
    plt.xlabel("Step")
    plt.ylabel("Combined Reward")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()

def plot_and_save_semantic_histogram(hg, filename):
    """
    Plot and save a histogram of semantic similarities in the hypergraph.
    
    Args:
        hg: Hypergraph instance
        filename: Path to save the plot
    """
    similarities = [edge.get("semantic_similarity", 0) for edge in hg.hyperedges.values()]
    plt.figure(figsize=(8, 4))
    plt.hist(similarities, bins=30, color='teal', edgecolor='black', alpha=0.7)
    plt.title("Distribution of Hyperedge Semantic Similarities")
    plt.xlabel("Semantic Similarity")
    plt.ylabel("Count")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()

def plot_and_save_laplacian_spectrum(G, filename):
    """
    Plot and save the Laplacian eigenvalue spectrum of a graph.
    
    Args:
        G: NetworkX graph
        filename: Path to save the plot
    """
    eigenvalues, _ = perform_spectral_analysis(G)
    plt.figure(figsize=(8, 4))
    plt.plot(eigenvalues, 'o-', color='purple')
    plt.title("Laplacian Eigenvalue Spectrum")
    plt.xlabel("Index")
    plt.ylabel("Eigenvalue")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()

def write_sorted_edges(hg, filename):
    """
    Write the hyperedges sorted by semantic similarity to a file.
    
    Args:
        hg: Hypergraph instance
        filename: Path to save the text file
    """
    sorted_edges = sorted(hg.hyperedges.items(), key=lambda x: x[1].get("semantic_similarity", 0))
    with open(filename, "w") as f:
        for edge_id, edge in sorted_edges:
            f.write(f"Edge {edge_id}: Nodes: {edge['nodes']}, Semantic Similarity: {edge.get('semantic_similarity', 0):.3f}\n")
