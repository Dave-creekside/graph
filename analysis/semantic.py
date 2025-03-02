"""
Semantic analysis functions for hypergraphs.
"""

import numpy as np
import spacy

# Load spaCy model
try:
    nlp = spacy.load("en_core_web_md")
except:
    print("Warning: spaCy model 'en_core_web_md' not found.")
    print("To install: python -m spacy download en_core_web_md")
    nlp = None

def get_semantic_similarity(word1, word2):
    """
    Calculate the semantic similarity between two words using spaCy.
    
    Args:
        word1: First word
        word2: Second word
        
    Returns:
        float: Semantic similarity score between 0 and 1
    """
    if nlp is None:
        return 0.0
        
    token1 = nlp(word1)
    token2 = nlp(word2)
    return token1.similarity(token2)

def contextualize_hypergraph(hg):
    """
    Calculate semantic similarities for all hyperedges in the hypergraph.
    
    For each hyperedge, this function calculates the average semantic similarity
    between all pairs of nodes in the hyperedge.
    
    Args:
        hg: Hypergraph instance
        
    Returns:
        Hypergraph instance with updated semantic similarities
    """
    if nlp is None:
        return hg
        
    for edge_id, edge_data in hg.hyperedges.items():
        nodes = list(edge_data["nodes"])
        if len(nodes) < 2:
            edge_data["semantic_similarity"] = 0.0
            continue
        sims = []
        for i in range(len(nodes)):
            for j in range(i+1, len(nodes)):
                sims.append(get_semantic_similarity(nodes[i], nodes[j]))
        edge_data["semantic_similarity"] = np.mean(sims) if sims else 0.0
    return hg

def compute_semantic_reward(hg):
    """
    Compute a reward based on the semantic similarities in the hypergraph.
    
    Args:
        hg: Hypergraph instance
        
    Returns:
        float: Average semantic similarity across all hyperedges
    """
    sims = [edge.get("semantic_similarity", 0) for edge in hg.hyperedges.values()]
    return np.mean(sims) if sims else 0.0

def combined_reward(hg, weight_structural=0.5, weight_semantic=0.5):
    """
    Compute a combined reward based on both structural and semantic properties.
    
    Args:
        hg: Hypergraph instance
        weight_structural: Weight for the structural component of the reward
        weight_semantic: Weight for the semantic component of the reward
        
    Returns:
        float: Combined reward
    """
    import networkx as nx
    from hypergraph.core import hypergraph_to_graph
    
    struct_r = 0.0
    G = hypergraph_to_graph(hg)
    if len(G.nodes()) > 2:
        struct_r = nx.average_clustering(G)
    sem_r = compute_semantic_reward(hg)
    return weight_structural * struct_r + weight_semantic * sem_r

def load_vocabulary_from_file(filename):
    """
    Load vocabulary from a text file.
    
    Args:
        filename: Path to the vocabulary file (one word per line)
        
    Returns:
        list: List of words
    """
    vocab = []
    try:
        with open(filename, "r", encoding="utf-8") as f:
            for line in f:
                word = line.strip()
                if word:
                    vocab.append(word)
    except Exception as e:
        print(f"Error loading vocabulary from {filename}: {e}")
    return vocab
