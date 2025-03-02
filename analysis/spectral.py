"""
Spectral analysis functions for hypergraphs.
"""

import numpy as np
from scipy.linalg import eigh
import networkx as nx

def perform_spectral_analysis(G):
    """
    Perform spectral analysis on a graph by computing the eigenvalues and eigenvectors
    of the normalized Laplacian matrix.
    
    Args:
        G: NetworkX graph
        
    Returns:
        tuple: (eigenvalues, eigenvectors)
    """
    L = nx.normalized_laplacian_matrix(G).todense()
    eigenvalues, eigenvectors = eigh(L)
    return eigenvalues, eigenvectors
