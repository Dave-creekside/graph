"""
Utility functions for the Hypergraph Evolution application.
"""

import os
import json
import random
import string

def ensure_directory_exists(filepath):
    """
    Ensure that the directory for the given filepath exists.
    
    Args:
        filepath: Path to a file
    """
    directory = os.path.dirname(filepath)
    if directory and not os.path.exists(directory):
        os.makedirs(directory)

def generate_random_id(length=8):
    """
    Generate a random ID string.
    
    Args:
        length: Length of the ID string
        
    Returns:
        str: Random ID string
    """
    return ''.join(random.choices(string.ascii_lowercase + string.digits, k=length))

def format_hyperedge(edge_id, edge_data):
    """
    Format a hyperedge for display.
    
    Args:
        edge_id: ID of the hyperedge
        edge_data: Data for the hyperedge
        
    Returns:
        str: Formatted string representation of the hyperedge
    """
    nodes_str = ", ".join(edge_data["nodes"])
    label = f" (Label: {edge_data['label']})" if edge_data.get("label") else ""
    sim = f" (Similarity: {edge_data.get('semantic_similarity', 'N/A'):.3f})" if "semantic_similarity" in edge_data else ""
    return f"Edge {edge_id}: [{nodes_str}]{label}{sim}"

def get_file_extension(filename, default_ext=".json"):
    """
    Get the file extension from a filename, or add a default extension if none exists.
    
    Args:
        filename: Filename to check
        default_ext: Default extension to add if none exists
        
    Returns:
        str: Filename with extension
    """
    if not filename:
        return None
    
    if "." not in filename:
        return filename + default_ext
    
    return filename
