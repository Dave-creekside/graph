"""
Functions for saving and loading hypergraphs to/from files.
"""

import json
import config

def save_hypergraph(hg, filename):
    """
    Save a hypergraph to a JSON file.
    
    Args:
        hg: Hypergraph instance to save
        filename: Path to save the hypergraph to
    """
    with open(filename, 'w', encoding='utf-8') as f:
        json.dump(hg.to_dict(), f, indent=2)
    
    print(f"Hypergraph and reasoning parameters saved to {filename}")

def load_hypergraph(filename):
    """
    Load a hypergraph from a JSON file.
    Also loads and applies any saved reasoning parameters.
    
    Args:
        filename: Path to the JSON file to load
        
    Returns:
        Hypergraph instance or None if loading failed
    """
    from hypergraph.core import Hypergraph
    
    try:
        with open(filename, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        hg, saved_params = Hypergraph.from_dict(data)
        
        # Apply saved parameters if present
        if saved_params:
            print("Loading saved reasoning parameters:")
            for param, value in saved_params.items():
                if param in config.REASONING_PARAMS:
                    old_value = config.REASONING_PARAMS[param]
                    config.REASONING_PARAMS[param] = value
                    print(f"  {param}: {old_value} -> {value}")
        
        print(f"Hypergraph loaded from {filename}")
        return hg
    except Exception as e:
        print(f"Error loading hypergraph from {filename}: {e}")
        return None
