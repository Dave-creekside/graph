"""
Feedback reasoning functions for hypergraph evolution.
"""

import re
from llm.reasoning import query_llm
from config import REASONING_PARAMS

def generate_feedback_reasoning_prompt(hg, query, focus_node=None):
    """
    Generate a prompt for the LLM to reason about a query and provide feedback
    on how to update the hypergraph.
    
    Args:
        hg: Hypergraph instance
        query: The reasoning query
        focus_node: Optional node to focus on
        
    Returns:
        str: The generated prompt
    """
    prompt = "You are a semantic reasoning engine that helps evolve a hypergraph of concepts.\n\n"
    
    # Add information about the hypergraph
    nodes = list(hg.get_nodes())
    edges = hg.get_hyperedges()
    
    prompt += f"The hypergraph contains {len(nodes)} nodes and {len(edges)} hyperedges.\n\n"
    
    # Add information about the focus node if provided
    if focus_node:
        prompt += f"Focus node: {focus_node}\n\n"
    
    # Add information about the nodes
    prompt += "Nodes:\n"
    for node in nodes[:REASONING_PARAMS["max_relevant_nodes"]]:
        prompt += f"- {node}\n"
    if len(nodes) > REASONING_PARAMS["max_relevant_nodes"]:
        prompt += f"- ... and {len(nodes) - REASONING_PARAMS['max_relevant_nodes']} more\n"
    prompt += "\n"
    
    # Add information about the hyperedges
    prompt += "Hyperedges:\n"
    for edge_id, edge in list(edges.items())[:REASONING_PARAMS["max_relevant_edges"]]:
        nodes_str = ", ".join(edge["nodes"])
        sim = edge.get("semantic_similarity", "N/A")
        prompt += f"- Edge {edge_id}: [{nodes_str}] (Semantic similarity: {sim})\n"
    if len(edges) > REASONING_PARAMS["max_relevant_edges"]:
        prompt += f"- ... and {len(edges) - REASONING_PARAMS['max_relevant_edges']} more\n"
    prompt += "\n"
    
    # Add the query
    prompt += f"Query: {query}\n\n"
    
    # Add the task
    prompt += """Task: 
1. Reason about the query based on the hypergraph.
2. Provide feedback on how to update the hypergraph based on your reasoning.

Your response should be structured as follows:

REASONING:
[Your reasoning about the query]

HYPERGRAPH UPDATES:
- ADD NODE: [node1], [node2], ... (Add new nodes to the hypergraph)
- ADD EDGE: [node1, node2, ...], [node3, node4, ...], ... (Add new hyperedges connecting the specified nodes)
- REMOVE NODE: [node1], [node2], ... (Remove nodes from the hypergraph)
- REMOVE EDGE: [edge_id1], [edge_id2], ... (Remove hyperedges from the hypergraph)
- ADJUST SIMILARITY: [edge_id1]=[new_similarity1], [edge_id2]=[new_similarity2], ... (Adjust semantic similarities of hyperedges)

Only include the update types that are needed. For each update type, provide a comma-separated list of items.
"""
    
    return prompt

def parse_hypergraph_updates(response):
    """
    Parse the hypergraph updates from the LLM's response.
    
    Args:
        response: The LLM's response
        
    Returns:
        dict: Dictionary of updates to apply to the hypergraph
    """
    updates = {
        "add_nodes": [],
        "add_edges": [],
        "remove_nodes": [],
        "remove_edges": [],
        "adjust_similarities": {}
    }
    
    # Extract the HYPERGRAPH UPDATES section
    match = re.search(r"HYPERGRAPH UPDATES:(.*?)(?:\n\n|$)", response, re.DOTALL)
    if not match:
        return updates
    
    updates_section = match.group(1).strip()
    
    # Parse ADD NODE
    add_node_match = re.search(r"- ADD NODE: (.*?)(?:\n|$)", updates_section)
    if add_node_match:
        nodes_str = add_node_match.group(1).strip()
        nodes = [node.strip() for node in nodes_str.split(",")]
        updates["add_nodes"] = nodes
    
    # Parse ADD EDGE
    add_edge_match = re.search(r"- ADD EDGE: (.*?)(?:\n|$)", updates_section)
    if add_edge_match:
        edges_str = add_edge_match.group(1).strip()
        # Split by "], [" to get individual edges
        edges_parts = edges_str.replace("[", "").replace("]", "").split(", ")
        edges = []
        current_edge = []
        for part in edges_parts:
            current_edge.append(part.strip())
            if len(current_edge) >= 2:  # At least 2 nodes per edge
                edges.append(current_edge)
                current_edge = []
        if current_edge:  # Add any remaining nodes
            edges.append(current_edge)
        updates["add_edges"] = edges
    
    # Parse REMOVE NODE
    remove_node_match = re.search(r"- REMOVE NODE: (.*?)(?:\n|$)", updates_section)
    if remove_node_match:
        nodes_str = remove_node_match.group(1).strip()
        nodes = [node.strip() for node in nodes_str.split(",")]
        updates["remove_nodes"] = nodes
    
    # Parse REMOVE EDGE
    remove_edge_match = re.search(r"- REMOVE EDGE: (.*?)(?:\n|$)", updates_section)
    if remove_edge_match:
        edges_str = remove_edge_match.group(1).strip()
        edge_ids = []
        for edge_id_str in edges_str.split(","):
            try:
                edge_id = int(edge_id_str.strip())
                edge_ids.append(edge_id)
            except ValueError:
                pass
        updates["remove_edges"] = edge_ids
    
    # Parse ADJUST SIMILARITY
    adjust_sim_match = re.search(r"- ADJUST SIMILARITY: (.*?)(?:\n|$)", updates_section)
    if adjust_sim_match:
        sims_str = adjust_sim_match.group(1).strip()
        for sim_pair in sims_str.split(","):
            parts = sim_pair.strip().split("=")
            if len(parts) == 2:
                try:
                    edge_id = int(parts[0].strip())
                    similarity = float(parts[1].strip())
                    updates["adjust_similarities"][edge_id] = similarity
                except ValueError:
                    pass
    
    return updates

def apply_hypergraph_updates(hg, updates):
    """
    Apply the parsed updates to the hypergraph.
    
    Args:
        hg: Hypergraph instance
        updates: Dictionary of updates to apply
        
    Returns:
        tuple: (updated_hypergraph, summary_of_changes)
    """
    summary = []
    
    # Add nodes
    for node in updates["add_nodes"]:
        hg.add_node(node)
        summary.append(f"Added node: {node}")
    
    # Add edges
    for nodes in updates["add_edges"]:
        hg.add_hyperedge(nodes)
        summary.append(f"Added hyperedge connecting: {', '.join(nodes)}")
    
    # Remove nodes
    for node in updates["remove_nodes"]:
        if node in hg.nodes:
            hg.nodes.remove(node)
            # Also remove any hyperedges containing this node
            edges_to_remove = []
            for edge_id, edge in hg.hyperedges.items():
                if node in edge["nodes"]:
                    edges_to_remove.append(edge_id)
            for edge_id in edges_to_remove:
                hg.hyperedges.pop(edge_id)
            summary.append(f"Removed node: {node}")
    
    # Remove edges
    for edge_id in updates["remove_edges"]:
        if edge_id in hg.hyperedges:
            hg.hyperedges.pop(edge_id)
            summary.append(f"Removed hyperedge {edge_id}")
    
    # Adjust similarities
    for edge_id, similarity in updates["adjust_similarities"].items():
        if edge_id in hg.hyperedges:
            hg.hyperedges[edge_id]["semantic_similarity"] = similarity
            summary.append(f"Adjusted similarity of hyperedge {edge_id} to {similarity}")
    
    return hg, summary

def reason_with_feedback(hg, query, focus_node=None):
    """
    Use an LLM to reason about a query and provide feedback to update the hypergraph.
    
    Args:
        hg: Hypergraph instance
        query: The reasoning query
        focus_node: Optional node to focus on
        
    Returns:
        tuple: (updated_hypergraph, reasoning_response, summary_of_changes)
    """
    # Generate the prompt
    prompt = generate_feedback_reasoning_prompt(hg, query, focus_node)
    
    # Query the LLM
    response = query_llm(prompt)
    
    # Parse the hypergraph updates
    updates = parse_hypergraph_updates(response)
    
    # Apply the updates
    updated_hg, summary = apply_hypergraph_updates(hg, updates)
    
    # Extract just the reasoning part
    reasoning_match = re.search(r"REASONING:(.*?)(?:HYPERGRAPH UPDATES:|$)", response, re.DOTALL)
    reasoning = reasoning_match.group(1).strip() if reasoning_match else response
    
    return updated_hg, reasoning, summary
