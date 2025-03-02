"""
Dream with feedback functions for hypergraph evolution.
"""

import re
from llm.reasoning import query_llm
from config import REASONING_PARAMS

def generate_dream_fb_prompt_llm1(hg, iterations=5):
    """
    Generate a prompt for the first LLM to start a dream session with feedback.
    
    Args:
        hg: Hypergraph instance
        iterations: Number of iterations for the dream session
        
    Returns:
        str: The generated prompt
    """
    prompt = "You are a semantic reasoning engine that explores connections between concepts in a hypergraph.\n\n"
    
    # Add information about the hypergraph
    nodes = list(hg.get_nodes())
    edges = hg.get_hyperedges()
    
    prompt += f"The hypergraph contains {len(nodes)} nodes and {len(edges)} hyperedges.\n\n"
    
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
    
    # Add the task
    prompt += f"""Task: You are participating in a conversation with another AI about concepts in this hypergraph. 
You will have {iterations} exchanges. For each of your turns:

1. Select a node or pair of nodes from the hypergraph
2. Explore potential connections, analogies, or insights related to these nodes
3. Suggest new concepts or relationships that could emerge
4. End with a question or thought that invites your conversation partner to respond

Additionally, after your main response, include a separate section with suggestions for updating the hypergraph:

HYPERGRAPH UPDATES:
- ADD NODE: [node1], [node2], ... (Add new nodes to the hypergraph)
- ADD EDGE: [node1, node2, ...], [node3, node4, ...], ... (Add new hyperedges connecting the specified nodes)
- REMOVE NODE: [node1], [node2], ... (Remove nodes from the hypergraph)
- REMOVE EDGE: [edge_id1], [edge_id2], ... (Remove hyperedges from the hypergraph)
- ADJUST SIMILARITY: [edge_id1]=[new_similarity1], [edge_id2]=[new_similarity2], ... (Adjust semantic similarities of hyperedges)

Only include the update types that are needed. For each update type, provide a comma-separated list of items.

Start the conversation by selecting your first node(s) and exploring them.
"""
    
    return prompt

def generate_dream_fb_prompt_llm2(conversation_history, hg, iteration, iterations_total):
    """
    Generate a prompt for the second LLM to respond in a dream session with feedback.
    
    Args:
        conversation_history: List of previous exchanges
        hg: Hypergraph instance (updated with any changes)
        iteration: Current iteration number
        iterations_total: Total number of iterations
        
    Returns:
        str: The generated prompt
    """
    prompt = "You are a thoughtful AI participating in a philosophical conversation about concepts and their relationships.\n\n"
    
    # Add information about the hypergraph
    nodes = list(hg.get_nodes())
    edges = hg.get_hyperedges()
    
    prompt += f"The hypergraph contains {len(nodes)} nodes and {len(edges)} hyperedges.\n\n"
    
    # Add information about the nodes
    prompt += "Nodes:\n"
    for node in nodes[:REASONING_PARAMS["max_relevant_nodes"]]:
        prompt += f"- {node}\n"
    if len(nodes) > REASONING_PARAMS["max_relevant_nodes"]:
        prompt += f"- ... and {len(nodes) - REASONING_PARAMS['max_relevant_nodes']} more\n"
    prompt += "\n"
    
    # Add the conversation history
    prompt += "Conversation so far:\n"
    for i, (speaker, message) in enumerate(conversation_history):
        # Extract just the main part of the message, not the HYPERGRAPH UPDATES section
        main_message = message.split("HYPERGRAPH UPDATES:")[0].strip()
        prompt += f"{speaker}: {main_message}\n\n"
    
    # Add the task
    prompt += f"""Task: You are in turn {iteration} of a {iterations_total}-turn conversation. 
Respond to your conversation partner's last message by:

1. Engaging with the concepts they've mentioned
2. Adding your own insights, analogies, or perspectives
3. Exploring potential new connections or relationships
4. If this is not the final turn, end with a thought that continues the conversation

Additionally, after your main response, include a separate section with suggestions for updating the hypergraph:

HYPERGRAPH UPDATES:
- ADD NODE: [node1], [node2], ... (Add new nodes to the hypergraph)
- ADD EDGE: [node1, node2, ...], [node3, node4, ...], ... (Add new hyperedges connecting the specified nodes)
- REMOVE NODE: [node1], [node2], ... (Remove nodes from the hypergraph)
- REMOVE EDGE: [edge_id1], [edge_id2], ... (Remove hyperedges from the hypergraph)
- ADJUST SIMILARITY: [edge_id1]=[new_similarity1], [edge_id2]=[new_similarity2], ... (Adjust semantic similarities of hyperedges)

Only include the update types that are needed. For each update type, provide a comma-separated list of items.

Your response should be thoughtful and insightful, building on what has been discussed so far.
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

def run_dream_feedback_session(hg, iterations=None):
    """
    Run a dream session with feedback using two LLMs in conversation,
    updating the hypergraph based on their suggestions.
    
    Args:
        hg: Hypergraph instance
        iterations: Number of iterations (defaults to config value)
        
    Returns:
        tuple: (updated_hypergraph, formatted_conversation_output)
    """
    # Use provided value or default from config
    iterations = iterations or REASONING_PARAMS["dream_iterations"]
    
    # Get LLM configurations
    llm1_provider = REASONING_PARAMS["llm_provider"]
    llm1_model = REASONING_PARAMS["llm_model"]
    llm1_api_key = REASONING_PARAMS["llm_api_key"]
    
    llm2_provider = REASONING_PARAMS["llm2_provider"]
    llm2_model = REASONING_PARAMS["llm2_model"]
    llm2_api_key = REASONING_PARAMS["llm2_api_key"]
    
    # Format model names for display
    llm1_name = f"{llm1_provider.capitalize()} ({llm1_model})"
    llm2_name = f"{llm2_provider.capitalize()} ({llm2_model})"
    
    # Generate the initial prompt for LLM1
    prompt = generate_dream_fb_prompt_llm1(hg, iterations)
    
    # Get the initial response from LLM1
    llm1_response = query_llm(
        prompt,
        provider=llm1_provider,
        model=llm1_model,
        api_key=llm1_api_key,
        max_tokens=REASONING_PARAMS["llm_max_tokens"] * 2,
        temperature=REASONING_PARAMS["llm_temperature"] + 0.1
    )
    
    # Parse and apply updates from LLM1
    updates = parse_hypergraph_updates(llm1_response)
    hg, summary = apply_hypergraph_updates(hg, updates)
    
    # Initialize conversation history
    conversation = [(llm1_name, llm1_response)]
    
    # Format the output
    output = "# Dream Session with Feedback\n\n"
    output += f"## Iteration 1\n\n"
    output += f"[{llm1_name}]\n{llm1_response}\n\n"
    
    # Add summary of changes
    if summary:
        output += "Changes to hypergraph:\n"
        for change in summary:
            output += f"  - {change}\n"
        output += "\n"
    
    # Continue the conversation for the specified number of iterations
    for i in range(2, iterations + 1):
        # Generate prompt for LLM2
        prompt = generate_dream_fb_prompt_llm2(conversation, hg, i, iterations)
        
        # Get response from LLM2
        llm2_response = query_llm(
            prompt,
            provider=llm2_provider,
            model=llm2_model,
            api_key=llm2_api_key,
            max_tokens=REASONING_PARAMS["llm_max_tokens"] * 2,
            temperature=REASONING_PARAMS["llm_temperature"] + 0.1
        )
        
        # Parse and apply updates from LLM2
        updates = parse_hypergraph_updates(llm2_response)
        hg, summary = apply_hypergraph_updates(hg, updates)
        
        # Add to conversation history
        conversation.append((llm2_name, llm2_response))
        
        # Add to output
        output += f"[{llm2_name}]\n{llm2_response}\n\n"
        
        # Add summary of changes
        if summary:
            output += "Changes to hypergraph:\n"
            for change in summary:
                output += f"  - {change}\n"
            output += "\n"
        
        # If this isn't the last iteration, get another response from LLM1
        if i < iterations:
            # Generate prompt for LLM1
            prompt = generate_dream_fb_prompt_llm2(conversation, hg, i + 1, iterations)
            
            # Get response from LLM1
            llm1_response = query_llm(
                prompt,
                provider=llm1_provider,
                model=llm1_model,
                api_key=llm1_api_key,
                max_tokens=REASONING_PARAMS["llm_max_tokens"] * 2,
                temperature=REASONING_PARAMS["llm_temperature"] + 0.1
            )
            
            # Parse and apply updates from LLM1
            updates = parse_hypergraph_updates(llm1_response)
            hg, summary = apply_hypergraph_updates(hg, updates)
            
            # Add to conversation history
            conversation.append((llm1_name, llm1_response))
            
            # Add to output
            output += f"## Iteration {i + 1}\n\n"
            output += f"[{llm1_name}]\n{llm1_response}\n\n"
            
            # Add summary of changes
            if summary:
                output += "Changes to hypergraph:\n"
                for change in summary:
                    output += f"  - {change}\n"
                output += "\n"
    
    return hg, output
