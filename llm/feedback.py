"""
LLM feedback functions for hypergraph evolution.
"""

from llm.reasoning import query_llm
from config import REASONING_PARAMS

def generate_feedback_prompt(hg, suggestion, focus_node=None):
    """
    Generate a prompt for the LLM to provide feedback on a suggestion.
    
    Args:
        hg: Hypergraph instance
        suggestion: The suggestion to evaluate
        focus_node: Optional node to focus on
        
    Returns:
        str: The generated prompt
    """
    prompt = "You are a semantic reasoning engine that evaluates suggestions for a hypergraph of concepts.\n\n"
    
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
    
    # Add information about the suggestion
    prompt += f"Suggestion: {suggestion}\n\n"
    
    # Add the task
    prompt += "Task: Evaluate this suggestion in the context of the hypergraph. Is it semantically coherent? Does it add value? Provide a score from 1-10 and explain your reasoning.\n"
    
    return prompt

def evaluate_suggestion(hg, suggestion, focus_node=None):
    """
    Use an LLM to evaluate a suggestion for the hypergraph.
    
    Args:
        hg: Hypergraph instance
        suggestion: The suggestion to evaluate
        focus_node: Optional node to focus on
        
    Returns:
        str: The LLM's evaluation
    """
    # Generate the prompt
    prompt = generate_feedback_prompt(hg, suggestion, focus_node)
    
    # Query the LLM
    response = query_llm(prompt)
    
    return response
