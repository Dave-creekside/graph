"""
LLM dream/self-talk functions for hypergraph evolution.
"""

from llm.reasoning import query_llm
from config import REASONING_PARAMS

def generate_dream_prompt(hg, iterations=5):
    """
    Generate a prompt for the LLM to engage in a self-talk/dream session.
    
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
    
    # Add the task
    prompt += f"""Task: Engage in a self-talk session for {iterations} iterations. In each iteration:
1. Select a node or pair of nodes from the hypergraph
2. Explore potential connections, analogies, or insights related to these nodes
3. Suggest new concepts or relationships that could emerge

Format your response as a dialogue with yourself, with each iteration clearly marked.
"""
    
    return prompt

def run_dream_session(hg, iterations=None):
    """
    Run a dream/self-talk session using the LLM.
    
    Args:
        hg: Hypergraph instance
        iterations: Number of iterations (defaults to config value)
        
    Returns:
        str: The LLM's dream session output
    """
    # Use provided value or default from config
    iterations = iterations or REASONING_PARAMS["dream_iterations"]
    
    # Generate the prompt
    prompt = generate_dream_prompt(hg, iterations)
    
    # Query the LLM
    # For dream sessions, we might want to use a different model
    provider = REASONING_PARAMS["llm2_provider"]
    model = REASONING_PARAMS["llm2_model"]
    api_key = REASONING_PARAMS["llm2_api_key"]
    
    response = query_llm(
        prompt,
        provider=provider,
        model=model,
        api_key=api_key,
        max_tokens=REASONING_PARAMS["llm_max_tokens"] * 2,  # Double the tokens for dream sessions
        temperature=REASONING_PARAMS["llm_temperature"] + 0.1  # Slightly higher temperature for creativity
    )
    
    return response
