"""
LLM reasoning functions for hypergraph evolution.
"""

import config
from config import REASONING_PARAMS
import importlib.util

# Import the appropriate LLM provider based on configuration
def get_llm_provider(provider_name):
    """
    Get the LLM provider module based on the provider name.
    
    Args:
        provider_name: Name of the LLM provider
        
    Returns:
        module: The provider module if available, None otherwise
    """
    try:
        module_path = f"llm.providers.{provider_name}"
        provider_module = __import__(module_path, fromlist=["query_" + provider_name])
        return provider_module
    except ImportError:
        print(f"Error: Provider module {provider_name} not found.")
        return None

def check_provider_availability():
    """
    Check which LLM providers are available by trying to import their packages.
    Updates the global availability flags in config.
    """
    # Check OpenAI
    if importlib.util.find_spec("openai"):
        config.OPENAI_AVAILABLE = True
    
    # Check Anthropic
    if importlib.util.find_spec("anthropic"):
        config.ANTHROPIC_AVAILABLE = True
    
    # Check Groq
    if importlib.util.find_spec("groq"):
        config.GROQ_AVAILABLE = True
    
    # Check Google Generative AI (for Gemini)
    if importlib.util.find_spec("google.generativeai"):
        config.GEMINI_AVAILABLE = True
    
    # Check other dependencies
    if importlib.util.find_spec("matplotlib"):
        config.MATPLOTLIB_AVAILABLE = True
    
    if importlib.util.find_spec("networkx"):
        config.NETWORKX_AVAILABLE = True
    
    if importlib.util.find_spec("numpy"):
        config.NUMPY_AVAILABLE = True
    
    if importlib.util.find_spec("spacy"):
        config.SPACY_AVAILABLE = True
    
    if importlib.util.find_spec("scipy"):
        config.SCIPY_AVAILABLE = True

def query_llm(prompt, provider=None, model=None, max_tokens=None, temperature=None, api_key=None):
    """
    Query an LLM with the given prompt using the specified provider.
    
    Args:
        prompt: The prompt to send to the model
        provider: LLM provider (defaults to config value)
        model: Model to use (defaults to config value)
        max_tokens: Maximum tokens to generate (defaults to config value)
        temperature: Temperature for generation (defaults to config value)
        api_key: API key for the provider (defaults to config value)
        
    Returns:
        str: The model's response
    """
    # Use provided values or defaults from config
    provider = provider or REASONING_PARAMS["llm_provider"]
    model = model or REASONING_PARAMS["llm_model"]
    max_tokens = max_tokens or REASONING_PARAMS["llm_max_tokens"]
    temperature = temperature or REASONING_PARAMS["llm_temperature"]
    api_key = api_key or REASONING_PARAMS["llm_api_key"]
    
    # Get the provider module
    provider_module = get_llm_provider(provider)
    if not provider_module:
        return f"Error: Provider {provider} not available."
    
    # Get the query function
    query_func_name = f"query_{provider}"
    if not hasattr(provider_module, query_func_name):
        return f"Error: Provider {provider} does not have a query function."
    
    query_func = getattr(provider_module, query_func_name)
    
    # Call the query function with the appropriate arguments
    if provider == "ollama":
        return query_func(prompt, model, max_tokens, temperature)
    else:
        return query_func(prompt, model, max_tokens, temperature, api_key)

def generate_prompt_from_hypergraph(hg, focus_node=None, max_relevant_nodes=10, max_relevant_edges=7, max_ca_examples=3):
    """
    Generate a prompt for the LLM based on the current state of the hypergraph.
    
    Args:
        hg: Hypergraph instance
        focus_node: Optional node to focus on
        max_relevant_nodes: Maximum number of relevant nodes to include
        max_relevant_edges: Maximum number of relevant edges to include
        max_ca_examples: Maximum number of CA examples to include
        
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
    for node in nodes[:max_relevant_nodes]:
        prompt += f"- {node}\n"
    if len(nodes) > max_relevant_nodes:
        prompt += f"- ... and {len(nodes) - max_relevant_nodes} more\n"
    prompt += "\n"
    
    # Add information about the hyperedges
    prompt += "Hyperedges:\n"
    for edge_id, edge in list(edges.items())[:max_relevant_edges]:
        nodes_str = ", ".join(edge["nodes"])
        sim = edge.get("semantic_similarity", "N/A")
        prompt += f"- Edge {edge_id}: [{nodes_str}] (Semantic similarity: {sim})\n"
    if len(edges) > max_relevant_edges:
        prompt += f"- ... and {len(edges) - max_relevant_edges} more\n"
    prompt += "\n"
    
    # Add information about cellular automata rules if applicable
    if REASONING_PARAMS["apply_ca_rules"]:
        from analysis.cellular_automata import apply_ca_rule
        
        _, ca_examples = apply_ca_rule(hg, REASONING_PARAMS["ca_neighbors_threshold"])
        if ca_examples:
            prompt += "Cellular Automata Rule Examples:\n"
            for i, (node1, node2) in enumerate(ca_examples[:max_ca_examples]):
                prompt += f"- Nodes '{node1}' and '{node2}' share enough common neighbors to form a new connection.\n"
            if len(ca_examples) > max_ca_examples:
                prompt += f"- ... and {len(ca_examples) - max_ca_examples} more\n"
            prompt += "\n"
    
    # Add the task
    prompt += "Task: Based on the current hypergraph, suggest a new concept that would fit well with the existing nodes and edges. Explain your reasoning.\n"
    
    return prompt

def reason_with_llm(hg, focus_node=None):
    """
    Use an LLM to reason about the hypergraph and suggest new concepts.
    
    Args:
        hg: Hypergraph instance
        focus_node: Optional node to focus on
        
    Returns:
        str: The LLM's response
    """
    # Generate the prompt
    prompt = generate_prompt_from_hypergraph(
        hg, 
        focus_node,
        REASONING_PARAMS["max_relevant_nodes"],
        REASONING_PARAMS["max_relevant_edges"],
        REASONING_PARAMS["max_ca_examples"]
    )
    
    # Query the LLM
    response = query_llm(prompt)
    
    return response
