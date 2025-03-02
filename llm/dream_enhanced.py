"""
Enhanced dream/self-talk functions for hypergraph evolution.
"""

from llm.reasoning import query_llm
from config import REASONING_PARAMS

def generate_dream_prompt_llm1(hg, iterations=5):
    """
    Generate a prompt for the first LLM to start a dream session.
    
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
    prompt += f"""Task: You are participating in a conversation with another AI about concepts in this hypergraph. 
You will have {iterations} exchanges. For each of your turns:

1. Select a node or pair of nodes from the hypergraph
2. Explore potential connections, analogies, or insights related to these nodes
3. Suggest new concepts or relationships that could emerge
4. End with a question or thought that invites your conversation partner to respond

Start the conversation by selecting your first node(s) and exploring them.
"""
    
    return prompt

def generate_dream_prompt_llm2(conversation_history, iteration, iterations_total):
    """
    Generate a prompt for the second LLM to respond in a dream session.
    
    Args:
        conversation_history: List of previous exchanges
        iteration: Current iteration number
        iterations_total: Total number of iterations
        
    Returns:
        str: The generated prompt
    """
    prompt = "You are a thoughtful AI participating in a philosophical conversation about concepts and their relationships.\n\n"
    
    # Add the conversation history
    prompt += "Conversation so far:\n"
    for i, (speaker, message) in enumerate(conversation_history):
        prompt += f"{speaker}: {message}\n\n"
    
    # Add the task
    prompt += f"""Task: You are in turn {iteration} of a {iterations_total}-turn conversation. 
Respond to your conversation partner's last message by:

1. Engaging with the concepts they've mentioned
2. Adding your own insights, analogies, or perspectives
3. Exploring potential new connections or relationships
4. If this is not the final turn, end with a thought that continues the conversation

Your response should be thoughtful and insightful, building on what has been discussed so far.
"""
    
    return prompt

def run_enhanced_dream_session(hg, iterations=None):
    """
    Run an enhanced dream session using two LLMs in conversation.
    
    Args:
        hg: Hypergraph instance
        iterations: Number of iterations (defaults to config value)
        
    Returns:
        str: The formatted conversation output
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
    prompt = generate_dream_prompt_llm1(hg, iterations)
    
    # Get the initial response from LLM1
    llm1_response = query_llm(
        prompt,
        provider=llm1_provider,
        model=llm1_model,
        api_key=llm1_api_key,
        max_tokens=REASONING_PARAMS["llm_max_tokens"] * 2,
        temperature=REASONING_PARAMS["llm_temperature"] + 0.1
    )
    
    # Initialize conversation history
    conversation = [(llm1_name, llm1_response)]
    
    # Format the output
    output = "# Enhanced Dream Session\n\n"
    output += f"## Iteration 1\n\n"
    output += f"[{llm1_name}]\n{llm1_response}\n\n"
    
    # Continue the conversation for the specified number of iterations
    for i in range(2, iterations + 1):
        # Generate prompt for LLM2
        prompt = generate_dream_prompt_llm2(conversation, i, iterations)
        
        # Get response from LLM2
        llm2_response = query_llm(
            prompt,
            provider=llm2_provider,
            model=llm2_model,
            api_key=llm2_api_key,
            max_tokens=REASONING_PARAMS["llm_max_tokens"] * 2,
            temperature=REASONING_PARAMS["llm_temperature"] + 0.1
        )
        
        # Add to conversation history
        conversation.append((llm2_name, llm2_response))
        
        # Add to output
        output += f"[{llm2_name}]\n{llm2_response}\n\n"
        
        # If this isn't the last iteration, get another response from LLM1
        if i < iterations:
            # Generate prompt for LLM1
            prompt = generate_dream_prompt_llm2(conversation, i + 1, iterations)
            
            # Get response from LLM1
            llm1_response = query_llm(
                prompt,
                provider=llm1_provider,
                model=llm1_model,
                api_key=llm1_api_key,
                max_tokens=REASONING_PARAMS["llm_max_tokens"] * 2,
                temperature=REASONING_PARAMS["llm_temperature"] + 0.1
            )
            
            # Add to conversation history
            conversation.append((llm1_name, llm1_response))
            
            # Add to output
            output += f"## Iteration {i + 1}\n\n"
            output += f"[{llm1_name}]\n{llm1_response}\n\n"
    
    return output
