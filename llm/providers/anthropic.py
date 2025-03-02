"""
Anthropic LLM provider integration.
"""

def query_anthropic(prompt, model, max_tokens, temperature, api_key, timeout=30):
    """
    Query the Anthropic API with the given prompt.
    
    Args:
        prompt: The prompt to send to the model
        model: The Anthropic model to use
        max_tokens: Maximum number of tokens to generate
        temperature: Temperature for generation
        api_key: Anthropic API key
        timeout: Timeout in seconds
        
    Returns:
        str: The model's response
    """
    if not api_key:
        return "Error: Anthropic API key is required. Please set it in the customize menu."
    
    try:
        import anthropic
        print(f"Querying Anthropic API with model {model}...")
        
        # Configure the client
        client = anthropic.Anthropic(api_key=api_key)
        
        # Create the message
        response = client.messages.create(
            model=model,
            max_tokens=max_tokens,
            temperature=temperature,
            messages=[{"role": "user", "content": prompt}],
            timeout=timeout
        )
        
        # Extract the response content
        return response.content[0].text
    
    except ImportError:
        return "Error: anthropic package not installed. Please install it with 'pip install anthropic'."
    except Exception as e:
        return f"Error calling Anthropic API: {str(e)}"
