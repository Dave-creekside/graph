"""
Groq LLM provider integration.
"""

def query_groq(prompt, model, max_tokens, temperature, api_key, timeout=30):
    """
    Query the Groq API with the given prompt.
    
    Args:
        prompt: The prompt to send to the model
        model: The Groq model to use
        max_tokens: Maximum number of tokens to generate
        temperature: Temperature for generation
        api_key: Groq API key
        timeout: Timeout in seconds
        
    Returns:
        str: The model's response
    """
    if not api_key:
        return "Error: Groq API key is required. Please set it in the customize menu."
    
    try:
        import groq
        print(f"Querying Groq API with model {model}...")
        
        # Configure the client
        client = groq.Groq(api_key=api_key)
        
        # Create the chat completion
        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=temperature,
            max_tokens=max_tokens,
            timeout=timeout
        )
        
        # Extract the response content
        return response.choices[0].message.content
    
    except ImportError:
        return "Error: groq package not installed. Please install it with 'pip install groq'."
    except Exception as e:
        return f"Error calling Groq API: {str(e)}"
