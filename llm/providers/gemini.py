"""
Google Gemini LLM provider integration.
"""

def query_gemini(prompt, model, max_tokens, temperature, api_key, timeout=30):
    """
    Query the Google Gemini API with the given prompt.
    
    Args:
        prompt: The prompt to send to the model
        model: The Gemini model to use
        max_tokens: Maximum number of tokens to generate
        temperature: Temperature for generation
        api_key: Google API key
        timeout: Timeout in seconds
        
    Returns:
        str: The model's response
    """
    if not api_key:
        return "Error: Google API key is required. Please set it in the customize menu."
    
    try:
        import google.generativeai as genai
        print(f"Querying Google Gemini API with model {model}...")
        
        # Configure the API
        genai.configure(api_key=api_key)
        
        # Create the model
        generation_config = {
            "temperature": temperature,
            "max_output_tokens": max_tokens,
            "top_p": 0.95,
            "top_k": 0
        }
        
        # Initialize the model
        gemini_model = genai.GenerativeModel(
            model_name=model,
            generation_config=generation_config
        )
        
        # Generate content
        response = gemini_model.generate_content(prompt, timeout=timeout)
        
        # Extract the response content
        return response.text
    
    except ImportError:
        return "Error: google-generativeai package not installed. Please install it with 'pip install google-generativeai'."
    except Exception as e:
        return f"Error calling Google Gemini API: {str(e)}"
