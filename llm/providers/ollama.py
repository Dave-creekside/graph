"""
Ollama LLM provider integration.
"""

import json
import requests

def query_ollama(prompt, model, max_tokens, temperature, timeout=30):
    """
    Query the Ollama API with the given prompt.
    
    Args:
        prompt: The prompt to send to the model
        model: The Ollama model to use
        max_tokens: Maximum number of tokens to generate
        temperature: Temperature for generation
        timeout: Timeout in seconds
        
    Returns:
        str: The model's response
    """
    try:
        print(f"Querying Ollama API with model {model} (this may take a moment)...")
        
        # First try the newer chat endpoint
        chat_url = "http://localhost:11434/api/chat"
        chat_payload = {
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": temperature
        }
        
        try:
            response = requests.post(chat_url, json=chat_payload, timeout=timeout)
            response.raise_for_status()
            data = response.json()
            return data.get("message", {}).get("content", "No response provided.")
        except requests.exceptions.RequestException:
            # If chat endpoint fails, fall back to the generate endpoint
            pass
        
        # Fall back to the older generate endpoint
        generate_url = "http://localhost:11434/api/generate"
        generate_payload = {
            "model": model,
            "prompt": prompt,
            "stream": False,
            "max_tokens": max_tokens,
            "temperature": temperature
        }
        
        response = requests.post(generate_url, json=generate_payload, timeout=timeout)
        response.raise_for_status()
        
        # Try to parse as a single JSON object first
        try:
            data = response.json()
            return data.get("response", "No response provided.")
        except ValueError:
            # If that fails, try to handle streaming response format
            # (multiple JSON objects separated by newlines)
            full_response = ""
            for line in response.text.strip().split('\n'):
                try:
                    data = json.loads(line)
                    if "response" in data:
                        full_response += data["response"]
                except:
                    continue
            
            if full_response:
                return full_response
            else:
                return "Failed to parse Ollama API response."
    
    except requests.exceptions.Timeout:
        return "Error: Ollama API request timed out. Make sure Ollama is running and try again."
    except requests.exceptions.ConnectionError:
        return "Error: Could not connect to Ollama API. Make sure Ollama is running on localhost:11434."
    except Exception as e:
        return f"Error calling Ollama API: {str(e)}"
