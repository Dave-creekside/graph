"""
Global configuration and parameters for the Hypergraph Evolution application.
"""

# Global reasoning parameters that can be customized
REASONING_PARAMS = {
    "similarity_threshold": 0.6,        # Threshold for semantic similarity to consider nodes relevant
    "ca_neighbors_threshold": 2,        # Number of common neighbors required for CA rule
    "max_relevant_nodes": 10,           # Maximum number of relevant nodes to include in prompt
    "max_relevant_edges": 7,            # Maximum number of relevant edges to include in prompt
    "max_ca_examples": 3,               # Maximum number of CA examples to include in prompt
    "llm_temperature": 0.7,             # Temperature for LLM generation
    "llm_max_tokens": 250,              # Maximum tokens for LLM response
    "llm_provider": "ollama",           # LLM provider (ollama, openai, anthropic, groq, gemini)
    "llm_model": "deepseek-r1:latest",  # Model to use for the selected provider
    "llm_api_key": "",                  # API key for non-Ollama providers
    "apply_ca_rules": True,             # Whether to apply CA rules during reasoning
    "max_steps": 300,                   # Maximum number of steps for the environment
    # Second LLM for dream/self-talk mode
    "llm2_provider": "ollama",          # Second LLM provider
    "llm2_model": "llama3.2",             # Model for the second LLM
    "llm2_api_key": "",                 # API key for the second LLM (if not Ollama)
    "dream_iterations": 5,              # Default number of iterations for dream mode
}

# Provider-specific model options
LLM_PROVIDER_MODELS = {
    "ollama": ["deepseek-r1:latest", "llama3.2"],
    "openai": ["gpt-4o", "gpt-4-turbo", "gpt-3.5-turbo"],
    "anthropic": ["claude-3-7-sonnet-20250219", "claude-3-sonnet", "claude-3-haiku"],
    "groq": ["llama3-8b-8192", "llama3-70b-8192", "mixtral-8x7b-32768"],
    "gemini": ["gemini-2.0-flash", "gemini-1.5-pro"]
}

# Dependency availability flags
OPENAI_AVAILABLE = False
ANTHROPIC_AVAILABLE = False
GROQ_AVAILABLE = False
GEMINI_AVAILABLE = False
MATPLOTLIB_AVAILABLE = False
NETWORKX_AVAILABLE = False
NUMPY_AVAILABLE = False
SPACY_AVAILABLE = False
SCIPY_AVAILABLE = False
