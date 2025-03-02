#!/usr/bin/env python3
"""
CLI Application for Hypergraph Evolution with Semantic and Structural Analysis,
Interactive Mode, and Multi-Provider LLM Reasoning Integration.

This application demonstrates how hypergraph reasoning and cellular automata rules
can affect language model reasoning processes. It builds a semantic hypergraph from
vocabulary, evolves it using reinforcement learning and cellular automata rules,
and uses the hypergraph to guide LLM reasoning.

Usage:
    python hy.py --interactive --skip_init --model llama3
    python hy.py --max_steps 300 --verbose --small_init [--vocab_file vocab.txt]

Arguments:
    --max_steps INT       Maximum number of RL steps (default: 300)
    --verbose             Enable verbose output
    --interactive         Enter interactive mode after simulation
    --vocab_file PATH     Path to external vocabulary file (one word per line)
    --model STRING        Model to use for reasoning (default: deepseek-r1:latest)
    --skip_init           Skip initial hypergraph generation and go directly to interactive mode
    --small_init          Use a small initial hypergraph (faster startup)
    --num_edges INT       Number of initial hyperedges to generate (default: 50)

Interactive Mode Commands:
    help                  Show available commands
    step [n]              Execute n simulation steps (default is 1)
    add                   Force an 'add' action
    remove                Force a 'remove' action
    reason [query]        Ask a reasoning query (LLM will respond)
    reason_fb [query]     Ask a reasoning query with feedback to update the hypergraph
    save [file]           Save the current hypergraph and reasoning parameters to a file
    load [file]           Load a hypergraph and its reasoning parameters from a file
    customize             Customize hypergraph reasoning parameters and LLM provider settings
    evolve                Apply cellular automata rules to evolve the hypergraph
    explore [node]        Explore connections for a specific node
    plot                  Update and save plots (graph, histogram, spectrum)
    stats                 Display current hypergraph statistics
    undo                  Revert to previous hypergraph state (if available)
    exit                  Exit interactive mode

Core Dependencies:
    - networkx
    - matplotlib
    - numpy
    - spacy (with the 'en_core_web_md' model installed)
    - scipy
    - requests
    - argparse

LLM Provider Dependencies (Optional):
    - Ollama (for local models)
    - openai (for OpenAI models)
    - anthropic (for Claude models)
    - groq (for Groq models)
    - google-generativeai (for Gemini models)

Note:
    For Ollama, ensure that it is running locally and its API is accessible.
    For other providers, an API key must be set through the customize menu.
"""

import argparse
import json
import random
import requests
import sys
import time
import os

# Optional imports for additional LLM providers
try:
    import openai
    OPENAI_AVAILABLE = True
except ImportError:
    print("Warning: openai package not installed. OpenAI integration will be disabled.")
    print("To install: pip install openai")
    OPENAI_AVAILABLE = False

try:
    import anthropic
    ANTHROPIC_AVAILABLE = True
except ImportError:
    print("Warning: anthropic package not installed. Anthropic integration will be disabled.")
    print("To install: pip install anthropic")
    ANTHROPIC_AVAILABLE = False

try:
    import groq
    GROQ_AVAILABLE = True
except ImportError:
    print("Warning: groq package not installed. Groq integration will be disabled.")
    print("To install: pip install groq")
    GROQ_AVAILABLE = False

try:
    import google.generativeai as genai
    GEMINI_AVAILABLE = True
except ImportError:
    print("Warning: google-generativeai package not installed. Google Gemini integration will be disabled.")
    print("To install: pip install google-generativeai")
    GEMINI_AVAILABLE = False

# Try to import optional dependencies with graceful fallback
try:
    import matplotlib
    matplotlib.use("Agg")  # Non-interactive backend for PNG generation
    import matplotlib.pyplot as plt
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    print("Warning: matplotlib not installed. Plotting features will be disabled.")
    print("To install: pip install matplotlib")
    MATPLOTLIB_AVAILABLE = False

try:
    import networkx as nx
    NETWORKX_AVAILABLE = True
except ImportError:
    print("Warning: networkx not installed. Graph visualization will be disabled.")
    print("To install: pip install networkx")
    NETWORKX_AVAILABLE = False

try:
    import numpy as np
    NUMPY_AVAILABLE = True
except ImportError:
    print("Warning: numpy not installed. Some features will be disabled.")
    print("To install: pip install numpy")
    NUMPY_AVAILABLE = False

try:
    import spacy
    SPACY_AVAILABLE = True
    try:
        nlp = spacy.load("en_core_web_md")
    except OSError:
        print("Warning: spaCy model 'en_core_web_md' not found.")
        print("To install: python -m spacy download en_core_web_md")
        SPACY_AVAILABLE = False
except ImportError:
    print("Warning: spacy not installed. Semantic analysis will be disabled.")
    print("To install: pip install spacy")
    SPACY_AVAILABLE = False

try:
    from scipy.linalg import eigh
    SCIPY_AVAILABLE = True
except ImportError:
    print("Warning: scipy not installed. Spectral analysis will be disabled.")
    print("To install: pip install scipy")
    SCIPY_AVAILABLE = False

# Check if all required dependencies are available
if not all([NUMPY_AVAILABLE, SPACY_AVAILABLE]):
    print("Error: Essential dependencies are missing. Please install the required packages.")
    print("To install all dependencies: pip install matplotlib networkx numpy spacy scipy")
    print("After installing spacy: python -m spacy download en_core_web_md")
    sys.exit(1)

# ------------------------------------------
# Hypergraph Data Structure and Utilities
# ------------------------------------------

class Hypergraph:
    def __init__(self):
        self.nodes = set()
        self.hyperedges = {}
        self.edge_counter = 0

    def add_node(self, node):
        self.nodes.add(node)
    
    def add_hyperedge(self, nodes, label=None):
        for node in nodes:
            self.add_node(node)
        self.hyperedges[self.edge_counter] = {"nodes": set(nodes), "label": label}
        self.edge_counter += 1

    def get_nodes(self):
        return self.nodes

    def get_hyperedges(self):
        return self.hyperedges
    
    def to_dict(self):
        """
        Convert the hypergraph to a dictionary for serialization.
        Also includes the current reasoning parameters.
        """
        return {
            "nodes": list(self.nodes),
            "hyperedges": {
                str(edge_id): {
                    "nodes": list(edge_data["nodes"]),
                    "label": edge_data.get("label"),
                    "semantic_similarity": edge_data.get("semantic_similarity", 0.0)
                }
                for edge_id, edge_data in self.hyperedges.items()
            },
            "edge_counter": self.edge_counter,
            "parameters": REASONING_PARAMS  # Include the current reasoning parameters
        }
    
    @classmethod
    def from_dict(cls, data):
        """
        Create a hypergraph from a dictionary representation.
        Also extracts and returns any saved reasoning parameters.
        """
        hg = cls()
        hg.nodes = set(data["nodes"])
        hg.edge_counter = data["edge_counter"]
        
        for edge_id_str, edge_data in data["hyperedges"].items():
            edge_id = int(edge_id_str)
            nodes = set(edge_data["nodes"])
            label = edge_data.get("label")
            semantic_similarity = edge_data.get("semantic_similarity", 0.0)
            
            # Reconstruct the hyperedge
            hg.hyperedges[edge_id] = {
                "nodes": nodes,
                "label": label,
                "semantic_similarity": semantic_similarity
            }
        
        # Extract parameters if present
        saved_params = data.get("parameters", None)
        
        return hg, saved_params

def save_hypergraph(hg, filename):
    """
    Save a hypergraph to a JSON file.
    """
    import json
    
    with open(filename, 'w', encoding='utf-8') as f:
        json.dump(hg.to_dict(), f, indent=2)
    
    print(f"Hypergraph and reasoning parameters saved to {filename}")

def load_hypergraph(filename):
    """
    Load a hypergraph from a JSON file.
    Also loads and applies any saved reasoning parameters.
    """
    import json
    global REASONING_PARAMS
    
    try:
        with open(filename, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        hg, saved_params = Hypergraph.from_dict(data)
        
        # Apply saved parameters if present
        if saved_params:
            print("Loading saved reasoning parameters:")
            for param, value in saved_params.items():
                if param in REASONING_PARAMS:
                    old_value = REASONING_PARAMS[param]
                    REASONING_PARAMS[param] = value
                    print(f"  {param}: {old_value} -> {value}")
        
        print(f"Hypergraph loaded from {filename}")
        return hg
    except Exception as e:
        print(f"Error loading hypergraph from {filename}: {e}")
        return None

def hypergraph_to_graph(hg: Hypergraph):
    """
    Convert the hypergraph into a NetworkX graph via clique expansion.
    """
    G = nx.Graph()
    G.add_nodes_from(hg.get_nodes())
    for edge in hg.get_hyperedges().values():
        nodes = list(edge["nodes"])
        for i in range(len(nodes)):
            for j in range(i+1, len(nodes)):
                G.add_edge(nodes[i], nodes[j])
    return G

# ------------------------------------------
# Reinforcement Learning Environment
# ------------------------------------------

class HypergraphEvolutionEnv:
    def __init__(self, vocabulary=None, max_steps=10):
        if vocabulary is None:
            self.vocabulary = ["cat", "dog", "lion", "tiger", "mammal", "pet", "wild", "predator", "domestic"]
        else:
            self.vocabulary = vocabulary
        self.max_steps = max_steps
        self.current_step = 0
        self.hypergraph = None

    def reset(self, initial_hypergraph=None):
        """
        Reset the environment to an initial state.
        """
        if initial_hypergraph is not None:
            self.hypergraph = initial_hypergraph
        else:
            self.hypergraph = Hypergraph()
            self.hypergraph.add_hyperedge(["cat", "mammal", "pet"], label="Initial Edge")
        self.current_step = 0
        return self.hypergraph

    def compute_structural_reward(self):
        G = hypergraph_to_graph(self.hypergraph)
        if len(G.nodes()) > 2:
            return nx.average_clustering(G)
        else:
            return 0.0

    def step(self, action):
        """
        Execute an action ('add' or 'remove'):
          - 'add': add a hyperedge by randomly sampling 2 or 3 nodes.
          - 'remove': remove a random hyperedge (if available).
        """
        info = {}
        if action == 'add':
            num_nodes = random.choice([2, 3])
            nodes = random.sample(self.vocabulary, num_nodes)
            self.hypergraph.add_hyperedge(nodes, label="Added Edge")
            info["action"] = f"Added hyperedge with nodes: {nodes}"
        elif action == 'remove':
            if self.hypergraph.hyperedges:
                edge_id = random.choice(list(self.hypergraph.hyperedges.keys()))
                removed_edge = self.hypergraph.hyperedges.pop(edge_id)
                info["action"] = f"Removed hyperedge {edge_id} with nodes: {removed_edge['nodes']}"
            else:
                info["action"] = "No hyperedge to remove."
        else:
            info["action"] = "Invalid action provided."
        self.current_step += 1
        done = self.current_step >= self.max_steps
        return self.hypergraph, None, done, info

# ------------------------------------------
# Cellular Automata Update Rule
# ------------------------------------------

def apply_ca_rule(hypergraph, common_neighbors_threshold=2):
    G = hypergraph_to_graph(hypergraph)
    nodes = list(hypergraph.get_nodes())
    candidate_pairs = []
    for i in range(len(nodes)):
        for j in range(i + 1, len(nodes)):
            if not G.has_edge(nodes[i], nodes[j]):
                neighbors1 = set(G.neighbors(nodes[i]))
                neighbors2 = set(G.neighbors(nodes[j]))
                if len(neighbors1.intersection(neighbors2)) >= common_neighbors_threshold:
                    candidate_pairs.append((nodes[i], nodes[j]))
    for pair in candidate_pairs:
        hypergraph.add_hyperedge(list(pair), label="CA Rule Edge")
    return hypergraph, candidate_pairs

# ------------------------------------------
# Semantic Integration using spaCy
# ------------------------------------------

nlp = spacy.load("en_core_web_md")

def get_semantic_similarity(word1, word2):
    token1 = nlp(word1)
    token2 = nlp(word2)
    return token1.similarity(token2)

def contextualize_hypergraph(hg: Hypergraph):
    for edge_id, edge_data in hg.hyperedges.items():
        nodes = list(edge_data["nodes"])
        if len(nodes) < 2:
            edge_data["semantic_similarity"] = 0.0
            continue
        sims = []
        for i in range(len(nodes)):
            for j in range(i+1, len(nodes)):
                sims.append(get_semantic_similarity(nodes[i], nodes[j]))
        edge_data["semantic_similarity"] = np.mean(sims) if sims else 0.0
    return hg

def compute_semantic_reward(hg: Hypergraph):
    sims = [edge.get("semantic_similarity", 0) for edge in hg.hyperedges.values()]
    return np.mean(sims) if sims else 0.0

def combined_reward(hg: Hypergraph, weight_structural=0.5, weight_semantic=0.5):
    struct_r = 0.0
    G = hypergraph_to_graph(hg)
    if len(G.nodes()) > 2:
        struct_r = nx.average_clustering(G)
    sem_r = compute_semantic_reward(hg)
    return weight_structural * struct_r + weight_semantic * sem_r

# ------------------------------------------
# Spectral Analysis
# ------------------------------------------

def perform_spectral_analysis(G):
    L = nx.normalized_laplacian_matrix(G).todense()
    eigenvalues, eigenvectors = eigh(L)
    return eigenvalues, eigenvectors

# ------------------------------------------
# Plotting and Saving Figures
# ------------------------------------------

def plot_and_save_graph(G, filename, title="Graph"):
    plt.figure(figsize=(10, 8))
    pos = nx.spring_layout(G, seed=42)
    nx.draw_networkx_nodes(G, pos, node_size=500, node_color='skyblue')
    nx.draw_networkx_edges(G, pos, width=1.5)
    nx.draw_networkx_labels(G, pos, font_size=10)
    plt.title(title)
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()

def plot_and_save_reward_progression(rewards, filename):
    plt.figure(figsize=(8, 4))
    plt.plot(range(1, len(rewards)+1), rewards, marker='o', linestyle='-')
    plt.title("Reward Progression over RL Steps")
    plt.xlabel("Step")
    plt.ylabel("Combined Reward")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()

def plot_and_save_semantic_histogram(hg: Hypergraph, filename):
    similarities = [edge.get("semantic_similarity", 0) for edge in hg.hyperedges.values()]
    plt.figure(figsize=(8, 4))
    plt.hist(similarities, bins=30, color='teal', edgecolor='black', alpha=0.7)
    plt.title("Distribution of Hyperedge Semantic Similarities")
    plt.xlabel("Semantic Similarity")
    plt.ylabel("Count")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()

def plot_and_save_laplacian_spectrum(G, filename):
    eigenvalues, _ = perform_spectral_analysis(G)
    plt.figure(figsize=(8, 4))
    plt.plot(eigenvalues, 'o-', color='purple')
    plt.title("Laplacian Eigenvalue Spectrum")
    plt.xlabel("Index")
    plt.ylabel("Eigenvalue")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()

def write_sorted_edges(hg: Hypergraph, filename):
    sorted_edges = sorted(hg.hyperedges.items(), key=lambda x: x[1].get("semantic_similarity", 0))
    with open(filename, "w") as f:
        for edge_id, edge in sorted_edges:
            f.write(f"Edge {edge_id}: Nodes: {edge['nodes']}, Semantic Similarity: {edge.get('semantic_similarity', 0):.3f}\n")

# ------------------------------------------
# Policy Function for Informed Action
# ------------------------------------------

def choose_informed_action(hg: Hypergraph, threshold=0.5, min_edges=5):
    low_sim_edges = [eid for eid, edge in hg.hyperedges.items() if edge.get("semantic_similarity", 1) < threshold]
    if low_sim_edges and len(hg.hyperedges) > min_edges:
        return 'remove'
    else:
        return 'add'

def generate_large_hypergraph(vocabulary, num_hyperedges=50, min_edge_size=2, max_edge_size=4):
    hg = Hypergraph()
    random.seed(42)
    for _ in range(num_hyperedges):
        edge_size = random.randint(min_edge_size, max_edge_size)
        nodes = random.sample(vocabulary, edge_size)
        hg.add_hyperedge(nodes, label="Random Edge")
    return hg

# ------------------------------------------
# Local LLM Integration using Ollama's Local API
# ------------------------------------------

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
}

# Provider-specific model options
LLM_PROVIDER_MODELS = {
    "ollama": ["deepseek-r1:latest", "llama3", "mistral", "phi3", "gemma", "llama2"],
    "openai": ["gpt-4", "gpt-4-turbo", "gpt-3.5-turbo"],
    "anthropic": ["claude-3-opus", "claude-3-sonnet", "claude-3-haiku"],
    "groq": ["llama3-8b-8192", "llama3-70b-8192", "mixtral-8x7b-32768"],
    "gemini": ["gemini-pro", "gemini-1.5-pro"]
}

def apply_hypergraph_reasoning(hg: Hypergraph, query):
    """
    Apply cellular automata rules to the hypergraph before reasoning,
    focusing on the query-relevant parts of the graph.
    """
    # First, contextualize the hypergraph to ensure semantic similarities are up-to-date
    contextualize_hypergraph(hg)
    
    # Create a temporary copy of the hypergraph for reasoning
    import copy
    reasoning_hg = copy.deepcopy(hg)
    
    # Apply CA rules to evolve the graph if enabled
    new_connections = []
    if REASONING_PARAMS["apply_ca_rules"]:
        reasoning_hg, new_connections = apply_ca_rule(
            reasoning_hg, 
            common_neighbors_threshold=REASONING_PARAMS["ca_neighbors_threshold"]
        )
    
    # Find query-relevant nodes
    query_words = query.lower().split()
    relevant_nodes = []
    
    for node in reasoning_hg.nodes:
        # Check if the node is directly mentioned in the query
        if node.lower() in query_words:
            relevant_nodes.append(node)
        else:
            # Check semantic similarity to query words
            node_token = nlp(node)
            for word in query_words:
                if len(word) > 3:  # Only consider meaningful words
                    word_token = nlp(word)
                    if node_token.similarity(word_token) > REASONING_PARAMS["similarity_threshold"]:
                        relevant_nodes.append(node)
                        break
    
    # Extract the subgraph of relevant nodes and their connections
    relevant_edges = []
    for edge_id, edge in reasoning_hg.hyperedges.items():
        if any(node in relevant_nodes for node in edge["nodes"]):
            relevant_edges.append((edge_id, edge))
    
    # Sort relevant edges by semantic similarity
    relevant_edges.sort(key=lambda x: x[1].get("semantic_similarity", 0), reverse=True)
    
    return reasoning_hg, relevant_nodes, relevant_edges, new_connections

def query_llm(hg: Hypergraph, query, max_tokens=None, timeout=30, update_graph=False):
    """
    Create a prompt from the current hypergraph state and the user query,
    and use the selected LLM provider to generate a reasoning response.
    
    If update_graph is True, the LLM's response will be used to update the hypergraph
    by strengthening or weakening connections based on the reasoning.
    """
    # Use default max_tokens if not specified
    if max_tokens is None:
        max_tokens = REASONING_PARAMS["llm_max_tokens"]
        
    # Apply hypergraph reasoning to get relevant context
    reasoning_hg, relevant_nodes, relevant_edges, new_connections = apply_hypergraph_reasoning(hg, query)
    
    # Build a detailed context summary
    context = "# Hypergraph-Based Reasoning System\n\n"
    
    # Add information about the hypergraph
    context += "## Hypergraph Structure\n"
    context += f"- Total nodes: {len(reasoning_hg.nodes)}\n"
    context += f"- Total hyperedges: {len(reasoning_hg.hyperedges)}\n"
    
    # Add information about cellular automata evolution
    if new_connections:
        context += "\n## Cellular Automata Evolution\n"
        context += f"- New connections formed through CA rules: {len(new_connections)}\n"
        max_ca = REASONING_PARAMS["max_ca_examples"]
        for i, pair in enumerate(new_connections[:max_ca]):
            context += f"  - Connected: {pair[0]} and {pair[1]}\n"
        if len(new_connections) > max_ca:
            context += f"  - (and {len(new_connections) - max_ca} more)\n"
    
    # Add information about query-relevant nodes
    context += "\n## Query-Relevant Nodes\n"
    if relevant_nodes:
        max_nodes = REASONING_PARAMS["max_relevant_nodes"]
        for node in relevant_nodes[:max_nodes]:
            context += f"- {node}\n"
        if len(relevant_nodes) > max_nodes:
            context += f"- (and {len(relevant_nodes) - max_nodes} more)\n"
    else:
        context += "- No directly relevant nodes found\n"
    
    # Add information about relevant hyperedges
    context += "\n## Relevant Hyperedges (by semantic similarity)\n"
    if relevant_edges:
        max_edges = REASONING_PARAMS["max_relevant_edges"]
        for i, (edge_id, edge) in enumerate(relevant_edges[:max_edges]):
            context += f"- Edge {edge_id}: {edge['nodes']} (similarity: {edge.get('semantic_similarity', 0):.3f})\n"
        if len(relevant_edges) > max_edges:
            context += f"- (and {len(relevant_edges) - max_edges} more)\n"
    else:
        context += "- No relevant hyperedges found\n"
    
    # Add reasoning instructions with feedback mechanism if update_graph is True
    context += "\n## Reasoning Instructions\n"
    context += "1. Use the hypergraph structure to guide your reasoning\n"
    context += "2. Consider both direct connections and semantic similarities\n"
    context += "3. Pay attention to new connections formed by cellular automata rules\n"
    context += "4. Provide a structured response that shows your reasoning path\n"
    
    if update_graph:
        context += "\n## Feedback Instructions\n"
        context += "After your main response, please provide structured feedback about the connections in the hypergraph:\n"
        context += "1. Identify pairs of nodes that should be more strongly connected based on your reasoning\n"
        context += "2. Identify pairs of nodes that should be less strongly connected\n"
        context += "3. Suggest new connections that should be added to the hypergraph\n"
        context += "Format your feedback as follows:\n"
        context += "```\n"
        context += "STRENGTHEN: [node1, node2, weight_adjustment], [node3, node4, weight_adjustment], ...\n"
        context += "WEAKEN: [node1, node2, weight_adjustment], [node3, node4, weight_adjustment], ...\n"
        context += "NEW: [node1, node2], [node3, node4, node5], ...\n"
        context += "```\n"
        context += "Where weight_adjustment is a value between 0.1 and 0.5 indicating how much to adjust the connection.\n"
    
    # Finalize the prompt
    prompt = context + "\n## Query\n" + query + "\n\n## Response:\n"
    
    # Get the selected provider
    provider = REASONING_PARAMS.get("llm_provider", "ollama").lower()
    model = REASONING_PARAMS.get("llm_model", "deepseek-r1:latest")
    temperature = REASONING_PARAMS.get("llm_temperature", 0.7)
    api_key = REASONING_PARAMS.get("llm_api_key", "")
    
    # Use the appropriate provider
    if provider == "ollama":
        return query_ollama(prompt, model, max_tokens, temperature, timeout)
    elif provider == "openai" and OPENAI_AVAILABLE:
        return query_openai(prompt, model, max_tokens, temperature, api_key, timeout)
    elif provider == "anthropic" and ANTHROPIC_AVAILABLE:
        return query_anthropic(prompt, model, max_tokens, temperature, api_key, timeout)
    elif provider == "groq" and GROQ_AVAILABLE:
        return query_groq(prompt, model, max_tokens, temperature, api_key, timeout)
    elif provider == "gemini" and GEMINI_AVAILABLE:
        return query_gemini(prompt, model, max_tokens, temperature, api_key, timeout)
    else:
        if provider != "ollama":
            missing_pkg = {
                "openai": "openai",
                "anthropic": "anthropic",
                "groq": "groq",
                "gemini": "google-generativeai"
            }.get(provider, provider)
            return f"Error: The {provider} provider is selected but the {missing_pkg} package is not installed. Please install it with 'pip install {missing_pkg}' and try again."
        return "Error: No valid LLM provider available. Please check your configuration."

def query_ollama(prompt, model, max_tokens, temperature, timeout=30):
    """
    Query the Ollama API with the given prompt.
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

def query_openai(prompt, model, max_tokens, temperature, api_key, timeout=30):
    """
    Query the OpenAI API with the given prompt.
    """
    if not api_key:
        return "Error: OpenAI API key is required. Please set it in the customize menu."
    
    try:
        print(f"Querying OpenAI API with model {model}...")
        
        # Configure the client
        client = openai.OpenAI(api_key=api_key)
        
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
    
    except Exception as e:
        return f"Error calling OpenAI API: {str(e)}"

def query_anthropic(prompt, model, max_tokens, temperature, api_key, timeout=30):
    """
    Query the Anthropic API with the given prompt.
    """
    if not api_key:
        return "Error: Anthropic API key is required. Please set it in the customize menu."
    
    try:
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
    
    except Exception as e:
        return f"Error calling Anthropic API: {str(e)}"

def query_groq(prompt, model, max_tokens, temperature, api_key, timeout=30):
    """
    Query the Groq API with the given prompt.
    """
    if not api_key:
        return "Error: Groq API key is required. Please set it in the customize menu."
    
    try:
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
    
    except Exception as e:
        return f"Error calling Groq API: {str(e)}"

def query_gemini(prompt, model, max_tokens, temperature, api_key, timeout=30):
    """
    Query the Google Gemini API with the given prompt.
    """
    if not api_key:
        return "Error: Google API key is required. Please set it in the customize menu."
    
    try:
        print(f"Querying Google Gemini API with model {model}...")
        
        # Configure the API
        genai.configure(api_key=api_key)
        
        # Create the model
        model_obj = genai.GenerativeModel(model_name=model)
        
        # Generate content
        response = model_obj.generate_content(
            prompt,
            generation_config=genai.GenerationConfig(
                temperature=temperature,
                max_output_tokens=max_tokens
            )
        )
        
        # Extract the response content
        return response.text
    
    except Exception as e:
        return f"Error calling Google Gemini API: {str(e)}"

def customize_reasoning_parameters():
    """
    Allow users to customize the parameters used for hypergraph reasoning.
    Returns a dictionary of the updated parameters.
    """
    print("\n=== Customize Hypergraph Reasoning Parameters ===")
    print("Current parameters:")
    for param, value in REASONING_PARAMS.items():
        # Don't show API key for security
        if param == "llm_api_key":
            if value:
                print(f"  {param}: [API KEY SET]")
            else:
                print(f"  {param}: [NOT SET]")
        else:
            print(f"  {param}: {value}")
    
    print("\nEnter new values (or press Enter to keep current value):")
    
    # LLM Provider selection
    print("\n--- LLM Provider Configuration ---")
    
    # Show available providers
    available_providers = ["ollama"]
    if OPENAI_AVAILABLE:
        available_providers.append("openai")
    if ANTHROPIC_AVAILABLE:
        available_providers.append("anthropic")
    if GROQ_AVAILABLE:
        available_providers.append("groq")
    if GEMINI_AVAILABLE:
        available_providers.append("gemini")
    
    print(f"Available LLM providers: {', '.join(available_providers)}")
    
    # Select provider
    current_provider = REASONING_PARAMS.get("llm_provider", "ollama")
    new_provider = input(f"LLM provider [{current_provider}]: ").strip().lower()
    
    if new_provider and new_provider in available_providers:
        REASONING_PARAMS["llm_provider"] = new_provider
        
        # If provider changed, reset model to a default for that provider
        if new_provider != current_provider:
            if new_provider == "ollama":
                REASONING_PARAMS["llm_model"] = "deepseek-r1:latest"
            elif new_provider == "openai":
                REASONING_PARAMS["llm_model"] = "gpt-3.5-turbo"
            elif new_provider == "anthropic":
                REASONING_PARAMS["llm_model"] = "claude-3-haiku"
            elif new_provider == "groq":
                REASONING_PARAMS["llm_model"] = "llama3-8b-8192"
            elif new_provider == "gemini":
                REASONING_PARAMS["llm_model"] = "gemini-pro"
    elif new_provider and new_provider not in available_providers:
        print(f"Provider '{new_provider}' is not available. Install the required package first.")
        print(f"Keeping current provider: {current_provider}")
    
    # Show available models for the selected provider
    provider = REASONING_PARAMS.get("llm_provider", "ollama")
    if provider in LLM_PROVIDER_MODELS:
        models = LLM_PROVIDER_MODELS[provider]
        print(f"Available models for {provider}: {', '.join(models)}")
    
    # Select model
    current_model = REASONING_PARAMS.get("llm_model", "deepseek-r1:latest")
    new_model = input(f"LLM model [{current_model}]: ").strip()
    if new_model:
        REASONING_PARAMS["llm_model"] = new_model
    
    # API key (if not ollama)
    if provider != "ollama":
        current_key = REASONING_PARAMS.get("llm_api_key", "")
        key_display = "[API KEY SET]" if current_key else "[NOT SET]"
        new_key = input(f"API key for {provider} {key_display}: ").strip()
        if new_key:
            REASONING_PARAMS["llm_api_key"] = new_key
    
    # LLM temperature
    new_val = input(f"LLM temperature [{REASONING_PARAMS['llm_temperature']}]: ").strip()
    if new_val:
        try:
            REASONING_PARAMS["llm_temperature"] = float(new_val)
        except ValueError:
            print("Invalid value. Keeping current value.")
    
    # LLM max tokens
    new_val = input(f"LLM max tokens [{REASONING_PARAMS['llm_max_tokens']}]: ").strip()
    if new_val:
        try:
            REASONING_PARAMS["llm_max_tokens"] = int(new_val)
        except ValueError:
            print("Invalid value. Keeping current value.")
    
    print("\n--- Hypergraph Parameters ---")
    
    # Similarity threshold
    new_val = input(f"Similarity threshold [{REASONING_PARAMS['similarity_threshold']}]: ").strip()
    if new_val:
        try:
            REASONING_PARAMS["similarity_threshold"] = float(new_val)
        except ValueError:
            print("Invalid value. Keeping current value.")
    
    # CA neighbors threshold
    new_val = input(f"CA neighbors threshold [{REASONING_PARAMS['ca_neighbors_threshold']}]: ").strip()
    if new_val:
        try:
            REASONING_PARAMS["ca_neighbors_threshold"] = int(new_val)
        except ValueError:
            print("Invalid value. Keeping current value.")
    
    # Max relevant nodes
    new_val = input(f"Max relevant nodes [{REASONING_PARAMS['max_relevant_nodes']}]: ").strip()
    if new_val:
        try:
            REASONING_PARAMS["max_relevant_nodes"] = int(new_val)
        except ValueError:
            print("Invalid value. Keeping current value.")
    
    # Max relevant edges
    new_val = input(f"Max relevant edges [{REASONING_PARAMS['max_relevant_edges']}]: ").strip()
    if new_val:
        try:
            REASONING_PARAMS["max_relevant_edges"] = int(new_val)
        except ValueError:
            print("Invalid value. Keeping current value.")
    
    # Apply CA rules
    new_val = input(f"Apply CA rules (true/false) [{REASONING_PARAMS['apply_ca_rules']}]: ").strip().lower()
    if new_val in ('true', 'false'):
        REASONING_PARAMS["apply_ca_rules"] = (new_val == 'true')
    
    # Max steps
    new_val = input(f"Max steps [{REASONING_PARAMS['max_steps']}]: ").strip()
    if new_val:
        try:
            REASONING_PARAMS["max_steps"] = int(new_val)
        except ValueError:
            print("Invalid value. Keeping current value.")
    
    print("\nUpdated parameters:")
    for param, value in REASONING_PARAMS.items():
        # Don't show API key for security
        if param == "llm_api_key":
            if value:
                print(f"  {param}: [API KEY SET]")
            else:
                print(f"  {param}: [NOT SET]")
        else:
            print(f"  {param}: {value}")
    
    return REASONING_PARAMS

# ------------------------------------------
# External Vocabulary Loader
# ------------------------------------------

def load_vocabulary_from_file(filename):
    vocab = []
    try:
        with open(filename, "r", encoding="utf-8") as f:
            for line in f:
                word = line.strip()
                if word:
                    vocab.append(word)
    except Exception as e:
        print(f"Error loading vocabulary from {filename}: {e}")
    return vocab

# ------------------------------------------
# Interactive Mode Function
# ------------------------------------------

def update_hypergraph_from_llm_feedback(hg, feedback_text):
    """
    Update the hypergraph based on feedback from the LLM.
    
    The feedback can be in various formats, including:
    ```
    STRENGTHEN: [node1, node2, weight_adjustment], [node3, node4, weight_adjustment], ...
    WEAKEN: [node1, node2, weight_adjustment], [node3, node4, weight_adjustment], ...
    NEW: [node1, node2], [node3, node4, node5], ...
    ```
    
    Or in a more natural language format:
    ```
    STRENGTHEN:
    - node1, node2, weight=0.3
    - node3, node4, weight=0.4
    
    NEW:
    - node1, node2, node3
    ```
    
    Returns a tuple of (updated_hypergraph, changes_made) where changes_made is a dictionary
    with keys 'strengthened', 'weakened', and 'new' containing the changes made.
    """
    import re
    import copy
    
    # Create a copy of the hypergraph to modify
    updated_hg = copy.deepcopy(hg)
    
    # Track changes made
    changes = {
        'strengthened': [],
        'weakened': [],
        'new': []
    }
    
    # Extract feedback sections - more flexible patterns to handle different formats
    strengthen_pattern = r"STRENGTHEN:?\s*(.*?)(?=WEAKEN:|NEW:|```|$)"
    weaken_pattern = r"WEAKEN:?\s*(.*?)(?=STRENGTHEN:|NEW:|```|$)"
    new_pattern = r"NEW:?\s*(.*?)(?=STRENGTHEN:|WEAKEN:|```|$)"
    
    # Find all matches
    strengthen_match = re.search(strengthen_pattern, feedback_text, re.DOTALL)
    weaken_match = re.search(weaken_pattern, feedback_text, re.DOTALL)
    new_match = re.search(new_pattern, feedback_text, re.DOTALL)
    
    # Process STRENGTHEN instructions
    if strengthen_match:
        strengthen_text = strengthen_match.group(1).strip()
        
        # Try different formats for strengthen instructions
        
        # Format 1: [node1, node2, weight]
        pairs = re.findall(r'\[(.*?)\]', strengthen_text)
        for pair in pairs:
            try:
                # Split by comma and strip whitespace
                elements = [e.strip() for e in pair.split(',')]
                if len(elements) >= 3:
                    node1, node2 = elements[0], elements[1]
                    # Try to extract weight from the third element
                    weight_text = elements[2]
                    weight_adj = float(weight_text.replace('weight=', '').replace('weight_adjustment=', ''))
                    
                    # Find existing hyperedges containing both nodes
                    for edge_id, edge in updated_hg.hyperedges.items():
                        if node1 in edge["nodes"] and node2 in edge["nodes"]:
                            # Strengthen the connection
                            current_sim = edge.get("semantic_similarity", 0)
                            edge["semantic_similarity"] = min(1.0, current_sim + weight_adj)
                            changes['strengthened'].append((node1, node2, weight_adj, edge_id))
                            break
            except Exception as e:
                print(f"Error processing strengthen instruction format 1: {e}")
        
        # Format 2: - node1, node2, weight=0.3
        bullet_pairs = re.findall(r'-\s*(.*?)(?=\n-|\n\n|$)', strengthen_text)
        for pair in bullet_pairs:
            try:
                # Split by comma and strip whitespace
                elements = [e.strip() for e in pair.split(',')]
                if len(elements) >= 2:
                    node1, node2 = elements[0], elements[1]
                    
                    # Try to extract weight from the last element or use default
                    weight_adj = 0.3  # Default weight
                    for elem in elements[2:]:
                        if 'weight' in elem:
                            try:
                                weight_adj = float(re.search(r'[\d.]+', elem).group())
                            except:
                                pass
                    
                    # Find existing hyperedges containing both nodes
                    edge_found = False
                    for edge_id, edge in updated_hg.hyperedges.items():
                        if node1 in edge["nodes"] and node2 in edge["nodes"]:
                            # Strengthen the connection
                            current_sim = edge.get("semantic_similarity", 0)
                            edge["semantic_similarity"] = min(1.0, current_sim + weight_adj)
                            changes['strengthened'].append((node1, node2, weight_adj, edge_id))
                            edge_found = True
                            break
                    
                    # If no existing edge, create a new one
                    if not edge_found:
                        updated_hg.add_hyperedge([node1, node2], label="LLM Suggested Edge")
                        edge_id = updated_hg.edge_counter - 1
                        edge_data = updated_hg.hyperedges[edge_id]
                        
                        # Calculate semantic similarity for the new edge
                        nodes_list = list(edge_data["nodes"])
                        sims = []
                        for i in range(len(nodes_list)):
                            for j in range(i+1, len(nodes_list)):
                                sims.append(get_semantic_similarity(nodes_list[i], nodes_list[j]))
                        
                        # Set initial similarity plus adjustment
                        base_sim = np.mean(sims) if sims else 0.0
                        edge_data["semantic_similarity"] = min(1.0, base_sim + weight_adj)
                        changes['new'].append(([node1, node2], edge_id))
            except Exception as e:
                print(f"Error processing strengthen instruction format 2: {e}")
    
    # Process WEAKEN instructions
    if weaken_match:
        weaken_text = weaken_match.group(1).strip()
        
        # Format 1: [node1, node2, weight]
        pairs = re.findall(r'\[(.*?)\]', weaken_text)
        for pair in pairs:
            try:
                # Split by comma and strip whitespace
                elements = [e.strip() for e in pair.split(',')]
                if len(elements) >= 3:
                    node1, node2 = elements[0], elements[1]
                    # Try to extract weight from the third element
                    weight_text = elements[2]
                    weight_adj = float(weight_text.replace('weight=', '').replace('weight_adjustment=', ''))
                    
                    # Find existing hyperedges containing both nodes
                    for edge_id, edge in updated_hg.hyperedges.items():
                        if node1 in edge["nodes"] and node2 in edge["nodes"]:
                            # Weaken the connection
                            current_sim = edge.get("semantic_similarity", 0)
                            edge["semantic_similarity"] = max(0.0, current_sim - weight_adj)
                            changes['weakened'].append((node1, node2, weight_adj, edge_id))
                            break
            except Exception as e:
                print(f"Error processing weaken instruction format 1: {e}")
        
        # Format 2: - node1, node2, weight=0.3
        bullet_pairs = re.findall(r'-\s*(.*?)(?=\n-|\n\n|$)', weaken_text)
        for pair in bullet_pairs:
            try:
                # Split by comma and strip whitespace
                elements = [e.strip() for e in pair.split(',')]
                if len(elements) >= 2:
                    node1, node2 = elements[0], elements[1]
                    
                    # Try to extract weight from the last element or use default
                    weight_adj = 0.3  # Default weight
                    for elem in elements[2:]:
                        if 'weight' in elem:
                            try:
                                weight_adj = float(re.search(r'[\d.]+', elem).group())
                            except:
                                pass
                    
                    # Find existing hyperedges containing both nodes
                    for edge_id, edge in updated_hg.hyperedges.items():
                        if node1 in edge["nodes"] and node2 in edge["nodes"]:
                            # Weaken the connection
                            current_sim = edge.get("semantic_similarity", 0)
                            edge["semantic_similarity"] = max(0.0, current_sim - weight_adj)
                            changes['weakened'].append((node1, node2, weight_adj, edge_id))
                            break
            except Exception as e:
                print(f"Error processing weaken instruction format 2: {e}")
    
    # Process NEW instructions
    if new_match:
        new_text = new_match.group(1).strip()
        
        # Format 1: [node1, node2, node3]
        groups = re.findall(r'\[(.*?)\]', new_text)
        for group in groups:
            try:
                # Split by comma and strip whitespace
                nodes = [n.strip() for n in group.split(',')]
                if len(nodes) >= 2:
                    # Filter out any weight specifications
                    nodes = [n for n in nodes if not n.startswith('weight')]
                    
                    # Add new nodes to vocabulary if needed
                    for node in nodes:
                        if node not in updated_hg.nodes:
                            updated_hg.add_node(node)
                    
                    # Add new hyperedge
                    updated_hg.add_hyperedge(nodes, label="LLM Suggested Edge")
                    edge_id = updated_hg.edge_counter - 1
                    
                    # Calculate semantic similarity for the new edge
                    edge_data = updated_hg.hyperedges[edge_id]
                    nodes_list = list(edge_data["nodes"])
                    sims = []
                    for i in range(len(nodes_list)):
                        for j in range(i+1, len(nodes_list)):
                            sims.append(get_semantic_similarity(nodes_list[i], nodes_list[j]))
                    edge_data["semantic_similarity"] = np.mean(sims) if sims else 0.0
                    
                    changes['new'].append((nodes, edge_id))
            except Exception as e:
                print(f"Error processing new instruction format 1: {e}")
        
        # Format 2: - node1, node2, node3
        bullet_groups = re.findall(r'-\s*(.*?)(?=\n-|\n\n|$)', new_text)
        for group in bullet_groups:
            try:
                # Split by comma and strip whitespace
                nodes = [n.strip() for n in group.split(',')]
                
                # Extract weight if present
                weight_adj = None
                filtered_nodes = []
                for node in nodes:
                    if 'weight' in node:
                        try:
                            weight_adj = float(re.search(r'[\d.]+', node).group())
                        except:
                            filtered_nodes.append(node)
                    else:
                        filtered_nodes.append(node)
                
                nodes = filtered_nodes
                
                if len(nodes) >= 2:
                    # Add new nodes to vocabulary if needed
                    for node in nodes:
                        if node not in updated_hg.nodes:
                            updated_hg.add_node(node)
                    
                    # Add new hyperedge
                    updated_hg.add_hyperedge(nodes, label="LLM Suggested Edge")
                    edge_id = updated_hg.edge_counter - 1
                    
                    # Calculate semantic similarity for the new edge
                    edge_data = updated_hg.hyperedges[edge_id]
                    nodes_list = list(edge_data["nodes"])
                    sims = []
                    for i in range(len(nodes_list)):
                        for j in range(i+1, len(nodes_list)):
                            sims.append(get_semantic_similarity(nodes_list[i], nodes_list[j]))
                    
                    # Apply weight adjustment if specified
                    base_sim = np.mean(sims) if sims else 0.0
                    if weight_adj is not None:
                        edge_data["semantic_similarity"] = min(1.0, base_sim + weight_adj)
                    else:
                        edge_data["semantic_similarity"] = base_sim
                    
                    changes['new'].append((nodes, edge_id))
            except Exception as e:
                print(f"Error processing new instruction format 2: {e}")
    
    return updated_hg, changes

def interactive_mode(env):
    print("Entering interactive mode. Type 'help' for available commands.")
    
    # Track the evolution of the hypergraph
    hypergraph_history = []
    
    # Save initial state
    import copy
    hypergraph_history.append(copy.deepcopy(env.hypergraph))
    
    while True:
        command = input("\nEnter command: ").strip().lower()
        if command == "help":
            print("\nAvailable commands:")
            print("  step [n]       - Execute n simulation steps (default is 1)")
            print("  add            - Force an 'add' action")
            print("  remove         - Force a 'remove' action")
            print("  reason [query] - Ask a reasoning query (LLM will respond)")
            print("  reason_fb [q]  - Ask a reasoning query with feedback to update the hypergraph")
            print("  save [file]    - Save the current hypergraph and parameters to a file")
            print("  load [file]    - Load a hypergraph and its parameters from a file")
            print("  customize      - Customize hypergraph reasoning parameters and LLM provider settings")
            print("  evolve         - Apply cellular automata rules to evolve the hypergraph")
            print("  explore [node] - Explore connections for a specific node")
            print("  plot           - Update and save plots (graph, histogram, spectrum)")
            print("  stats          - Display current hypergraph statistics")
            print("  undo           - Revert to previous hypergraph state (if available)")
            print("  exit           - Exit interactive mode")
        elif command.startswith("step"):
            parts = command.split()
            n_steps = int(parts[1]) if len(parts) > 1 and parts[1].isdigit() else 1
            
            # Save current state before modifications
            hypergraph_history.append(copy.deepcopy(env.hypergraph))
            
            # Reset the environment's current step counter and max_steps
            env.current_step = 0
            env.max_steps = REASONING_PARAMS["max_steps"]
            
            for i in range(n_steps):
                contextualize_hypergraph(env.hypergraph)
                action = choose_informed_action(env.hypergraph, threshold=0.5, min_edges=10)
                env.hypergraph, _, done, info = env.step(action)
                contextualize_hypergraph(env.hypergraph)
                current_reward = combined_reward(env.hypergraph)
                print(f"Step executed: {info['action']} | Combined Reward: {current_reward:.4f} | Action: {action}")
                if done:
                    print(f"Maximum steps reached ({REASONING_PARAMS['max_steps']} steps).")
                    break
        elif command == "add":
            # Save current state before modifications
            hypergraph_history.append(copy.deepcopy(env.hypergraph))
            
            env.hypergraph, _, _, info = env.step("add")
            contextualize_hypergraph(env.hypergraph)
            print(f"Manual add executed: {info['action']}")
        elif command == "remove":
            # Save current state before modifications
            hypergraph_history.append(copy.deepcopy(env.hypergraph))
            
            env.hypergraph, _, _, info = env.step("remove")
            contextualize_hypergraph(env.hypergraph)
            print(f"Manual remove executed: {info['action']}")
        elif command.startswith("reason_fb"):
            # Reasoning with feedback to update the hypergraph
            query = command[len("reason_fb"):].strip()
            if not query:
                query = input("Enter your reasoning query (with feedback): ")
            
            print("\nProcessing query through hypergraph reasoning with feedback...")
            # Save current state before modifications
            hypergraph_history.append(copy.deepcopy(env.hypergraph))
            
            # Get the reasoning context first
            reasoning_hg, relevant_nodes, relevant_edges, new_connections = apply_hypergraph_reasoning(env.hypergraph, query)
            
            # Display reasoning context
            print(f"\nQuery: {query}")
            print(f"\nFound {len(relevant_nodes)} relevant nodes:")
            for node in relevant_nodes[:5]:
                print(f"  - {node}")
            if len(relevant_nodes) > 5:
                print(f"  - (and {len(relevant_nodes) - 5} more)")
                
            print(f"\nTop relevant hyperedges:")
            for i, (edge_id, edge) in enumerate(relevant_edges[:3]):
                print(f"  - Edge {edge_id}: {edge['nodes']} (similarity: {edge.get('semantic_similarity', 0):.3f})")
            if len(relevant_edges) > 3:
                print(f"  - (and {len(relevant_edges) - 3} more)")
                
            if new_connections:
                print(f"\nCellular automata formed {len(new_connections)} new connections")
            
            # Now get the LLM response with feedback
            response = query_llm(env.hypergraph, query, update_graph=True)
            provider = REASONING_PARAMS.get("llm_provider", "ollama").capitalize()
            print(f"\n{provider} Reasoning Output:\n", response)
            
            # Update the hypergraph based on the LLM's feedback
            updated_hg, changes = update_hypergraph_from_llm_feedback(env.hypergraph, response)
            
            # Display changes made to the hypergraph
            print("\nChanges made to the hypergraph:")
            
            if changes['strengthened']:
                print("\nStrengthened connections:")
                for node1, node2, weight_adj, edge_id in changes['strengthened']:
                    print(f"  - Edge {edge_id}: {node1} and {node2} (increased by {weight_adj:.2f})")
            
            if changes['weakened']:
                print("\nWeakened connections:")
                for node1, node2, weight_adj, edge_id in changes['weakened']:
                    print(f"  - Edge {edge_id}: {node1} and {node2} (decreased by {weight_adj:.2f})")
            
            if changes['new']:
                print("\nNew connections:")
                for nodes, edge_id in changes['new']:
                    print(f"  - Edge {edge_id}: {', '.join(nodes)}")
            
            if not any(changes.values()):
                print("  No changes were made to the hypergraph.")
            
            # Update the environment's hypergraph
            env.hypergraph = updated_hg
            
        elif command.startswith("reason") and not command.startswith("reason_fb"):
            query = command[len("reason"):].strip()
            if not query:
                query = input("Enter your reasoning query: ")
            
            print("\nProcessing query through hypergraph reasoning...")
            # Get the reasoning context first
            reasoning_hg, relevant_nodes, relevant_edges, new_connections = apply_hypergraph_reasoning(env.hypergraph, query)
            
            # Display reasoning context
            print(f"\nQuery: {query}")
            print(f"\nFound {len(relevant_nodes)} relevant nodes:")
            for node in relevant_nodes[:5]:
                print(f"  - {node}")
            if len(relevant_nodes) > 5:
                print(f"  - (and {len(relevant_nodes) - 5} more)")
                
            print(f"\nTop relevant hyperedges:")
            for i, (edge_id, edge) in enumerate(relevant_edges[:3]):
                print(f"  - Edge {edge_id}: {edge['nodes']} (similarity: {edge.get('semantic_similarity', 0):.3f})")
            if len(relevant_edges) > 3:
                print(f"  - (and {len(relevant_edges) - 3} more)")
                
            if new_connections:
                print(f"\nCellular automata formed {len(new_connections)} new connections")
            
            # Now get the LLM response
            response = query_llm(env.hypergraph, query)
            provider = REASONING_PARAMS.get("llm_provider", "ollama").capitalize()
            print(f"\n{provider} Reasoning Output:\n", response)
        elif command == "evolve":
            # Save current state before modifications
            hypergraph_history.append(copy.deepcopy(env.hypergraph))
            
            threshold = REASONING_PARAMS["ca_neighbors_threshold"]
            print(f"Applying cellular automata rules to evolve the hypergraph (threshold: {threshold})...")
            env.hypergraph, ca_pairs = apply_ca_rule(env.hypergraph, common_neighbors_threshold=threshold)
            contextualize_hypergraph(env.hypergraph)
            
            if ca_pairs:
                print(f"Created {len(ca_pairs)} new connections:")
                for i, pair in enumerate(ca_pairs[:5]):
                    print(f"  - Connected: {pair[0]} and {pair[1]}")
                if len(ca_pairs) > 5:
                    print(f"  - (and {len(ca_pairs) - 5} more)")
            else:
                print(f"No new connections were formed. Try lowering the CA neighbors threshold (currently {threshold}).")
        elif command.startswith("explore"):
            parts = command.split(maxsplit=1)
            if len(parts) < 2:
                print("Please specify a node to explore. Example: explore cat")
                continue
                
            node_to_explore = parts[1].strip()
            if node_to_explore not in env.hypergraph.nodes:
                print(f"Node '{node_to_explore}' not found in the hypergraph.")
                print(f"Available nodes: {', '.join(list(env.hypergraph.nodes)[:10])}...")
                continue
                
            print(f"\nExploring connections for node: {node_to_explore}")
            
            # Find all hyperedges containing this node
            connected_edges = []
            for edge_id, edge in env.hypergraph.hyperedges.items():
                if node_to_explore in edge["nodes"]:
                    connected_edges.append((edge_id, edge))
            
            # Sort by semantic similarity
            connected_edges.sort(key=lambda x: x[1].get("semantic_similarity", 0), reverse=True)
            
            print(f"Found in {len(connected_edges)} hyperedges:")
            for i, (edge_id, edge) in enumerate(connected_edges[:7]):
                other_nodes = [n for n in edge["nodes"] if n != node_to_explore]
                print(f"  - Edge {edge_id}: Connected to {', '.join(other_nodes)} (similarity: {edge.get('semantic_similarity', 0):.3f})")
            
            if len(connected_edges) > 7:
                print(f"  - (and {len(connected_edges) - 7} more)")
                
            # Find semantically similar nodes
            print("\nSemanticaly similar nodes:")
            similarities = []
            for other_node in env.hypergraph.nodes:
                if other_node != node_to_explore:
                    sim = get_semantic_similarity(node_to_explore, other_node)
                    similarities.append((other_node, sim))
            
            # Sort by similarity
            similarities.sort(key=lambda x: x[1], reverse=True)
            
            for node, sim in similarities[:5]:
                print(f"  - {node}: {sim:.3f}")
        elif command == "plot":
            G_current = hypergraph_to_graph(env.hypergraph)
            plot_and_save_graph(G_current, "interactive_hypergraph.png", title="Interactive Hypergraph (Clique Expansion)")
            plot_and_save_semantic_histogram(env.hypergraph, "interactive_semantic_histogram.png")
            plot_and_save_laplacian_spectrum(G_current, "interactive_laplacian_spectrum.png")
            print("Updated plots saved as:")
            print("  - interactive_hypergraph.png")
            print("  - interactive_semantic_histogram.png")
            print("  - interactive_laplacian_spectrum.png")
        elif command.startswith("save"):
            # Save the hypergraph to a file
            parts = command.split(maxsplit=1)
            if len(parts) < 2:
                filename = input("Enter filename to save hypergraph: ").strip()
            else:
                filename = parts[1].strip()
            
            # Add .json extension if not provided
            if not filename.endswith('.json'):
                filename += '.json'
            
            # Save the hypergraph
            save_hypergraph(env.hypergraph, filename)
            
        elif command.startswith("load"):
            # Load a hypergraph from a file
            parts = command.split(maxsplit=1)
            if len(parts) < 2:
                filename = input("Enter filename to load hypergraph: ").strip()
            else:
                filename = parts[1].strip()
            
            # Add .json extension if not provided
            if not filename.endswith('.json'):
                filename += '.json'
            
            # Load the hypergraph
            loaded_hg = load_hypergraph(filename)
            if loaded_hg:
                # Save current state before loading
                hypergraph_history.append(copy.deepcopy(env.hypergraph))
                
                # Update the environment's hypergraph
                env.hypergraph = loaded_hg
                
                # Update the vocabulary with new nodes
                for node in loaded_hg.nodes:
                    if node not in env.vocabulary:
                        env.vocabulary.append(node)
                
                print(f"Loaded hypergraph with {len(loaded_hg.nodes)} nodes and {len(loaded_hg.hyperedges)} hyperedges.")
                print(f"Vocabulary updated to {len(env.vocabulary)} words.")
            
        elif command == "customize":
            # Allow users to customize reasoning parameters
            customize_reasoning_parameters()
        elif command == "stats":
            print("\nCurrent Hypergraph Stats:")
            print(f"  Nodes: {len(env.hypergraph.nodes)}")
            print(f"  Hyperedges: {len(env.hypergraph.hyperedges)}")
            print(f"  Latest Combined Reward: {combined_reward(env.hypergraph):.4f}")
            
            # Show top 5 most semantically similar edges
            sorted_edges = sorted(env.hypergraph.hyperedges.items(), 
                                 key=lambda x: x[1].get("semantic_similarity", 0), 
                                 reverse=True)
            
            print("\n  Top 5 most semantically similar hyperedges:")
            for i, (edge_id, edge) in enumerate(sorted_edges[:5]):
                print(f"    Edge {edge_id}: {edge['nodes']} (similarity: {edge.get('semantic_similarity', 0):.3f})")
        elif command == "undo":
            if len(hypergraph_history) > 1:
                # Restore previous state
                env.hypergraph = hypergraph_history.pop()
                print("Reverted to previous hypergraph state.")
            else:
                print("No previous state available to restore.")
        elif command == "exit":
            print("Exiting interactive mode.")
            break
        else:
            print("Unknown command. Type 'help' to see available commands.")

# ------------------------------------------
# Main CLI Application
# ------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Hypergraph Evolution CLI Application")
    parser.add_argument("--max_steps", type=int, default=300, help="Maximum number of RL steps")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose output")
    parser.add_argument("--interactive", action="store_true", help="Enter interactive mode after simulation")
    parser.add_argument("--vocab_file", type=str, help="Path to external vocabulary file (one word per line)")
    parser.add_argument("--model", type=str, help="Model to use for reasoning (default: deepseek-r1:latest)")
    parser.add_argument("--skip_init", action="store_true", help="Skip initial hypergraph generation and go directly to interactive mode")
    parser.add_argument("--small_init", action="store_true", help="Use a small initial hypergraph (faster startup)")
    parser.add_argument("--num_edges", type=int, default=50, help="Number of initial hyperedges to generate")
    args = parser.parse_args()
    
    # Set the model if specified
    if args.model:
        REASONING_PARAMS["llm_model"] = args.model
        if args.verbose:
            print(f"Using model: {args.model}")

    # Load vocabulary from external file if provided, else use default expanded vocabulary.
    if args.vocab_file:
        expanded_vocabulary = load_vocabulary_from_file(args.vocab_file)
        if args.verbose:
            print(f"Loaded {len(expanded_vocabulary)} vocabulary words from {args.vocab_file}.")
    else:
        expanded_vocabulary = [
            "cat", "dog", "lion", "tiger", "wolf", "fox", "mammal", "reptile",
            "amphibian", "bird", "insect", "pet", "wild", "domestic", "predator",
            "prey", "feline", "canine", "avian", "scales", "feathers", "claws", 
            "fur", "wings", "nocturnal", "diurnal", "vertebrate", "invertebrate",
            "small", "large", "fast", "slow", "agile", "fierce", "gentle", "playful",
            "computer", "ocean", "mathematics", "philosophy", "music", "art", "history"
        ]
        if args.verbose:
            print("Using default vocabulary.")

    # Create environment with vocabulary
    env = HypergraphEvolutionEnv(vocabulary=expanded_vocabulary, max_steps=args.max_steps)
    
    # Skip initialization if requested
    if args.skip_init:
        print("Skipping initial hypergraph generation.")
        initial_hg = Hypergraph()
        initial_hg.add_hyperedge(["cat", "mammal", "pet"], label="Initial Edge")
        env.reset(initial_hypergraph=initial_hg)
        
        if args.interactive:
            print("Entering interactive mode directly.")
            interactive_mode(env)
            return
    else:
        # Determine number of edges for initial hypergraph
        num_edges = 10 if args.small_init else args.num_edges
        
        print(f"Generating initial hypergraph with {num_edges} edges...")
        initial_hg = generate_large_hypergraph(expanded_vocabulary, num_hyperedges=num_edges)
        
        print("Contextualizing hypergraph (calculating semantic similarities)...")
        print("This may take a moment for large hypergraphs...")
        contextualize_hypergraph(initial_hg)
        
        # Create an environment seeded with our hypergraph
        env.reset(initial_hypergraph=initial_hg)

    # RL loop with informed policy.
    rewards = []
    if args.verbose:
        print("Starting RL simulation with max_steps =", args.max_steps)
    for step in range(env.max_steps):
        contextualize_hypergraph(env.hypergraph)
        action = choose_informed_action(env.hypergraph, threshold=0.5, min_edges=10)
        env.hypergraph, _, done, info = env.step(action)
        contextualize_hypergraph(env.hypergraph)
        r = combined_reward(env.hypergraph)
        rewards.append(r)
        if args.verbose:
            print(f"Step {step+1}: {info['action']} | Combined Reward: {r:.4f} | Action: {action}")
        if done:
            break

    # Apply a cellular automata update.
    threshold = REASONING_PARAMS["ca_neighbors_threshold"]
    env.hypergraph, ca_pairs = apply_ca_rule(env.hypergraph, common_neighbors_threshold=threshold)
    if args.verbose:
        print(f"Applied CA rule (threshold: {threshold}); new hyperedge pairs added: {len(ca_pairs)}")

    # Final outputs and plots.
    G_final = hypergraph_to_graph(env.hypergraph)
    plot_and_save_graph(G_final, "final_hypergraph.png", title="Final Hypergraph (Clique Expansion)")
    if args.verbose:
        print("Saved final hypergraph plot as 'final_hypergraph.png'.")
    plot_and_save_reward_progression(rewards, "reward_progression.png")
    if args.verbose:
        print("Saved reward progression plot as 'reward_progression.png'.")
    plot_and_save_semantic_histogram(env.hypergraph, "semantic_histogram.png")
    if args.verbose:
        print("Saved semantic histogram as 'semantic_histogram.png'.")
    plot_and_save_laplacian_spectrum(G_final, "laplacian_spectrum.png")
    if args.verbose:
        print("Saved Laplacian spectrum as 'laplacian_spectrum.png'.")
    write_sorted_edges(env.hypergraph, "sorted_edges.txt")
    if args.verbose:
        print("Saved sorted hyperedges to 'sorted_edges.txt'.")

    print("\n--- Simulation Summary ---")
    print("Total hyperedges:", len(env.hypergraph.hyperedges))
    print("Total nodes:", len(env.hypergraph.nodes))
    print("Final combined reward:", rewards[-1] if rewards else "N/A")
    print("Simulation complete.")

    # Enter interactive mode if requested.
    if args.interactive:
        interactive_mode(env)

if __name__ == "__main__":
    main()
