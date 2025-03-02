"""
Interactive CLI interface for the Hypergraph Evolution application.
"""

import os
import sys
import cmd
import psutil
import config
import networkx as nx
from config import REASONING_PARAMS, LLM_PROVIDER_MODELS
from hypergraph.core import Hypergraph, hypergraph_to_graph
from hypergraph.serialization import save_hypergraph, load_hypergraph
from environment.env import HypergraphEvolutionEnv
from environment.policy import generate_large_hypergraph
from analysis.semantic import contextualize_hypergraph, load_vocabulary_from_file, combined_reward
from analysis.cellular_automata import apply_ca_rule
from llm.reasoning import reason_with_llm, check_provider_availability
from llm.feedback import evaluate_suggestion
from llm.dream import run_dream_session
from llm.dream_enhanced import run_enhanced_dream_session
from llm.dream_feedback import run_dream_feedback_session
from llm.feedback_reasoning import reason_with_feedback
from llm.chat import ChatDatabase, chat_with_llm
from llm.chat_feedback import chat_with_feedback
from visualization.plotting import (
    plot_and_save_graph, 
    plot_and_save_semantic_histogram,
    write_sorted_edges
)

class HypergraphShell(cmd.Cmd):
    """Interactive shell for the Hypergraph Evolution application."""
    
    intro = """
    ╔═══════════════════════════════════════════════════════════════╗
    ║                                                               ║
    ║   Hypergraph Evolution - Semantic Reasoning with LLMs         ║
    ║                                                               ║
    ║   Type 'menu' to show the main menu.                          ║
    ║   Type 'help' or '?' to list all commands.                    ║
    ║   Type 'exit' or 'quit' to exit.                              ║
    ║                                                               ║
    ╚═══════════════════════════════════════════════════════════════╝
    """
    prompt = "hypergraph> "
    
    def __init__(self):
        super().__init__()
        self.hypergraph = Hypergraph()
        self.env = None
        self.vocabulary = []
        
        # Check which providers are available
        check_provider_availability()
        
        # Print available providers
        print("Available LLM providers:")
        if config.OPENAI_AVAILABLE:
            print("  - OpenAI")
        if config.ANTHROPIC_AVAILABLE:
            print("  - Anthropic")
        if config.GROQ_AVAILABLE:
            print("  - Groq")
        if config.GEMINI_AVAILABLE:
            print("  - Gemini")
        print("  - Ollama (local)")
        print()
    
    def do_load(self, arg):
        """Load a hypergraph from a file: load <filename>"""
        if not arg:
            print("Please provide a filename.")
            return
        
        hg = load_hypergraph(arg)
        if hg:
            self.hypergraph = hg
    
    def do_save(self, arg):
        """Save the current hypergraph to a file: save <filename>"""
        if not arg:
            print("Please provide a filename.")
            return
        
        save_hypergraph(self.hypergraph, arg)
    
    def do_add_node(self, arg):
        """Add a node to the hypergraph: add_node <node>"""
        if not arg:
            print("Please provide a node name.")
            return
        
        self.hypergraph.add_node(arg)
        print(f"Added node: {arg}")
    
    def do_add_edge(self, arg):
        """Add a hyperedge connecting multiple nodes: add_edge <node1> <node2> ... [label]"""
        args = arg.split()
        if len(args) < 2:
            print("Please provide at least two nodes to connect.")
            return
        
        # Check if the last argument is a label (enclosed in quotes)
        if args[-1].startswith('"') and args[-1].endswith('"'):
            nodes = args[:-1]
            label = args[-1].strip('"')
        else:
            nodes = args
            label = None
        
        self.hypergraph.add_hyperedge(nodes, label)
        print(f"Added hyperedge connecting: {', '.join(nodes)}")
    
    def do_show(self, arg):
        """Show the current hypergraph"""
        nodes = self.hypergraph.get_nodes()
        edges = self.hypergraph.get_hyperedges()
        
        print(f"\nHypergraph with {len(nodes)} nodes and {len(edges)} hyperedges:")
        
        print("\nNodes:")
        for node in nodes:
            print(f"  - {node}")
        
        print("\nHyperedges:")
        for edge_id, edge in edges.items():
            nodes_str = ", ".join(edge["nodes"])
            label = f" (Label: {edge['label']})" if edge.get("label") else ""
            sim = f" (Similarity: {edge.get('semantic_similarity', 'N/A')})" if "semantic_similarity" in edge else ""
            print(f"  - Edge {edge_id}: [{nodes_str}]{label}{sim}")
        
        print()
    
    def do_load_vocabulary(self, arg):
        """Load vocabulary from a file: load_vocabulary <filename>"""
        if not arg:
            print("Please provide a filename.")
            return
        
        self.vocabulary = load_vocabulary_from_file(arg)
        print(f"Loaded {len(self.vocabulary)} words from {arg}")
    
    def do_generate(self, arg):
        """Generate a random hypergraph: generate [num_edges]"""
        if not self.vocabulary:
            print("Please load a vocabulary first using 'load_vocabulary'.")
            return
        
        try:
            num_edges = int(arg) if arg else 10
        except ValueError:
            print("Please provide a valid number of edges.")
            return
        
        self.hypergraph = generate_large_hypergraph(self.vocabulary, num_edges)
        print(f"Generated hypergraph with {num_edges} random hyperedges.")
    
    def do_contextualize(self, arg):
        """Calculate semantic similarities for all hyperedges"""
        if not config.SPACY_AVAILABLE:
            print("Error: spaCy is not available. Please install it with 'pip install spacy'.")
            print("Then download the model with 'python -m spacy download en_core_web_md'")
            return
        
        print("Calculating semantic similarities (this may take a moment)...")
        self.hypergraph = contextualize_hypergraph(self.hypergraph)
        print("Done.")
    
    def do_apply_ca(self, arg):
        """Apply cellular automata rules to evolve the hypergraph"""
        if not config.NETWORKX_AVAILABLE:
            print("Error: NetworkX is not available. Please install it with 'pip install networkx'.")
            return
        
        try:
            threshold = int(arg) if arg else REASONING_PARAMS["ca_neighbors_threshold"]
        except ValueError:
            print("Please provide a valid threshold.")
            return
        
        print(f"Applying CA rules with threshold {threshold}...")
        self.hypergraph, new_connections = apply_ca_rule(self.hypergraph, threshold)
        
        if new_connections:
            print(f"Created {len(new_connections)} new connections:")
            for node1, node2 in new_connections:
                print(f"  - Connected {node1} and {node2}")
        else:
            print("No new connections were created.")
    
    def do_reason(self, arg):
        """Use LLM to reason about the hypergraph and suggest new concepts"""
        focus_node = arg if arg else None
        
        print(f"Reasoning with {REASONING_PARAMS['llm_provider']} model {REASONING_PARAMS['llm_model']}...")
        response = reason_with_llm(self.hypergraph, focus_node)
        
        print("\nLLM Response:")
        print(response)
        print()
    
    def do_evaluate(self, arg):
        """Evaluate a suggestion using the LLM: evaluate <suggestion>"""
        if not arg:
            print("Please provide a suggestion to evaluate.")
            return
        
        print(f"Evaluating suggestion with {REASONING_PARAMS['llm_provider']} model {REASONING_PARAMS['llm_model']}...")
        response = evaluate_suggestion(self.hypergraph, arg)
        
        print("\nLLM Evaluation:")
        print(response)
        print()
    
    def do_dream(self, arg):
        """Run a classic dream/self-talk session with a single LLM"""
        try:
            iterations = int(arg) if arg else REASONING_PARAMS["dream_iterations"]
        except ValueError:
            print("Please provide a valid number of iterations.")
            return
        
        print(f"Running classic dream session with {REASONING_PARAMS['llm2_provider']} model {REASONING_PARAMS['llm2_model']}...")
        response = run_dream_session(self.hypergraph, iterations)
        
        print("\nDream Session:")
        print(response)
        print()
    
    def do_dream_enhanced(self, arg):
        """
        Run an enhanced dream session with two LLMs in conversation.
        
        This creates a dialogue between the primary and secondary LLMs,
        with each taking turns to explore concepts in the hypergraph.
        """
        try:
            iterations = int(arg) if arg else REASONING_PARAMS["dream_iterations"]
        except ValueError:
            print("Please provide a valid number of iterations.")
            return
        
        print(f"Running enhanced dream session between:")
        print(f"- Primary LLM: {REASONING_PARAMS['llm_provider']} model {REASONING_PARAMS['llm_model']}")
        print(f"- Secondary LLM: {REASONING_PARAMS['llm2_provider']} model {REASONING_PARAMS['llm2_model']}")
        
        response = run_enhanced_dream_session(self.hypergraph, iterations)
        
        print("\nEnhanced Dream Session:")
        print(response)
        print()
    
    def do_dream_fb(self, arg):
        """
        Run a dream session with feedback that updates the hypergraph.
        
        This creates a dialogue between the primary and secondary LLMs,
        with each taking turns to explore concepts and suggest updates to the hypergraph.
        """
        try:
            iterations = int(arg) if arg else REASONING_PARAMS["dream_iterations"]
        except ValueError:
            print("Please provide a valid number of iterations.")
            return
        
        print(f"Running dream session with feedback between:")
        print(f"- Primary LLM: {REASONING_PARAMS['llm_provider']} model {REASONING_PARAMS['llm_model']}")
        print(f"- Secondary LLM: {REASONING_PARAMS['llm2_provider']} model {REASONING_PARAMS['llm2_model']}")
        
        self.hypergraph, response = run_dream_feedback_session(self.hypergraph, iterations)
        
        print("\nDream Session with Feedback:")
        print(response)
        print()
    
    def do_visualize(self, arg):
        """Visualize the hypergraph: visualize [filename]"""
        if not config.MATPLOTLIB_AVAILABLE or not config.NETWORKX_AVAILABLE:
            print("Error: Matplotlib and/or NetworkX are not available.")
            print("Please install them with 'pip install matplotlib networkx'.")
            return
        
        from hypergraph.core import hypergraph_to_graph
        
        filename = arg if arg else "hypergraph.png"
        
        print(f"Visualizing hypergraph to {filename}...")
        G = hypergraph_to_graph(self.hypergraph)
        plot_and_save_graph(G, filename)
        print(f"Visualization saved to {filename}")
    
    def do_histogram(self, arg):
        """Plot a histogram of semantic similarities: histogram [filename]"""
        if not config.MATPLOTLIB_AVAILABLE:
            print("Error: Matplotlib is not available. Please install it with 'pip install matplotlib'.")
            return
        
        filename = arg if arg else "histogram.png"
        
        print(f"Plotting semantic similarity histogram to {filename}...")
        plot_and_save_semantic_histogram(self.hypergraph, filename)
        print(f"Histogram saved to {filename}")
    
    def do_write_edges(self, arg):
        """Write sorted edges to a file: write_edges [filename]"""
        filename = arg if arg else "edges.txt"
        
        print(f"Writing sorted edges to {filename}...")
        write_sorted_edges(self.hypergraph, filename)
        print(f"Edges written to {filename}")
    
    def do_customize(self, arg):
        """Customize reasoning parameters"""
        self._show_customize_menu()
    
    def _show_customize_menu(self):
        """Show the customization menu"""
        while True:
            print("\nCustomize Reasoning Parameters:")
            print("  1. LLM Provider and Model")
            print("  2. LLM Temperature")
            print("  3. LLM Max Tokens")
            print("  4. Similarity Threshold")
            print("  5. CA Neighbors Threshold")
            print("  6. Apply CA Rules")
            print("  7. Max Steps")
            print("  8. Second LLM (for Dream Mode)")
            print("  9. Dream Iterations")
            print("  0. Back to main menu")
            
            choice = input("\nEnter your choice (0-9): ")
            
            if choice == "0":
                break
            elif choice == "1":
                self._customize_llm_provider()
            elif choice == "2":
                self._customize_temperature()
            elif choice == "3":
                self._customize_max_tokens()
            elif choice == "4":
                self._customize_similarity_threshold()
            elif choice == "5":
                self._customize_ca_threshold()
            elif choice == "6":
                self._customize_apply_ca()
            elif choice == "7":
                self._customize_max_steps()
            elif choice == "8":
                self._customize_second_llm()
            elif choice == "9":
                self._customize_dream_iterations()
    
    def _customize_llm_provider(self):
        """Customize the LLM provider and model"""
        print("\nAvailable LLM Providers:")
        print("  1. Ollama (local)")
        print("  2. OpenAI" + (" (not installed)" if not config.OPENAI_AVAILABLE else ""))
        print("  3. Anthropic" + (" (not installed)" if not config.ANTHROPIC_AVAILABLE else ""))
        print("  4. Groq" + (" (not installed)" if not config.GROQ_AVAILABLE else ""))
        print("  5. Google Gemini" + (" (not installed)" if not config.GEMINI_AVAILABLE else ""))
        
        choice = input("\nSelect provider (1-5): ")
        
        provider_map = {
            "1": "ollama",
            "2": "openai",
            "3": "anthropic",
            "4": "groq",
            "5": "gemini"
        }
        
        if choice in provider_map:
            provider = provider_map[choice]
            
            # Check if the provider is available
            if provider == "openai" and not config.OPENAI_AVAILABLE:
                print("OpenAI package not installed. Please install it with 'pip install openai'.")
                return
            elif provider == "anthropic" and not config.ANTHROPIC_AVAILABLE:
                print("Anthropic package not installed. Please install it with 'pip install anthropic'.")
                return
            elif provider == "groq" and not config.GROQ_AVAILABLE:
                print("Groq package not installed. Please install it with 'pip install groq'.")
                return
            elif provider == "gemini" and not config.GEMINI_AVAILABLE:
                print("Google Generative AI package not installed. Please install it with 'pip install google-generativeai'.")
                return
            
            # Show available models for the selected provider
            print(f"\nAvailable models for {provider}:")
            for i, model in enumerate(LLM_PROVIDER_MODELS[provider], 1):
                print(f"  {i}. {model}")
            
            model_choice = input(f"\nSelect model (1-{len(LLM_PROVIDER_MODELS[provider])}): ")
            try:
                model_idx = int(model_choice) - 1
                if 0 <= model_idx < len(LLM_PROVIDER_MODELS[provider]):
                    model = LLM_PROVIDER_MODELS[provider][model_idx]
                    
                    # Set the provider and model
                    REASONING_PARAMS["llm_provider"] = provider
                    REASONING_PARAMS["llm_model"] = model
                    
                    # If not Ollama, ask for API key
                    if provider != "ollama":
                        api_key = input(f"\nEnter your {provider} API key: ")
                        REASONING_PARAMS["llm_api_key"] = api_key
                    else:
                        # Clear any existing API key if switching to Ollama
                        REASONING_PARAMS["llm_api_key"] = ""
                    
                    print(f"\nLLM provider set to {provider} with model {model}")
                else:
                    print("Invalid model selection.")
            except ValueError:
                print("Invalid input.")
        else:
            print("Invalid provider selection.")
    
    def _customize_second_llm(self):
        """Customize the second LLM for dream mode"""
        print("\nAvailable LLM Providers for Dream Mode:")
        print("  1. Ollama (local)")
        print("  2. OpenAI" + (" (not installed)" if not config.OPENAI_AVAILABLE else ""))
        print("  3. Anthropic" + (" (not installed)" if not config.ANTHROPIC_AVAILABLE else ""))
        print("  4. Groq" + (" (not installed)" if not config.GROQ_AVAILABLE else ""))
        print("  5. Google Gemini" + (" (not installed)" if not config.GEMINI_AVAILABLE else ""))
        
        choice = input("\nSelect provider (1-5): ")
        
        provider_map = {
            "1": "ollama",
            "2": "openai",
            "3": "anthropic",
            "4": "groq",
            "5": "gemini"
        }
        
        if choice in provider_map:
            provider = provider_map[choice]
            
            # Check if the provider is available
            if provider == "openai" and not config.OPENAI_AVAILABLE:
                print("OpenAI package not installed. Please install it with 'pip install openai'.")
                return
            elif provider == "anthropic" and not config.ANTHROPIC_AVAILABLE:
                print("Anthropic package not installed. Please install it with 'pip install anthropic'.")
                return
            elif provider == "groq" and not config.GROQ_AVAILABLE:
                print("Groq package not installed. Please install it with 'pip install groq'.")
                return
            elif provider == "gemini" and not config.GEMINI_AVAILABLE:
                print("Google Generative AI package not installed. Please install it with 'pip install google-generativeai'.")
                return
            
            # Show available models for the selected provider
            print(f"\nAvailable models for {provider}:")
            for i, model in enumerate(LLM_PROVIDER_MODELS[provider], 1):
                print(f"  {i}. {model}")
            
            model_choice = input(f"\nSelect model (1-{len(LLM_PROVIDER_MODELS[provider])}): ")
            try:
                model_idx = int(model_choice) - 1
                if 0 <= model_idx < len(LLM_PROVIDER_MODELS[provider]):
                    model = LLM_PROVIDER_MODELS[provider][model_idx]
                    
                    # Set the provider and model
                    REASONING_PARAMS["llm2_provider"] = provider
                    REASONING_PARAMS["llm2_model"] = model
                    
                    # If not Ollama, ask for API key
                    if provider != "ollama":
                        api_key = input(f"\nEnter your {provider} API key: ")
                        REASONING_PARAMS["llm2_api_key"] = api_key
                    else:
                        # Clear any existing API key if switching to Ollama
                        REASONING_PARAMS["llm2_api_key"] = ""
                    
                    print(f"\nSecond LLM provider set to {provider} with model {model}")
                else:
                    print("Invalid model selection.")
            except ValueError:
                print("Invalid input.")
        else:
            print("Invalid provider selection.")
    
    def _customize_temperature(self):
        """Customize the LLM temperature"""
        current = REASONING_PARAMS["llm_temperature"]
        print(f"\nCurrent temperature: {current}")
        
        try:
            temp = float(input("Enter new temperature (0.0-1.0): "))
            if 0.0 <= temp <= 1.0:
                REASONING_PARAMS["llm_temperature"] = temp
                print(f"Temperature set to {temp}")
            else:
                print("Temperature must be between 0.0 and 1.0.")
        except ValueError:
            print("Invalid input.")
    
    def _customize_max_tokens(self):
        """Customize the LLM max tokens"""
        current = REASONING_PARAMS["llm_max_tokens"]
        print(f"\nCurrent max tokens: {current}")
        
        try:
            tokens = int(input("Enter new max tokens: "))
            if tokens > 0:
                REASONING_PARAMS["llm_max_tokens"] = tokens
                print(f"Max tokens set to {tokens}")
            else:
                print("Max tokens must be positive.")
        except ValueError:
            print("Invalid input.")
    
    def _customize_similarity_threshold(self):
        """Customize the similarity threshold"""
        current = REASONING_PARAMS["similarity_threshold"]
        print(f"\nCurrent similarity threshold: {current}")
        
        try:
            threshold = float(input("Enter new threshold (0.0-1.0): "))
            if 0.0 <= threshold <= 1.0:
                REASONING_PARAMS["similarity_threshold"] = threshold
                print(f"Similarity threshold set to {threshold}")
            else:
                print("Threshold must be between 0.0 and 1.0.")
        except ValueError:
            print("Invalid input.")
    
    def _customize_ca_threshold(self):
        """Customize the CA neighbors threshold"""
        current = REASONING_PARAMS["ca_neighbors_threshold"]
        print(f"\nCurrent CA neighbors threshold: {current}")
        
        try:
            threshold = int(input("Enter new threshold: "))
            if threshold > 0:
                REASONING_PARAMS["ca_neighbors_threshold"] = threshold
                print(f"CA neighbors threshold set to {threshold}")
            else:
                print("Threshold must be positive.")
        except ValueError:
            print("Invalid input.")
    
    def _customize_apply_ca(self):
        """Customize whether to apply CA rules"""
        current = REASONING_PARAMS["apply_ca_rules"]
        print(f"\nCurrently {'applying' if current else 'not applying'} CA rules")
        
        choice = input("Apply CA rules? (y/n): ")
        if choice.lower() in ["y", "yes"]:
            REASONING_PARAMS["apply_ca_rules"] = True
            print("Will apply CA rules.")
        elif choice.lower() in ["n", "no"]:
            REASONING_PARAMS["apply_ca_rules"] = False
            print("Will not apply CA rules.")
        else:
            print("Invalid input.")
    
    def _customize_max_steps(self):
        """Customize the max steps"""
        current = REASONING_PARAMS["max_steps"]
        print(f"\nCurrent max steps: {current}")
        
        try:
            steps = int(input("Enter new max steps: "))
            if steps > 0:
                REASONING_PARAMS["max_steps"] = steps
                print(f"Max steps set to {steps}")
            else:
                print("Max steps must be positive.")
        except ValueError:
            print("Invalid input.")
    
    def _customize_dream_iterations(self):
        """Customize the dream iterations"""
        current = REASONING_PARAMS["dream_iterations"]
        print(f"\nCurrent dream iterations: {current}")
        
        try:
            iterations = int(input("Enter new dream iterations: "))
            if iterations > 0:
                REASONING_PARAMS["dream_iterations"] = iterations
                print(f"Dream iterations set to {iterations}")
            else:
                print("Dream iterations must be positive.")
        except ValueError:
            print("Invalid input.")
    
    def do_reason_fb(self, arg):
        """
        Use LLM to reason about the hypergraph and provide feedback to update it.
        
        This creates a feedback loop where the LLM's reasoning affects the hypergraph structure.
        Type '#__#' to exit the feedback reasoning mode.
        """
        if not arg:
            print("Please provide an initial query.")
            return
        
        print(f"Starting feedback reasoning with {REASONING_PARAMS['llm_provider']} model {REASONING_PARAMS['llm_model']}...")
        print("Type '#__#' to exit the feedback reasoning mode.")
        print()
        
        query = arg
        while True:
            # Reason with feedback
            self.hypergraph, reasoning, summary = reason_with_feedback(self.hypergraph, query)
            
            # Print the reasoning
            print("\nLLM Reasoning:")
            print(reasoning)
            print()
            
            # Print the summary of changes
            if summary:
                print("Changes to hypergraph:")
                for change in summary:
                    print(f"  - {change}")
                print()
            else:
                print("No changes were made to the hypergraph.")
                print()
            
            # Get the next query
            query = input("Enter next query (or '#__#' to exit): ")
            if query == "#__#":
                break
    
    def do_chat(self, arg):
        """
        Start a chat session with the LLM.
        
        The LLM will have access to the hypergraph as context.
        Type '#__#' to exit the chat mode.
        """
        print(f"Starting chat with {REASONING_PARAMS['llm_provider']} model {REASONING_PARAMS['llm_model']}...")
        print("The LLM has access to the hypergraph as context.")
        print("Type '#__#' to exit the chat mode.")
        print()
        
        # Initialize the chat database
        db = ChatDatabase()
        session_id = db.create_session()
        
        messages = []
        while True:
            # Get user input
            user_input = input("You: ")
            if user_input == "#__#":
                break
            
            # Add user message to the database and messages list
            db.add_message(session_id, "user", user_input)
            messages.append(("user", user_input))
            
            # Get LLM response
            response = chat_with_llm(messages, self.hypergraph)
            
            # Add assistant message to the database and messages list
            db.add_message(session_id, "assistant", response)
            messages.append(("assistant", response))
            
            # Print the response
            print(f"Assistant: {response}")
            print()
    
    def do_chat_fb(self, arg):
        """
        Start a chat session with the LLM that also updates the hypergraph.
        
        This combines chat functionality with feedback reasoning, maintaining conversation
        history while also updating the hypergraph based on the conversation.
        Type '#__#' to exit the chat with feedback mode.
        """
        print(f"Starting chat with feedback using {REASONING_PARAMS['llm_provider']} model {REASONING_PARAMS['llm_model']}...")
        print("The LLM will have a conversation with you while updating the hypergraph.")
        print("Type '#__#' to exit the chat with feedback mode.")
        print()
        
        # Initialize the chat database
        db = ChatDatabase()
        session_id = db.create_session()
        
        messages = []
        while True:
            # Get user input
            user_input = input("You: ")
            if user_input == "#__#":
                break
            
            # Add user message to the database and messages list
            db.add_message(session_id, "user", user_input)
            messages.append(("user", user_input))
            
            # Get LLM response with feedback
            self.hypergraph, chat_response, summary = chat_with_feedback(self.hypergraph, messages)
            
            # Add assistant message to the database and messages list
            db.add_message(session_id, "assistant", chat_response)
            messages.append(("assistant", chat_response))
            
            # Print the response
            print(f"Assistant: {chat_response}")
            print()
            
            # Print the summary of changes
            if summary:
                print("Changes to hypergraph:")
                for change in summary:
                    print(f"  - {change}")
                print()
    
    def do_stats(self, arg):
        """Display statistics about the current hypergraph and system"""
        nodes = self.hypergraph.get_nodes()
        edges = self.hypergraph.get_hyperedges()
        
        print("\n╔═══════════════════════════════════════════════════════════════╗")
        print("║                     Hypergraph Statistics                     ║")
        print("╚═══════════════════════════════════════════════════════════════╝")
        
        # Basic statistics
        print("\n[Basic Statistics]")
        print(f"Nodes: {len(nodes)}")
        print(f"Hyperedges: {len(edges)}")
        
        # Node statistics
        if nodes:
            node_degrees = {}
            for node in nodes:
                node_degrees[node] = 0
                for edge in edges.values():
                    if node in edge["nodes"]:
                        node_degrees[node] += 1
            
            avg_degree = sum(node_degrees.values()) / len(node_degrees)
            max_degree_node = max(node_degrees.items(), key=lambda x: x[1])
            
            print(f"Average node degree: {avg_degree:.2f}")
            print(f"Highest degree node: '{max_degree_node[0]}' with {max_degree_node[1]} connections")
        
        # Edge statistics
        if edges:
            edge_sizes = [len(edge["nodes"]) for edge in edges.values()]
            avg_edge_size = sum(edge_sizes) / len(edge_sizes)
            max_edge_size = max(edge_sizes)
            min_edge_size = min(edge_sizes)
            
            print(f"Average hyperedge size: {avg_edge_size:.2f}")
            print(f"Largest hyperedge size: {max_edge_size}")
            print(f"Smallest hyperedge size: {min_edge_size}")
        
        # Semantic statistics
        semantic_edges = [edge for edge in edges.values() if "semantic_similarity" in edge]
        if semantic_edges:
            similarities = [edge["semantic_similarity"] for edge in semantic_edges]
            avg_similarity = sum(similarities) / len(similarities)
            max_similarity = max(similarities)
            min_similarity = min(similarities)
            
            print("\n[Semantic Statistics]")
            print(f"Edges with semantic similarity: {len(semantic_edges)}/{len(edges)}")
            print(f"Average semantic similarity: {avg_similarity:.4f}")
            print(f"Highest semantic similarity: {max_similarity:.4f}")
            print(f"Lowest semantic similarity: {min_similarity:.4f}")
        
        # Structural statistics
        if config.NETWORKX_AVAILABLE and nodes:
            print("\n[Structural Statistics]")
            G = hypergraph_to_graph(self.hypergraph)
            
            # Calculate connected components
            try:
                num_components = nx.number_connected_components(G)
                largest_component = max(nx.connected_components(G), key=len)
                print(f"Connected components: {num_components}")
                print(f"Largest component size: {len(largest_component)}/{len(nodes)}")
            except:
                print("Could not calculate connected components.")
            
            # Calculate clustering coefficient
            try:
                clustering = nx.average_clustering(G)
                print(f"Average clustering coefficient: {clustering:.4f}")
            except:
                print("Could not calculate clustering coefficient.")
            
            # Calculate diameter (only for the largest connected component)
            try:
                largest_cc = G.subgraph(largest_component)
                diameter = nx.diameter(largest_cc)
                print(f"Diameter of largest component: {diameter}")
            except:
                print("Could not calculate diameter.")
        
        # LLM provider information
        print("\n[LLM Configuration]")
        print(f"Primary LLM: {REASONING_PARAMS['llm_provider']} - {REASONING_PARAMS['llm_model']}")
        print(f"Secondary LLM: {REASONING_PARAMS['llm2_provider']} - {REASONING_PARAMS['llm2_model']}")
        print(f"Temperature: {REASONING_PARAMS['llm_temperature']}")
        print(f"Max tokens: {REASONING_PARAMS['llm_max_tokens']}")
        
        # System information
        print("\n[System Information]")
        process = psutil.Process(os.getpid())
        memory_usage = process.memory_info().rss / 1024 / 1024  # Convert to MB
        print(f"Memory usage: {memory_usage:.2f} MB")
        print(f"Available providers: ", end="")
        providers = []
        if config.OPENAI_AVAILABLE:
            providers.append("OpenAI")
        if config.ANTHROPIC_AVAILABLE:
            providers.append("Anthropic")
        if config.GROQ_AVAILABLE:
            providers.append("Groq")
        if config.GEMINI_AVAILABLE:
            providers.append("Gemini")
        providers.append("Ollama")
        print(", ".join(providers))
        
        print()
    
    def do_exit(self, arg):
        """Exit the program"""
        print("Goodbye!")
        return True
    
    def do_quit(self, arg):
        """Exit the program"""
        return self.do_exit(arg)
    
    def do_menu(self, arg):
        """Show the main menu with organized command categories"""
        while True:
            print("\n╔═══════════════════════════════════════════════════════════════╗")
            print("║                     Hypergraph Evolution                      ║")
            print("╚═══════════════════════════════════════════════════════════════╝")
            print("\nMain Menu:")
            print("  1. Hypergraph Management")
            print("  2. Node & Edge Operations")
            print("  3. Analysis & Evolution")
            print("  4. LLM Interaction")
            print("  5. Settings")
            print("  0. Return to command line")
            
            choice = input("\nEnter your choice (0-5): ")
            
            if choice == "0":
                break
            elif choice == "1":
                self._show_hypergraph_management_menu()
            elif choice == "2":
                self._show_node_edge_menu()
            elif choice == "3":
                self._show_analysis_menu()
            elif choice == "4":
                self._show_llm_menu()
            elif choice == "5":
                self._show_customize_menu()
            else:
                print("Invalid choice. Please try again.")
    
    def _show_hypergraph_management_menu(self):
        """Show the hypergraph management submenu"""
        while True:
            print("\nHypergraph Management:")
            print("  1. Load hypergraph from file")
            print("  2. Save hypergraph to file")
            print("  3. Show current hypergraph")
            print("  4. Display statistics")
            print("  0. Back to main menu")
            
            choice = input("\nEnter your choice (0-4): ")
            
            if choice == "0":
                break
            elif choice == "1":
                filename = input("Enter filename to load: ")
                self.do_load(filename)
            elif choice == "2":
                filename = input("Enter filename to save: ")
                self.do_save(filename)
            elif choice == "3":
                self.do_show("")
            elif choice == "4":
                self.do_stats("")
            else:
                print("Invalid choice. Please try again.")
    
    def _show_node_edge_menu(self):
        """Show the node and edge operations submenu"""
        while True:
            print("\nNode & Edge Operations:")
            print("  1. Add node")
            print("  2. Add edge")
            print("  3. Load vocabulary")
            print("  4. Generate random hypergraph")
            print("  0. Back to main menu")
            
            choice = input("\nEnter your choice (0-4): ")
            
            if choice == "0":
                break
            elif choice == "1":
                node = input("Enter node name: ")
                self.do_add_node(node)
            elif choice == "2":
                nodes = input("Enter nodes (space-separated): ")
                self.do_add_edge(nodes)
            elif choice == "3":
                filename = input("Enter vocabulary filename: ")
                self.do_load_vocabulary(filename)
            elif choice == "4":
                num_edges = input("Enter number of edges (default: 10): ")
                self.do_generate(num_edges)
            else:
                print("Invalid choice. Please try again.")
    
    def _show_analysis_menu(self):
        """Show the analysis and evolution submenu"""
        while True:
            print("\nAnalysis & Evolution:")
            print("  1. Calculate semantic similarities")
            print("  2. Apply cellular automata rules")
            print("  3. Visualize hypergraph")
            print("  4. Plot semantic histogram")
            print("  5. Write edges to file")
            print("  0. Back to main menu")
            
            choice = input("\nEnter your choice (0-5): ")
            
            if choice == "0":
                break
            elif choice == "1":
                self.do_contextualize("")
            elif choice == "2":
                threshold = input("Enter threshold (default: from settings): ")
                self.do_apply_ca(threshold)
            elif choice == "3":
                filename = input("Enter filename (default: hypergraph.png): ")
                self.do_visualize(filename)
            elif choice == "4":
                filename = input("Enter filename (default: histogram.png): ")
                self.do_histogram(filename)
            elif choice == "5":
                filename = input("Enter filename (default: edges.txt): ")
                self.do_write_edges(filename)
            else:
                print("Invalid choice. Please try again.")
    
    def _show_llm_menu(self):
        """Show the LLM interaction submenu"""
        while True:
            print("\nLLM Interaction:")
            print("  1. Basic reasoning")
            print("  2. Reasoning with feedback")
            print("  3. Evaluate suggestion")
            print("  4. Chat mode")
            print("  5. Chat with feedback")
            print("  6. Classic dream mode")
            print("  7. Enhanced dream mode")
            print("  8. Dream with feedback")
            print("  0. Back to main menu")
            
            choice = input("\nEnter your choice (0-8): ")
            
            if choice == "0":
                break
            elif choice == "1":
                focus_node = input("Enter focus node (optional): ")
                self.do_reason(focus_node)
            elif choice == "2":
                query = input("Enter initial query: ")
                if query:
                    self.do_reason_fb(query)
            elif choice == "3":
                suggestion = input("Enter suggestion to evaluate: ")
                if suggestion:
                    self.do_evaluate(suggestion)
            elif choice == "4":
                self.do_chat("")
            elif choice == "5":
                self.do_chat_fb("")
            elif choice == "6":
                iterations = input("Enter number of iterations (default: from settings): ")
                self.do_dream(iterations)
            elif choice == "7":
                iterations = input("Enter number of iterations (default: from settings): ")
                self.do_dream_enhanced(iterations)
            elif choice == "8":
                iterations = input("Enter number of iterations (default: from settings): ")
                self.do_dream_fb(iterations)
            else:
                print("Invalid choice. Please try again.")
    
    def emptyline(self):
        """Do nothing on empty line"""
        pass

def main():
    """Main function to run the interactive shell"""
    HypergraphShell().cmdloop()

if __name__ == "__main__":
    main()
