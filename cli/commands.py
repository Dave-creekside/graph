"""
Command-line interface commands for the Hypergraph Evolution application.
"""

import argparse
import sys
import os
import config
from config import REASONING_PARAMS
from hypergraph.core import Hypergraph
from hypergraph.serialization import save_hypergraph, load_hypergraph
from environment.env import HypergraphEvolutionEnv
from environment.policy import generate_large_hypergraph, choose_informed_action
from analysis.semantic import contextualize_hypergraph, load_vocabulary_from_file, combined_reward
from analysis.cellular_automata import apply_ca_rule
from llm.reasoning import reason_with_llm, check_provider_availability
from llm.feedback import evaluate_suggestion
from llm.dream import run_dream_session
from visualization.plotting import (
    plot_and_save_graph, 
    plot_and_save_semantic_histogram,
    plot_and_save_reward_progression,
    plot_and_save_laplacian_spectrum,
    write_sorted_edges
)

def create_parser():
    """Create the command-line argument parser."""
    parser = argparse.ArgumentParser(description="Hypergraph Evolution - Semantic Reasoning with LLMs")
    
    subparsers = parser.add_subparsers(dest="command", help="Command to run")
    
    # Interactive mode
    interactive_parser = subparsers.add_parser("interactive", help="Run in interactive mode")
    
    # Load hypergraph
    load_parser = subparsers.add_parser("load", help="Load a hypergraph from a file")
    load_parser.add_argument("filename", help="Path to the hypergraph file")
    
    # Save hypergraph
    save_parser = subparsers.add_parser("save", help="Save a hypergraph to a file")
    save_parser.add_argument("filename", help="Path to save the hypergraph to")
    save_parser.add_argument("--hypergraph", help="Path to the hypergraph file to save (if not provided, a new one will be created)")
    
    # Generate hypergraph
    generate_parser = subparsers.add_parser("generate", help="Generate a random hypergraph")
    generate_parser.add_argument("--vocabulary", help="Path to the vocabulary file")
    generate_parser.add_argument("--num-edges", type=int, default=10, help="Number of hyperedges to generate")
    generate_parser.add_argument("--output", help="Path to save the generated hypergraph")
    
    # Contextualize hypergraph
    contextualize_parser = subparsers.add_parser("contextualize", help="Calculate semantic similarities for all hyperedges")
    contextualize_parser.add_argument("hypergraph", help="Path to the hypergraph file")
    contextualize_parser.add_argument("--output", help="Path to save the contextualized hypergraph")
    
    # Apply CA rules
    ca_parser = subparsers.add_parser("apply-ca", help="Apply cellular automata rules to evolve the hypergraph")
    ca_parser.add_argument("hypergraph", help="Path to the hypergraph file")
    ca_parser.add_argument("--threshold", type=int, help="Common neighbors threshold")
    ca_parser.add_argument("--output", help="Path to save the evolved hypergraph")
    
    # Reason with LLM
    reason_parser = subparsers.add_parser("reason", help="Use LLM to reason about the hypergraph")
    reason_parser.add_argument("hypergraph", help="Path to the hypergraph file")
    reason_parser.add_argument("--focus-node", help="Node to focus on")
    reason_parser.add_argument("--provider", help="LLM provider to use")
    reason_parser.add_argument("--model", help="LLM model to use")
    reason_parser.add_argument("--api-key", help="API key for the LLM provider")
    
    # Evaluate suggestion
    evaluate_parser = subparsers.add_parser("evaluate", help="Evaluate a suggestion using the LLM")
    evaluate_parser.add_argument("hypergraph", help="Path to the hypergraph file")
    evaluate_parser.add_argument("suggestion", help="Suggestion to evaluate")
    evaluate_parser.add_argument("--focus-node", help="Node to focus on")
    
    # Dream session
    dream_parser = subparsers.add_parser("dream", help="Run a dream/self-talk session with the LLM")
    dream_parser.add_argument("hypergraph", help="Path to the hypergraph file")
    dream_parser.add_argument("--iterations", type=int, help="Number of iterations")
    
    # Visualize hypergraph
    visualize_parser = subparsers.add_parser("visualize", help="Visualize the hypergraph")
    visualize_parser.add_argument("hypergraph", help="Path to the hypergraph file")
    visualize_parser.add_argument("--output", default="hypergraph.png", help="Path to save the visualization")
    
    # Plot semantic histogram
    histogram_parser = subparsers.add_parser("histogram", help="Plot a histogram of semantic similarities")
    histogram_parser.add_argument("hypergraph", help="Path to the hypergraph file")
    histogram_parser.add_argument("--output", default="histogram.png", help="Path to save the histogram")
    
    # Plot Laplacian spectrum
    spectrum_parser = subparsers.add_parser("spectrum", help="Plot the Laplacian eigenvalue spectrum")
    spectrum_parser.add_argument("hypergraph", help="Path to the hypergraph file")
    spectrum_parser.add_argument("--output", default="spectrum.png", help="Path to save the spectrum plot")
    
    # Write sorted edges
    write_edges_parser = subparsers.add_parser("write-edges", help="Write sorted edges to a file")
    write_edges_parser.add_argument("hypergraph", help="Path to the hypergraph file")
    write_edges_parser.add_argument("--output", default="edges.txt", help="Path to save the edges")
    
    # Run RL environment
    rl_parser = subparsers.add_parser("run-rl", help="Run the RL environment")
    rl_parser.add_argument("--vocabulary", help="Path to the vocabulary file")
    rl_parser.add_argument("--steps", type=int, default=10, help="Number of steps to run")
    rl_parser.add_argument("--output", help="Path to save the final hypergraph")
    rl_parser.add_argument("--plot-rewards", help="Path to save the reward progression plot")
    
    return parser

def run_interactive():
    """Run the interactive shell."""
    from cli.interactive import main as interactive_main
    interactive_main()

def run_load(args):
    """Load a hypergraph from a file."""
    hg = load_hypergraph(args.filename)
    if hg:
        print(f"Loaded hypergraph with {len(hg.get_nodes())} nodes and {len(hg.get_hyperedges())} hyperedges.")
    return hg

def run_save(args):
    """Save a hypergraph to a file."""
    if args.hypergraph:
        hg = load_hypergraph(args.hypergraph)
        if not hg:
            return None
    else:
        hg = Hypergraph()
    
    save_hypergraph(hg, args.filename)
    return hg

def run_generate(args):
    """Generate a random hypergraph."""
    if args.vocabulary:
        vocabulary = load_vocabulary_from_file(args.vocabulary)
        if not vocabulary:
            print(f"Error: Could not load vocabulary from {args.vocabulary}")
            return None
    else:
        vocabulary = ["cat", "dog", "lion", "tiger", "mammal", "pet", "wild", "predator", "domestic"]
    
    hg = generate_large_hypergraph(vocabulary, args.num_edges)
    print(f"Generated hypergraph with {args.num_edges} random hyperedges.")
    
    if args.output:
        save_hypergraph(hg, args.output)
        print(f"Saved generated hypergraph to {args.output}")
    
    return hg

def run_contextualize(args):
    """Calculate semantic similarities for all hyperedges."""
    if not config.SPACY_AVAILABLE:
        print("Error: spaCy is not available. Please install it with 'pip install spacy'.")
        print("Then download the model with 'python -m spacy download en_core_web_md'")
        return None
    
    hg = load_hypergraph(args.hypergraph)
    if not hg:
        return None
    
    print("Calculating semantic similarities (this may take a moment)...")
    hg = contextualize_hypergraph(hg)
    print("Done.")
    
    if args.output:
        save_hypergraph(hg, args.output)
        print(f"Saved contextualized hypergraph to {args.output}")
    
    return hg

def run_apply_ca(args):
    """Apply cellular automata rules to evolve the hypergraph."""
    if not config.NETWORKX_AVAILABLE:
        print("Error: NetworkX is not available. Please install it with 'pip install networkx'.")
        return None
    
    hg = load_hypergraph(args.hypergraph)
    if not hg:
        return None
    
    threshold = args.threshold if args.threshold else REASONING_PARAMS["ca_neighbors_threshold"]
    
    print(f"Applying CA rules with threshold {threshold}...")
    hg, new_connections = apply_ca_rule(hg, threshold)
    
    if new_connections:
        print(f"Created {len(new_connections)} new connections:")
        for node1, node2 in new_connections:
            print(f"  - Connected {node1} and {node2}")
    else:
        print("No new connections were created.")
    
    if args.output:
        save_hypergraph(hg, args.output)
        print(f"Saved evolved hypergraph to {args.output}")
    
    return hg

def run_reason(args):
    """Use LLM to reason about the hypergraph."""
    hg = load_hypergraph(args.hypergraph)
    if not hg:
        return None
    
    # Override parameters if provided
    if args.provider:
        REASONING_PARAMS["llm_provider"] = args.provider
    if args.model:
        REASONING_PARAMS["llm_model"] = args.model
    if args.api_key:
        REASONING_PARAMS["llm_api_key"] = args.api_key
    
    focus_node = args.focus_node if args.focus_node else None
    
    print(f"Reasoning with {REASONING_PARAMS['llm_provider']} model {REASONING_PARAMS['llm_model']}...")
    response = reason_with_llm(hg, focus_node)
    
    print("\nLLM Response:")
    print(response)
    print()
    
    return response

def run_evaluate(args):
    """Evaluate a suggestion using the LLM."""
    hg = load_hypergraph(args.hypergraph)
    if not hg:
        return None
    
    focus_node = args.focus_node if args.focus_node else None
    
    print(f"Evaluating suggestion with {REASONING_PARAMS['llm_provider']} model {REASONING_PARAMS['llm_model']}...")
    response = evaluate_suggestion(hg, args.suggestion, focus_node)
    
    print("\nLLM Evaluation:")
    print(response)
    print()
    
    return response

def run_dream(args):
    """Run a dream/self-talk session with the LLM."""
    hg = load_hypergraph(args.hypergraph)
    if not hg:
        return None
    
    iterations = args.iterations if args.iterations else REASONING_PARAMS["dream_iterations"]
    
    print(f"Running dream session with {REASONING_PARAMS['llm2_provider']} model {REASONING_PARAMS['llm2_model']}...")
    response = run_dream_session(hg, iterations)
    
    print("\nDream Session:")
    print(response)
    print()
    
    return response

def run_visualize(args):
    """Visualize the hypergraph."""
    if not config.MATPLOTLIB_AVAILABLE or not config.NETWORKX_AVAILABLE:
        print("Error: Matplotlib and/or NetworkX are not available.")
        print("Please install them with 'pip install matplotlib networkx'.")
        return None
    
    hg = load_hypergraph(args.hypergraph)
    if not hg:
        return None
    
    from hypergraph.core import hypergraph_to_graph
    
    print(f"Visualizing hypergraph to {args.output}...")
    G = hypergraph_to_graph(hg)
    plot_and_save_graph(G, args.output)
    print(f"Visualization saved to {args.output}")
    
    return G

def run_histogram(args):
    """Plot a histogram of semantic similarities."""
    if not config.MATPLOTLIB_AVAILABLE:
        print("Error: Matplotlib is not available. Please install it with 'pip install matplotlib'.")
        return None
    
    hg = load_hypergraph(args.hypergraph)
    if not hg:
        return None
    
    print(f"Plotting semantic similarity histogram to {args.output}...")
    plot_and_save_semantic_histogram(hg, args.output)
    print(f"Histogram saved to {args.output}")
    
    return True

def run_spectrum(args):
    """Plot the Laplacian eigenvalue spectrum."""
    if not config.MATPLOTLIB_AVAILABLE or not config.NETWORKX_AVAILABLE or not config.SCIPY_AVAILABLE:
        print("Error: Matplotlib, NetworkX, and/or SciPy are not available.")
        print("Please install them with 'pip install matplotlib networkx scipy'.")
        return None
    
    hg = load_hypergraph(args.hypergraph)
    if not hg:
        return None
    
    from hypergraph.core import hypergraph_to_graph
    
    print(f"Plotting Laplacian spectrum to {args.output}...")
    G = hypergraph_to_graph(hg)
    plot_and_save_laplacian_spectrum(G, args.output)
    print(f"Spectrum saved to {args.output}")
    
    return True

def run_write_edges(args):
    """Write sorted edges to a file."""
    hg = load_hypergraph(args.hypergraph)
    if not hg:
        return None
    
    print(f"Writing sorted edges to {args.output}...")
    write_sorted_edges(hg, args.output)
    print(f"Edges written to {args.output}")
    
    return True

def run_rl(args):
    """Run the RL environment."""
    if args.vocabulary:
        vocabulary = load_vocabulary_from_file(args.vocabulary)
        if not vocabulary:
            print(f"Error: Could not load vocabulary from {args.vocabulary}")
            return None
    else:
        vocabulary = ["cat", "dog", "lion", "tiger", "mammal", "pet", "wild", "predator", "domestic"]
    
    env = HypergraphEvolutionEnv(vocabulary, args.steps)
    hg = env.reset()
    
    print(f"Running RL environment for {args.steps} steps...")
    
    rewards = []
    for step in range(args.steps):
        action = choose_informed_action(hg)
        hg, _, done, info = env.step(action)
        
        # Calculate reward
        reward = combined_reward(hg)
        rewards.append(reward)
        
        print(f"Step {step+1}/{args.steps}: Action: {action}, Reward: {reward:.4f}")
        print(f"  {info['action']}")
        
        if done:
            break
    
    print(f"RL run completed with final reward: {rewards[-1]:.4f}")
    
    if args.output:
        save_hypergraph(hg, args.output)
        print(f"Saved final hypergraph to {args.output}")
    
    if args.plot_rewards:
        if not config.MATPLOTLIB_AVAILABLE:
            print("Error: Matplotlib is not available. Please install it with 'pip install matplotlib'.")
        else:
            plot_and_save_reward_progression(rewards, args.plot_rewards)
            print(f"Reward progression plot saved to {args.plot_rewards}")
    
    return hg

def main():
    """Main function to run the command-line interface."""
    # Check which providers are available
    check_provider_availability()
    
    parser = create_parser()
    args = parser.parse_args()
    
    if args.command == "interactive":
        run_interactive()
    elif args.command == "load":
        run_load(args)
    elif args.command == "save":
        run_save(args)
    elif args.command == "generate":
        run_generate(args)
    elif args.command == "contextualize":
        run_contextualize(args)
    elif args.command == "apply-ca":
        run_apply_ca(args)
    elif args.command == "reason":
        run_reason(args)
    elif args.command == "evaluate":
        run_evaluate(args)
    elif args.command == "dream":
        run_dream(args)
    elif args.command == "visualize":
        run_visualize(args)
    elif args.command == "histogram":
        run_histogram(args)
    elif args.command == "spectrum":
        run_spectrum(args)
    elif args.command == "write-edges":
        run_write_edges(args)
    elif args.command == "run-rl":
        run_rl(args)
    else:
        parser.print_help()

if __name__ == "__main__":
    main()
