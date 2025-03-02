# Hypergraph Evolution with LLM Reasoning

This application demonstrates how hypergraph reasoning and cellular automata rules can affect language model reasoning processes. It builds a semantic hypergraph from vocabulary, evolves it using reinforcement learning and cellular automata rules, and uses the hypergraph to guide LLM reasoning.

## Features

- **Hypergraph Construction**: Build semantic networks from vocabulary words
- **Reinforcement Learning**: Evolve the hypergraph through RL-based actions
- **Cellular Automata Rules**: Apply CA rules to form new connections based on graph structure
- **Semantic Analysis**: Calculate semantic similarities between nodes using spaCy
- **Multi-LLM Integration**: Use various LLM providers (Ollama, OpenAI, Anthropic, Groq, Google Gemini) to perform reasoning
- **Interactive Mode**: Explore and manipulate the hypergraph through a command-line interface
- **Visualization**: Generate plots of the hypergraph, semantic histograms, and spectral analysis
- **Save/Load**: Persist and restore hypergraphs between sessions

## Requirements

### Core Dependencies
- Python 3.6+
- NetworkX
- Matplotlib
- NumPy
- spaCy (with the 'en_core_web_md' model installed)
- SciPy
- Requests

### LLM Provider Dependencies (Optional)
- Ollama (for local models)
- OpenAI Python package (`pip install openai`)
- Anthropic Python package (`pip install anthropic`)
- Groq Python package (`pip install groq`)
- Google Generative AI package (`pip install google-generativeai`)

## Installation

1. Clone this repository
2. Install the core dependencies:
   ```
   pip install networkx matplotlib numpy spacy scipy requests
   python -m spacy download en_core_web_md
   ```
3. Install LLM provider packages as needed:
   ```
   # For Ollama (local models)
   # Install from https://ollama.ai/

   # For OpenAI models (GPT-4, GPT-3.5, etc.)
   pip install openai

   # For Anthropic models (Claude)
   pip install anthropic

   # For Groq models
   pip install groq

   # For Google Gemini models
   pip install google-generativeai
   ```

## Usage

### Basic Usage

```
python hy.py --interactive --skip_init --model llama3
```

### Command-Line Arguments

- `--max_steps INT`: Maximum number of RL steps (default: 300)
- `--verbose`: Enable verbose output
- `--interactive`: Enter interactive mode after simulation
- `--vocab_file PATH`: Path to external vocabulary file (one word per line)
- `--model STRING`: Model to use for reasoning (default: deepseek-r1:latest)
- `--skip_init`: Skip initial hypergraph generation and go directly to interactive mode
- `--small_init`: Use a small initial hypergraph (faster startup)
- `--num_edges INT`: Number of initial hyperedges to generate (default: 50)

### Interactive Mode Commands

- `help`: Show available commands
- `step [n]`: Execute n simulation steps (default: 1)
- `add`: Force an 'add' action
- `remove`: Force a 'remove' action
- `reason [query]`: Ask a reasoning query (LLM will respond)
- `reason_fb [query]`: Ask a reasoning query with feedback to update the hypergraph
- `save [file]`: Save the current hypergraph to a file
- `load [file]`: Load a hypergraph from a file
- `customize`: Customize hypergraph reasoning parameters and LLM provider settings
- `evolve`: Apply cellular automata rules to evolve the hypergraph
- `explore [node]`: Explore connections for a specific node
- `plot`: Update and save plots (graph, histogram, spectrum)
- `stats`: Display current hypergraph statistics
- `undo`: Revert to previous hypergraph state (if available)
- `exit`: Exit interactive mode

## LLM Provider Configuration

The application supports multiple LLM providers that can be configured through the `customize` command in interactive mode:

### Ollama (Local Models)
- No API key required
- Models: deepseek-r1:latest, llama3, mistral, phi3, gemma, llama2, etc.
- Requires Ollama to be installed and running locally

### OpenAI
- Requires API key from [OpenAI Platform](https://platform.openai.com/)
- Models: gpt-4, gpt-4-turbo, gpt-3.5-turbo, etc.

### Anthropic
- Requires API key from [Anthropic Console](https://console.anthropic.com/)
- Models: claude-3-opus, claude-3-sonnet, claude-3-haiku, etc.

### Groq
- Requires API key from [Groq Cloud](https://console.groq.com/)
- Models: llama3-8b-8192, llama3-70b-8192, mixtral-8x7b-32768, etc.

### Google Gemini
- Requires API key from [Google AI Studio](https://makersuite.google.com/app/apikey)
- Models: gemini-pro, gemini-1.5-pro, etc.

## Understanding the Evolve Command

The `evolve` command applies cellular automata (CA) rules to the hypergraph. These rules look for patterns in the graph structure and create new connections based on those patterns.

Specifically, the CA rule implemented in this application looks for pairs of nodes that:
1. Are not directly connected to each other
2. Share a certain number of common neighbors (defined by the `ca_neighbors_threshold` parameter, which defaults to 2)

When such pairs are found, a new hyperedge is created connecting them.

If the `evolve` command doesn't create any new connections, it could be because:
- There are no pairs of nodes that meet the criteria (no pairs have enough common neighbors)
- The hypergraph is too small or not well-connected enough for the CA rules to find patterns
- The threshold for common neighbors might be too high for the current hypergraph structure

You can adjust the CA neighbors threshold using the `customize` command.

## Reasoning with Feedback

The `reason_fb` command allows the LLM to not only reason based on the hypergraph but also provide feedback to update the hypergraph structure. This creates a complete feedback loop where:

1. The hypergraph guides the LLM's reasoning
2. The LLM's reasoning affects the hypergraph structure
3. The updated hypergraph then influences future reasoning

The LLM can:
- Strengthen existing connections
- Weaken existing connections
- Create new connections

This feedback mechanism allows the hypergraph to evolve based on the LLM's reasoning, creating a dynamic knowledge representation that adapts over time.

## Save and Load

The `save` and `load` commands allow you to persist and restore hypergraphs between sessions:

- `save myhypergraph`: Saves the current hypergraph to myhypergraph.json
- `load myhypergraph`: Loads a hypergraph from myhypergraph.json

When loading a hypergraph, any new nodes are automatically added to the vocabulary, ensuring that the loaded hypergraph can be properly manipulated and evolved.

## Examples

### Basic Reasoning

```
Enter command: reason What is the relationship between mammals and birds?
```

### Reasoning with Feedback

```
Enter command: reason_fb How do domestic and wild animals differ?
```

### Exploring a Node

```
Enter command: explore cat
```

### Evolving the Hypergraph

```
Enter command: evolve
```

### Saving and Loading

```
Enter command: save myhypergraph
Enter command: load myhypergraph
```

## License

This project is open source and available under the MIT License.
