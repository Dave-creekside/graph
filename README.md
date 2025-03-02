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
# Run in interactive mode
python main.py

# Run with command-line arguments
python main.py <command> [options]
```

### Command-Line Interface

The application now uses a modular command-line interface with subcommands:

```
# Show help
python main.py --help

# Run in interactive mode
python main.py interactive

# Generate a random hypergraph
python main.py generate --num-edges 20 --output myhypergraph.json

# Load a hypergraph and reason about it
python main.py reason myhypergraph.json

# Apply cellular automata rules
python main.py apply-ca myhypergraph.json --output evolved.json

# Visualize a hypergraph
python main.py visualize myhypergraph.json --output graph.png
```

### Available Commands

- `interactive`: Run in interactive mode
- `load`: Load a hypergraph from a file
- `save`: Save a hypergraph to a file
- `generate`: Generate a random hypergraph
- `contextualize`: Calculate semantic similarities for all hyperedges
- `apply-ca`: Apply cellular automata rules to evolve the hypergraph
- `reason`: Use LLM to reason about the hypergraph
- `evaluate`: Evaluate a suggestion using the LLM
- `dream`: Run a dream/self-talk session with the LLM
- `visualize`: Visualize the hypergraph
- `histogram`: Plot a histogram of semantic similarities
- `spectrum`: Plot the Laplacian eigenvalue spectrum
- `write-edges`: Write sorted edges to a file
- `run-rl`: Run the RL environment

### Interactive Mode Commands

- `help`: Show available commands
- `add_node <node>`: Add a node to the hypergraph
- `add_edge <node1> <node2> ...`: Add a hyperedge connecting multiple nodes
- `show`: Show the current hypergraph
- `load <file>`: Load a hypergraph from a file
- `save <file>`: Save the current hypergraph to a file
- `load_vocabulary <file>`: Load vocabulary from a file
- `generate [num_edges]`: Generate a random hypergraph
- `contextualize`: Calculate semantic similarities for all hyperedges
- `apply_ca [threshold]`: Apply cellular automata rules to evolve the hypergraph
- `reason [focus_node]`: Use LLM to reason about the hypergraph
- `reason_fb <query>`: Interactive reasoning with feedback to update the hypergraph
- `evaluate <suggestion>`: Evaluate a suggestion using the LLM
- `dream [iterations]`: Run a dream/self-talk session with the LLM
- `chat`: Start an interactive chat session with the LLM using the hypergraph as context
- `chat_fb`: Start a chat session that also updates the hypergraph based on the conversation
- `visualize [filename]`: Visualize the hypergraph
- `histogram [filename]`: Plot a histogram of semantic similarities
- `write_edges [filename]`: Write sorted edges to a file
- `stats`: Display comprehensive statistics about the hypergraph and system
- `customize`: Customize reasoning parameters and LLM provider settings
- `exit` or `quit`: Exit the program

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
- Add new nodes to the hypergraph
- Add new hyperedges connecting nodes
- Remove existing nodes
- Remove existing hyperedges
- Adjust semantic similarities of hyperedges

This feedback mechanism allows the hypergraph to evolve based on the LLM's reasoning, creating a dynamic knowledge representation that adapts over time. The interactive mode allows for continuous refinement through multiple queries, with each response updating the hypergraph structure.

To exit the feedback reasoning mode, type `#__#`.

## Chat Modes

### Standard Chat

The `chat` command starts an interactive chat session with the LLM, using the hypergraph as context. This allows you to have a conversation with the LLM that's informed by the knowledge represented in the hypergraph.

Features of the standard chat mode:
- Persistent conversation history stored in a SQLite database
- Hypergraph context provided to the LLM
- Natural conversational interface

To exit the chat mode, type `#__#`.

### Chat with Feedback

The `chat_fb` command combines chat functionality with feedback reasoning. It maintains a conversation history while also updating the hypergraph based on the conversation.

Features of the chat with feedback mode:
- All features of standard chat mode
- Automatic updates to the hypergraph based on the conversation
- Visible summary of changes made to the hypergraph after each response
- Continuous evolution of the hypergraph as the conversation progresses

This creates a dynamic interaction where your conversation with the LLM directly shapes the hypergraph structure, which in turn influences the LLM's responses in subsequent turns.

To exit the chat with feedback mode, type `#__#`.

## Statistics

The `stats` command provides comprehensive statistics about the current hypergraph and system:

- **Basic Statistics**: Number of nodes, hyperedges, average node degree, etc.
- **Semantic Statistics**: Distribution of semantic similarities, average/highest/lowest similarity
- **Structural Statistics**: Connected components, clustering coefficient, diameter
- **LLM Configuration**: Current provider, model, temperature, max tokens
- **System Information**: Memory usage, available providers

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

### Chat with Feedback

```
Enter command: chat_fb
You: Tell me about the relationship between mammals and reptiles
Assistant: [LLM response about mammals and reptiles]
Changes to hypergraph:
  - Added node: mammals
  - Added node: reptiles
  - Added hyperedge connecting: mammals, reptiles
```

### Saving and Loading

```
Enter command: save myhypergraph
Enter command: load myhypergraph
```

## License

This project is open source and available under the MIT License.
