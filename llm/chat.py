"""
Chat functions for interactive conversations with LLMs.
"""

import os
import sqlite3
import datetime
from llm.reasoning import query_llm
from config import REASONING_PARAMS

class ChatDatabase:
    """Database for storing chat history."""
    
    def __init__(self, db_path="chat_history.db"):
        """
        Initialize the chat database.
        
        Args:
            db_path: Path to the SQLite database file
        """
        self.db_path = db_path
        self._create_tables_if_not_exist()
    
    def _create_tables_if_not_exist(self):
        """Create the necessary tables if they don't exist."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Create sessions table
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS sessions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            provider TEXT,
            model TEXT
        )
        ''')
        
        # Create messages table
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS messages (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id INTEGER,
            role TEXT,
            content TEXT,
            timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (session_id) REFERENCES sessions (id)
        )
        ''')
        
        conn.commit()
        conn.close()
    
    def create_session(self, name=None):
        """
        Create a new chat session.
        
        Args:
            name: Optional name for the session
            
        Returns:
            int: ID of the created session
        """
        if name is None:
            name = f"Chat {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute(
            "INSERT INTO sessions (name, provider, model) VALUES (?, ?, ?)",
            (name, REASONING_PARAMS["llm_provider"], REASONING_PARAMS["llm_model"])
        )
        
        session_id = cursor.lastrowid
        conn.commit()
        conn.close()
        
        return session_id
    
    def add_message(self, session_id, role, content):
        """
        Add a message to a chat session.
        
        Args:
            session_id: ID of the session
            role: Role of the message sender ('user' or 'assistant')
            content: Content of the message
            
        Returns:
            int: ID of the created message
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute(
            "INSERT INTO messages (session_id, role, content) VALUES (?, ?, ?)",
            (session_id, role, content)
        )
        
        message_id = cursor.lastrowid
        conn.commit()
        conn.close()
        
        return message_id
    
    def get_session_messages(self, session_id):
        """
        Get all messages for a session.
        
        Args:
            session_id: ID of the session
            
        Returns:
            list: List of (role, content) tuples
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute(
            "SELECT role, content FROM messages WHERE session_id = ? ORDER BY timestamp",
            (session_id,)
        )
        
        messages = cursor.fetchall()
        conn.close()
        
        return messages
    
    def get_sessions(self):
        """
        Get all chat sessions.
        
        Returns:
            list: List of (id, name, created_at, provider, model) tuples
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute(
            "SELECT id, name, created_at, provider, model FROM sessions ORDER BY created_at DESC"
        )
        
        sessions = cursor.fetchall()
        conn.close()
        
        return sessions
    
    def delete_session(self, session_id):
        """
        Delete a chat session and all its messages.
        
        Args:
            session_id: ID of the session to delete
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Delete messages first (foreign key constraint)
        cursor.execute("DELETE FROM messages WHERE session_id = ?", (session_id,))
        
        # Delete the session
        cursor.execute("DELETE FROM sessions WHERE id = ?", (session_id,))
        
        conn.commit()
        conn.close()

def generate_chat_prompt(messages, hg=None):
    """
    Generate a prompt for the LLM to continue a chat conversation.
    
    Args:
        messages: List of (role, content) tuples
        hg: Optional hypergraph instance to provide context
        
    Returns:
        str: The generated prompt
    """
    prompt = "You are a helpful assistant engaged in a conversation with a user.\n\n"
    
    # Add hypergraph context if provided
    if hg:
        nodes = list(hg.get_nodes())
        edges = hg.get_hyperedges()
        
        prompt += f"You have access to a hypergraph with {len(nodes)} nodes and {len(edges)} hyperedges.\n\n"
        
        # Add information about the nodes
        prompt += "Nodes in the hypergraph:\n"
        for node in nodes[:REASONING_PARAMS["max_relevant_nodes"]]:
            prompt += f"- {node}\n"
        if len(nodes) > REASONING_PARAMS["max_relevant_nodes"]:
            prompt += f"- ... and {len(nodes) - REASONING_PARAMS['max_relevant_nodes']} more\n"
        prompt += "\n"
        
        # Add information about some hyperedges
        prompt += "Some hyperedges in the hypergraph:\n"
        for edge_id, edge in list(edges.items())[:REASONING_PARAMS["max_relevant_edges"]]:
            nodes_str = ", ".join(edge["nodes"])
            sim = edge.get("semantic_similarity", "N/A")
            prompt += f"- Edge {edge_id}: [{nodes_str}] (Semantic similarity: {sim})\n"
        if len(edges) > REASONING_PARAMS["max_relevant_edges"]:
            prompt += f"- ... and {len(edges) - REASONING_PARAMS['max_relevant_edges']} more\n"
        prompt += "\n"
        
        prompt += "Use this hypergraph as context for your responses when relevant.\n\n"
    
    # Add conversation history
    prompt += "Conversation history:\n"
    for role, content in messages:
        if role == "user":
            prompt += f"User: {content}\n"
        else:
            prompt += f"Assistant: {content}\n"
    
    # Add the final instruction
    prompt += "\nRespond to the user's most recent message in a helpful and informative way."
    
    return prompt

def chat_with_llm(messages, hg=None):
    """
    Continue a chat conversation with the LLM.
    
    Args:
        messages: List of (role, content) tuples
        hg: Optional hypergraph instance to provide context
        
    Returns:
        str: The LLM's response
    """
    # Generate the prompt
    prompt = generate_chat_prompt(messages, hg)
    
    # Query the LLM
    response = query_llm(prompt)
    
    return response
