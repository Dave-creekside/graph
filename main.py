#!/usr/bin/env python3
"""
Main entry point for the Hypergraph Evolution application.
"""

import sys
from cli.commands import main as commands_main
from cli.interactive import main as interactive_main
from llm.reasoning import check_provider_availability

def main():
    """Main function to run the application."""
    # Check which providers are available
    check_provider_availability()
    
    # If no arguments are provided, run in interactive mode
    if len(sys.argv) == 1:
        interactive_main()
    else:
        commands_main()

if __name__ == "__main__":
    main()
