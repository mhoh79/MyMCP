#!/usr/bin/env python3
"""
Wrapper script to start the Math Calculator MCP Server for Claude Desktop.

This script ensures the project root is in Python's module search path,
allowing the server to be run with proper module imports.

Usage:
    python start_math_server.py [--config config.yaml]

For Claude Desktop configuration, use:
    "command": "/path/to/python",
    "args": ["/path/to/MyMCP/start_math_server.py"]
"""

import sys
from pathlib import Path

# Add project root to Python path
# This wrapper script is located in the project root, so Path(__file__).parent
# gives us the correct project root directory that needs to be in sys.path
project_root = Path(__file__).parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

# Import and run the server
from src.builtin.math_server.server import main

if __name__ == "__main__":
    main()
