#!/usr/bin/env python3
"""
Wrapper script to start the Engineering Math MCP Server.

This script provides a convenient way to start the Engineering Math Server
with both stdio (for Claude Desktop) and HTTP (for remote access) transports.

Usage:
    # Start in stdio mode (default, for Claude Desktop)
    python start_engineering_math_server.py
    
    # Start in HTTP mode on port 8002
    python start_engineering_math_server.py --transport http --port 8002
    
    # Start in HTTP mode with development logging
    python start_engineering_math_server.py --transport http --port 8002 --dev
"""

import sys
from pathlib import Path

# Add src directory to path
src_dir = Path(__file__).parent / "src"
if str(src_dir) not in sys.path:
    sys.path.insert(0, str(src_dir))

from builtin.engineering_math_server import main

if __name__ == "__main__":
    main()
