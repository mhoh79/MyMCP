"""
Entry point for running Engineering Math Server as a module.

Usage:
    python -m src.builtin.engineering_math_server
    python -m src.builtin.engineering_math_server --transport http --port 8002
"""

from .server import main

if __name__ == "__main__":
    main()
