"""
Built-in MCP servers using the base framework.

This package contains production MCP servers that inherit from
the BaseMCPServer framework for minimal boilerplate code.
"""

from .math_server.server import MathServer
from .stats_server.server import StatsServer

__all__ = ["MathServer", "StatsServer"]
