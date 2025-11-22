"""
Tool registration and management utilities for MCP servers.

This module provides the ToolRegistry class for managing MCP tools.
"""

from typing import Dict, Callable, Any, Optional, List
from mcp.types import Tool


class ToolRegistry:
    """
    Registry for managing MCP tools and their handlers.
    
    This class provides a centralized way to register and manage tools
    in an MCP server, including validation and dispatch.
    
    Attributes:
        tools (Dict[str, Tool]): Dictionary mapping tool names to Tool definitions
        handlers (Dict[str, Callable]): Dictionary mapping tool names to handler functions
        
    Examples:
        >>> registry = ToolRegistry()
        >>> registry.register_tool(my_tool, my_handler)
        >>> tools = registry.list_tools()
        >>> handler = registry.get_handler("my_tool")
    """
    
    def __init__(self):
        """Initialize an empty tool registry."""
        self.tools: Dict[str, Tool] = {}
        self.handlers: Dict[str, Callable] = {}
    
    def register_tool(self, tool: Tool, handler: Callable) -> None:
        """
        Register a tool with its handler function.
        
        Args:
            tool: MCP Tool definition
            handler: Async function to handle tool calls
            
        Raises:
            ValueError: If tool with same name already registered
            TypeError: If handler is not callable
        """
        if not callable(handler):
            raise TypeError(f"Handler for tool '{tool.name}' must be callable")
        
        if tool.name in self.tools:
            raise ValueError(f"Tool '{tool.name}' is already registered")
        
        self.tools[tool.name] = tool
        self.handlers[tool.name] = handler
    
    def get_handler(self, tool_name: str) -> Optional[Callable]:
        """
        Get the handler function for a tool.
        
        Args:
            tool_name: Name of the tool
            
        Returns:
            Handler function if found, None otherwise
        """
        return self.handlers.get(tool_name)
    
    def get_tool(self, tool_name: str) -> Optional[Tool]:
        """
        Get the Tool definition for a tool.
        
        Args:
            tool_name: Name of the tool
            
        Returns:
            Tool definition if found, None otherwise
        """
        return self.tools.get(tool_name)
    
    def list_tools(self) -> List[Tool]:
        """
        Get a list of all registered tools.
        
        Returns:
            List of Tool definitions
        """
        return list(self.tools.values())
    
    def tool_exists(self, tool_name: str) -> bool:
        """
        Check if a tool is registered.
        
        Args:
            tool_name: Name of the tool
            
        Returns:
            True if tool exists, False otherwise
        """
        return tool_name in self.tools
    
    def unregister_tool(self, tool_name: str) -> bool:
        """
        Unregister a tool and its handler.
        
        Args:
            tool_name: Name of the tool to unregister
            
        Returns:
            True if tool was unregistered, False if not found
        """
        if tool_name in self.tools:
            del self.tools[tool_name]
            del self.handlers[tool_name]
            return True
        return False
    
    def clear(self) -> None:
        """Clear all registered tools and handlers."""
        self.tools.clear()
        self.handlers.clear()
    
    def count(self) -> int:
        """
        Get the number of registered tools.
        
        Returns:
            Number of tools in the registry
        """
        return len(self.tools)
