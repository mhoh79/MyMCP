"""
Skeleton MCP Server - Template for Creating New MCP Servers.

This is a minimal, working example that demonstrates how to create
an MCP server using the BaseMCPServer class. Copy this template
to create your own custom MCP servers.

Key Concepts:
1. Inherit from BaseMCPServer
2. Implement three abstract methods: register_tools(), get_server_name(), get_server_version()
3. Define tools with proper MCP Tool schema
4. Create async handler functions for each tool
5. Register tools with the tool registry

This template includes two example tools (echo and reverse) to show
common patterns for tool implementation.
"""

import sys
from pathlib import Path
from mcp.types import Tool, TextContent, CallToolResult

# Add parent directory to path for imports
_parent_dir = Path(__file__).parent.parent.parent
if str(_parent_dir) not in sys.path:
    sys.path.insert(0, str(_parent_dir))

from core import BaseMCPServer


class SkeletonServer(BaseMCPServer):
    """
    Skeleton MCP Server - A minimal template for creating new servers.
    
    This server provides two example tools:
    - echo: Returns the input text unchanged
    - reverse: Returns the input text reversed
    
    To create your own server:
    1. Copy this file to a new location (e.g., src/custom/my_server/)
    2. Rename the class (e.g., MyServer)
    3. Update get_server_name() and get_server_version()
    4. Define your own tools in register_tools()
    5. Create handler functions for your tools
    """
    
    def register_tools(self) -> None:
        """
        Register all tools provided by this server.
        
        This method is called during server initialization to register
        all available tools. Each tool needs:
        1. A Tool definition with name, description, and input schema
        2. An async handler function that processes tool calls
        """
        
        # ============================================================
        # TOOL 1: Echo Tool
        # ============================================================
        # This tool demonstrates basic input/output handling
        
        echo_tool = Tool(
            name="echo",
            description="Returns the input text unchanged. Useful for testing the server connection.",
            inputSchema={
                "type": "object",
                "properties": {
                    "text": {
                        "type": "string",
                        "description": "The text to echo back"
                    }
                },
                "required": ["text"]
            }
        )
        
        async def handle_echo(arguments: dict) -> CallToolResult:
            """
            Handler for the echo tool.
            
            Args:
                arguments: Dictionary containing tool arguments
                          Expected keys: text (str)
            
            Returns:
                CallToolResult with the echoed text
            """
            try:
                # Extract the text argument
                text = arguments.get("text", "")
                
                # Validate input
                if not text:
                    return CallToolResult(
                        content=[TextContent(
                            type="text",
                            text="Error: 'text' parameter is required and cannot be empty"
                        )],
                        isError=True
                    )
                
                # Return the echoed text
                return CallToolResult(
                    content=[TextContent(
                        type="text",
                        text=f"Echo: {text}"
                    )],
                    isError=False
                )
                
            except Exception as e:
                # Handle any unexpected errors
                self.logger.error(f"Error in echo handler: {e}", exc_info=True)
                return CallToolResult(
                    content=[TextContent(
                        type="text",
                        text=f"Error processing echo request: {str(e)}"
                    )],
                    isError=True
                )
        
        # Register the echo tool with its handler
        self.tool_registry.register_tool(echo_tool, handle_echo)
        
        # ============================================================
        # TOOL 2: Reverse Tool
        # ============================================================
        # This tool demonstrates string manipulation
        
        reverse_tool = Tool(
            name="reverse",
            description="Reverses the input text. Useful for demonstrating text transformation.",
            inputSchema={
                "type": "object",
                "properties": {
                    "text": {
                        "type": "string",
                        "description": "The text to reverse"
                    }
                },
                "required": ["text"]
            }
        )
        
        async def handle_reverse(arguments: dict) -> CallToolResult:
            """
            Handler for the reverse tool.
            
            Args:
                arguments: Dictionary containing tool arguments
                          Expected keys: text (str)
            
            Returns:
                CallToolResult with the reversed text
            """
            try:
                # Extract the text argument
                text = arguments.get("text", "")
                
                # Validate input
                if not text:
                    return CallToolResult(
                        content=[TextContent(
                            type="text",
                            text="Error: 'text' parameter is required and cannot be empty"
                        )],
                        isError=True
                    )
                
                # Reverse the text
                reversed_text = text[::-1]
                
                # Return the result
                return CallToolResult(
                    content=[TextContent(
                        type="text",
                        text=f"Reversed: {reversed_text}"
                    )],
                    isError=False
                )
                
            except Exception as e:
                # Handle any unexpected errors
                self.logger.error(f"Error in reverse handler: {e}", exc_info=True)
                return CallToolResult(
                    content=[TextContent(
                        type="text",
                        text=f"Error processing reverse request: {str(e)}"
                    )],
                    isError=True
                )
        
        # Register the reverse tool with its handler
        self.tool_registry.register_tool(reverse_tool, handle_reverse)
    
    def get_server_name(self) -> str:
        """
        Return the server name.
        
        This name is used in:
        - MCP protocol initialization
        - Logging
        - Health/metrics endpoints
        
        Returns:
            Server name (use lowercase with hyphens)
        """
        return "skeleton-server"
    
    def get_server_version(self) -> str:
        """
        Return the server version.
        
        Use semantic versioning (MAJOR.MINOR.PATCH):
        - MAJOR: Breaking changes
        - MINOR: New features (backward compatible)
        - PATCH: Bug fixes
        
        Returns:
            Server version string
        """
        return "1.0.0"


def main():
    """
    Entry point for the Skeleton MCP Server.
    
    This function:
    1. Creates an argument parser with standard options
    2. Parses command line arguments
    3. Creates the server instance
    4. Runs the server with the specified transport mode
    
    Usage:
        # Run in stdio mode (default, for Claude Desktop)
        python server.py
        
        # Run in HTTP mode for remote access
        python server.py --transport http --host 0.0.0.0 --port 8000
        
        # Run in HTTP mode with debug logging
        python server.py --transport http --host 0.0.0.0 --port 8000 --dev
    """
    # Create argument parser with server description
    parser = BaseMCPServer.create_argument_parser(
        description="Skeleton MCP Server - Template for creating new MCP servers"
    )
    args = parser.parse_args()
    
    # Create and run the server
    server = SkeletonServer(config_path=args.config)
    server.run(
        transport=args.transport,
        host=args.host,
        port=args.port,
        dev_mode=args.dev
    )


if __name__ == "__main__":
    main()
