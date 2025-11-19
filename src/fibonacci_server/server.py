"""
Main MCP server implementation for Fibonacci calculations.
This server exposes a tool that calculates Fibonacci numbers.
"""

import asyncio
import logging
from typing import Any

# MCP SDK imports for building the server
from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.types import (
    Tool,
    TextContent,
    CallToolResult,
    INVALID_PARAMS,
    INTERNAL_ERROR,
)

# Configure logging to stderr (stdout is reserved for MCP protocol messages)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler()],  # Logs go to stderr by default
)
logger = logging.getLogger("fibonacci-server")


def calculate_fibonacci(n: int) -> int:
    """
    Calculate the nth Fibonacci number using iterative approach.
    
    Args:
        n: The position in the Fibonacci sequence (0-indexed)
        
    Returns:
        The Fibonacci number at position n
        
    Raises:
        ValueError: If n is negative
        
    Examples:
        fib(0) = 0
        fib(1) = 1
        fib(5) = 5
        fib(10) = 55
    """
    if n < 0:
        raise ValueError("Fibonacci is not defined for negative numbers")
    
    if n <= 1:
        return n
    
    # Iterative approach is more efficient than recursive for large n
    # Avoids stack overflow and has O(n) time complexity
    prev, curr = 0, 1
    for _ in range(2, n + 1):
        prev, curr = curr, prev + curr
    
    return curr


def calculate_fibonacci_sequence(n: int) -> list[int]:
    """
    Calculate the Fibonacci sequence up to the nth number.
    
    Args:
        n: The number of Fibonacci numbers to generate
        
    Returns:
        List containing the first n Fibonacci numbers
        
    Examples:
        fib_sequence(5) = [0, 1, 1, 2, 3]
        fib_sequence(10) = [0, 1, 1, 2, 3, 5, 8, 13, 21, 34]
    """
    if n <= 0:
        return []
    
    sequence = [0]
    if n == 1:
        return sequence
    
    sequence.append(1)
    for i in range(2, n):
        sequence.append(sequence[i-1] + sequence[i-2])
    
    return sequence


# Initialize the MCP server with a descriptive name
app = Server("fibonacci-calculator")


@app.list_tools()
async def list_tools() -> list[Tool]:
    """
    Register and list all available tools that this MCP server provides.
    
    This function is called by MCP clients to discover what tools are available.
    Each tool has a name, description, and input schema defined using JSON Schema.
    
    Returns:
        List of Tool objects describing available tools
    """
    logger.info("Client requested tool list")
    
    return [
        Tool(
            name="calculate_fibonacci",
            description=(
                "Calculate the nth Fibonacci number or generate a Fibonacci sequence. "
                "Supports both single value calculation and sequence generation. "
                "The Fibonacci sequence starts: 0, 1, 1, 2, 3, 5, 8, 13, 21, 34..."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "n": {
                        "type": "integer",
                        "description": "The position in the Fibonacci sequence (0-indexed) or count of numbers to generate",
                        "minimum": 0,
                        "maximum": 1000,  # Reasonable limit to prevent performance issues
                    },
                    "return_sequence": {
                        "type": "boolean",
                        "description": "If true, returns the entire sequence up to n. If false, returns only the nth number.",
                        "default": False,
                    },
                },
                "required": ["n"],
            },
        ),
    ]


@app.call_tool()
async def call_tool(name: str, arguments: Any) -> CallToolResult:
    """
    Handle tool execution requests from MCP clients.
    
    This function is called when an MCP client (like Claude) wants to use one of
    the tools that this server provides. It validates inputs, executes the tool,
    and returns results in the proper MCP format.
    
    Args:
        name: The name of the tool to execute
        arguments: Dictionary containing the tool's input parameters
        
    Returns:
        CallToolResult containing the tool's output or error information
    """
    logger.info(f"Tool call requested: {name} with arguments: {arguments}")
    
    # Validate that the requested tool exists
    if name != "calculate_fibonacci":
        logger.error(f"Unknown tool requested: {name}")
        return CallToolResult(
            content=[TextContent(type="text", text=f"Unknown tool: {name}")],
            isError=True,
        )
    
    try:
        # Extract and validate parameters
        n = arguments.get("n")
        return_sequence = arguments.get("return_sequence", False)
        
        # Validate required parameter
        if n is None:
            logger.error("Missing required parameter: n")
            return CallToolResult(
                content=[TextContent(
                    type="text",
                    text="Missing required parameter 'n'"
                )],
                isError=True,
            )
        
        # Validate parameter type and range
        if not isinstance(n, int):
            logger.error(f"Invalid parameter type for n: {type(n)}")
            return CallToolResult(
                content=[TextContent(
                    type="text",
                    text=f"Parameter 'n' must be an integer, got {type(n).__name__}"
                )],
                isError=True,
            )
        
        if n < 0 or n > 1000:
            logger.error(f"Parameter n out of range: {n}")
            return CallToolResult(
                content=[TextContent(
                    type="text",
                    text="Parameter 'n' must be between 0 and 1000"
                )],
                isError=True,
            )
        
        # Execute the appropriate calculation
        if return_sequence:
            logger.info(f"Calculating Fibonacci sequence up to position {n}")
            result = calculate_fibonacci_sequence(n)
            result_text = (
                f"Fibonacci sequence (first {n} numbers):\n"
                f"{result}\n\n"
                f"Last number: {result[-1] if result else 'N/A'}"
            )
        else:
            logger.info(f"Calculating Fibonacci number at position {n}")
            result = calculate_fibonacci(n)
            result_text = f"Fibonacci number at position {n}: {result}"
        
        logger.info(f"Calculation successful: {result}")
        
        # Return successful result in MCP format
        return CallToolResult(
            content=[TextContent(type="text", text=result_text)],
            isError=False,
        )
        
    except ValueError as e:
        # Handle calculation errors (e.g., negative numbers)
        logger.error(f"Calculation error: {e}")
        return CallToolResult(
            content=[TextContent(type="text", text=f"Calculation error: {str(e)}")],
            isError=True,
        )
    except Exception as e:
        # Handle unexpected errors
        logger.exception(f"Unexpected error during tool execution: {e}")
        return CallToolResult(
            content=[TextContent(type="text", text=f"Internal error: {str(e)}")],
            isError=True,
        )


async def main():
    """
    Main entry point for the MCP server.
    
    This function starts the server using stdio transport, which means:
    - The server reads MCP protocol messages from stdin
    - The server writes MCP protocol messages to stdout
    - All logging and debugging output goes to stderr
    
    This is the standard transport for MCP servers that will be launched
    by client applications like Claude Desktop.
    """
    logger.info("Starting Fibonacci MCP server")
    
    # Run the server using stdio transport
    # This will block until the server receives a shutdown signal
    async with stdio_server() as (read_stream, write_stream):
        logger.info("Server started, waiting for requests...")
        await app.run(
            read_stream,
            write_stream,
            app.create_initialization_options()
        )
    
    logger.info("Server shut down successfully")


# Entry point when running as a script
if __name__ == "__main__":
    # Run the async main function
    asyncio.run(main())
