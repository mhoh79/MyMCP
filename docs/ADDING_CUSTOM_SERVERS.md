# Adding Custom MCP Servers

This guide walks you through creating your own custom MCP (Model Context Protocol) servers using the MyMCP framework.

## Table of Contents

- [Overview](#overview)
- [Prerequisites](#prerequisites)
- [Quick Start](#quick-start)
- [Step-by-Step Guide](#step-by-step-guide)
- [Tool Development](#tool-development)
- [Testing Your Server](#testing-your-server)
- [Integration with Claude Desktop](#integration-with-claude-desktop)
- [Advanced Topics](#advanced-topics)
- [Best Practices](#best-practices)
- [Troubleshooting](#troubleshooting)

## Overview

The MyMCP framework provides a flexible base for building MCP servers with:
- **Dual transport support**: stdio (for Claude Desktop) and HTTP (for web/remote clients)
- **Built-in infrastructure**: Health checks, metrics, logging, error handling
- **Type safety**: Full type hints using the MCP SDK
- **Easy tool registration**: Simple API for adding custom tools

## Prerequisites

Before you begin:
- Python 3.10 or higher installed
- Basic understanding of Python async functions
- Familiarity with JSON Schema (for tool input definitions)
- MyMCP repository cloned and dependencies installed

```bash
# Install dependencies
pip install -r requirements.txt
```

## Quick Start

The fastest way to create a custom server:

```bash
# 1. Copy the skeleton template
cp -r src/templates/skeleton_server src/custom/my_awesome_server

# 2. Navigate to your server
cd src/custom/my_awesome_server

# 3. Edit server.py to customize it
# 4. Test it
python -m src.custom.my_awesome_server.server --transport stdio
```

That's it! You now have a working MCP server.

## Step-by-Step Guide

### Step 1: Create Your Server Directory

Start with the skeleton template:

```bash
cp -r src/templates/skeleton_server src/custom/my_server
```

Your new server structure:
```
src/custom/my_server/
├── __init__.py          # Package initialization
├── server.py            # Main server implementation
└── README.md            # Documentation
```

### Step 2: Customize the Server Class

Edit `src/custom/my_server/server.py`:

```python
from src.core import BaseMCPServer
from mcp.types import Tool, TextContent, CallToolResult

class MyServer(BaseMCPServer):
    """My custom MCP server."""
    
    def register_tools(self) -> None:
        """Register all tools for this server."""
        # Your tools will be registered here
        pass
    
    def get_server_name(self) -> str:
        """Return the server name (use lowercase-with-hyphens)."""
        return "my-server"
    
    def get_server_version(self) -> str:
        """Return the server version (use semantic versioning)."""
        return "1.0.0"
```

**Important naming conventions:**
- Server name: lowercase with hyphens (e.g., `my-awesome-server`)
- Version: semantic versioning (e.g., `1.0.0`, `2.1.3`)

### Step 3: Define Your Tools

Tools are the functions that your MCP server exposes. Each tool needs:
1. A Tool definition (metadata + JSON Schema)
2. A handler function (the actual implementation)

Example tool definition:

```python
def register_tools(self) -> None:
    # Define the tool
    calculate_tool = Tool(
        name="calculate",
        description="Performs basic arithmetic calculations",
        inputSchema={
            "type": "object",
            "properties": {
                "operation": {
                    "type": "string",
                    "description": "The operation to perform",
                    "enum": ["add", "subtract", "multiply", "divide"]
                },
                "a": {
                    "type": "number",
                    "description": "First number"
                },
                "b": {
                    "type": "number",
                    "description": "Second number"
                }
            },
            "required": ["operation", "a", "b"]
        }
    )
    
    # Register it
    self.tool_registry.register_tool(calculate_tool, self.handle_calculate)

async def handle_calculate(self, arguments: dict) -> CallToolResult:
    """Handle the calculate tool."""
    try:
        operation = arguments.get("operation")
        a = arguments.get("a", 0)
        b = arguments.get("b", 0)
        
        # Perform calculation
        if operation == "add":
            result = a + b
        elif operation == "subtract":
            result = a - b
        elif operation == "multiply":
            result = a * b
        elif operation == "divide":
            if b == 0:
                return CallToolResult(
                    content=[TextContent(
                        type="text",
                        text="Error: Division by zero"
                    )],
                    isError=True
                )
            result = a / b
        else:
            return CallToolResult(
                content=[TextContent(
                    type="text",
                    text=f"Error: Unknown operation '{operation}'"
                )],
                isError=True
            )
        
        return CallToolResult(
            content=[TextContent(
                type="text",
                text=f"Result: {result}"
            )],
            isError=False
        )
    except Exception as e:
        self.logger.error(f"Error in calculate: {e}", exc_info=True)
        return CallToolResult(
            content=[TextContent(
                type="text",
                text=f"Error: {str(e)}"
            )],
            isError=True
        )
```

### Step 4: Update the Main Entry Point

At the bottom of `server.py`, update the main function:

```python
def main():
    """Main entry point for the server."""
    parser = BaseMCPServer.create_argument_parser(
        description="My Custom MCP Server - Does amazing things"
    )
    args = parser.parse_args()
    
    server = MyServer(config_path=args.config)
    server.run(
        transport=args.transport,
        host=args.host,
        port=args.port,
        dev_mode=args.dev
    )

if __name__ == "__main__":
    main()
```

### Step 5: Update Package Initialization

Edit `src/custom/my_server/__init__.py`:

```python
"""My custom MCP server."""

from .server import MyServer

__all__ = ["MyServer"]
```

## Tool Development

### Input Schema Design

JSON Schema defines what parameters your tool accepts:

```python
inputSchema={
    "type": "object",
    "properties": {
        # String parameter
        "name": {
            "type": "string",
            "description": "User's name"
        },
        # Number parameter
        "age": {
            "type": "number",
            "description": "User's age",
            "minimum": 0,
            "maximum": 150
        },
        # Boolean parameter
        "active": {
            "type": "boolean",
            "description": "Whether user is active"
        },
        # Array parameter
        "tags": {
            "type": "array",
            "description": "List of tags",
            "items": {"type": "string"}
        },
        # Enum parameter
        "status": {
            "type": "string",
            "description": "User status",
            "enum": ["active", "inactive", "pending"]
        }
    },
    "required": ["name", "age"]  # Required parameters
}
```

### Handler Function Pattern

All handler functions follow this pattern:

```python
async def handle_my_tool(self, arguments: dict) -> CallToolResult:
    """
    Handle the my_tool tool.
    
    Args:
        arguments: Dictionary of tool arguments
        
    Returns:
        CallToolResult with content and error status
    """
    try:
        # 1. Extract parameters
        param = arguments.get("param", "default")
        
        # 2. Validate input
        if not param:
            return CallToolResult(
                content=[TextContent(
                    type="text",
                    text="Error: param is required"
                )],
                isError=True
            )
        
        # 3. Process the request
        result = do_something(param)
        
        # 4. Return success
        return CallToolResult(
            content=[TextContent(
                type="text",
                text=f"Success: {result}"
            )],
            isError=False
        )
        
    except Exception as e:
        # 5. Handle errors
        self.logger.error(f"Error in my_tool: {e}", exc_info=True)
        return CallToolResult(
            content=[TextContent(
                type="text",
                text=f"Error: {str(e)}"
            )],
            isError=True
        )
```

### Multiple Tools

Register multiple tools in a single server:

```python
def register_tools(self) -> None:
    # Tool 1
    tool1 = Tool(name="tool1", ...)
    self.tool_registry.register_tool(tool1, self.handle_tool1)
    
    # Tool 2
    tool2 = Tool(name="tool2", ...)
    self.tool_registry.register_tool(tool2, self.handle_tool2)
    
    # Tool 3
    tool3 = Tool(name="tool3", ...)
    self.tool_registry.register_tool(tool3, self.handle_tool3)
```

## Testing Your Server

### Test in stdio Mode

The default mode for Claude Desktop:

```bash
cd /home/runner/work/MyMCP/MyMCP
python -m src.custom.my_server.server
```

You can test it interactively by sending JSON-RPC messages via stdin.

### Test in HTTP Mode

For remote access and debugging:

```bash
# Start the server on port 8002
python -m src.custom.my_server.server --transport http --port 8002

# In another terminal, test it
curl -X POST http://localhost:8002/messages \
  -H "Content-Type: application/json" \
  -d '{
    "jsonrpc": "2.0",
    "method": "tools/list",
    "id": 1
  }'
```

### Development Mode

Enable debug logging:

```bash
python -m src.custom.my_server.server --transport http --port 8002 --dev
```

### Health Checks

When running in HTTP mode, your server automatically provides:

```bash
# Health check
curl http://localhost:8002/health

# Readiness check
curl http://localhost:8002/ready

# Metrics
curl http://localhost:8002/metrics
```

## Integration with Claude Desktop

### Configure Claude Desktop

1. Find your Claude Desktop config file:
   - **macOS**: `~/Library/Application Support/Claude/config.json`
   - **Windows**: `%APPDATA%\Claude\config.json`
   - **Linux**: `~/.config/Claude/config.json`

2. Add your server to the `mcpServers` section:

```json
{
  "mcpServers": {
    "my-server": {
      "command": "python",
      "args": [
        "-m",
        "src.custom.my_server.server"
      ],
      "cwd": "/absolute/path/to/MyMCP"
    }
  }
}
```

**Important:** Use the absolute path to your MyMCP directory.

3. Restart Claude Desktop

4. Your custom tools should now be available!

### Verify Integration

In Claude Desktop, ask:
```
What tools do you have available from my-server?
```

Claude should list your custom tools.

## Advanced Topics

### Shared State

Use instance variables for shared state between tools:

```python
class MyServer(BaseMCPServer):
    def __init__(self, config_path=None):
        super().__init__(config_path)
        self.cache = {}  # Shared cache
        self.counter = 0  # Shared counter
    
    async def handle_tool1(self, arguments: dict) -> CallToolResult:
        # Access shared state
        self.counter += 1
        self.cache[key] = value
        ...
```

### Configuration

Access configuration in your handlers:

```python
async def handle_tool(self, arguments: dict) -> CallToolResult:
    # Get config value with default
    max_items = self.config.get("max_items", 100)
    api_key = self.config.get("api_key", "")
    
    # Use configuration
    ...
```

Create a custom config file:

```yaml
# config/my_server.yaml
max_items: 200
api_key: "your-key-here"
enable_caching: true
```

Use it:

```bash
python -m src.custom.my_server.server --config config/my_server.yaml
```

### External Dependencies

Add external libraries to your server:

```python
import httpx  # For HTTP requests
import numpy as np  # For numerical computation
from mylib import custom_function  # Your custom library

async def handle_fetch(self, arguments: dict) -> CallToolResult:
    url = arguments.get("url")
    async with httpx.AsyncClient() as client:
        response = await client.get(url)
        return CallToolResult(
            content=[TextContent(text=response.text)],
            isError=False
        )
```

Don't forget to add dependencies to your requirements:

```bash
# In your server's README or requirements file
httpx>=0.24.0
numpy>=1.24.0
```

### Logging

Use the built-in logger:

```python
async def handle_tool(self, arguments: dict) -> CallToolResult:
    # Debug level
    self.logger.debug(f"Processing with args: {arguments}")
    
    # Info level
    self.logger.info("Tool executed successfully")
    
    # Warning level
    self.logger.warning("Deprecated parameter used")
    
    # Error level
    self.logger.error("Failed to process", exc_info=True)
```

## Best Practices

### 1. Input Validation

Always validate user inputs:

```python
async def handle_tool(self, arguments: dict) -> CallToolResult:
    value = arguments.get("value")
    
    # Check for presence
    if value is None:
        return error_result("value is required")
    
    # Check type
    if not isinstance(value, int):
        return error_result("value must be an integer")
    
    # Check range
    if value < 0 or value > 100:
        return error_result("value must be between 0 and 100")
    
    # Process...
```

### 2. Error Handling

Provide clear, actionable error messages:

```python
# Bad
return error_result("Error")

# Good
return error_result("Invalid email format. Expected: user@example.com")
```

### 3. Documentation

Document your tools thoroughly:

```python
Tool(
    name="calculate_tax",
    description="Calculates tax amount based on income and region. "
                "Supports US, UK, and EU tax systems. "
                "Returns both tax amount and effective rate.",
    inputSchema={
        "type": "object",
        "properties": {
            "income": {
                "type": "number",
                "description": "Annual income in local currency (USD, GBP, EUR)"
            },
            "region": {
                "type": "string",
                "description": "Tax region: 'US', 'UK', or 'EU'",
                "enum": ["US", "UK", "EU"]
            }
        },
        "required": ["income", "region"]
    }
)
```

### 4. Testing

Test your tools thoroughly before deployment:

```python
# Create a test file: tests/custom/test_my_server.py
import pytest
from src.custom.my_server import MyServer

@pytest.mark.asyncio
async def test_my_tool():
    server = MyServer()
    result = await server.handle_my_tool({"param": "value"})
    assert not result.isError
    assert "expected" in result.content[0].text
```

Run tests:

```bash
pytest tests/custom/test_my_server.py
```

### 5. Performance

For expensive operations, consider caching:

```python
class MyServer(BaseMCPServer):
    def __init__(self, config_path=None):
        super().__init__(config_path)
        self.cache = {}
    
    async def handle_expensive_tool(self, arguments: dict) -> CallToolResult:
        key = arguments.get("key")
        
        # Check cache first
        if key in self.cache:
            self.logger.debug(f"Cache hit for {key}")
            return self.cache[key]
        
        # Compute result
        result = expensive_computation(key)
        
        # Cache it
        self.cache[key] = result
        return result
```

## Troubleshooting

### Server Won't Start

**Problem:** `ModuleNotFoundError` or `ImportError`

**Solution:**
```bash
# Ensure you're in the project root
cd /path/to/MyMCP

# Install dependencies
pip install -r requirements.txt

# Run with full module path
python -m src.custom.my_server.server
```

### Tools Not Appearing

**Problem:** Tools don't show up in Claude Desktop

**Solution:**
1. Check that `register_tools()` is called
2. Verify tool names are unique
3. Restart Claude Desktop
4. Check Claude Desktop logs for errors

### Tool Execution Fails

**Problem:** Tool returns an error when called

**Solution:**
1. Add debug logging:
   ```python
   self.logger.debug(f"Arguments received: {arguments}")
   ```
2. Run in dev mode:
   ```bash
   python -m src.custom.my_server.server --dev
   ```
3. Check for missing parameter validation

### HTTP Mode Issues

**Problem:** Server doesn't respond to HTTP requests

**Solution:**
```bash
# Check if port is available
lsof -i :8002

# Try a different port
python -m src.custom.my_server.server --transport http --port 8003

# Check firewall settings
```

### Claude Desktop Integration

**Problem:** Server not showing in Claude Desktop

**Solution:**
1. Verify config.json syntax (valid JSON)
2. Use absolute paths in `cwd`
3. Check that Python is in PATH
4. Review Claude Desktop logs
5. Restart Claude Desktop

## Next Steps

- Review example servers: `src/builtin/math_server/` and `src/builtin/stats_server/`
- Read the [Architecture documentation](./ARCHITECTURE.md)
- Join the MCP community
- Share your custom servers with others!

## Additional Resources

- [MCP Protocol Specification](https://modelcontextprotocol.io/)
- [JSON Schema Documentation](https://json-schema.org/)
- [Python asyncio Guide](https://docs.python.org/3/library/asyncio.html)
- [BaseMCPServer API Reference](../src/core/README.md)

---

**Happy building! 🚀**

If you create something useful, consider contributing it back to the project or sharing it with the community!
