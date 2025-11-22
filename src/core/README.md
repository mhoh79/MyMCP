# Core MCP Server Framework

This package provides reusable base classes and utilities for building MCP (Model Context Protocol) servers with dual transport support (stdio/HTTP).

## Components

### ServerState

Tracks server operational metrics for monitoring endpoints.

```python
from src.core import ServerState

state = ServerState()
state.increment_requests()
state.increment_connections()
uptime = state.get_uptime_seconds()
```

**Attributes:**
- `start_time`: Unix timestamp when server started
- `total_requests`: Total number of requests processed
- `active_connections`: Number of active SSE connections
- `mcp_initialized`: Whether MCP protocol is initialized

**Methods:**
- `get_uptime_seconds()`: Get server uptime in seconds
- `increment_requests()`: Increment total request counter
- `increment_connections()`: Increment active connection counter
- `decrement_connections()`: Decrement active connection counter

### ToolRegistry

Registry for managing MCP tools and their handlers.

```python
from src.core import ToolRegistry
from mcp.types import Tool

registry = ToolRegistry()

# Register a tool
tool = Tool(name="my_tool", description="My tool", inputSchema={...})
async def my_handler(arguments):
    # Handle tool call
    return CallToolResult(...)

registry.register_tool(tool, my_handler)

# Use the registry
tools = registry.list_tools()
handler = registry.get_handler("my_tool")
```

**Methods:**
- `register_tool(tool, handler)`: Register a tool with its handler
- `get_handler(tool_name)`: Get handler function for a tool
- `get_tool(tool_name)`: Get Tool definition
- `list_tools()`: Get list of all registered tools
- `tool_exists(tool_name)`: Check if tool is registered
- `unregister_tool(tool_name)`: Unregister a tool
- `clear()`: Clear all registered tools
- `count()`: Get number of registered tools

### BaseMCPServer

Abstract base class for MCP servers with dual transport support.

```python
from src.core import BaseMCPServer
from mcp.types import Tool, CallToolResult, TextContent

class MyServer(BaseMCPServer):
    def register_tools(self):
        """Register server-specific tools."""
        tool = Tool(
            name="my_tool",
            description="Does something useful",
            inputSchema={
                "type": "object",
                "properties": {
                    "value": {"type": "string"}
                },
                "required": ["value"]
            }
        )
        
        async def my_handler(arguments):
            value = arguments.get("value")
            return CallToolResult(
                content=[TextContent(
                    type="text",
                    text=f"Result: {value}"
                )],
                isError=False
            )
        
        self.tool_registry.register_tool(tool, my_handler)
    
    def get_server_name(self):
        return "my-server"
    
    def get_server_version(self):
        return "1.0.0"

# Run the server
if __name__ == "__main__":
    server = MyServer()
    
    # Stdio mode (for Claude Desktop)
    server.run(transport="stdio")
    
    # HTTP mode (for remote access)
    server.run(transport="http", host="0.0.0.0", port=8000)
```

**Required Methods (Abstract):**
- `register_tools()`: Register server-specific tools
- `get_server_name()`: Return server name
- `get_server_version()`: Return server version

**Available Methods:**
- `run(transport, host, port, dev_mode)`: Run server with specified transport
- `run_stdio_server()`: Run using stdio transport
- `run_http_server(host, port, dev_mode)`: Run using HTTP/SSE transport
- `create_argument_parser(description)`: Create CLI argument parser

**Attributes:**
- `config`: Server configuration
- `logger`: Logger instance
- `server_state`: ServerState instance
- `tool_registry`: ToolRegistry instance
- `app`: MCP Server instance

## HTTP Transport Features

When running in HTTP mode, the server provides:

### Endpoints

- **POST /messages**: JSON-RPC 2.0 endpoint for MCP protocol messages
- **GET /sse**: Server-Sent Events endpoint for streaming
- **GET /health**: Health check (liveness)
- **GET /ready**: Readiness check
- **GET /metrics**: Operational metrics

### Middleware

Automatically configured:
- CORS support
- Authentication (if enabled in config)
- Rate limiting (if enabled in config)
- Request logging

### Graceful Shutdown

Handles SIGTERM and SIGINT signals for graceful shutdown, closing all active connections.

## Command Line Interface

```bash
# Stdio mode (default)
python server.py

# HTTP mode
python server.py --transport http --host 0.0.0.0 --port 8000

# Development mode (debug logging)
python server.py --transport http --dev

# Custom configuration
python server.py --config /path/to/config.yaml
```

## Testing

The framework includes comprehensive tests:

```bash
# Run core tests
pytest tests/core/

# Run with coverage
pytest tests/core/ --cov=src/core --cov-report=term-missing
```

## Design Principles

1. **Minimal changes**: The framework extracts common patterns without changing existing server behavior
2. **Dual transport**: Supports both stdio (Claude Desktop) and HTTP (remote access) out of the box
3. **Extensible**: Easy to add new tools by implementing `register_tools()`
4. **Type-safe**: Uses MCP SDK types throughout
5. **Well-tested**: Comprehensive test coverage for core functionality
6. **Production-ready**: Includes monitoring, logging, and graceful shutdown

## Example Servers

See `src/math_server/` and `src/stats_server/` for complete examples of servers using this framework (once migrated).

## Migration Guide

To migrate an existing server to use this framework:

1. Import BaseMCPServer:
   ```python
   from src.core import BaseMCPServer
   ```

2. Change server class to inherit from BaseMCPServer:
   ```python
   class MyServer(BaseMCPServer):
   ```

3. Implement required abstract methods:
   - `register_tools()`: Move tool registration here
   - `get_server_name()`: Return server name
   - `get_server_version()`: Return server version

4. Remove duplicate code:
   - ServerState class (use `self.server_state`)
   - Tool registration pattern (use `self.tool_registry`)
   - HTTP/stdio server setup (use `self.run()`)
   - Health/ready/metrics endpoints (automatically provided)

5. Update entry point:
   ```python
   if __name__ == "__main__":
       parser = BaseMCPServer.create_argument_parser("My Server")
       args = parser.parse_args()
       
       server = MyServer(config_path=args.config)
       server.run(
           transport=args.transport,
           host=args.host,
           port=args.port,
           dev_mode=args.dev
       )
   ```

## Future Enhancements

- [ ] Support for additional transport protocols
- [ ] Built-in metrics collection (Prometheus)
- [ ] Advanced logging configuration
- [ ] Plugin system for middleware
- [ ] Server clustering support
