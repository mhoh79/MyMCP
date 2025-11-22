"""
Base MCP server implementation with dual transport support.

This module provides the BaseMCPServer abstract class that handles
common server functionality for both stdio and HTTP transports.
"""

import argparse
import asyncio
import json
import logging
import signal
import sys
from abc import ABC, abstractmethod
from datetime import datetime
from pathlib import Path
from typing import Optional, Any

from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.types import Tool, CallToolResult

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
import uvicorn
from sse_starlette.sse import EventSourceResponse

# Import configuration and middleware
from pathlib import Path as _Path
import sys as _sys

# Add parent directory to path if not already there
_parent_dir = _Path(__file__).parent.parent
if str(_parent_dir) not in _sys.path:
    _sys.path.insert(0, str(_parent_dir))

from config import load_config, Config
from middleware import setup_middleware
from .server_state import ServerState
from .tool_registry import ToolRegistry


class BaseMCPServer(ABC):
    """
    Abstract base class for MCP servers with dual transport support.
    
    This class provides common functionality for MCP servers that support
    both stdio (for Claude Desktop) and HTTP/SSE (for remote access) transports.
    
    Subclasses must implement:
    - register_tools(): Register server-specific tools
    - get_server_name(): Return the server name
    - get_server_version(): Return the server version
    
    Attributes:
        config (Config): Server configuration
        logger (logging.Logger): Logger instance
        server_state (ServerState): Server state tracker
        tool_registry (ToolRegistry): Tool registry
        app (Server): MCP Server instance
        
    Examples:
        >>> class MyServer(BaseMCPServer):
        ...     def register_tools(self):
        ...         # Register tools
        ...     def get_server_name(self):
        ...         return "my-server"
        ...     def get_server_version(self):
        ...         return "1.0.0"
        >>> server = MyServer()
        >>> server.run(transport="http", host="0.0.0.0", port=8000)
    """
    
    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize the base MCP server.
        
        Args:
            config_path: Optional path to configuration file
        """
        # Load configuration
        self.config = load_config(config_path)
        
        # Setup logging
        self.logger = self._setup_logging()
        
        # Initialize server state
        self.server_state = ServerState()
        
        # Initialize tool registry
        self.tool_registry = ToolRegistry()
        
        # Create MCP server instance
        self.app = Server(self.get_server_name())
        
        # Register MCP protocol handlers
        self._setup_mcp_handlers()
        
        # Register server-specific tools
        self.register_tools()
        
        self.logger.info(f"{self.get_server_name()} initialized with {self.tool_registry.count()} tools")
    
    def _setup_logging(self) -> logging.Logger:
        """
        Setup logging based on configuration.
        
        Returns:
            Logger instance
        """
        logger = logging.getLogger(self.get_server_name())
        log_level = getattr(logging, self.config.logging.level)
        logger.setLevel(log_level)
        
        # Only add handler if none exist
        if not logger.handlers:
            handler = logging.StreamHandler()
            handler.setFormatter(
                logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
            )
            logger.addHandler(handler)
        
        return logger
    
    def _setup_mcp_handlers(self) -> None:
        """Setup MCP protocol handlers."""
        
        @self.app.list_tools()
        async def list_tools() -> list[Tool]:
            """List all available tools."""
            return self.tool_registry.list_tools()
        
        @self.app.call_tool()
        async def call_tool(name: str, arguments: Any) -> CallToolResult:
            """
            Call a specific tool.
            
            Args:
                name: Tool name
                arguments: Tool arguments
                
            Returns:
                CallToolResult with tool execution results
            """
            handler = self.tool_registry.get_handler(name)
            if handler is None:
                from mcp.types import TextContent
                return CallToolResult(
                    content=[TextContent(
                        type="text",
                        text=f"Unknown tool: {name}"
                    )],
                    isError=True
                )
            
            try:
                return await handler(arguments)
            except Exception as e:
                from mcp.types import TextContent
                self.logger.error(f"Error executing tool {name}: {e}", exc_info=True)
                return CallToolResult(
                    content=[TextContent(
                        type="text",
                        text=f"Error executing tool: {str(e)}"
                    )],
                    isError=True
                )
    
    @abstractmethod
    def register_tools(self) -> None:
        """
        Register server-specific tools.
        
        This method must be implemented by subclasses to register
        their specific tools using self.tool_registry.register_tool().
        """
        pass
    
    @abstractmethod
    def get_server_name(self) -> str:
        """
        Get the server name.
        
        Returns:
            Server name (e.g., "math-calculator", "stats-server")
        """
        pass
    
    @abstractmethod
    def get_server_version(self) -> str:
        """
        Get the server version.
        
        Returns:
            Server version (e.g., "1.0.0")
        """
        pass
    
    async def run_stdio_server(self) -> None:
        """
        Run the server using stdio transport.
        
        This function starts the server using stdio transport, which means:
        - The server reads MCP protocol messages from stdin
        - The server writes MCP protocol messages to stdout
        - All logging and debugging output goes to stderr
        
        This is the standard transport for MCP servers that will be launched
        by client applications like Claude Desktop.
        """
        self.logger.info(f"Starting {self.get_server_name()} MCP server (stdio mode)")
        self.logger.info(f"Log level: {self.config.logging.level}")
        
        try:
            async with stdio_server() as (read_stream, write_stream):
                self.logger.info("Server started, waiting for requests...")
                await self.app.run(
                    read_stream,
                    write_stream,
                    self.app.create_initialization_options()
                )
        except Exception as e:
            self.logger.error(f"Server error: {e}", exc_info=True)
            sys.exit(1)
        
        self.logger.info("Server shut down successfully")
    
    async def run_http_server(
        self,
        host: str = "0.0.0.0",
        port: int = 8000,
        dev_mode: bool = False
    ) -> None:
        """
        Run the server using HTTP/SSE transport.
        
        This function starts the server using FastAPI with:
        - SSE endpoint at /sse for server-to-client messages
        - POST endpoint at /messages for client-to-server JSON-RPC requests
        - Health/ready/metrics endpoints for monitoring
        
        This allows remote access from tools like GitHub Codespaces while
        maintaining the MCP protocol over HTTP.
        
        Args:
            host: Host to bind to (default: 0.0.0.0)
            port: Port to bind to (default: 8000)
            dev_mode: Enable development mode with debug logging (default: False)
        """
        self.logger.info(f"Starting {self.get_server_name()} MCP server (HTTP mode)")
        self.logger.info(f"Server will listen on {host}:{port}")
        
        # Create FastAPI app
        fastapi_app = FastAPI(
            title=f"{self.get_server_name()} MCP Server",
            version=self.get_server_version()
        )
        
        # Setup middleware (CORS, authentication, rate limiting, logging)
        setup_middleware(fastapi_app, self.config)
        
        # Store for SSE connections
        sse_connections = []
        
        @fastapi_app.get("/sse")
        async def sse_endpoint(request: Request):
            """
            Server-Sent Events endpoint for streaming server-to-client messages.
            This endpoint streams MCP protocol messages from the server to the client.
            """
            async def event_generator():
                try:
                    # Register this connection
                    connection_id = id(request)
                    sse_connections.append(connection_id)
                    self.server_state.increment_connections()
                    self.logger.info(f"New SSE connection established: {connection_id}")
                    
                    # Send initial connection message
                    yield {
                        "event": "connected",
                        "data": json.dumps({
                            "status": "connected",
                            "server": self.get_server_name()
                        })
                    }
                    
                    # Keep the connection alive and send events
                    while True:
                        # Check if client disconnected
                        if await request.is_disconnected():
                            self.logger.info(f"SSE connection disconnected: {connection_id}")
                            break
                        
                        # Send keepalive ping every 30 seconds
                        yield {
                            "event": "ping",
                            "data": json.dumps({"timestamp": datetime.now().isoformat()})
                        }
                        
                        await asyncio.sleep(30)
                        
                except asyncio.CancelledError:
                    self.logger.info(f"SSE connection cancelled: {connection_id}")
                except Exception as e:
                    self.logger.error(f"Error in SSE endpoint: {e}", exc_info=True)
                finally:
                    if connection_id in sse_connections:
                        sse_connections.remove(connection_id)
                    self.server_state.decrement_connections()
                    self.logger.info(f"SSE connection closed: {connection_id}")
            
            return EventSourceResponse(event_generator())
        
        @fastapi_app.post("/messages")
        async def messages_endpoint(request: Request):
            """
            JSON-RPC 2.0 endpoint for client-to-server requests.
            Handles MCP protocol messages sent from the client.
            """
            # Track request
            self.server_state.increment_requests()
            
            try:
                # Parse JSON-RPC request
                body = await request.json()
                self.logger.debug(f"Received JSON-RPC request: {body}")
                
                # Validate JSON-RPC 2.0 format
                if "jsonrpc" not in body or body["jsonrpc"] != "2.0":
                    return {
                        "jsonrpc": "2.0",
                        "error": {
                            "code": -32600,
                            "message": "Invalid Request: missing or invalid jsonrpc version"
                        },
                        "id": body.get("id")
                    }
                
                method = body.get("method")
                params = body.get("params", {})
                request_id = body.get("id")
                
                # Handle different MCP methods
                if method == "initialize":
                    result = {
                        "protocolVersion": "2024-11-05",
                        "capabilities": {
                            "tools": {}
                        },
                        "serverInfo": {
                            "name": self.get_server_name(),
                            "version": self.get_server_version()
                        }
                    }
                elif method == "tools/list":
                    # Return list of available tools
                    tools_list = self.tool_registry.list_tools()
                    result = {"tools": [tool.model_dump() for tool in tools_list]}
                elif method == "tools/call":
                    # Call a specific tool
                    tool_name = params.get("name")
                    tool_arguments = params.get("arguments", {})
                    
                    # Execute the tool
                    handler = self.tool_registry.get_handler(tool_name)
                    if handler is None:
                        return {
                            "jsonrpc": "2.0",
                            "error": {
                                "code": -32601,
                                "message": f"Tool not found: {tool_name}"
                            },
                            "id": request_id
                        }
                    
                    tool_result = await handler(tool_arguments)
                    result = {
                        "content": [content.model_dump() for content in tool_result.content],
                        "isError": tool_result.isError
                    }
                else:
                    return {
                        "jsonrpc": "2.0",
                        "error": {
                            "code": -32601,
                            "message": f"Method not found: {method}"
                        },
                        "id": request_id
                    }
                
                # Return successful response
                return {
                    "jsonrpc": "2.0",
                    "result": result,
                    "id": request_id
                }
                
            except json.JSONDecodeError:
                return {
                    "jsonrpc": "2.0",
                    "error": {
                        "code": -32700,
                        "message": "Parse error: invalid JSON"
                    },
                    "id": None
                }
            except Exception as e:
                self.logger.error(f"Error handling message: {e}", exc_info=True)
                return {
                    "jsonrpc": "2.0",
                    "error": {
                        "code": -32603,
                        "message": f"Internal error: {str(e)}"
                    },
                    "id": body.get("id") if isinstance(body, dict) else None
                }
        
        @fastapi_app.get("/health")
        async def health_check():
            """
            Health check endpoint - basic liveness check.
            Always returns 200 OK with server status and uptime.
            """
            return {
                "status": "ok",
                "server": self.get_server_name(),
                "uptime_seconds": round(self.server_state.get_uptime_seconds(), 2),
                "timestamp": datetime.now().isoformat() + "Z"
            }
        
        @fastapi_app.get("/ready")
        async def readiness_check():
            """
            Readiness check endpoint - indicates if server is ready to accept requests.
            Returns 200 if ready, 503 if not ready.
            """
            if self.server_state.mcp_initialized:
                return {
                    "status": "ready",
                    "mcp_initialized": True,
                    "tools_count": self.tool_registry.count()
                }
            else:
                return JSONResponse(
                    status_code=503,
                    content={
                        "status": "not_ready",
                        "mcp_initialized": False,
                        "tools_count": 0
                    }
                )
        
        @fastapi_app.get("/metrics")
        async def metrics_endpoint():
            """
            Metrics endpoint - provides operational statistics.
            Returns request counts, active connections, tools available, and uptime.
            """
            uptime_seconds = self.server_state.get_uptime_seconds()
            
            # Calculate requests per minute
            uptime_minutes = uptime_seconds / 60.0
            requests_per_minute = (
                self.server_state.total_requests / uptime_minutes if uptime_minutes > 0 else 0.0
            )
            
            return {
                "total_requests": self.server_state.total_requests,
                "active_connections": self.server_state.active_connections,
                "tools_available": self.tool_registry.count(),
                "uptime_seconds": round(uptime_seconds, 2),
                "requests_per_minute": round(requests_per_minute, 2)
            }
        
        # Graceful shutdown handler
        shutdown_event = asyncio.Event()
        
        def signal_handler(signum, frame):
            """Handle shutdown signals."""
            sig_name = signal.Signals(signum).name
            self.logger.info(f"Received signal {sig_name}, initiating graceful shutdown...")
            shutdown_event.set()
        
        # Register signal handlers
        signal.signal(signal.SIGTERM, signal_handler)
        signal.signal(signal.SIGINT, signal_handler)
        
        # Set up uvicorn configuration
        if dev_mode:
            self.logger.info("🔥 Development mode enabled - debug logging active")
            self.logger.warning("Dev mode is for development only - do not use in production!")
            log_level = "debug"
        else:
            log_level = "info"
        
        config_uvicorn = uvicorn.Config(
            fastapi_app,
            host=host,
            port=port,
            log_level=log_level
        )
        server = uvicorn.Server(config_uvicorn)
        
        # Create server task
        server_task = asyncio.create_task(server.serve())
        
        try:
            # Wait for either server completion or shutdown signal
            _, _ = await asyncio.wait(
                [server_task, asyncio.create_task(shutdown_event.wait())],
                return_when=asyncio.FIRST_COMPLETED
            )
            
            # If shutdown was signaled, perform graceful cleanup
            if shutdown_event.is_set():
                self.logger.info("Shutting down server...")
                self.logger.info(f"Closing {len(sse_connections)} active SSE connections...")
                
                # Signal server to shutdown
                server.should_exit = True
                
                # Wait for server to shutdown (max 30 seconds)
                try:
                    await asyncio.wait_for(server_task, timeout=30.0)
                except asyncio.TimeoutError:
                    self.logger.warning("Server shutdown timed out after 30 seconds")
                
                self.logger.info("All connections closed")
                self.logger.info("Shutdown complete")
            
        except Exception as e:
            self.logger.error(f"Server error: {e}", exc_info=True)
            sys.exit(1)
    
    def run(
        self,
        transport: str = "stdio",
        host: str = "0.0.0.0",
        port: int = 8000,
        dev_mode: bool = False
    ) -> None:
        """
        Run the MCP server with specified transport.
        
        Args:
            transport: Transport mode ("stdio" or "http")
            host: Host to bind to (HTTP mode only)
            port: Port to bind to (HTTP mode only)
            dev_mode: Enable development mode (HTTP mode only)
        """
        if transport == "http":
            asyncio.run(self.run_http_server(host, port, dev_mode))
        elif transport == "stdio":
            if dev_mode:
                self.logger.warning("--dev flag is only supported in HTTP mode, ignoring")
            asyncio.run(self.run_stdio_server())
        else:
            raise ValueError(f"Invalid transport mode: {transport}. Must be 'stdio' or 'http'")
    
    @classmethod
    def create_argument_parser(cls, description: Optional[str] = None) -> argparse.ArgumentParser:
        """
        Create a standard argument parser for the server.
        
        Args:
            description: Server description for help text
            
        Returns:
            Configured ArgumentParser instance
        """
        if description is None:
            description = "MCP Server"
        
        parser = argparse.ArgumentParser(
            description=description,
            formatter_class=argparse.RawDescriptionHelpFormatter,
            epilog="""
Examples:
  # Run with default configuration (stdio mode)
  python server.py
  
  # Run in HTTP mode
  python server.py --transport http --host 0.0.0.0 --port 8000
  
  # Run in HTTP mode with development logging
  python server.py --transport http --host 0.0.0.0 --port 8000 --dev
  
  # Run with custom configuration file
  python server.py --config /path/to/config.yaml
            """
        )
        
        parser.add_argument(
            "--config",
            type=str,
            default=None,
            help="Path to YAML configuration file (optional)"
        )
        parser.add_argument(
            "--transport",
            type=str,
            choices=["stdio", "http"],
            default="stdio",
            help="Transport mode: stdio (default, for Claude Desktop) or http (for remote access)"
        )
        parser.add_argument(
            "--host",
            type=str,
            default="0.0.0.0",
            help="Host to bind to in HTTP mode (default: 0.0.0.0)"
        )
        parser.add_argument(
            "--port",
            type=int,
            default=8000,
            help="Port to bind to in HTTP mode (default: 8000)"
        )
        parser.add_argument(
            "--dev",
            action="store_true",
            help="Enable development mode with debug logging (HTTP mode only)"
        )
        
        return parser
