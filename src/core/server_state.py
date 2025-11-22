"""
Server state management for MCP servers.

This module provides the ServerState class for tracking operational metrics.
"""

import time


class ServerState:
    """
    Track server operational metrics for monitoring endpoints.
    
    This class maintains server state including:
    - Server start time for uptime calculation
    - Total request count
    - Active SSE connection count
    - MCP initialization status
    
    Attributes:
        start_time (float): Unix timestamp when the server started
        total_requests (int): Total number of requests processed
        active_connections (int): Number of active SSE connections
        mcp_initialized (bool): Whether MCP protocol is initialized
        
    Examples:
        >>> state = ServerState()
        >>> state.increment_requests()
        >>> state.get_uptime_seconds()
        0.001234
    """
    
    def __init__(self):
        """Initialize server state with default values."""
        self.start_time = time.time()
        self.total_requests = 0
        self.active_connections = 0
        self.mcp_initialized = True  # MCP is initialized when HTTP server starts
        
    def get_uptime_seconds(self) -> float:
        """
        Get server uptime in seconds.
        
        Returns:
            float: Number of seconds since server started
        """
        return time.time() - self.start_time
    
    def increment_requests(self):
        """Increment total request counter."""
        self.total_requests += 1
    
    def increment_connections(self):
        """Increment active connection counter."""
        self.active_connections += 1
    
    def decrement_connections(self):
        """Decrement active connection counter."""
        if self.active_connections > 0:
            self.active_connections -= 1
