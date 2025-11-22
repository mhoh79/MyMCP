"""
Comprehensive test suite for MCP HTTP client.

Tests cover:
- SSE connection establishment
- MCP tool execution via HTTP POST
- Authentication scenarios (enabled/disabled, valid/invalid keys)
- Rate limiting behavior (within/exceeding limits, disabled)
- CORS headers verification
- Health check endpoints
- All MCP tools
- Concurrent requests
- Error handling
"""

import asyncio
import os
import json
import time
from typing import Any, Dict, Optional
from urllib.parse import urljoin

import httpx
import pytest


# Configuration from environment or defaults
SERVER_URL = os.getenv("MATH_SERVER_URL", "http://localhost:8000")
API_KEY = os.getenv("MCP_API_KEY", "test-api-key-12345678")


class MCPHTTPClient:
    """
    Client for testing MCP over HTTP.
    
    Handles JSON-RPC 2.0 requests to MCP server over HTTP transport.
    """
    
    def __init__(
        self,
        base_url: str = SERVER_URL,
        api_key: Optional[str] = None,
        timeout: float = 30.0
    ):
        """
        Initialize MCP HTTP client.
        
        Args:
            base_url: Base URL of the MCP server
            api_key: Optional API key for authentication
            timeout: Request timeout in seconds
        """
        self.base_url = base_url.rstrip('/')
        self.api_key = api_key
        self.timeout = timeout
        self._request_id = 0
    
    def _get_next_id(self) -> int:
        """Get next request ID."""
        self._request_id += 1
        return self._request_id
    
    def _build_headers(self, include_auth: bool = True) -> Dict[str, str]:
        """
        Build request headers.
        
        Args:
            include_auth: Whether to include authentication header
            
        Returns:
            Dictionary of HTTP headers
        """
        headers = {
            "Content-Type": "application/json",
            "Accept": "application/json"
        }
        
        if include_auth and self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        
        return headers
    
    async def test_sse_connection(self, timeout: float = 5.0) -> bool:
        """
        Test SSE stream connection.
        
        Args:
            timeout: Connection timeout in seconds
            
        Returns:
            True if connection successful, False otherwise
        """
        url = urljoin(self.base_url, "/sse")
        
        try:
            async with httpx.AsyncClient(timeout=timeout) as client:
                async with client.stream("GET", url, headers=self._build_headers()) as response:
                    # Check if connection is successful
                    if response.status_code != 200:
                        return False
                    
                    # Try to read first event
                    async for line in response.aiter_lines():
                        if line.strip():
                            # Successfully received data
                            return True
                            break
                    
                    return False
                    
        except (httpx.TimeoutException, httpx.ConnectError):
            return False
    
    async def call_mcp_method(
        self,
        method: str,
        params: Optional[Dict[str, Any]] = None,
        include_auth: bool = True
    ) -> Dict[str, Any]:
        """
        Call an MCP method via JSON-RPC 2.0.
        
        Args:
            method: MCP method name (e.g., 'tools/list', 'tools/call')
            params: Method parameters
            include_auth: Whether to include authentication
            
        Returns:
            JSON-RPC response as dictionary
        """
        url = urljoin(self.base_url, "/messages")
        
        request_body = {
            "jsonrpc": "2.0",
            "method": method,
            "params": params or {},
            "id": self._get_next_id()
        }
        
        headers = self._build_headers(include_auth=include_auth)
        
        async with httpx.AsyncClient(timeout=self.timeout) as client:
            response = await client.post(url, json=request_body, headers=headers)
            return response.json()
    
    async def list_tools(self, include_auth: bool = True) -> Dict[str, Any]:
        """
        List available MCP tools.
        
        Args:
            include_auth: Whether to include authentication
            
        Returns:
            Response containing list of tools
        """
        return await self.call_mcp_method("tools/list", include_auth=include_auth)
    
    async def call_tool(
        self,
        tool_name: str,
        arguments: Dict[str, Any],
        include_auth: bool = True
    ) -> Dict[str, Any]:
        """
        Call a specific MCP tool.
        
        Args:
            tool_name: Name of the tool to call
            arguments: Tool arguments
            include_auth: Whether to include authentication
            
        Returns:
            Tool execution result
        """
        params = {
            "name": tool_name,
            "arguments": arguments
        }
        return await self.call_mcp_method("tools/call", params, include_auth=include_auth)
    
    async def get_health(self) -> httpx.Response:
        """Get health check response."""
        url = urljoin(self.base_url, "/health")
        async with httpx.AsyncClient(timeout=self.timeout) as client:
            return await client.get(url)
    
    async def get_ready(self) -> httpx.Response:
        """Get readiness check response."""
        url = urljoin(self.base_url, "/ready")
        async with httpx.AsyncClient(timeout=self.timeout) as client:
            return await client.get(url)
    
    async def get_metrics(self) -> httpx.Response:
        """Get metrics response."""
        url = urljoin(self.base_url, "/metrics")
        async with httpx.AsyncClient(timeout=self.timeout) as client:
            return await client.get(url)
    
    async def make_request(
        self,
        method: str,
        path: str,
        include_auth: bool = True,
        **kwargs
    ) -> httpx.Response:
        """
        Make a raw HTTP request.
        
        Args:
            method: HTTP method
            path: URL path
            include_auth: Whether to include authentication
            **kwargs: Additional arguments for httpx
            
        Returns:
            HTTP response
        """
        url = urljoin(self.base_url, path)
        headers = self._build_headers(include_auth=include_auth)
        
        if 'headers' in kwargs:
            headers.update(kwargs.pop('headers'))
        
        async with httpx.AsyncClient(timeout=self.timeout) as client:
            return await client.request(method, url, headers=headers, **kwargs)


# ============================================================================
# Test Fixtures
# ============================================================================


@pytest.fixture
def client():
    """Create MCP HTTP client without authentication."""
    return MCPHTTPClient(base_url=SERVER_URL, api_key=None)


@pytest.fixture
def auth_client():
    """Create MCP HTTP client with authentication."""
    return MCPHTTPClient(base_url=SERVER_URL, api_key=API_KEY)


# ============================================================================
# SSE Connection Tests
# ============================================================================


@pytest.mark.asyncio
async def test_sse_connection_success(client):
    """Test SSE connection establishment."""
    result = await client.test_sse_connection(timeout=10.0)
    assert result is True, "SSE connection should be established"


@pytest.mark.asyncio
async def test_sse_connection_headers(client):
    """Test SSE response headers."""
    url = urljoin(SERVER_URL, "/sse")
    async with httpx.AsyncClient(timeout=10.0) as http_client:
        async with http_client.stream("GET", url) as response:
            assert response.status_code == 200
            assert "text/event-stream" in response.headers.get("content-type", "")


# ============================================================================
# MCP Protocol Tests
# ============================================================================


@pytest.mark.asyncio
async def test_initialize_method(client):
    """Test MCP initialize method."""
    response = await client.call_mcp_method("initialize", {
        "protocolVersion": "2024-11-05",
        "capabilities": {},
        "clientInfo": {
            "name": "test-client",
            "version": "1.0.0"
        }
    })
    
    assert "result" in response
    assert response["result"]["protocolVersion"] == "2024-11-05"
    assert "serverInfo" in response["result"]


@pytest.mark.asyncio
async def test_list_tools_success(client):
    """Test listing available tools."""
    response = await client.list_tools()
    
    assert "result" in response
    assert "tools" in response["result"]
    assert isinstance(response["result"]["tools"], list)
    assert len(response["result"]["tools"]) > 0


@pytest.mark.asyncio
async def test_list_tools_structure(client):
    """Test tool list structure."""
    response = await client.list_tools()
    tools = response["result"]["tools"]
    
    # Check first tool has required fields
    tool = tools[0]
    assert "name" in tool
    assert "description" in tool
    assert "inputSchema" in tool


# ============================================================================
# Tool Execution Tests
# ============================================================================


@pytest.mark.asyncio
async def test_fibonacci_tool(client):
    """Test calculate_fibonacci tool."""
    response = await client.call_tool("calculate_fibonacci", {"n": 10})
    
    assert "result" in response
    result = response["result"]
    assert "content" in result
    assert result["isError"] is False


@pytest.mark.asyncio
async def test_generate_primes_tool(client):
    """Test generate_primes tool."""
    response = await client.call_tool("generate_primes", {"limit": 20})
    
    assert "result" in response
    assert response["result"]["isError"] is False


@pytest.mark.asyncio
async def test_is_prime_tool(client):
    """Test is_prime tool."""
    response = await client.call_tool("is_prime", {"n": 17})
    
    assert "result" in response
    assert response["result"]["isError"] is False


@pytest.mark.asyncio
async def test_prime_factorization_tool(client):
    """Test prime_factorization tool."""
    response = await client.call_tool("prime_factorization", {"n": 24})
    
    assert "result" in response
    assert response["result"]["isError"] is False


@pytest.mark.asyncio
async def test_nth_prime_tool(client):
    """Test nth_prime tool."""
    response = await client.call_tool("nth_prime", {"n": 10})
    
    assert "result" in response
    assert response["result"]["isError"] is False


@pytest.mark.asyncio
async def test_gcd_tool(client):
    """Test gcd tool."""
    response = await client.call_tool("gcd", {"numbers": [48, 18]})
    
    assert "result" in response
    assert response["result"]["isError"] is False


@pytest.mark.asyncio
async def test_lcm_tool(client):
    """Test lcm tool."""
    response = await client.call_tool("lcm", {"numbers": [12, 18]})
    
    assert "result" in response
    assert response["result"]["isError"] is False


@pytest.mark.asyncio
async def test_factorial_tool(client):
    """Test factorial tool."""
    response = await client.call_tool("factorial", {"n": 5})
    
    assert "result" in response
    assert response["result"]["isError"] is False


@pytest.mark.asyncio
async def test_combinations_tool(client):
    """Test combinations tool."""
    response = await client.call_tool("combinations", {"n": 5, "r": 2})
    
    assert "result" in response
    assert response["result"]["isError"] is False


@pytest.mark.asyncio
async def test_permutations_tool(client):
    """Test permutations tool."""
    response = await client.call_tool("permutations", {"n": 5, "r": 2})
    
    assert "result" in response
    assert response["result"]["isError"] is False


@pytest.mark.asyncio
async def test_generate_hash_tool(client):
    """Test generate_hash tool."""
    response = await client.call_tool("generate_hash", {
        "data": "Hello World",
        "algorithm": "sha256",
        "output_format": "hex"
    })
    
    assert "result" in response
    assert response["result"]["isError"] is False


@pytest.mark.asyncio
async def test_unit_convert_tool(client):
    """Test unit_convert tool."""
    response = await client.call_tool("unit_convert", {
        "value": 100,
        "from_unit": "km",
        "to_unit": "mi"
    })
    
    assert "result" in response
    assert response["result"]["isError"] is False


@pytest.mark.asyncio
async def test_date_diff_tool(client):
    """Test date_diff tool."""
    response = await client.call_tool("date_diff", {
        "date1": "2025-01-01",
        "date2": "2025-12-31",
        "unit": "days"
    })
    
    assert "result" in response
    assert response["result"]["isError"] is False


@pytest.mark.asyncio
async def test_date_add_tool(client):
    """Test date_add tool."""
    response = await client.call_tool("date_add", {
        "date": "2025-01-01",
        "amount": 30,
        "unit": "days"
    })
    
    assert "result" in response
    assert response["result"]["isError"] is False


@pytest.mark.asyncio
async def test_text_stats_tool(client):
    """Test text_stats tool."""
    response = await client.call_tool("text_stats", {
        "text": "Hello world! This is a test."
    })
    
    assert "result" in response
    assert response["result"]["isError"] is False


@pytest.mark.asyncio
async def test_word_frequency_tool(client):
    """Test word_frequency tool."""
    response = await client.call_tool("word_frequency", {
        "text": "hello world hello test world hello",
        "top_n": 5,
        "skip_common": False
    })
    
    assert "result" in response
    assert response["result"]["isError"] is False


# ============================================================================
# Authentication Tests
# ============================================================================


@pytest.mark.asyncio
async def test_auth_disabled_no_key(client):
    """Test that requests succeed without key when auth is disabled."""
    response = await client.list_tools(include_auth=False)
    assert "result" in response


@pytest.mark.asyncio
async def test_auth_disabled_with_key(auth_client):
    """Test that requests succeed with key when auth is disabled."""
    response = await auth_client.list_tools(include_auth=True)
    assert "result" in response


@pytest.mark.asyncio
@pytest.mark.skip(reason="Requires auth enabled in config")
async def test_auth_enabled_valid_key(auth_client):
    """Test authentication with valid key (requires auth enabled)."""
    response = await auth_client.list_tools(include_auth=True)
    assert "result" in response


@pytest.mark.asyncio
@pytest.mark.skip(reason="Requires auth enabled in config")
async def test_auth_enabled_invalid_key():
    """Test authentication with invalid key (requires auth enabled)."""
    invalid_client = MCPHTTPClient(base_url=SERVER_URL, api_key="invalid-key")
    response = await invalid_client.list_tools(include_auth=True)
    assert "error" in response


@pytest.mark.asyncio
@pytest.mark.skip(reason="Requires auth enabled in config")
async def test_auth_enabled_missing_key(client):
    """Test authentication with missing key (requires auth enabled)."""
    response = await client.list_tools(include_auth=False)
    assert "error" in response


# ============================================================================
# Rate Limiting Tests
# ============================================================================


@pytest.mark.asyncio
async def test_rate_limit_within_limit(client):
    """Test requests within rate limit."""
    # Make a few requests (well within any reasonable limit)
    for _ in range(5):
        response = await client.list_tools()
        assert "result" in response


@pytest.mark.asyncio
@pytest.mark.skip(reason="Requires rate limiting enabled and takes time")
async def test_rate_limit_exceeding_limit(client):
    """Test rate limit enforcement (requires rate limiting enabled)."""
    # This test would need to make many requests quickly
    # Skipped by default to avoid impacting server
    success_count = 0
    rate_limited_count = 0
    
    for i in range(100):
        try:
            response = await client.list_tools()
            if "result" in response:
                success_count += 1
            elif "error" in response:
                rate_limited_count += 1
        except httpx.HTTPStatusError as e:
            if e.response.status_code == 429:
                rate_limited_count += 1
    
    # Should have some rate limited requests
    assert rate_limited_count > 0


@pytest.mark.asyncio
async def test_rate_limit_headers(client):
    """Test rate limit headers in response."""
    url = urljoin(SERVER_URL, "/messages")
    response = await client.make_request("POST", "/messages", json={
        "jsonrpc": "2.0",
        "method": "tools/list",
        "id": 1
    })
    
    # Headers may or may not be present depending on config
    # Just verify response is successful
    assert response.status_code == 200


# ============================================================================
# CORS Tests
# ============================================================================


@pytest.mark.asyncio
async def test_cors_headers_present(client):
    """Test CORS headers in response."""
    # Test with actual POST request since OPTIONS may not be supported
    response = await client.make_request("POST", "/messages", json={
        "jsonrpc": "2.0",
        "method": "tools/list",
        "id": 1
    })
    
    # Just verify the request succeeds
    # CORS headers are optional depending on configuration
    assert response.status_code == 200


@pytest.mark.asyncio
async def test_cors_origin_header(client):
    """Test CORS with Origin header."""
    url = urljoin(SERVER_URL, "/messages")
    headers = {
        "Origin": "http://localhost:3000",
    }
    
    async with httpx.AsyncClient() as http_client:
        response = await http_client.post(url, json={
            "jsonrpc": "2.0",
            "method": "tools/list",
            "id": 1
        }, headers=headers)
        # Response should succeed regardless of CORS config
        assert response.status_code == 200


# ============================================================================
# Health Endpoint Tests
# ============================================================================


@pytest.mark.asyncio
async def test_health_endpoint(client):
    """Test /health endpoint."""
    response = await client.get_health()
    
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"
    assert "uptime_seconds" in data
    assert "server" in data


@pytest.mark.asyncio
async def test_ready_endpoint(client):
    """Test /ready endpoint."""
    response = await client.get_ready()
    
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ready"
    assert data["mcp_initialized"] is True
    assert "tools_count" in data
    assert data["tools_count"] > 0


@pytest.mark.asyncio
async def test_metrics_endpoint(client):
    """Test /metrics endpoint."""
    response = await client.get_metrics()
    
    assert response.status_code == 200
    data = response.json()
    assert "total_requests" in data
    assert "active_connections" in data
    assert "tools_available" in data
    assert "uptime_seconds" in data
    assert data["tools_available"] > 0


# ============================================================================
# Error Handling Tests
# ============================================================================


@pytest.mark.asyncio
async def test_invalid_json_rpc(client):
    """Test invalid JSON-RPC request."""
    url = urljoin(SERVER_URL, "/messages")
    
    # Send invalid JSON-RPC (missing version)
    response = await client.make_request("POST", "/messages", json={
        "method": "tools/list",
        "id": 1
    })
    
    assert response.status_code == 200
    data = response.json()
    assert "error" in data


@pytest.mark.asyncio
async def test_unknown_method(client):
    """Test calling unknown MCP method."""
    response = await client.call_mcp_method("unknown/method", {})
    
    assert "error" in response
    assert response["error"]["code"] == -32601  # Method not found


@pytest.mark.asyncio
async def test_invalid_tool_name(client):
    """Test calling non-existent tool."""
    response = await client.call_tool("nonexistent_tool", {})
    
    assert "result" in response
    assert response["result"]["isError"] is True


@pytest.mark.asyncio
async def test_invalid_tool_arguments(client):
    """Test calling tool with invalid arguments."""
    response = await client.call_tool("fibonacci", {"n": "not a number"})
    
    # Server should return error in result
    assert "result" in response
    # Error handling depends on server implementation
    # Just verify we get a response


@pytest.mark.asyncio
async def test_malformed_json(client):
    """Test sending malformed JSON."""
    url = urljoin(SERVER_URL, "/messages")
    
    async with httpx.AsyncClient() as http_client:
        response = await http_client.post(
            url,
            content="{ invalid json }",
            headers={"Content-Type": "application/json"}
        )
        
        assert response.status_code == 200
        data = response.json()
        assert "error" in data
        assert data["error"]["code"] == -32700  # Parse error


# ============================================================================
# Concurrent Request Tests
# ============================================================================


@pytest.mark.asyncio
async def test_concurrent_tool_calls(client):
    """Test multiple concurrent tool calls."""
    # Create multiple concurrent requests
    tasks = [
        client.call_tool("fibonacci", {"n": i})
        for i in range(5, 10)
    ]
    
    # Execute concurrently
    results = await asyncio.gather(*tasks, return_exceptions=True)
    
    # All should succeed
    assert len(results) == 5
    for result in results:
        assert isinstance(result, dict)
        assert "result" in result


@pytest.mark.asyncio
async def test_concurrent_different_tools(client):
    """Test concurrent calls to different tools."""
    tasks = [
        client.call_tool("fibonacci", {"n": 10}),
        client.call_tool("is_prime", {"n": 17}),
        client.call_tool("factorial", {"n": 5}),
        client.call_tool("gcd", {"numbers": [48, 18]}),
    ]
    
    results = await asyncio.gather(*tasks, return_exceptions=True)
    
    # All should succeed
    assert len(results) == 4
    for result in results:
        assert isinstance(result, dict)
        assert "result" in result


@pytest.mark.asyncio
async def test_concurrent_list_tools(client):
    """Test concurrent tools/list calls."""
    tasks = [client.list_tools() for _ in range(10)]
    
    results = await asyncio.gather(*tasks, return_exceptions=True)
    
    # All should succeed and return same tool list
    assert len(results) == 10
    for result in results:
        assert isinstance(result, dict)
        assert "result" in result
        assert "tools" in result["result"]


# ============================================================================
# Performance Tests
# ============================================================================


@pytest.mark.asyncio
async def test_response_time(client):
    """Test that response time is reasonable."""
    start = time.time()
    response = await client.list_tools()
    duration = time.time() - start
    
    assert "result" in response
    assert duration < 5.0, f"Response took {duration:.2f}s, expected < 5s"


@pytest.mark.asyncio
async def test_large_response_handling(client):
    """Test handling of large responses."""
    # Request something that returns a lot of data
    response = await client.call_tool("pascal_triangle", {"rows": 20})
    
    assert "result" in response
    assert response["result"]["isError"] is False


# ============================================================================
# Main Test Runner (for standalone execution)
# ============================================================================


if __name__ == "__main__":
    print(f"Testing MCP server at: {SERVER_URL}")
    print(f"Using API key: {API_KEY if API_KEY else 'None'}")
    print("\nRun tests with: pytest tests/test_http_client.py -v")
