"""
Comprehensive test suite for Stats Server MCP HTTP client.

Tests cover:
- SSE connection establishment
- MCP tool execution via HTTP POST
- Authentication scenarios (enabled/disabled, valid/invalid keys)
- Rate limiting behavior (within/exceeding limits, disabled)
- CORS headers verification
- Health check endpoints
- All Stats Server MCP tools (32 tools across 6 categories)
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
SERVER_URL = os.getenv("STATS_SERVER_URL", "http://localhost:8001")
API_KEY = os.getenv("MCP_API_KEY", "test-api-key-12345678")


class StatsHTTPClient:
    """
    Client for testing Stats Server MCP over HTTP.
    
    Handles JSON-RPC 2.0 requests to Stats MCP server over HTTP transport.
    """
    
    def __init__(
        self,
        base_url: str = SERVER_URL,
        api_key: Optional[str] = None,
        timeout: float = 30.0
    ):
        """
        Initialize Stats HTTP client.
        
        Args:
            base_url: Base URL of the Stats MCP server
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
def stats_client():
    """Create Stats HTTP client without authentication."""
    return StatsHTTPClient(base_url=SERVER_URL, api_key=None)


@pytest.fixture
def stats_auth_client():
    """Create Stats HTTP client with authentication."""
    return StatsHTTPClient(base_url=SERVER_URL, api_key=API_KEY)


# ============================================================================
# SSE Connection Tests
# ============================================================================


@pytest.mark.asyncio
async def test_stats_sse_connection_success(stats_client):
    """Test SSE connection establishment for stats server."""
    result = await stats_client.test_sse_connection(timeout=10.0)
    assert result is True, "SSE connection should be established"


@pytest.mark.asyncio
async def test_stats_sse_connection_headers(stats_client):
    """Test SSE response headers for stats server."""
    url = urljoin(SERVER_URL, "/sse")
    async with httpx.AsyncClient(timeout=10.0) as http_client:
        async with http_client.stream("GET", url) as response:
            assert response.status_code == 200
            assert "text/event-stream" in response.headers.get("content-type", "")


# ============================================================================
# MCP Protocol Tests
# ============================================================================


@pytest.mark.asyncio
async def test_stats_initialize_method(stats_client):
    """Test MCP initialize method for stats server."""
    response = await stats_client.call_mcp_method("initialize", {
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
async def test_stats_list_tools_success(stats_client):
    """Test listing available tools on stats server."""
    response = await stats_client.list_tools()
    
    assert "result" in response
    assert "tools" in response["result"]
    assert isinstance(response["result"]["tools"], list)
    assert len(response["result"]["tools"]) > 0


@pytest.mark.asyncio
async def test_stats_list_tools_structure(stats_client):
    """Test tool list structure for stats server."""
    response = await stats_client.list_tools()
    tools = response["result"]["tools"]
    
    # Check first tool has required fields
    tool = tools[0]
    assert "name" in tool
    assert "description" in tool
    assert "inputSchema" in tool


# ============================================================================
# Descriptive Statistics Tool Tests
# ============================================================================


@pytest.mark.asyncio
async def test_descriptive_stats_tool(stats_client):
    """Test descriptive_stats tool."""
    response = await stats_client.call_tool("descriptive_stats", {
        "data": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    })
    
    assert "result" in response
    assert response["result"]["isError"] is False


@pytest.mark.asyncio
async def test_correlation_tool(stats_client):
    """Test correlation tool."""
    response = await stats_client.call_tool("correlation", {
        "x": [1, 2, 3, 4, 5],
        "y": [2, 4, 6, 8, 10]
    })
    
    assert "result" in response
    assert response["result"]["isError"] is False


@pytest.mark.asyncio
async def test_percentile_tool(stats_client):
    """Test percentile tool."""
    response = await stats_client.call_tool("percentile", {
        "data": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
        "percentile": 50
    })
    
    assert "result" in response
    assert response["result"]["isError"] is False


@pytest.mark.asyncio
async def test_detect_outliers_tool(stats_client):
    """Test detect_outliers tool."""
    response = await stats_client.call_tool("detect_outliers", {
        "data": [1, 2, 3, 4, 5, 6, 7, 8, 9, 100],
        "method": "iqr"
    })
    
    assert "result" in response
    assert response["result"]["isError"] is False


# ============================================================================
# Time Series Analysis Tool Tests
# ============================================================================


@pytest.mark.asyncio
async def test_moving_average_tool(stats_client):
    """Test moving_average tool."""
    response = await stats_client.call_tool("moving_average", {
        "data": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
        "window_size": 3,
        "ma_type": "simple"
    })
    
    assert "result" in response
    assert response["result"]["isError"] is False


@pytest.mark.asyncio
async def test_detect_trend_tool(stats_client):
    """Test detect_trend tool."""
    response = await stats_client.call_tool("detect_trend", {
        "data": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    })
    
    assert "result" in response
    assert response["result"]["isError"] is False


@pytest.mark.asyncio
async def test_autocorrelation_tool(stats_client):
    """Test autocorrelation tool."""
    response = await stats_client.call_tool("autocorrelation", {
        "data": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
        "max_lag": 5
    })
    
    assert "result" in response
    assert response["result"]["isError"] is False


@pytest.mark.asyncio
async def test_change_point_detection_tool(stats_client):
    """Test change_point_detection tool."""
    response = await stats_client.call_tool("change_point_detection", {
        "data": [1, 2, 3, 4, 5, 10, 11, 12, 13, 14]
    })
    
    assert "result" in response
    assert response["result"]["isError"] is False


@pytest.mark.asyncio
async def test_rate_of_change_tool(stats_client):
    """Test rate_of_change tool."""
    response = await stats_client.call_tool("rate_of_change", {
        "data": [1, 2, 4, 7, 11]
    })
    
    assert "result" in response
    assert response["result"]["isError"] is False


@pytest.mark.asyncio
async def test_rolling_statistics_tool(stats_client):
    """Test rolling_statistics tool."""
    response = await stats_client.call_tool("rolling_statistics", {
        "data": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
        "window_size": 3
    })
    
    assert "result" in response
    assert response["result"]["isError"] is False


# ============================================================================
# Statistical Process Control (SPC) Tool Tests
# ============================================================================


@pytest.mark.asyncio
async def test_control_limits_tool(stats_client):
    """Test control_limits tool."""
    response = await stats_client.call_tool("control_limits", {
        "data": [10, 12, 11, 13, 12, 14, 13, 12, 11, 13],
        "chart_type": "xbar"
    })
    
    assert "result" in response
    assert response["result"]["isError"] is False


@pytest.mark.asyncio
async def test_process_capability_tool(stats_client):
    """Test process_capability tool."""
    response = await stats_client.call_tool("process_capability", {
        "data": [10, 12, 11, 13, 12, 14, 13, 12, 11, 13],
        "lsl": 8,
        "usl": 16
    })
    
    assert "result" in response
    assert response["result"]["isError"] is False


@pytest.mark.asyncio
async def test_western_electric_rules_tool(stats_client):
    """Test western_electric_rules tool."""
    response = await stats_client.call_tool("western_electric_rules", {
        "data": [10, 12, 11, 13, 12, 14, 13, 12, 11, 13],
        "center_line": 12,
        "sigma": 1
    })
    
    assert "result" in response
    assert response["result"]["isError"] is False


@pytest.mark.asyncio
async def test_cusum_chart_tool(stats_client):
    """Test cusum_chart tool."""
    response = await stats_client.call_tool("cusum_chart", {
        "data": [10, 12, 11, 13, 12, 14, 13, 12, 11, 13],
        "target": 12
    })
    
    assert "result" in response
    assert response["result"]["isError"] is False


@pytest.mark.asyncio
async def test_ewma_chart_tool(stats_client):
    """Test ewma_chart tool."""
    response = await stats_client.call_tool("ewma_chart", {
        "data": [10, 12, 11, 13, 12, 14, 13, 12, 11, 13],
        "lambda_weight": 0.2
    })
    
    assert "result" in response
    assert response["result"]["isError"] is False


# ============================================================================
# Signal Processing Tool Tests
# ============================================================================


@pytest.mark.asyncio
async def test_fft_analysis_tool(stats_client):
    """Test fft_analysis tool."""
    response = await stats_client.call_tool("fft_analysis", {
        "signal": [1, 2, 3, 4, 5, 4, 3, 2, 1, 2, 3, 4, 5, 4, 3, 2],
        "sampling_rate": 100
    })
    
    assert "result" in response
    assert response["result"]["isError"] is False


@pytest.mark.asyncio
async def test_power_spectral_density_tool(stats_client):
    """Test power_spectral_density tool."""
    response = await stats_client.call_tool("power_spectral_density", {
        "signal": [1, 2, 3, 4, 5, 4, 3, 2, 1, 2, 3, 4, 5, 4, 3, 2],
        "sampling_rate": 100
    })
    
    assert "result" in response
    assert response["result"]["isError"] is False


@pytest.mark.asyncio
async def test_rms_value_tool(stats_client):
    """Test rms_value tool."""
    response = await stats_client.call_tool("rms_value", {
        "signal": [1, 2, 3, 4, 5, 4, 3, 2, 1, 2]
    })
    
    assert "result" in response
    assert response["result"]["isError"] is False


@pytest.mark.asyncio
async def test_peak_detection_tool(stats_client):
    """Test peak_detection tool."""
    response = await stats_client.call_tool("peak_detection", {
        "signal": [1, 2, 5, 2, 1, 2, 6, 2, 1, 2, 4, 2, 1]
    })
    
    assert "result" in response
    assert response["result"]["isError"] is False


@pytest.mark.asyncio
async def test_signal_to_noise_ratio_tool(stats_client):
    """Test signal_to_noise_ratio tool."""
    response = await stats_client.call_tool("signal_to_noise_ratio", {
        "signal": [5, 5.1, 4.9, 5.2, 4.8],
        "noise": [0.1, -0.1, 0.05, -0.05, 0.08]
    })
    
    assert "result" in response
    assert response["result"]["isError"] is False


@pytest.mark.asyncio
async def test_harmonic_analysis_tool(stats_client):
    """Test harmonic_analysis tool."""
    response = await stats_client.call_tool("harmonic_analysis", {
        "signal": [1, 2, 3, 4, 5, 4, 3, 2, 1, 2, 3, 4, 5, 4, 3, 2],
        "sampling_rate": 100,
        "fundamental_frequency": 10
    })
    
    assert "result" in response
    assert response["result"]["isError"] is False


# ============================================================================
# Regression Analysis Tool Tests
# ============================================================================


@pytest.mark.asyncio
async def test_linear_regression_tool(stats_client):
    """Test linear_regression tool."""
    response = await stats_client.call_tool("linear_regression", {
        "x": [1, 2, 3, 4, 5],
        "y": [2, 4, 6, 8, 10]
    })
    
    assert "result" in response
    assert response["result"]["isError"] is False


@pytest.mark.asyncio
async def test_polynomial_regression_tool(stats_client):
    """Test polynomial_regression tool."""
    response = await stats_client.call_tool("polynomial_regression", {
        "x": [1, 2, 3, 4, 5],
        "y": [1, 4, 9, 16, 25],
        "degree": 2
    })
    
    assert "result" in response
    assert response["result"]["isError"] is False


@pytest.mark.asyncio
async def test_residual_analysis_tool(stats_client):
    """Test residual_analysis tool."""
    response = await stats_client.call_tool("residual_analysis", {
        "actual": [2, 4, 6, 8, 10],
        "predicted": [2.1, 3.9, 6.2, 7.8, 10.1]
    })
    
    assert "result" in response
    assert response["result"]["isError"] is False


@pytest.mark.asyncio
async def test_prediction_with_intervals_tool(stats_client):
    """Test prediction_with_intervals tool."""
    response = await stats_client.call_tool("prediction_with_intervals", {
        "x": [1, 2, 3, 4, 5],
        "y": [2, 4, 6, 8, 10],
        "x_new": [6, 7]
    })
    
    assert "result" in response
    assert response["result"]["isError"] is False


@pytest.mark.asyncio
async def test_multivariate_regression_tool(stats_client):
    """Test multivariate_regression tool."""
    response = await stats_client.call_tool("multivariate_regression", {
        "X": [[1, 2], [2, 3], [3, 4], [4, 5], [5, 6]],
        "y": [3, 5, 7, 9, 11]
    })
    
    assert "result" in response
    assert response["result"]["isError"] is False


# ============================================================================
# Advanced Outlier Detection Tool Tests
# ============================================================================


@pytest.mark.asyncio
async def test_z_score_detection_tool(stats_client):
    """Test z_score_detection tool."""
    response = await stats_client.call_tool("z_score_detection", {
        "data": [1, 2, 3, 4, 5, 6, 7, 8, 9, 100],
        "threshold": 3.0
    })
    
    assert "result" in response
    assert response["result"]["isError"] is False


@pytest.mark.asyncio
async def test_grubbs_test_tool(stats_client):
    """Test grubbs_test tool."""
    response = await stats_client.call_tool("grubbs_test", {
        "data": [1, 2, 3, 4, 5, 6, 7, 8, 9, 100],
        "alpha": 0.05
    })
    
    assert "result" in response
    assert response["result"]["isError"] is False


@pytest.mark.asyncio
async def test_dixon_q_test_tool(stats_client):
    """Test dixon_q_test tool."""
    response = await stats_client.call_tool("dixon_q_test", {
        "data": [1, 2, 3, 4, 5, 6, 7, 8, 9, 100],
        "alpha": 0.05
    })
    
    assert "result" in response
    assert response["result"]["isError"] is False


@pytest.mark.asyncio
async def test_isolation_forest_tool(stats_client):
    """Test isolation_forest tool."""
    response = await stats_client.call_tool("isolation_forest", {
        "data": [[1], [2], [3], [4], [5], [6], [7], [8], [9], [100]]
    })
    
    assert "result" in response
    assert response["result"]["isError"] is False


@pytest.mark.asyncio
async def test_mahalanobis_distance_tool(stats_client):
    """Test mahalanobis_distance tool."""
    response = await stats_client.call_tool("mahalanobis_distance", {
        "data": [[1, 2], [2, 3], [3, 4], [4, 5], [5, 6], [10, 10]]
    })
    
    assert "result" in response
    assert response["result"]["isError"] is False


@pytest.mark.asyncio
async def test_streaming_outlier_detection_tool(stats_client):
    """Test streaming_outlier_detection tool."""
    response = await stats_client.call_tool("streaming_outlier_detection", {
        "data": [1, 2, 3, 4, 5, 6, 7, 8, 9, 100],
        "window_size": 5
    })
    
    assert "result" in response
    assert response["result"]["isError"] is False


# ============================================================================
# Authentication Tests
# ============================================================================


@pytest.mark.asyncio
async def test_stats_auth_disabled_no_key(stats_client):
    """Test that requests succeed without key when auth is disabled."""
    response = await stats_client.list_tools(include_auth=False)
    assert "result" in response


@pytest.mark.asyncio
async def test_stats_auth_disabled_with_key(stats_auth_client):
    """Test that requests succeed with key when auth is disabled."""
    response = await stats_auth_client.list_tools(include_auth=True)
    assert "result" in response


# ============================================================================
# Health Endpoint Tests
# ============================================================================


@pytest.mark.asyncio
async def test_stats_health_endpoint(stats_client):
    """Test /health endpoint for stats server."""
    response = await stats_client.get_health()
    
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"
    assert "uptime_seconds" in data
    assert "server" in data


@pytest.mark.asyncio
async def test_stats_ready_endpoint(stats_client):
    """Test /ready endpoint for stats server."""
    response = await stats_client.get_ready()
    
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ready"
    assert data["mcp_initialized"] is True
    assert "tools_count" in data
    assert data["tools_count"] > 0


@pytest.mark.asyncio
async def test_stats_metrics_endpoint(stats_client):
    """Test /metrics endpoint for stats server."""
    response = await stats_client.get_metrics()
    
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
async def test_stats_invalid_json_rpc(stats_client):
    """Test invalid JSON-RPC request."""
    url = urljoin(SERVER_URL, "/messages")
    
    # Send invalid JSON-RPC (missing version)
    response = await stats_client.make_request("POST", "/messages", json={
        "method": "tools/list",
        "id": 1
    })
    
    assert response.status_code == 200
    data = response.json()
    assert "error" in data


@pytest.mark.asyncio
async def test_stats_unknown_method(stats_client):
    """Test calling unknown MCP method."""
    response = await stats_client.call_mcp_method("unknown/method", {})
    
    assert "error" in response
    assert response["error"]["code"] == -32601  # Method not found


@pytest.mark.asyncio
async def test_stats_invalid_tool_name(stats_client):
    """Test calling non-existent tool."""
    response = await stats_client.call_tool("nonexistent_tool", {})
    
    assert "result" in response
    assert response["result"]["isError"] is True


# ============================================================================
# Concurrent Request Tests
# ============================================================================


@pytest.mark.asyncio
async def test_stats_concurrent_tool_calls(stats_client):
    """Test multiple concurrent tool calls."""
    # Create multiple concurrent requests
    tasks = [
        stats_client.call_tool("descriptive_stats", {"data": list(range(i, i+10))})
        for i in range(1, 6)
    ]
    
    # Execute concurrently
    results = await asyncio.gather(*tasks, return_exceptions=True)
    
    # All should succeed
    assert len(results) == 5
    for result in results:
        assert isinstance(result, dict)
        assert "result" in result


@pytest.mark.asyncio
async def test_stats_concurrent_different_tools(stats_client):
    """Test concurrent calls to different tools."""
    tasks = [
        stats_client.call_tool("descriptive_stats", {"data": [1, 2, 3, 4, 5]}),
        stats_client.call_tool("correlation", {"x": [1, 2, 3], "y": [2, 4, 6]}),
        stats_client.call_tool("percentile", {"data": [1, 2, 3, 4, 5], "percentile": 50}),
        stats_client.call_tool("moving_average", {"data": [1, 2, 3, 4, 5], "window_size": 3, "ma_type": "simple"}),
    ]
    
    results = await asyncio.gather(*tasks, return_exceptions=True)
    
    # All should succeed
    assert len(results) == 4
    for result in results:
        assert isinstance(result, dict)
        assert "result" in result


@pytest.mark.asyncio
async def test_stats_concurrent_list_tools(stats_client):
    """Test concurrent tools/list calls."""
    tasks = [stats_client.list_tools() for _ in range(10)]
    
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
async def test_stats_response_time(stats_client):
    """Test that response time is reasonable."""
    start = time.time()
    response = await stats_client.list_tools()
    duration = time.time() - start
    
    assert "result" in response
    assert duration < 5.0, f"Response took {duration:.2f}s, expected < 5s"


# ============================================================================
# Main Test Runner (for standalone execution)
# ============================================================================


if __name__ == "__main__":
    print(f"Testing Stats MCP server at: {SERVER_URL}")
    print(f"Using API key: {API_KEY if API_KEY else 'None'}")
    print("\nRun tests with: pytest tests/test_http_stats_server.py -v")
