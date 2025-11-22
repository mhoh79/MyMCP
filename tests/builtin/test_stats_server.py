"""
Tests for the refactored Statistics MCP Server.

These tests verify that the builtin stats_server properly inherits from
BaseMCPServer and maintains all functionality.
"""

import pytest
from src.builtin.stats_server import StatsServer


def test_stats_server_initialization():
    """Test that StatsServer can be instantiated."""
    server = StatsServer()
    assert server is not None
    assert server.get_server_name() == "stats-server"
    assert server.get_server_version() == "2.0.0"


def test_stats_server_tools_registration():
    """Test that all stats tools are properly registered."""
    server = StatsServer()
    
    # Should have 32 tools registered
    assert server.tool_registry.count() == 32
    
    # Check some key tools exist
    expected_tools = [
        "descriptive_stats",
        "correlation",
        "percentile",
        "detect_outliers",
        "moving_average",
        "fft_analysis",
        "linear_regression",
        "control_limits"
    ]
    
    for tool_name in expected_tools:
        assert server.tool_registry.tool_exists(tool_name), f"Tool {tool_name} not found"
        assert server.tool_registry.get_handler(tool_name) is not None


def test_stats_server_tool_list():
    """Test that list_tools returns all tools."""
    server = StatsServer()
    tools = server.tool_registry.list_tools()
    
    assert len(tools) == 32
    assert all(hasattr(tool, 'name') for tool in tools)
    assert all(hasattr(tool, 'description') for tool in tools)
    assert all(hasattr(tool, 'inputSchema') for tool in tools)


@pytest.mark.asyncio
async def test_descriptive_stats_handler():
    """Test that descriptive statistics calculation works."""
    server = StatsServer()
    handler = server.tool_registry.get_handler("descriptive_stats")
    
    data = [1.0, 2.0, 3.0, 4.0, 5.0]
    result = await handler({"data": data})
    
    assert result.isError is False
    # Mean should be 3.0
    assert "3.0" in result.content[0].text


@pytest.mark.asyncio
async def test_correlation_handler():
    """Test that correlation calculation works."""
    server = StatsServer()
    handler = server.tool_registry.get_handler("correlation")
    
    x = [1.0, 2.0, 3.0, 4.0, 5.0]
    y = [2.0, 4.0, 6.0, 8.0, 10.0]
    result = await handler({"x": x, "y": y})
    
    assert result.isError is False
    # Perfect positive correlation should be 1.0
    assert "1.0" in result.content[0].text


@pytest.mark.asyncio
async def test_percentile_handler():
    """Test that percentile calculation works."""
    server = StatsServer()
    handler = server.tool_registry.get_handler("percentile")
    
    data = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]
    result = await handler({"data": data, "percentile": 50})
    
    assert result.isError is False
    # Median (50th percentile) should be 5.5
    assert "5.5" in result.content[0].text


@pytest.mark.asyncio
async def test_error_handling():
    """Test that error handling works properly."""
    server = StatsServer()
    handler = server.tool_registry.get_handler("descriptive_stats")
    
    # Test missing required parameter
    result = await handler({})
    assert result.isError is True
    assert "missing" in result.content[0].text.lower() or "required" in result.content[0].text.lower()
    
    # Test empty data
    result = await handler({"data": []})
    assert result.isError is True
