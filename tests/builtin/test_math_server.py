"""
Tests for the refactored Math Calculator MCP Server.

These tests verify that the builtin math_server properly inherits from
BaseMCPServer and maintains all functionality.
"""

import pytest
from src.builtin.math_server import MathServer


def test_math_server_initialization():
    """Test that MathServer can be instantiated."""
    server = MathServer()
    assert server is not None
    assert server.get_server_name() == "math-calculator"
    assert server.get_server_version() == "2.0.0"


def test_math_server_tools_registration():
    """Test that all math tools are properly registered."""
    server = MathServer()
    
    # Should have 25 tools registered
    assert server.tool_registry.count() == 25
    
    # Check some key tools exist
    expected_tools = [
        "calculate_fibonacci",
        "is_prime",
        "generate_primes",
        "factorial",
        "gcd",
        "lcm",
        "generate_hash",
        "unit_convert",
        "date_diff",
        "text_stats"
    ]
    
    for tool_name in expected_tools:
        assert server.tool_registry.tool_exists(tool_name), f"Tool {tool_name} not found"
        assert server.tool_registry.get_handler(tool_name) is not None


def test_math_server_tool_list():
    """Test that list_tools returns all tools."""
    server = MathServer()
    tools = server.tool_registry.list_tools()
    
    assert len(tools) == 25
    assert all(hasattr(tool, 'name') for tool in tools)
    assert all(hasattr(tool, 'description') for tool in tools)
    assert all(hasattr(tool, 'inputSchema') for tool in tools)


@pytest.mark.asyncio
async def test_fibonacci_handler():
    """Test that fibonacci calculation works."""
    server = MathServer()
    handler = server.tool_registry.get_handler("calculate_fibonacci")
    
    # Test calculating single fibonacci number
    result = await handler({"n": 10, "return_sequence": False})
    assert result.isError is False
    assert "55" in result.content[0].text
    
    # Test calculating fibonacci sequence
    result = await handler({"n": 5, "return_sequence": True})
    assert result.isError is False
    assert "[0, 1, 1, 2, 3]" in result.content[0].text


@pytest.mark.asyncio
async def test_is_prime_handler():
    """Test that prime checking works."""
    server = MathServer()
    handler = server.tool_registry.get_handler("is_prime")
    
    # Test prime number
    result = await handler({"n": 17})
    assert result.isError is False
    assert "prime" in result.content[0].text.lower() and "17" in result.content[0].text
    
    # Test non-prime number
    result = await handler({"n": 20})
    assert result.isError is False
    assert "not" in result.content[0].text.lower() and "prime" in result.content[0].text.lower()


@pytest.mark.asyncio
async def test_gcd_handler():
    """Test that GCD calculation works."""
    server = MathServer()
    handler = server.tool_registry.get_handler("gcd")
    
    result = await handler({"numbers": [48, 18]})
    assert result.isError is False
    assert "6" in result.content[0].text


@pytest.mark.asyncio
async def test_error_handling():
    """Test that error handling works properly."""
    server = MathServer()
    handler = server.tool_registry.get_handler("calculate_fibonacci")
    
    # Test missing required parameter
    result = await handler({})
    assert result.isError is True
    assert "missing" in result.content[0].text.lower() or "required" in result.content[0].text.lower()
