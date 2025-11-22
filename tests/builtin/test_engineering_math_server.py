"""
Tests for the Engineering Math MCP Server.

These tests verify that the engineering_math_server properly inherits from
BaseMCPServer and all tools are registered correctly.
"""

import pytest
import numpy as np
from src.builtin.engineering_math_server import EngineeringMathServer


def test_engineering_math_server_initialization():
    """Test that EngineeringMathServer can be instantiated."""
    server = EngineeringMathServer()
    assert server is not None
    assert server.get_server_name() == "engineering-math-server"
    assert server.get_server_version() == "1.0.0"


def test_engineering_math_server_tools_registration():
    """Test that all 31 engineering math tools are properly registered."""
    server = EngineeringMathServer()
    
    # Should have 31 tools registered
    assert server.tool_registry.count() == 31
    
    # Check linear algebra tools exist
    linear_algebra_tools = [
        "matrix_operations",
        "matrix_inverse",
        "matrix_decomposition",
        "solve_linear_system",
        "vector_operations",
        "least_squares_fit"
    ]
    
    for tool_name in linear_algebra_tools:
        assert server.tool_registry.tool_exists(tool_name), f"Tool {tool_name} not found"
        assert server.tool_registry.get_handler(tool_name) is not None
    
    # Check calculus tools exist
    calculus_tools = [
        "numerical_derivative",
        "numerical_integration",
        "ode_solver"
    ]
    
    for tool_name in calculus_tools:
        assert server.tool_registry.tool_exists(tool_name), f"Tool {tool_name} not found"
    
    # Check numerical methods tools exist
    numerical_tools = [
        "root_finding",
        "interpolation"
    ]
    
    for tool_name in numerical_tools:
        assert server.tool_registry.tool_exists(tool_name), f"Tool {tool_name} not found"
    
    # Check polynomial tools exist
    polynomial_tools = [
        "polynomial_arithmetic",
        "polynomial_roots"
    ]
    
    for tool_name in polynomial_tools:
        assert server.tool_registry.tool_exists(tool_name), f"Tool {tool_name} not found"
    
    # Check utility tools exist
    utility_tools = [
        "unit_converter",
        "precision_calculator",
        "significant_figures"
    ]
    
    for tool_name in utility_tools:
        assert server.tool_registry.tool_exists(tool_name), f"Tool {tool_name} not found"


def test_engineering_math_server_tool_list():
    """Test that list_tools returns all tools."""
    server = EngineeringMathServer()
    tools = server.tool_registry.list_tools()
    
    assert len(tools) == 31
    
    # Verify tool names are unique
    tool_names = [tool.name for tool in tools]
    assert len(tool_names) == len(set(tool_names))


@pytest.mark.asyncio
async def test_matrix_operations_handler():
    """Test matrix operations tool handler."""
    server = EngineeringMathServer()
    handler = server.tool_registry.get_handler("matrix_operations")
    
    # Test matrix addition
    result = await handler({
        "operation": "add",
        "matrix_a": [[1, 2], [3, 4]],
        "matrix_b": [[5, 6], [7, 8]]
    })
    
    assert result.isError is False
    assert "result" in result.content[0].text
    
    # Test matrix transpose
    result = await handler({
        "operation": "transpose",
        "matrix_a": [[1, 2, 3], [4, 5, 6]]
    })
    
    assert result.isError is False


@pytest.mark.asyncio
async def test_vector_operations_handler():
    """Test vector operations tool handler."""
    server = EngineeringMathServer()
    handler = server.tool_registry.get_handler("vector_operations")
    
    # Test dot product
    result = await handler({
        "operation": "dot",
        "vector_a": [1, 2, 3],
        "vector_b": [4, 5, 6]
    })
    
    assert result.isError is False
    assert "dot_product" in result.content[0].text
    
    # Test vector normalization
    result = await handler({
        "operation": "normalize",
        "vector_a": [3, 4]
    })
    
    assert result.isError is False


@pytest.mark.asyncio
async def test_numerical_derivative_handler():
    """Test numerical derivative tool handler."""
    server = EngineeringMathServer()
    handler = server.tool_registry.get_handler("numerical_derivative")
    
    # Test central difference on quadratic function f(x) = x^2
    # f'(x) = 2x, so at x=2, f'(2) = 4
    x_values = [1.0, 2.0, 3.0]
    f_values = [1.0, 4.0, 9.0]  # x^2
    
    result = await handler({
        "f_values": f_values,
        "x_values": x_values,
        "method": "central"
    })
    
    assert result.isError is False
    assert "derivative" in result.content[0].text


@pytest.mark.asyncio
async def test_numerical_integration_handler():
    """Test numerical integration tool handler."""
    server = EngineeringMathServer()
    handler = server.tool_registry.get_handler("numerical_integration")
    
    # Test integration of f(x) = 1 from 0 to 1 (should be 1.0)
    result = await handler({
        "f_values": [1.0, 1.0, 1.0],
        "x_values": [0.0, 0.5, 1.0],
        "method": "trapz"
    })
    
    assert result.isError is False
    assert "integral" in result.content[0].text


@pytest.mark.asyncio
async def test_polynomial_arithmetic_handler():
    """Test polynomial arithmetic tool handler."""
    server = EngineeringMathServer()
    handler = server.tool_registry.get_handler("polynomial_arithmetic")
    
    # Test polynomial addition: (x + 1) + (2x + 3) = 3x + 4
    result = await handler({
        "operation": "add",
        "poly_a": [1, 1],  # x + 1
        "poly_b": [2, 3]   # 2x + 3
    })
    
    assert result.isError is False


@pytest.mark.asyncio
async def test_polynomial_roots_handler():
    """Test polynomial roots tool handler."""
    server = EngineeringMathServer()
    handler = server.tool_registry.get_handler("polynomial_roots")
    
    # Test finding roots of x^2 - 5x + 6 = (x-2)(x-3), roots are 2 and 3
    result = await handler({
        "coefficients": [1, -5, 6]
    })
    
    assert result.isError is False
    assert "real_roots" in result.content[0].text or "complex_roots" in result.content[0].text


@pytest.mark.asyncio
async def test_unit_converter_handler():
    """Test unit converter tool handler."""
    server = EngineeringMathServer()
    handler = server.tool_registry.get_handler("unit_converter")
    
    # Test length conversion: 1 km to meters
    result = await handler({
        "value": 1.0,
        "from_unit": "km",
        "to_unit": "m"
    })
    
    assert result.isError is False
    assert "1000" in result.content[0].text
    
    # Test temperature conversion: 0 C to F
    result = await handler({
        "value": 0.0,
        "from_unit": "C",
        "to_unit": "F"
    })
    
    assert result.isError is False
    assert "32" in result.content[0].text


@pytest.mark.asyncio
async def test_precision_calculator_handler():
    """Test precision calculator tool handler."""
    server = EngineeringMathServer()
    handler = server.tool_registry.get_handler("precision_calculator")
    
    # Test high precision addition
    result = await handler({
        "operation": "add",
        "value_a": "0.1",
        "value_b": "0.2",
        "precision": 50
    })
    
    assert result.isError is False
    assert "0.3" in result.content[0].text


@pytest.mark.asyncio
async def test_significant_figures_handler():
    """Test significant figures tool handler."""
    server = EngineeringMathServer()
    handler = server.tool_registry.get_handler("significant_figures")
    
    # Test rounding to 3 significant figures
    result = await handler({
        "value": 123.456,
        "sig_figs": 3
    })
    
    assert result.isError is False
    assert "123" in result.content[0].text


@pytest.mark.asyncio
async def test_interpolation_handler():
    """Test interpolation tool handler."""
    server = EngineeringMathServer()
    handler = server.tool_registry.get_handler("interpolation")
    
    # Test linear interpolation
    result = await handler({
        "x": [0, 1, 2],
        "y": [0, 1, 4],
        "x_new": [0.5, 1.5],
        "method": "linear"
    })
    
    assert result.isError is False
    assert "interpolated_values" in result.content[0].text


@pytest.mark.asyncio
async def test_error_handling():
    """Test that tools handle errors gracefully."""
    server = EngineeringMathServer()
    handler = server.tool_registry.get_handler("matrix_operations")
    
    # Test with mismatched matrix dimensions
    result = await handler({
        "operation": "add",
        "matrix_a": [[1, 2]],
        "matrix_b": [[1, 2, 3]]
    })
    
    assert result.isError is True
    assert "Error" in result.content[0].text or "shape" in result.content[0].text.lower()
