"""
Calculus & Analysis Tools for Engineering Math Server.

This module provides 9 calculus tools for derivatives, integrals, and ODEs.
"""

import logging
import numpy as np
from scipy import integrate, optimize
from typing import Any, Dict, List, Callable
import sympy as sp

from mcp.types import Tool, TextContent, CallToolResult

logger = logging.getLogger("engineering-math-server")


# ============================================================================
# Tool 7: Numerical Derivative
# ============================================================================

def numerical_derivative(
    f_values: List[float],
    x_values: List[float] = None,
    method: str = "central",
    h: float = None
) -> Dict[str, Any]:
    """
    Compute numerical derivatives using finite difference methods.
    
    Args:
        f_values: Function values or expression
        x_values: X coordinates (optional, assumed equally spaced if not provided)
        method: Finite difference method ('forward', 'backward', 'central', 'five_point')
        h: Step size (auto-computed if not provided)
        
    Returns:
        Dictionary with derivative values
    """
    f = np.array(f_values, dtype=float)
    
    if x_values is None:
        x = np.arange(len(f), dtype=float)
    else:
        x = np.array(x_values, dtype=float)
    
    if len(f) != len(x):
        raise ValueError(f"Length mismatch: f has {len(f)} values, x has {len(x)}")
    
    if h is None:
        h = x[1] - x[0] if len(x) > 1 else 1.0
    
    if method == "forward":
        # Forward difference: f'(x) ≈ (f(x+h) - f(x)) / h
        derivative = np.diff(f) / h
        x_deriv = x[:-1]
    
    elif method == "backward":
        # Backward difference: f'(x) ≈ (f(x) - f(x-h)) / h
        derivative = np.diff(f) / h
        x_deriv = x[1:]
    
    elif method == "central":
        # Central difference: f'(x) ≈ (f(x+h) - f(x-h)) / (2h)
        derivative = (f[2:] - f[:-2]) / (2 * h)
        x_deriv = x[1:-1]
    
    elif method == "five_point":
        # Five-point stencil: O(h^4) accuracy
        if len(f) < 5:
            raise ValueError("Five-point stencil requires at least 5 points")
        
        derivative = (-f[4:] + 8*f[3:-1] - 8*f[1:-3] + f[:-4]) / (12 * h)
        x_deriv = x[2:-2]
    
    else:
        raise ValueError(f"Unknown method: {method}")
    
    return {
        "derivative": derivative.tolist(),
        "x_values": x_deriv.tolist(),
        "method": method,
        "step_size": float(h)
    }


# Placeholder implementations for remaining calculus tools
def numerical_differentiation_advanced(**kwargs):
    """Richardson extrapolation for higher accuracy derivatives."""
    return {"error": "Not yet implemented", "tool": "numerical_differentiation_advanced"}


def numerical_integration(
    f_values: List[float] = None,
    x_values: List[float] = None,
    method: str = "trapz",
    a: float = None,
    b: float = None
) -> Dict[str, Any]:
    """Numerical integration using trapezoidal, Simpson's, or Romberg methods."""
    if f_values is None or x_values is None:
        raise ValueError("f_values and x_values are required")
    
    f = np.array(f_values, dtype=float)
    x = np.array(x_values, dtype=float)
    
    if method == "trapz":
        result = float(np.trapz(f, x))
        return {"integral": result, "method": "Trapezoidal"}
    
    elif method == "simpson":
        result = float(integrate.simpson(f, x=x))
        return {"integral": result, "method": "Simpson's Rule"}
    
    else:
        raise ValueError(f"Unknown method: {method}")


def symbolic_derivative(**kwargs):
    """Symbolic differentiation with simplification."""
    return {"error": "Not yet implemented", "tool": "symbolic_derivative"}


def partial_derivatives(**kwargs):
    """Compute gradient, Hessian, and Jacobian matrices."""
    return {"error": "Not yet implemented", "tool": "partial_derivatives"}


def limit_calculator(**kwargs):
    """Compute limits including L'Hôpital's rule."""
    return {"error": "Not yet implemented", "tool": "limit_calculator"}


def taylor_series(**kwargs):
    """Taylor and Maclaurin series expansion."""
    return {"error": "Not yet implemented", "tool": "taylor_series"}


def ode_solver(**kwargs):
    """Solve initial value ODEs using Euler, RK4, RK45 methods."""
    return {"error": "Not yet implemented", "tool": "ode_solver"}


def ode_boundary_value(**kwargs):
    """Solve boundary value problems."""
    return {"error": "Not yet implemented", "tool": "ode_boundary_value"}


# ============================================================================
# Tool Definitions
# ============================================================================

CALCULUS_TOOLS = [
    Tool(
        name="numerical_derivative",
        description="Compute numerical derivatives using finite difference methods (forward, backward, central, five-point)",
        inputSchema={
            "type": "object",
            "properties": {
                "f_values": {
                    "type": "array",
                    "items": {"type": "number"},
                    "description": "Function values"
                },
                "x_values": {
                    "type": "array",
                    "items": {"type": "number"},
                    "description": "X coordinates (optional)"
                },
                "method": {
                    "type": "string",
                    "enum": ["forward", "backward", "central", "five_point"],
                    "description": "Finite difference method"
                },
                "h": {
                    "type": "number",
                    "description": "Step size (optional)"
                }
            },
            "required": ["f_values"]
        }
    ),
    Tool(
        name="numerical_differentiation_advanced",
        description="Advanced numerical differentiation with Richardson extrapolation",
        inputSchema={"type": "object", "properties": {}}
    ),
    Tool(
        name="numerical_integration",
        description="Numerical integration using trapezoidal, Simpson's, or Romberg methods",
        inputSchema={
            "type": "object",
            "properties": {
                "f_values": {
                    "type": "array",
                    "items": {"type": "number"},
                    "description": "Function values"
                },
                "x_values": {
                    "type": "array",
                    "items": {"type": "number"},
                    "description": "X coordinates"
                },
                "method": {
                    "type": "string",
                    "enum": ["trapz", "simpson", "romberg"],
                    "description": "Integration method"
                }
            },
            "required": ["f_values", "x_values"]
        }
    ),
    Tool(
        name="symbolic_derivative",
        description="Symbolic differentiation with simplification",
        inputSchema={"type": "object", "properties": {}}
    ),
    Tool(
        name="partial_derivatives",
        description="Compute gradient, Hessian, and Jacobian matrices",
        inputSchema={"type": "object", "properties": {}}
    ),
    Tool(
        name="limit_calculator",
        description="Compute limits including L'Hôpital's rule",
        inputSchema={"type": "object", "properties": {}}
    ),
    Tool(
        name="taylor_series",
        description="Taylor and Maclaurin series expansion",
        inputSchema={"type": "object", "properties": {}}
    ),
    Tool(
        name="ode_solver",
        description="Solve initial value ODEs using Euler, RK4, RK45 methods",
        inputSchema={"type": "object", "properties": {}}
    ),
    Tool(
        name="ode_boundary_value",
        description="Solve boundary value problems",
        inputSchema={"type": "object", "properties": {}}
    )
]


# ============================================================================
# Tool Handlers
# ============================================================================

async def handle_numerical_derivative(arguments: Dict[str, Any]) -> CallToolResult:
    """Handle numerical_derivative tool calls."""
    try:
        result = numerical_derivative(
            f_values=arguments["f_values"],
            x_values=arguments.get("x_values"),
            method=arguments.get("method", "central"),
            h=arguments.get("h")
        )
        
        import json
        return CallToolResult(
            content=[TextContent(type="text", text=json.dumps(result, indent=2))],
            isError=False
        )
    except Exception as e:
        logger.error(f"Error in numerical_derivative: {e}", exc_info=True)
        return CallToolResult(
            content=[TextContent(type="text", text=f"Error: {str(e)}")],
            isError=True
        )


async def handle_numerical_differentiation_advanced(arguments: Dict[str, Any]) -> CallToolResult:
    """Handle numerical_differentiation_advanced tool calls."""
    result = numerical_differentiation_advanced(**arguments)
    import json
    return CallToolResult(
        content=[TextContent(type="text", text=json.dumps(result))],
        isError="error" in result
    )


async def handle_numerical_integration(arguments: Dict[str, Any]) -> CallToolResult:
    """Handle numerical_integration tool calls."""
    try:
        result = numerical_integration(
            f_values=arguments.get("f_values"),
            x_values=arguments.get("x_values"),
            method=arguments.get("method", "trapz")
        )
        
        import json
        return CallToolResult(
            content=[TextContent(type="text", text=json.dumps(result, indent=2))],
            isError=False
        )
    except Exception as e:
        logger.error(f"Error in numerical_integration: {e}", exc_info=True)
        return CallToolResult(
            content=[TextContent(type="text", text=f"Error: {str(e)}")],
            isError=True
        )


async def handle_symbolic_derivative(arguments: Dict[str, Any]) -> CallToolResult:
    """Handle symbolic_derivative tool calls."""
    result = symbolic_derivative(**arguments)
    import json
    return CallToolResult(
        content=[TextContent(type="text", text=json.dumps(result))],
        isError="error" in result
    )


async def handle_partial_derivatives(arguments: Dict[str, Any]) -> CallToolResult:
    """Handle partial_derivatives tool calls."""
    result = partial_derivatives(**arguments)
    import json
    return CallToolResult(
        content=[TextContent(type="text", text=json.dumps(result))],
        isError="error" in result
    )


async def handle_limit_calculator(arguments: Dict[str, Any]) -> CallToolResult:
    """Handle limit_calculator tool calls."""
    result = limit_calculator(**arguments)
    import json
    return CallToolResult(
        content=[TextContent(type="text", text=json.dumps(result))],
        isError="error" in result
    )


async def handle_taylor_series(arguments: Dict[str, Any]) -> CallToolResult:
    """Handle taylor_series tool calls."""
    result = taylor_series(**arguments)
    import json
    return CallToolResult(
        content=[TextContent(type="text", text=json.dumps(result))],
        isError="error" in result
    )


async def handle_ode_solver(arguments: Dict[str, Any]) -> CallToolResult:
    """Handle ode_solver tool calls."""
    result = ode_solver(**arguments)
    import json
    return CallToolResult(
        content=[TextContent(type="text", text=json.dumps(result))],
        isError="error" in result
    )


async def handle_ode_boundary_value(arguments: Dict[str, Any]) -> CallToolResult:
    """Handle ode_boundary_value tool calls."""
    result = ode_boundary_value(**arguments)
    import json
    return CallToolResult(
        content=[TextContent(type="text", text=json.dumps(result))],
        isError="error" in result
    )
