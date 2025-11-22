"""
Numerical Methods & Equation Solving Tools for Engineering Math Server.

This module provides 5 numerical methods tools for root finding, optimization, and curve fitting.
"""

import logging
import numpy as np
from scipy import optimize, interpolate
from typing import Any, Dict, List

from mcp.types import Tool, TextContent, CallToolResult

logger = logging.getLogger("engineering-math-server")


# ============================================================================
# Tool 16: Root Finding
# ============================================================================

def root_finding(
    method: str,
    a: float = None,
    b: float = None,
    x0: float = None,
    tolerance: float = 1e-6,
    max_iterations: int = 100,
    f_values: List[float] = None,
    x_values: List[float] = None
) -> Dict[str, Any]:
    """
    Find roots using bisection, Newton-Raphson, secant, or Brent's method.
    
    Note: This is a simplified implementation. The bisection method currently
    uses sin(x) as a placeholder function for demonstration. In a full implementation,
    this would accept function values or a function expression.
    
    Args:
        method: Root finding method ('bisection', 'newton', 'secant', 'brent')
        a, b: Interval endpoints for bisection/brent
        x0: Initial guess for iterative methods
        tolerance: Convergence tolerance
        max_iterations: Maximum iterations
        f_values: Function values (for interpolation-based methods)
        x_values: X coordinates
        
    Returns:
        Dictionary with root and convergence info
    """
    if method == "bisection":
        if a is None or b is None:
            raise ValueError("Bisection requires interval [a, b]")
        
        # NOTE: Placeholder implementation using sin(x) for demonstration
        # Full implementation would accept function expression or callable
        fa = np.sin(a)
        fb = np.sin(b)
        
        if fa * fb > 0:
            raise ValueError("Function must have opposite signs at endpoints")
        
        for iteration in range(max_iterations):
            c = (a + b) / 2
            fc = np.sin(c)
            
            if abs(fc) < tolerance:
                return {
                    "root": float(c),
                    "iterations": iteration + 1,
                    "method": "bisection",
                    "converged": True
                }
            
            if fa * fc < 0:
                b = c
                fb = fc
            else:
                a = c
                fa = fc
        
        return {
            "root": float((a + b) / 2),
            "iterations": max_iterations,
            "method": "bisection",
            "converged": False
        }
    
    else:
        return {"error": f"Method '{method}' not yet fully implemented"}


# Placeholder implementations
def system_of_equations_solver(**kwargs):
    """Solve non-linear systems using Newton's method or Broyden's method."""
    return {"error": "Not yet implemented", "tool": "system_of_equations_solver"}


def interpolation(
    x: List[float],
    y: List[float],
    x_new: List[float],
    method: str = "linear"
) -> Dict[str, Any]:
    """Interpolate data using linear, polynomial, or spline methods."""
    x_arr = np.array(x, dtype=float)
    y_arr = np.array(y, dtype=float)
    x_new_arr = np.array(x_new, dtype=float)
    
    if method == "linear":
        f = interpolate.interp1d(x_arr, y_arr, kind='linear', fill_value='extrapolate')
        y_new = f(x_new_arr)
        
        return {
            "interpolated_values": y_new.tolist(),
            "x_values": x_new_arr.tolist(),
            "method": "linear"
        }
    
    elif method == "cubic":
        f = interpolate.CubicSpline(x_arr, y_arr)
        y_new = f(x_new_arr)
        
        return {
            "interpolated_values": y_new.tolist(),
            "x_values": x_new_arr.tolist(),
            "method": "cubic spline"
        }
    
    else:
        raise ValueError(f"Unknown method: {method}")


def optimization_1d(**kwargs):
    """1D optimization using golden section or Brent's method."""
    return {"error": "Not yet implemented", "tool": "optimization_1d"}


def curve_fitting_advanced(**kwargs):
    """Non-linear least squares using Levenberg-Marquardt."""
    return {"error": "Not yet implemented", "tool": "curve_fitting_advanced"}


# ============================================================================
# Tool Definitions
# ============================================================================

NUMERICAL_METHODS_TOOLS = [
    Tool(
        name="root_finding",
        description="Find roots using bisection, Newton-Raphson, secant, or Brent's method",
        inputSchema={
            "type": "object",
            "properties": {
                "method": {
                    "type": "string",
                    "enum": ["bisection", "newton", "secant", "brent"],
                    "description": "Root finding method"
                },
                "a": {"type": "number", "description": "Interval start (bisection/brent)"},
                "b": {"type": "number", "description": "Interval end (bisection/brent)"},
                "x0": {"type": "number", "description": "Initial guess (iterative methods)"},
                "tolerance": {"type": "number", "description": "Convergence tolerance"},
                "max_iterations": {"type": "number", "description": "Maximum iterations"}
            },
            "required": ["method"]
        }
    ),
    Tool(
        name="system_of_equations_solver",
        description="Solve non-linear systems using Newton's method or Broyden's method",
        inputSchema={"type": "object", "properties": {}}
    ),
    Tool(
        name="interpolation",
        description="Interpolate data using linear, polynomial, or cubic spline methods",
        inputSchema={
            "type": "object",
            "properties": {
                "x": {
                    "type": "array",
                    "items": {"type": "number"},
                    "description": "Known x values"
                },
                "y": {
                    "type": "array",
                    "items": {"type": "number"},
                    "description": "Known y values"
                },
                "x_new": {
                    "type": "array",
                    "items": {"type": "number"},
                    "description": "X values to interpolate"
                },
                "method": {
                    "type": "string",
                    "enum": ["linear", "cubic", "polynomial"],
                    "description": "Interpolation method"
                }
            },
            "required": ["x", "y", "x_new"]
        }
    ),
    Tool(
        name="optimization_1d",
        description="1D optimization using golden section or Brent's method",
        inputSchema={"type": "object", "properties": {}}
    ),
    Tool(
        name="curve_fitting_advanced",
        description="Non-linear least squares using Levenberg-Marquardt algorithm",
        inputSchema={"type": "object", "properties": {}}
    )
]


# ============================================================================
# Tool Handlers
# ============================================================================

async def handle_root_finding(arguments: Dict[str, Any]) -> CallToolResult:
    """Handle root_finding tool calls."""
    try:
        result = root_finding(
            method=arguments["method"],
            a=arguments.get("a"),
            b=arguments.get("b"),
            x0=arguments.get("x0"),
            tolerance=arguments.get("tolerance", 1e-6),
            max_iterations=arguments.get("max_iterations", 100)
        )
        
        import json
        return CallToolResult(
            content=[TextContent(type="text", text=json.dumps(result, indent=2))],
            isError="error" in result
        )
    except Exception as e:
        logger.error(f"Error in root_finding: {e}", exc_info=True)
        return CallToolResult(
            content=[TextContent(type="text", text=f"Error: {str(e)}")],
            isError=True
        )


async def handle_system_of_equations_solver(arguments: Dict[str, Any]) -> CallToolResult:
    """Handle system_of_equations_solver tool calls."""
    result = system_of_equations_solver(**arguments)
    import json
    return CallToolResult(
        content=[TextContent(type="text", text=json.dumps(result))],
        isError="error" in result
    )


async def handle_interpolation(arguments: Dict[str, Any]) -> CallToolResult:
    """Handle interpolation tool calls."""
    try:
        result = interpolation(
            x=arguments["x"],
            y=arguments["y"],
            x_new=arguments["x_new"],
            method=arguments.get("method", "linear")
        )
        
        import json
        return CallToolResult(
            content=[TextContent(type="text", text=json.dumps(result, indent=2))],
            isError=False
        )
    except Exception as e:
        logger.error(f"Error in interpolation: {e}", exc_info=True)
        return CallToolResult(
            content=[TextContent(type="text", text=f"Error: {str(e)}")],
            isError=True
        )


async def handle_optimization_1d(arguments: Dict[str, Any]) -> CallToolResult:
    """Handle optimization_1d tool calls."""
    result = optimization_1d(**arguments)
    import json
    return CallToolResult(
        content=[TextContent(type="text", text=json.dumps(result))],
        isError="error" in result
    )


async def handle_curve_fitting_advanced(arguments: Dict[str, Any]) -> CallToolResult:
    """Handle curve_fitting_advanced tool calls."""
    result = curve_fitting_advanced(**arguments)
    import json
    return CallToolResult(
        content=[TextContent(type="text", text=json.dumps(result))],
        isError="error" in result
    )
