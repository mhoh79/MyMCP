"""
Polynomial & Algebraic Structures Tools for Engineering Math Server.

This module provides 4 polynomial tools for arithmetic, roots, interpolation, and analysis.
"""

import logging
import numpy as np
from typing import Any, Dict, List

from mcp.types import Tool, TextContent, CallToolResult

logger = logging.getLogger("engineering-math-server")


# ============================================================================
# Tool 21: Polynomial Arithmetic
# ============================================================================

def polynomial_arithmetic(
    operation: str,
    poly_a: List[float],
    poly_b: List[float] = None
) -> Dict[str, Any]:
    """
    Perform polynomial arithmetic: add, subtract, multiply, divide.
    
    Args:
        operation: Operation to perform ('add', 'subtract', 'multiply', 'divide')
        poly_a: First polynomial coefficients (highest degree first)
        poly_b: Second polynomial coefficients
        
    Returns:
        Dictionary with result polynomial
    """
    p1 = np.poly1d(poly_a)
    
    if operation == "degree":
        return {
            "degree": int(p1.order),
            "coefficients": poly_a
        }
    
    if poly_b is None:
        raise ValueError(f"Operation '{operation}' requires poly_b")
    
    p2 = np.poly1d(poly_b)
    
    if operation == "add":
        result = p1 + p2
    elif operation == "subtract":
        result = p1 - p2
    elif operation == "multiply":
        result = p1 * p2
    elif operation == "divide":
        quotient, remainder = np.polydiv(poly_a, poly_b)
        return {
            "quotient": quotient.tolist(),
            "remainder": remainder.tolist(),
            "operation": "divide"
        }
    else:
        raise ValueError(f"Unknown operation: {operation}")
    
    return {
        "result": result.coeffs.tolist(),
        "degree": int(result.order),
        "operation": operation
    }


# ============================================================================
# Tool 22: Polynomial Roots
# ============================================================================

def polynomial_roots(
    coefficients: List[float],
    method: str = "numpy"
) -> Dict[str, Any]:
    """
    Find polynomial roots using analytical or numerical methods.
    
    Args:
        coefficients: Polynomial coefficients (highest degree first)
        method: Solution method ('numpy', 'analytical')
        
    Returns:
        Dictionary with roots
    """
    coeffs = np.array(coefficients, dtype=float)
    
    if method == "numpy":
        roots = np.roots(coeffs)
        
        # Separate real and complex roots
        real_roots = []
        complex_roots = []
        
        for root in roots:
            if abs(root.imag) < 1e-10:
                real_roots.append(float(root.real))
            else:
                complex_roots.append({
                    "real": float(root.real),
                    "imag": float(root.imag)
                })
        
        return {
            "real_roots": real_roots,
            "complex_roots": complex_roots,
            "total_roots": len(roots),
            "method": "numpy"
        }
    
    else:
        raise ValueError(f"Unknown method: {method}")


# Placeholder implementations
def polynomial_interpolation(**kwargs):
    """Construct polynomial through points using Lagrange or Newton method."""
    return {"error": "Not yet implemented", "tool": "polynomial_interpolation"}


def polynomial_analysis(**kwargs):
    """Analyze polynomial: critical points, inflection points, end behavior."""
    return {"error": "Not yet implemented", "tool": "polynomial_analysis"}


# ============================================================================
# Tool Definitions
# ============================================================================

POLYNOMIAL_TOOLS = [
    Tool(
        name="polynomial_arithmetic",
        description="Perform polynomial arithmetic: add, subtract, multiply, divide",
        inputSchema={
            "type": "object",
            "properties": {
                "operation": {
                    "type": "string",
                    "enum": ["add", "subtract", "multiply", "divide", "degree"],
                    "description": "Arithmetic operation"
                },
                "poly_a": {
                    "type": "array",
                    "items": {"type": "number"},
                    "description": "First polynomial coefficients (highest degree first)"
                },
                "poly_b": {
                    "type": "array",
                    "items": {"type": "number"},
                    "description": "Second polynomial coefficients"
                }
            },
            "required": ["operation", "poly_a"]
        }
    ),
    Tool(
        name="polynomial_roots",
        description="Find polynomial roots using analytical or numerical methods",
        inputSchema={
            "type": "object",
            "properties": {
                "coefficients": {
                    "type": "array",
                    "items": {"type": "number"},
                    "description": "Polynomial coefficients (highest degree first)"
                },
                "method": {
                    "type": "string",
                    "enum": ["numpy", "analytical"],
                    "description": "Solution method"
                }
            },
            "required": ["coefficients"]
        }
    ),
    Tool(
        name="polynomial_interpolation",
        description="Construct polynomial through points using Lagrange or Newton method",
        inputSchema={"type": "object", "properties": {}}
    ),
    Tool(
        name="polynomial_analysis",
        description="Analyze polynomial: critical points, inflection points, end behavior",
        inputSchema={"type": "object", "properties": {}}
    )
]


# ============================================================================
# Tool Handlers
# ============================================================================

async def handle_polynomial_arithmetic(arguments: Dict[str, Any]) -> CallToolResult:
    """Handle polynomial_arithmetic tool calls."""
    try:
        result = polynomial_arithmetic(
            operation=arguments["operation"],
            poly_a=arguments["poly_a"],
            poly_b=arguments.get("poly_b")
        )
        
        import json
        return CallToolResult(
            content=[TextContent(type="text", text=json.dumps(result, indent=2))],
            isError=False
        )
    except Exception as e:
        logger.error(f"Error in polynomial_arithmetic: {e}", exc_info=True)
        return CallToolResult(
            content=[TextContent(type="text", text=f"Error: {str(e)}")],
            isError=True
        )


async def handle_polynomial_roots(arguments: Dict[str, Any]) -> CallToolResult:
    """Handle polynomial_roots tool calls."""
    try:
        result = polynomial_roots(
            coefficients=arguments["coefficients"],
            method=arguments.get("method", "numpy")
        )
        
        import json
        return CallToolResult(
            content=[TextContent(type="text", text=json.dumps(result, indent=2))],
            isError=False
        )
    except Exception as e:
        logger.error(f"Error in polynomial_roots: {e}", exc_info=True)
        return CallToolResult(
            content=[TextContent(type="text", text=f"Error: {str(e)}")],
            isError=True
        )


async def handle_polynomial_interpolation(arguments: Dict[str, Any]) -> CallToolResult:
    """Handle polynomial_interpolation tool calls."""
    result = polynomial_interpolation(**arguments)
    import json
    return CallToolResult(
        content=[TextContent(type="text", text=json.dumps(result))],
        isError="error" in result
    )


async def handle_polynomial_analysis(arguments: Dict[str, Any]) -> CallToolResult:
    """Handle polynomial_analysis tool calls."""
    result = polynomial_analysis(**arguments)
    import json
    return CallToolResult(
        content=[TextContent(type="text", text=json.dumps(result))],
        isError="error" in result
    )
