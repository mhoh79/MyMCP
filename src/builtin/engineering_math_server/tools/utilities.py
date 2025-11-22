"""
Cross-Cutting Utility Tools for Engineering Math Server.

This module provides 7 utility tools for unit conversion, precision arithmetic, and error propagation.
"""

import logging
from decimal import Decimal, getcontext
from typing import Any, Dict, List
import re

from mcp.types import Tool, TextContent, CallToolResult

logger = logging.getLogger("engineering-math-server")


# ============================================================================
# Tool 25: Unit Converter (Enhanced)
# ============================================================================

# Unit conversion tables
UNIT_CONVERSIONS = {
    # Length
    "length": {
        "m": 1.0,
        "km": 1000.0,
        "cm": 0.01,
        "mm": 0.001,
        "ft": 0.3048,
        "in": 0.0254,
        "mi": 1609.34,
    },
    # Weight/Mass
    "mass": {
        "kg": 1.0,
        "g": 0.001,
        "mg": 0.000001,
        "lb": 0.453592,
        "oz": 0.0283495,
        "ton": 1000.0,
    },
    # Temperature (special handling needed)
    "temperature": {
        "C": "celsius",
        "F": "fahrenheit",
        "K": "kelvin",
    },
    # Volume
    "volume": {
        "L": 1.0,
        "mL": 0.001,
        "m3": 1000.0,
        "gal": 3.78541,
        "qt": 0.946353,
        "pt": 0.473176,
    },
    # Time
    "time": {
        "s": 1.0,
        "min": 60.0,
        "h": 3600.0,
        "day": 86400.0,
        "week": 604800.0,
    },
    # Angles
    "angle": {
        "rad": 1.0,
        "deg": 0.0174533,  # pi/180
        "grad": 0.015708,  # pi/200
    },
    # Pressure
    "pressure": {
        "Pa": 1.0,
        "kPa": 1000.0,
        "bar": 100000.0,
        "psi": 6894.76,
        "atm": 101325.0,
        "mmHg": 133.322,
    },
    # Energy
    "energy": {
        "J": 1.0,
        "kJ": 1000.0,
        "kWh": 3600000.0,
        "BTU": 1055.06,
        "cal": 4.184,
        "kcal": 4184.0,
    },
    # Power
    "power": {
        "W": 1.0,
        "kW": 1000.0,
        "hp": 745.7,
        "BTU/hr": 0.293071,
    },
    # Force
    "force": {
        "N": 1.0,
        "kN": 1000.0,
        "lbf": 4.44822,
        "kgf": 9.80665,
    },
}


def unit_converter(
    value: float,
    from_unit: str,
    to_unit: str,
    category: str = None
) -> Dict[str, Any]:
    """
    Convert units across multiple categories.
    
    Args:
        value: Value to convert
        from_unit: Source unit
        to_unit: Target unit
        category: Unit category (auto-detected if not provided)
        
    Returns:
        Dictionary with converted value
    """
    # Special handling for temperature
    if category == "temperature" or from_unit in ["C", "F", "K"]:
        if from_unit == "C" and to_unit == "F":
            result = value * 9/5 + 32
        elif from_unit == "F" and to_unit == "C":
            result = (value - 32) * 5/9
        elif from_unit == "C" and to_unit == "K":
            result = value + 273.15
        elif from_unit == "K" and to_unit == "C":
            result = value - 273.15
        elif from_unit == "F" and to_unit == "K":
            result = (value - 32) * 5/9 + 273.15
        elif from_unit == "K" and to_unit == "F":
            result = (value - 273.15) * 9/5 + 32
        else:
            result = value  # Same unit
        
        return {
            "value": float(result),
            "from_unit": from_unit,
            "to_unit": to_unit,
            "category": "temperature"
        }
    
    # Find category if not provided
    if category is None:
        for cat, units in UNIT_CONVERSIONS.items():
            if from_unit in units and to_unit in units:
                category = cat
                break
        
        if category is None:
            raise ValueError(f"Cannot find category for units: {from_unit}, {to_unit}")
    
    if category not in UNIT_CONVERSIONS:
        raise ValueError(f"Unknown category: {category}")
    
    units = UNIT_CONVERSIONS[category]
    
    if from_unit not in units:
        raise ValueError(f"Unknown unit '{from_unit}' in category '{category}'")
    
    if to_unit not in units:
        raise ValueError(f"Unknown unit '{to_unit}' in category '{category}'")
    
    # Convert to base unit, then to target unit
    base_value = value * units[from_unit]
    result = base_value / units[to_unit]
    
    return {
        "value": float(result),
        "from_unit": from_unit,
        "to_unit": to_unit,
        "category": category
    }


# ============================================================================
# Tool 26: Precision Calculator
# ============================================================================

def precision_calculator(
    operation: str,
    value_a: str,
    value_b: str = None,
    precision: int = 50
) -> Dict[str, Any]:
    """
    Perform arbitrary precision arithmetic.
    
    Args:
        operation: Arithmetic operation ('add', 'subtract', 'multiply', 'divide')
        value_a: First value (as string for precision)
        value_b: Second value (as string for precision)
        precision: Decimal precision
        
    Returns:
        Dictionary with high-precision result
    """
    getcontext().prec = precision
    
    a = Decimal(value_a)
    
    if operation == "sqrt":
        result = a.sqrt()
        return {
            "result": str(result),
            "precision": precision,
            "operation": operation
        }
    
    if value_b is None:
        raise ValueError(f"Operation '{operation}' requires value_b")
    
    b = Decimal(value_b)
    
    if operation == "add":
        result = a + b
    elif operation == "subtract":
        result = a - b
    elif operation == "multiply":
        result = a * b
    elif operation == "divide":
        if b == 0:
            raise ValueError("Division by zero")
        result = a / b
    else:
        raise ValueError(f"Unknown operation: {operation}")
    
    return {
        "result": str(result),
        "precision": precision,
        "operation": operation
    }


# Placeholder implementations
def expression_parser(**kwargs):
    """Parse and evaluate mathematical expressions."""
    return {"error": "Not yet implemented", "tool": "expression_parser"}


def symbolic_simplification(**kwargs):
    """Simplify algebraic and trigonometric expressions."""
    return {"error": "Not yet implemented", "tool": "symbolic_simplification"}


def equation_balancer(**kwargs):
    """Balance chemical and physical equations."""
    return {"error": "Not yet implemented", "tool": "equation_balancer"}


def significant_figures(
    value: float,
    sig_figs: int
) -> Dict[str, Any]:
    """
    Round value to specified significant figures.
    
    Args:
        value: Value to round
        sig_figs: Number of significant figures
        
    Returns:
        Dictionary with rounded value
    """
    if value == 0:
        return {"result": 0.0, "significant_figures": sig_figs}
    
    from math import log10, floor
    
    # Calculate the order of magnitude
    magnitude = floor(log10(abs(value)))
    
    # Round to sig_figs
    rounded = round(value, -magnitude + sig_figs - 1)
    
    return {
        "result": float(rounded),
        "significant_figures": sig_figs,
        "original_value": float(value)
    }


def error_propagation(**kwargs):
    """Calculate uncertainty propagation through calculations."""
    return {"error": "Not yet implemented", "tool": "error_propagation"}


# ============================================================================
# Tool Definitions
# ============================================================================

UTILITY_TOOLS = [
    Tool(
        name="unit_converter",
        description="Convert units across multiple categories (length, mass, temperature, volume, time, angles, pressure, energy, power, force)",
        inputSchema={
            "type": "object",
            "properties": {
                "value": {
                    "type": "number",
                    "description": "Value to convert"
                },
                "from_unit": {
                    "type": "string",
                    "description": "Source unit"
                },
                "to_unit": {
                    "type": "string",
                    "description": "Target unit"
                },
                "category": {
                    "type": "string",
                    "enum": ["length", "mass", "temperature", "volume", "time", "angle", "pressure", "energy", "power", "force"],
                    "description": "Unit category (optional, auto-detected)"
                }
            },
            "required": ["value", "from_unit", "to_unit"]
        }
    ),
    Tool(
        name="precision_calculator",
        description="Perform arbitrary precision arithmetic to avoid floating point errors",
        inputSchema={
            "type": "object",
            "properties": {
                "operation": {
                    "type": "string",
                    "enum": ["add", "subtract", "multiply", "divide", "sqrt"],
                    "description": "Arithmetic operation"
                },
                "value_a": {
                    "type": "string",
                    "description": "First value (as string for precision)"
                },
                "value_b": {
                    "type": "string",
                    "description": "Second value (as string for precision)"
                },
                "precision": {
                    "type": "number",
                    "description": "Decimal precision (default: 50)"
                }
            },
            "required": ["operation", "value_a"]
        }
    ),
    Tool(
        name="expression_parser",
        description="Parse and evaluate mathematical expressions with variables",
        inputSchema={"type": "object", "properties": {}}
    ),
    Tool(
        name="symbolic_simplification",
        description="Simplify algebraic and trigonometric expressions",
        inputSchema={"type": "object", "properties": {}}
    ),
    Tool(
        name="equation_balancer",
        description="Balance chemical and physical equations",
        inputSchema={"type": "object", "properties": {}}
    ),
    Tool(
        name="significant_figures",
        description="Round values to specified significant figures",
        inputSchema={
            "type": "object",
            "properties": {
                "value": {
                    "type": "number",
                    "description": "Value to round"
                },
                "sig_figs": {
                    "type": "number",
                    "description": "Number of significant figures"
                }
            },
            "required": ["value", "sig_figs"]
        }
    ),
    Tool(
        name="error_propagation",
        description="Calculate uncertainty propagation through calculations",
        inputSchema={"type": "object", "properties": {}}
    )
]


# ============================================================================
# Tool Handlers
# ============================================================================

async def handle_unit_converter(arguments: Dict[str, Any]) -> CallToolResult:
    """Handle unit_converter tool calls."""
    try:
        result = unit_converter(
            value=arguments["value"],
            from_unit=arguments["from_unit"],
            to_unit=arguments["to_unit"],
            category=arguments.get("category")
        )
        
        import json
        return CallToolResult(
            content=[TextContent(type="text", text=json.dumps(result, indent=2))],
            isError=False
        )
    except Exception as e:
        logger.error(f"Error in unit_converter: {e}", exc_info=True)
        return CallToolResult(
            content=[TextContent(type="text", text=f"Error: {str(e)}")],
            isError=True
        )


async def handle_precision_calculator(arguments: Dict[str, Any]) -> CallToolResult:
    """Handle precision_calculator tool calls."""
    try:
        result = precision_calculator(
            operation=arguments["operation"],
            value_a=arguments["value_a"],
            value_b=arguments.get("value_b"),
            precision=arguments.get("precision", 50)
        )
        
        import json
        return CallToolResult(
            content=[TextContent(type="text", text=json.dumps(result, indent=2))],
            isError=False
        )
    except Exception as e:
        logger.error(f"Error in precision_calculator: {e}", exc_info=True)
        return CallToolResult(
            content=[TextContent(type="text", text=f"Error: {str(e)}")],
            isError=True
        )


async def handle_expression_parser(arguments: Dict[str, Any]) -> CallToolResult:
    """Handle expression_parser tool calls."""
    result = expression_parser(**arguments)
    import json
    return CallToolResult(
        content=[TextContent(type="text", text=json.dumps(result))],
        isError="error" in result
    )


async def handle_symbolic_simplification(arguments: Dict[str, Any]) -> CallToolResult:
    """Handle symbolic_simplification tool calls."""
    result = symbolic_simplification(**arguments)
    import json
    return CallToolResult(
        content=[TextContent(type="text", text=json.dumps(result))],
        isError="error" in result
    )


async def handle_equation_balancer(arguments: Dict[str, Any]) -> CallToolResult:
    """Handle equation_balancer tool calls."""
    result = equation_balancer(**arguments)
    import json
    return CallToolResult(
        content=[TextContent(type="text", text=json.dumps(result))],
        isError="error" in result
    )


async def handle_significant_figures(arguments: Dict[str, Any]) -> CallToolResult:
    """Handle significant_figures tool calls."""
    try:
        result = significant_figures(
            value=arguments["value"],
            sig_figs=arguments["sig_figs"]
        )
        
        import json
        return CallToolResult(
            content=[TextContent(type="text", text=json.dumps(result, indent=2))],
            isError=False
        )
    except Exception as e:
        logger.error(f"Error in significant_figures: {e}", exc_info=True)
        return CallToolResult(
            content=[TextContent(type="text", text=f"Error: {str(e)}")],
            isError=True
        )


async def handle_error_propagation(arguments: Dict[str, Any]) -> CallToolResult:
    """Handle error_propagation tool calls."""
    result = error_propagation(**arguments)
    import json
    return CallToolResult(
        content=[TextContent(type="text", text=json.dumps(result))],
        isError="error" in result
    )
