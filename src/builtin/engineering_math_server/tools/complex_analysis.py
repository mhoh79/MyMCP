"""
Complex Analysis Tools for Engineering Math Server.

This module provides 4 complex number analysis tools for AC circuit analysis,
control system pole/zero analysis, frequency response calculations, and signal processing.
"""

import logging
import numpy as np
import cmath
from typing import Any, Dict, List, Tuple, Union

from mcp.types import Tool, TextContent, CallToolResult

logger = logging.getLogger("engineering-math-server")


# ============================================================================
# Helper Functions
# ============================================================================

def parse_complex_number(s: str) -> complex:
    """
    Parse complex number from string in various formats.
    
    Supported formats:
    - Rectangular: '3+4j', '3-4j', '3.5+2.1j'
    - Polar: '5∠30°', '5∠0.524rad', '5∠30deg'
    - Exponential: implicit via polar conversion
    
    Args:
        s: String representation of complex number
        
    Returns:
        Complex number
    """
    s = s.strip()
    
    # Polar format: magnitude∠angle
    if '∠' in s:
        parts = s.split('∠')
        magnitude = float(parts[0])
        angle_str = parts[1].strip()
        
        # Check for degree symbol or 'deg'
        if '°' in angle_str or 'deg' in angle_str.lower():
            angle = float(angle_str.replace('°', '').replace('deg', '').replace('Deg', '').strip())
            angle_rad = np.deg2rad(angle)
        else:
            # Assume radians
            angle_rad = float(angle_str.replace('rad', '').strip())
        
        return cmath.rect(magnitude, angle_rad)
    
    # Rectangular format: a+bj
    else:
        return complex(s)


def format_complex_result(z: complex, angle_unit: str = "degrees") -> str:
    """
    Format complex number result with both rectangular and polar forms.
    
    Args:
        z: Complex number
        angle_unit: 'degrees' or 'radians'
        
    Returns:
        Formatted string with multiple representations
    """
    magnitude = abs(z)
    phase_rad = cmath.phase(z)
    
    if angle_unit == "degrees":
        phase = np.rad2deg(phase_rad)
        angle_str = f"{phase:.4f}°"
    else:
        phase = phase_rad
        angle_str = f"{phase:.4f} rad"
    
    output = f"""Complex Number Result:

Rectangular Form: {z.real:.4f} + {z.imag:.4f}j
Polar Form: {magnitude:.4f}∠{angle_str}
Exponential Form: {magnitude:.4f}e^(i·{phase_rad:.4f})

Magnitude: {magnitude:.4f}
Phase: {angle_str}
"""
    return output


# ============================================================================
# Tool 1: Complex Operations
# ============================================================================

def complex_operations(
    z1: str,
    operation: str,
    z2: str = None,
    angle_unit: str = "degrees"
) -> Dict[str, Any]:
    """
    Perform complex number arithmetic and conversions.
    
    Args:
        z1: First complex number (rectangular or polar)
        operation: Operation to perform
        z2: Second complex number (optional for some operations)
        angle_unit: Unit for angles ('degrees' or 'radians')
        
    Returns:
        Dictionary with operation result
    """
    # Parse first complex number
    c1 = parse_complex_number(z1)
    
    # Single-operand operations
    if operation == "conjugate":
        result = c1.conjugate()
        return {
            "result": complex(result),
            "rectangular": f"{result.real:.4f} + {result.imag:.4f}j",
            "magnitude": abs(result),
            "phase_rad": cmath.phase(result),
            "phase_deg": np.rad2deg(cmath.phase(result))
        }
    
    elif operation == "magnitude":
        result = abs(c1)
        return {
            "magnitude": result,
            "input": z1
        }
    
    elif operation == "phase":
        phase_rad = cmath.phase(c1)
        phase_deg = np.rad2deg(phase_rad)
        return {
            "phase_radians": phase_rad,
            "phase_degrees": phase_deg,
            "input": z1
        }
    
    elif operation == "rect_to_polar":
        magnitude = abs(c1)
        phase_rad = cmath.phase(c1)
        
        if angle_unit == "degrees":
            phase = np.rad2deg(phase_rad)
            return {
                "magnitude": magnitude,
                "phase_degrees": phase,
                "polar_form": f"{magnitude:.4f}∠{phase:.4f}°"
            }
        else:
            return {
                "magnitude": magnitude,
                "phase_radians": phase_rad,
                "polar_form": f"{magnitude:.4f}∠{phase_rad:.4f} rad"
            }
    
    elif operation == "polar_to_rect":
        return {
            "real": c1.real,
            "imaginary": c1.imag,
            "rectangular_form": f"{c1.real:.4f} + {c1.imag:.4f}j"
        }
    
    # Binary operations
    if z2 is None:
        raise ValueError(f"Operation '{operation}' requires z2")
    
    c2 = parse_complex_number(z2)
    
    if operation == "add":
        result = c1 + c2
    elif operation == "subtract":
        result = c1 - c2
    elif operation == "multiply":
        result = c1 * c2
    elif operation == "divide":
        if abs(c2) < 1e-15:
            raise ValueError("Division by zero")
        result = c1 / c2
    elif operation == "power":
        # z2 should be a real number for standard power operation
        if isinstance(c2, complex):
            # Complex power
            result = c1 ** c2
        else:
            result = c1 ** c2
    else:
        raise ValueError(f"Unknown operation: {operation}")
    
    # Format result
    magnitude = abs(result)
    phase_rad = cmath.phase(result)
    
    if angle_unit == "degrees":
        phase = np.rad2deg(phase_rad)
        angle_str = f"{phase:.4f}°"
    else:
        phase = phase_rad
        angle_str = f"{phase:.4f} rad"
    
    return {
        "result": complex(result),
        "rectangular": f"{result.real:.4f} + {result.imag:.4f}j",
        "polar": f"{magnitude:.4f}∠{angle_str}",
        "magnitude": magnitude,
        "phase_rad": phase_rad,
        "phase_deg": np.rad2deg(phase_rad),
        "operation": operation
    }


# ============================================================================
# Tool 2: Complex Functions
# ============================================================================

def complex_functions(
    z: str,
    function: str,
    **kwargs
) -> Dict[str, Any]:
    """
    Compute complex elementary, trigonometric, and hyperbolic functions.
    
    Args:
        z: Complex number input
        function: Function to compute
        **kwargs: Additional function-specific parameters
        
    Returns:
        Dictionary with function result
    """
    c = parse_complex_number(z)
    
    # Elementary functions
    if function == "exp":
        result = cmath.exp(c)
    elif function == "log":
        if abs(c) < 1e-15:
            raise ValueError("Logarithm of zero is undefined")
        result = cmath.log(c)
    elif function == "sqrt":
        result = cmath.sqrt(c)
    
    # Trigonometric functions
    elif function == "sin":
        result = cmath.sin(c)
    elif function == "cos":
        result = cmath.cos(c)
    elif function == "tan":
        result = cmath.tan(c)
    elif function == "asin":
        result = cmath.asin(c)
    elif function == "acos":
        result = cmath.acos(c)
    elif function == "atan":
        result = cmath.atan(c)
    
    # Hyperbolic functions
    elif function == "sinh":
        result = cmath.sinh(c)
    elif function == "cosh":
        result = cmath.cosh(c)
    elif function == "tanh":
        result = cmath.tanh(c)
    elif function == "asinh":
        result = cmath.asinh(c)
    elif function == "acosh":
        result = cmath.acosh(c)
    elif function == "atanh":
        result = cmath.atanh(c)
    
    else:
        raise ValueError(f"Unknown function: {function}")
    
    # Format result
    magnitude = abs(result)
    phase_rad = cmath.phase(result)
    phase_deg = np.rad2deg(phase_rad)
    
    return {
        "result": complex(result),
        "rectangular": f"{result.real:.6f} + {result.imag:.6f}j",
        "polar": f"{magnitude:.6f}∠{phase_deg:.4f}°",
        "magnitude": magnitude,
        "phase_rad": phase_rad,
        "phase_deg": phase_deg,
        "function": function,
        "input": z
    }


# ============================================================================
# Tool 3: Roots of Unity
# ============================================================================

def roots_of_unity(
    n: int,
    z: str = None,
    root_index: int = None
) -> Dict[str, Any]:
    """
    Calculate nth roots of unity or nth roots of any complex number.
    
    Args:
        n: Order of roots (must be positive integer)
        z: Complex number to find roots of (default: 1, for roots of unity)
        root_index: Specific root index to return (0 to n-1), or None for all
        
    Returns:
        Dictionary with roots information
    """
    if n <= 0:
        raise ValueError("n must be a positive integer")
    
    # Parse input number (default to 1 for roots of unity)
    if z is None:
        c = complex(1, 0)
        is_unity = True
    else:
        c = parse_complex_number(z)
        is_unity = False
    
    # Calculate magnitude and phase of input
    r = abs(c)
    theta = cmath.phase(c)
    
    # Calculate all nth roots
    roots = []
    for k in range(n):
        # r^(1/n) * e^(i*(theta + 2πk)/n)
        root_magnitude = r ** (1/n)
        root_phase = (theta + 2 * np.pi * k) / n
        root = cmath.rect(root_magnitude, root_phase)
        
        roots.append({
            "index": k,
            "value": complex(root),
            "rectangular": f"{root.real:.6f} + {root.imag:.6f}j",
            "polar": f"{abs(root):.6f}∠{np.rad2deg(cmath.phase(root)):.4f}°",
            "magnitude": abs(root),
            "phase_rad": cmath.phase(root),
            "phase_deg": np.rad2deg(cmath.phase(root))
        })
    
    # Calculate properties
    angular_spacing = 360.0 / n
    
    # Sum of roots (should be 0 for n > 1 for roots of unity)
    root_sum = sum(r["value"] for r in roots)
    
    # Product of roots
    root_product = 1
    for r in roots:
        root_product *= r["value"]
    
    result = {
        "n": n,
        "input": z if z else "1+0j",
        "is_roots_of_unity": is_unity,
        "number_of_roots": n,
        "angular_spacing_degrees": angular_spacing,
        "roots": roots
    }
    
    if is_unity:
        result["sum_of_roots"] = {
            "value": complex(root_sum),
            "magnitude": abs(root_sum),
            "note": "Sum is 0 for n > 1" if n > 1 else "Sum is 1 for n = 1"
        }
        result["product_of_roots"] = {
            "value": complex(root_product),
            "magnitude": abs(root_product),
            "expected": (-1) ** (n - 1)
        }
    
    # If specific root requested, highlight it
    if root_index is not None:
        if 0 <= root_index < n:
            result["requested_root"] = roots[root_index]
        else:
            raise ValueError(f"root_index must be between 0 and {n-1}")
    
    return result


# ============================================================================
# Tool 4: Complex Conjugate Operations
# ============================================================================

def complex_conjugate_operations(
    z: str,
    operation: str,
    z2: str = None
) -> Dict[str, Any]:
    """
    Perform operations involving complex conjugates.
    
    Args:
        z: Complex number
        operation: Operation to perform
        z2: Second complex number (for binary operations)
        
    Returns:
        Dictionary with operation result
    """
    c = parse_complex_number(z)
    
    if operation == "conjugate":
        conj = c.conjugate()
        return {
            "original": complex(c),
            "conjugate": complex(conj),
            "original_rect": f"{c.real:.4f} + {c.imag:.4f}j",
            "conjugate_rect": f"{conj.real:.4f} + {conj.imag:.4f}j",
            "magnitude_unchanged": abs(c),
            "phase_original": np.rad2deg(cmath.phase(c)),
            "phase_conjugate": np.rad2deg(cmath.phase(conj))
        }
    
    elif operation == "real_part":
        # Re(z) = (z + z*)/2
        real_part = (c + c.conjugate()) / 2
        return {
            "real_part": real_part.real,
            "verification": c.real,
            "input": z
        }
    
    elif operation == "imaginary_part":
        # Im(z) = (z - z*)/(2i)
        imag_part = (c - c.conjugate()) / (2j)
        return {
            "imaginary_part": imag_part.real,  # Result is real
            "verification": c.imag,
            "input": z
        }
    
    elif operation == "magnitude_squared":
        # |z|² = z·z*
        mag_sq = (c * c.conjugate()).real
        return {
            "magnitude_squared": mag_sq,
            "magnitude": np.sqrt(mag_sq),
            "verification": abs(c) ** 2,
            "input": z
        }
    
    elif operation == "conjugate_product":
        if z2 is None:
            raise ValueError("Operation 'conjugate_product' requires z2")
        
        c2 = parse_complex_number(z2)
        # z1 * conj(z2)
        result = c * c2.conjugate()
        
        return {
            "result": complex(result),
            "rectangular": f"{result.real:.4f} + {result.imag:.4f}j",
            "magnitude": abs(result),
            "phase_deg": np.rad2deg(cmath.phase(result)),
            "z1": z,
            "z2": z2,
            "application": "Used in power calculations (S = V·I*)"
        }
    
    elif operation == "conjugate_sum":
        if z2 is None:
            raise ValueError("Operation 'conjugate_sum' requires z2")
        
        c2 = parse_complex_number(z2)
        # Conjugate of sum: (z1 + z2)*
        sum_conj = (c + c2).conjugate()
        # Sum of conjugates: z1* + z2*
        conj_sum = c.conjugate() + c2.conjugate()
        
        return {
            "conjugate_of_sum": complex(sum_conj),
            "sum_of_conjugates": complex(conj_sum),
            "are_equal": abs(sum_conj - conj_sum) < 1e-10,
            "property": "(z1 + z2)* = z1* + z2*"
        }
    
    elif operation == "conjugate_product_property":
        if z2 is None:
            raise ValueError("Operation 'conjugate_product_property' requires z2")
        
        c2 = parse_complex_number(z2)
        # Conjugate of product: (z1 · z2)*
        prod_conj = (c * c2).conjugate()
        # Product of conjugates: z1* · z2*
        conj_prod = c.conjugate() * c2.conjugate()
        
        return {
            "conjugate_of_product": complex(prod_conj),
            "product_of_conjugates": complex(conj_prod),
            "are_equal": abs(prod_conj - conj_prod) < 1e-10,
            "property": "(z1 · z2)* = z1* · z2*"
        }
    
    else:
        raise ValueError(f"Unknown operation: {operation}")


# ============================================================================
# Tool Definitions
# ============================================================================

COMPLEX_ANALYSIS_TOOLS = [
    Tool(
        name="complex_operations",
        description="""Perform complex number arithmetic and conversions.
        
Supports:
- Arithmetic: add, subtract, multiply, divide, power
- Conversions: rectangular ↔ polar ↔ exponential
- Properties: magnitude, phase, conjugate

Input formats:
- Rectangular: '3+4j' or '3-4j'
- Polar: '5∠30°' or '5∠0.524rad'
- Mixed: Automatic detection and conversion

Applications:
- AC circuit impedance calculations
- Phasor analysis
- Control system pole/zero analysis
- Frequency domain operations""",
        inputSchema={
            "type": "object",
            "properties": {
                "z1": {
                    "type": "string",
                    "description": "First complex number (rectangular or polar format)"
                },
                "operation": {
                    "type": "string",
                    "enum": [
                        "add", "subtract", "multiply", "divide", "power",
                        "conjugate", "magnitude", "phase",
                        "rect_to_polar", "polar_to_rect"
                    ],
                    "description": "Operation to perform"
                },
                "z2": {
                    "type": "string",
                    "description": "Second complex number (required for binary operations)"
                },
                "angle_unit": {
                    "type": "string",
                    "enum": ["degrees", "radians"],
                    "description": "Unit for angles in polar form (default: degrees)"
                }
            },
            "required": ["z1", "operation"]
        }
    ),
    Tool(
        name="complex_functions",
        description="""Compute complex elementary, trigonometric, and hyperbolic functions.

Elementary Functions:
- exp(z): Complex exponential
- log(z): Complex natural logarithm (principal branch)
- sqrt(z): Complex square root (principal branch)

Trigonometric Functions:
- sin(z), cos(z), tan(z)
- asin(z), acos(z), atan(z)

Hyperbolic Functions:
- sinh(z), cosh(z), tanh(z)
- asinh(z), acosh(z), atanh(z)

Branch Cuts:
- log(z): branch cut on negative real axis
- sqrt(z): branch cut on negative real axis
- asin, acos: branch cuts on real axis outside [-1,1]

Applications:
- Transfer function evaluation
- Frequency response calculations
- Complex impedance analysis
- Laplace/Fourier transform evaluation""",
        inputSchema={
            "type": "object",
            "properties": {
                "z": {
                    "type": "string",
                    "description": "Complex number input (rectangular or polar)"
                },
                "function": {
                    "type": "string",
                    "enum": [
                        "exp", "log", "sqrt",
                        "sin", "cos", "tan", "asin", "acos", "atan",
                        "sinh", "cosh", "tanh", "asinh", "acosh", "atanh"
                    ],
                    "description": "Complex function to compute"
                }
            },
            "required": ["z", "function"]
        }
    ),
    Tool(
        name="roots_of_unity",
        description="""Calculate nth roots of unity or nth roots of any complex number.

Nth Roots of Unity:
- Returns all n solutions to z^n = 1
- Evenly spaced on unit circle
- Angular spacing: 360°/n

Nth Roots of Complex Number:
- Returns all n solutions to w^n = z
- Magnitude: |z|^(1/n)
- Angular spacing: 360°/n

Properties:
- All roots have equal magnitude
- Roots form regular n-gon on complex plane
- Sum of roots of unity = 0 (for n > 1)
- Product of roots of unity = (-1)^(n-1)

Applications:
- Polynomial root finding
- FFT and DFT calculations
- Geometric constructions
- Symmetry analysis""",
        inputSchema={
            "type": "object",
            "properties": {
                "n": {
                    "type": "integer",
                    "description": "Order of roots (must be positive integer)",
                    "minimum": 1
                },
                "z": {
                    "type": "string",
                    "description": "Complex number to find roots of (default: 1 for roots of unity)"
                },
                "root_index": {
                    "type": "integer",
                    "description": "Specific root index to highlight (0 to n-1)"
                }
            },
            "required": ["n"]
        }
    ),
    Tool(
        name="complex_conjugate_operations",
        description="""Perform operations involving complex conjugates.

Conjugate Properties:
- z* = a - bi (reflection across real axis)
- (z1 + z2)* = z1* + z2*
- (z1 · z2)* = z1* · z2*
- |z|² = z·z*

Operations:
- conjugate: Compute complex conjugate
- real_part: Re(z) = (z + z*)/2
- imaginary_part: Im(z) = (z - z*)/(2i)
- magnitude_squared: |z|² = z·z*
- conjugate_product: z1·z2* (power calculations)
- conjugate_sum: Verify (z1 + z2)* = z1* + z2*
- conjugate_product_property: Verify (z1·z2)* = z1*·z2*

Applications:
- AC power calculations: S = V·I*
- Real/imaginary part extraction
- Magnitude calculations
- Conjugate symmetry in Fourier transforms
- Polynomial root pairing""",
        inputSchema={
            "type": "object",
            "properties": {
                "z": {
                    "type": "string",
                    "description": "Complex number (rectangular or polar)"
                },
                "operation": {
                    "type": "string",
                    "enum": [
                        "conjugate", "real_part", "imaginary_part",
                        "magnitude_squared", "conjugate_product",
                        "conjugate_sum", "conjugate_product_property"
                    ],
                    "description": "Operation to perform"
                },
                "z2": {
                    "type": "string",
                    "description": "Second complex number (required for binary operations)"
                }
            },
            "required": ["z", "operation"]
        }
    )
]


# ============================================================================
# Handler Functions
# ============================================================================

async def handle_complex_operations(arguments: Dict[str, Any]) -> CallToolResult:
    """Handle complex_operations tool calls."""
    try:
        result = complex_operations(
            z1=arguments["z1"],
            operation=arguments["operation"],
            z2=arguments.get("z2"),
            angle_unit=arguments.get("angle_unit", "degrees")
        )
        
        import json
        return CallToolResult(
            content=[TextContent(type="text", text=json.dumps(result, indent=2, default=str))],
            isError=False
        )
    except Exception as e:
        logger.error(f"Error in complex_operations: {e}", exc_info=True)
        return CallToolResult(
            content=[TextContent(type="text", text=f"Error: {str(e)}")],
            isError=True
        )


async def handle_complex_functions(arguments: Dict[str, Any]) -> CallToolResult:
    """Handle complex_functions tool calls."""
    try:
        result = complex_functions(
            z=arguments["z"],
            function=arguments["function"]
        )
        
        import json
        return CallToolResult(
            content=[TextContent(type="text", text=json.dumps(result, indent=2, default=str))],
            isError=False
        )
    except Exception as e:
        logger.error(f"Error in complex_functions: {e}", exc_info=True)
        return CallToolResult(
            content=[TextContent(type="text", text=f"Error: {str(e)}")],
            isError=True
        )


async def handle_roots_of_unity(arguments: Dict[str, Any]) -> CallToolResult:
    """Handle roots_of_unity tool calls."""
    try:
        result = roots_of_unity(
            n=arguments["n"],
            z=arguments.get("z"),
            root_index=arguments.get("root_index")
        )
        
        import json
        return CallToolResult(
            content=[TextContent(type="text", text=json.dumps(result, indent=2, default=str))],
            isError=False
        )
    except Exception as e:
        logger.error(f"Error in roots_of_unity: {e}", exc_info=True)
        return CallToolResult(
            content=[TextContent(type="text", text=f"Error: {str(e)}")],
            isError=True
        )


async def handle_complex_conjugate_operations(arguments: Dict[str, Any]) -> CallToolResult:
    """Handle complex_conjugate_operations tool calls."""
    try:
        result = complex_conjugate_operations(
            z=arguments["z"],
            operation=arguments["operation"],
            z2=arguments.get("z2")
        )
        
        import json
        return CallToolResult(
            content=[TextContent(type="text", text=json.dumps(result, indent=2, default=str))],
            isError=False
        )
    except Exception as e:
        logger.error(f"Error in complex_conjugate_operations: {e}", exc_info=True)
        return CallToolResult(
            content=[TextContent(type="text", text=f"Error: {str(e)}")],
            isError=True
        )
