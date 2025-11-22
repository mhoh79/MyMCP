# Issue #80: Implement Complex Analysis Tools

**Priority**: High (required by Control Systems, Signal Processing)  
**Dependencies**: #79 (Engineering Math Server for numerical methods)  
**Labels**: enhancement, math-tools, complex-numbers  
**Estimated Effort**: 3-5 days

## Overview

Implement comprehensive complex number analysis tools essential for AC circuit analysis, control system pole/zero analysis, frequency response calculations, and signal processing in the frequency domain.

## Objectives

- Provide complete complex number arithmetic and conversions
- Support phasor analysis for AC systems
- Enable frequency domain calculations
- Integrate with Engineering Math Server

## Scope

### Group 5: Complex Analysis (4 major tools)

#### 1. `complex_operations`

**Arithmetic Operations**:
- Addition, subtraction, multiplication, division
- Power and roots of complex numbers
- Conjugate, absolute value (magnitude)
- Argument (phase angle)

**Conversions**:
- Rectangular ↔ Polar (r∠θ)
- Rectangular ↔ Exponential (re^(iθ))
- Degrees ↔ Radians for angles
- Principal argument calculation

**Input Formats Supported**:
```python
# Rectangular: a + bj
complex_operations(z1="3+4j", z2="1-2j", operation="multiply")

# Polar: magnitude∠angle
complex_operations(z1="5∠36.87°", z2="2.236∠-63.43°", operation="multiply")

# Mixed formats
complex_operations(z1="3+4j", z2="5∠30°", operation="add")
```

**Operations**:
- `add`, `subtract`, `multiply`, `divide`
- `power` (z^n)
- `conjugate`
- `magnitude`, `phase`
- `rect_to_polar`, `polar_to_rect`

#### 2. `complex_functions`

**Elementary Functions**:
- Exponential: e^z = e^x(cos(y) + i·sin(y))
- Natural logarithm: ln(z) = ln(|z|) + i·arg(z)
- General powers: z^w
- Square root (principal and all branches)

**Trigonometric Functions**:
- sin(z), cos(z), tan(z)
- csc(z), sec(z), cot(z)
- Inverse trig functions

**Hyperbolic Functions**:
- sinh(z), cosh(z), tanh(z)
- csch(z), sech(z), coth(z)
- Inverse hyperbolic functions

**Branch Cut Handling**:
- Principal branch specifications
- Multi-valued function warnings
- Branch cut visualization support

#### 3. `roots_of_unity`

**Nth Roots of Unity**:
- Calculate all nth roots of unity
- Roots of any complex number
- Geometric representation on unit circle
- Applications to polynomial roots

**Features**:
```python
roots_of_unity(n=5)
# Returns: [1, e^(i·2π/5), e^(i·4π/5), e^(i·6π/5), e^(i·8π/5)]

nth_root(z="16+0j", n=4)
# Returns: All 4 fourth roots of 16
```

**Properties Computed**:
- Magnitude (all = 1 for roots of unity)
- Angular spacing: 2π/n
- Sum of roots = 0 (except n=1)
- Product of roots = (-1)^(n-1)

#### 4. `complex_conjugate_operations`

**Conjugate Properties**:
- Complex conjugate: z* = a - bi
- Conjugate of sum/product/quotient
- Conjugate pairs in polynomial roots
- Symmetry properties

**Applications**:
- Real part extraction: Re(z) = (z + z*)/2
- Imaginary part: Im(z) = (z - z*)/(2i)
- Magnitude squared: |z|² = z·z*
- Reflection across real axis

**Magnitude and Argument**:
- |z| = √(a² + b²)
- arg(z) = atan2(b, a)
- Principal argument (-π, π]
- Argument in degrees or radians

## Technical Implementation

### Integration with Engineering Math Server

Add to `src/builtin/engineering_math_server/tools/complex_analysis.py`:

```python
"""Complex analysis tools for engineering applications."""

import numpy as np
import cmath
from typing import Union, List, Tuple, Dict
from mcp.types import Tool, TextContent, CallToolResult

def create_complex_operations_tool() -> Tool:
    """Create complex number operations tool."""
    return Tool(
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
        - Frequency domain operations
        """,
        inputSchema={
            "type": "object",
            "properties": {
                "z1": {
                    "type": "string",
                    "description": "First complex number (rectangular or polar)"
                },
                "z2": {
                    "type": "string", 
                    "description": "Second complex number (optional for some operations)"
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
                "angle_unit": {
                    "type": "string",
                    "enum": ["degrees", "radians"],
                    "default": "degrees",
                    "description": "Unit for angles in polar form"
                }
            },
            "required": ["z1", "operation"]
        }
    )

async def handle_complex_operations(arguments: dict) -> CallToolResult:
    """Handle complex number operations."""
    try:
        # Parse complex numbers from various formats
        z1 = parse_complex_number(arguments["z1"])
        
        operation = arguments["operation"]
        angle_unit = arguments.get("angle_unit", "degrees")
        
        # Perform operation
        if operation == "add":
            z2 = parse_complex_number(arguments["z2"])
            result = z1 + z2
            
        # ... (other operations)
        
        # Format output
        output = format_complex_result(result, angle_unit)
        
        return CallToolResult(
            content=[TextContent(type="text", text=output)]
        )
        
    except Exception as e:
        return CallToolResult(
            content=[TextContent(
                type="text",
                text=f"Error: {str(e)}"
            )],
            isError=True
        )
```

### Helper Functions

```python
def parse_complex_number(s: str) -> complex:
    """Parse complex number from string in various formats."""
    s = s.strip()
    
    # Polar format: magnitude∠angle
    if '∠' in s:
        parts = s.split('∠')
        magnitude = float(parts[0])
        angle_str = parts[1]
        
        # Check for degree symbol or 'deg'
        if '°' in angle_str or 'deg' in angle_str.lower():
            angle = float(angle_str.replace('°', '').replace('deg', ''))
            angle_rad = np.deg2rad(angle)
        else:
            # Assume radians
            angle_rad = float(angle_str.replace('rad', ''))
        
        return cmath.rect(magnitude, angle_rad)
    
    # Rectangular format: a+bj
    else:
        return complex(s)

def format_complex_result(z: complex, angle_unit: str = "degrees") -> str:
    """Format complex number result with both rectangular and polar forms."""
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
```

## Key Engineering Applications

### 1. AC Circuit Analysis (Impedance)
```python
# Calculate total impedance: Z = R + jωL + 1/(jωC)
complex_operations(
    z1="100+0j",           # Resistance (100Ω)
    z2="0+314.159j",       # Inductive reactance (ωL)
    operation="add"
)
# Then subtract capacitive reactance
```

### 2. Phasor Analysis
```python
# Voltage phasor: V = 120∠0° V
# Current phasor: I = 10∠-30° A
# Power: S = V·I* (conjugate of current)

complex_operations(
    z1="120∠0°",
    z2="10∠-30°",
    operation="multiply"
)
# Then take conjugate of z2 first
```

### 3. Control System Poles/Zeros
```python
# Pole location: s = -2 + 3j
# Magnitude gives distance from origin
# Phase gives angle in s-plane

complex_operations(
    z1="-2+3j",
    operation="magnitude"  # Distance from origin
)

complex_operations(
    z1="-2+3j", 
    operation="phase"      # Angle in complex plane
)
```

### 4. Frequency Response
```python
# Transfer function: H(jω) = 1/(1 + jω/ω₀)
# Evaluate at ω = 100 rad/s, ω₀ = 10 rad/s

complex_operations(
    z1="1+0j",
    z2="1+10j",  # 1 + j(100/10)
    operation="divide"
)
```

## Testing Requirements

### Unit Tests
- Arithmetic operations with known results
- Conversion accuracy (rectangular ↔ polar)
- Special values (0, 1, i, -1, -i)
- Branch cut behavior for logarithm and power
- Roots of unity properties

### Validation Tests
```python
# Test Euler's formula: e^(iπ) + 1 = 0
assert abs(cmath.exp(1j * cmath.pi) + 1) < 1e-10

# Test De Moivre's theorem: (cos θ + i sin θ)^n = cos(nθ) + i sin(nθ)

# Test conjugate properties: (z₁ + z₂)* = z₁* + z₂*
```

### Engineering Application Tests
- AC circuit impedance calculations
- Phasor arithmetic validation
- Transfer function evaluation
- Bode plot magnitude/phase calculations

## Documentation Requirements

1. **Mathematical Background**
   - Complex number theory
   - Euler's formula
   - De Moivre's theorem
   - Branch cuts and principal values

2. **Application Examples**
   - AC circuit analysis examples
   - Control system pole placement
   - Frequency response calculation
   - Filter design applications

3. **Common Pitfalls**
   - Branch cut discontinuities
   - Principal vs. multi-valued functions
   - Angle wrapping (-π to π)
   - Precision loss in subtraction

## Deliverables

- [ ] Complex operations tool implementation
- [ ] Complex functions tool implementation
- [ ] Roots of unity tool implementation
- [ ] Complex conjugate operations tool implementation
- [ ] Comprehensive test suite
- [ ] Documentation with engineering examples
- [ ] Integration with Engineering Math Server

## Success Criteria

- ✅ All 4 complex analysis tools functional
- ✅ Accurate conversions between representations
- ✅ Proper branch cut handling
- ✅ Engineering application examples working
- ✅ Integration with existing tools verified
- ✅ Tests pass with numerical accuracy < 1e-10

## Timeline

**Days 1-2**: Implement complex_operations and complex_functions  
**Days 3-4**: Implement roots_of_unity and conjugate operations  
**Day 5**: Testing, documentation, integration

## Related Issues

- Requires: #79 (Engineering Math Server)
- Blocks: #83 (Control Systems Server)
- Blocks: #85 (Signal Processing Server)
- Related: #81 (Transform Methods - uses complex analysis)

## References

- Complex Analysis for Engineers (Churchill & Brown)
- AC Circuit Analysis references
- Control Systems Engineering (Nise)
- Signal Processing using MATLAB (Ingle & Proakis)
