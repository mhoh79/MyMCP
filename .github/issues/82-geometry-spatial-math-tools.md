# Issue #82: Implement Geometry & Spatial Math Tools

**Priority**: Medium (required by Structural Analysis Server)  
**Dependencies**: #79 (Engineering Math Server for trig functions)  
**Labels**: enhancement, math-tools, geometry, spatial-analysis  
**Estimated Effort**: 1 week

## Overview

Implement comprehensive geometry and spatial mathematics tools for mechanical engineering, structural analysis, robot kinematics, and CAD calculations. These tools handle spatial relationships, coordinate transformations, and geometric property calculations.

## Objectives

- Provide complete geometric calculation capabilities
- Support multiple coordinate systems
- Enable spatial transformations and rotations
- Calculate geometric properties (areas, volumes, centroids, moments)
- Support 2D and 3D operations

## Scope

### Group 7: Geometry, Trigonometry & Spatial Mathematics (6 tools)

#### 1. `triangle_solver`

**Solve Triangles from Given Information**:
- SSS (Side-Side-Side)
- SAS (Side-Angle-Side)
- ASA (Angle-Side-Angle)
- AAS (Angle-Angle-Side)
- SSA (Side-Side-Angle) - ambiguous case handled

**Features**:
```python
triangle_solver(
    case="SAS",
    side_a=5.0,
    side_b=7.0,
    angle_C=60.0,          # degrees
    angle_unit="degrees"
)
```

**Computed Properties**:
- All three sides (a, b, c)
- All three angles (A, B, C)
- Area (multiple formulas)
- Perimeter
- Inradius (inscribed circle)
- Circumradius (circumscribed circle)
- Heights (altitudes)
- Medians
- Angle bisectors

**Special Triangles**:
- Right triangle (Pythagoras)
- Isosceles triangle
- Equilateral triangle
- Obtuse/acute classification

**Applications**:
- Force vector decomposition
- Truss analysis
- Surveying calculations
- Linkage kinematics

#### 2. `coordinate_transforms`

**Multi-System Conversions**:
- Cartesian (x, y, z)
- Cylindrical (r, θ, z)
- Spherical (r, θ, φ)
- Polar (2D: r, θ)

**Transformation Matrix Support**:
- Translation matrices
- Rotation matrices (2D and 3D)
- Scaling matrices
- Homogeneous coordinates (4×4 matrices)

**Features**:
```python
coordinate_transforms(
    from_system="cartesian",
    to_system="spherical",
    coordinates=[3.0, 4.0, 5.0],
    angle_convention="physics"  # or "mathematics"
)
```

**2D Transformations**:
- Cartesian ↔ Polar
- Rotation about origin
- Translation
- Reflection (x-axis, y-axis, origin)

**3D Transformations**:
- Cartesian ↔ Cylindrical ↔ Spherical
- Euler angle rotations (XYZ, ZYX, etc.)
- Quaternion support
- Axis-angle representation

**Applications**:
- Robot kinematics (DH parameters)
- Antenna positioning
- Geographic coordinates
- Field calculations (electric, magnetic)

#### 3. `geometric_calculations`

**2D Shapes**:
- Triangle: area, perimeter, centroid
- Rectangle/square: properties
- Circle: area, circumference, arc length, sector
- Ellipse: area, perimeter (approximate)
- Polygon (regular/irregular): area, centroid
- Composite shapes

**3D Solids**:
- Cube/rectangular prism
- Sphere: volume, surface area
- Cylinder: volume, surface area, centroid
- Cone: volume, surface area
- Pyramid: volume, surface area
- Torus: volume, surface area
- Composite solids

**Advanced Properties**:
- Centroid (center of mass)
- Moments of inertia (Iₓ, Iᵧ, Iₓᵧ, polar J)
- Radius of gyration
- Section modulus
- Parallel axis theorem application
- Product of inertia

**Features**:
```python
geometric_calculations(
    shape="I_beam",
    dimensions={
        "width": 200,      # mm
        "height": 400,     # mm
        "flange_thickness": 15,
        "web_thickness": 10
    },
    properties=["area", "centroid", "moment_of_inertia_x", "moment_of_inertia_y"]
)
```

**Standard Structural Shapes**:
- I-beam (wide flange)
- C-channel
- Angle (L-shape)
- T-section
- Hollow rectangular/circular sections
- Custom composite sections

**Applications**:
- Structural beam design
- Tank sizing
- Material takeoff calculations
- Center of gravity calculations
- Piping volume calculations

#### 4. `angle_calculations`

**Angle Operations**:
- Unit conversions (degrees, radians, gradians)
- Angle normalization (0-360°, -180 to 180°)
- Angle addition/subtraction with wrapping
- Trigonometric function values
- Inverse trig with quadrant determination

**Features**:
```python
angle_calculations(
    operation="normalize",
    angle=450.0,
    unit="degrees",
    range_type="positive"  # 0-360 or "signed" -180 to 180
)
```

**Trigonometric Functions**:
- Standard: sin, cos, tan
- Reciprocal: csc, sec, cot
- Inverse: arcsin, arccos, arctan, atan2
- Exact values for special angles (0°, 30°, 45°, 60°, 90°)

**Angle Between Vectors**:
- 2D angle between vectors
- 3D angle between vectors
- Signed angle (with reference direction)
- Dihedral angles

**Applications**:
- Bearing calculations
- Phase angle computations
- Slope angles
- Joint angles in linkages

#### 5. `rotation_matrices`

**2D Rotations**:
- Rotation about origin
- Rotation about arbitrary point
- Multiple rotation composition

**3D Rotations**:
- Rotation about principal axes (X, Y, Z)
- Euler angles (12 conventions)
- Rotation about arbitrary axis (Rodrigues' formula)
- Quaternion to rotation matrix
- Rotation matrix to Euler angles

**Features**:
```python
rotation_matrices(
    dimension=3,
    rotation_type="euler_angles",
    angles=[30, 45, 60],   # degrees
    sequence="XYZ",        # rotation order
    angle_unit="degrees"
)
```

**Rotation Representations**:
- 3×3 rotation matrices
- Euler angles (gimbal lock awareness)
- Quaternions (unit quaternions)
- Axis-angle representation
- Direction cosine matrices

**Operations**:
- Compose rotations (matrix multiplication)
- Inverse rotation (transpose)
- Interpolation (SLERP for quaternions)
- Convert between representations

**Applications**:
- Robot orientation
- Aircraft attitude (roll, pitch, yaw)
- Camera transformations
- 3D graphics/CAD
- Gyroscope data processing

#### 6. `distance_calculations`

**Distance Metrics**:
- Euclidean distance (L2 norm)
- Manhattan distance (L1 norm)
- Chebyshev distance (Linf norm)
- Geodesic distance (sphere surface)

**Point-to-Geometry Distances**:
- Point to point
- Point to line (2D and 3D)
- Point to plane
- Point to circle/sphere
- Point to polygon/polyhedron

**Line-to-Geometry Distances**:
- Line to line (skew lines in 3D)
- Line to plane
- Closest approach between lines

**Features**:
```python
distance_calculations(
    calculation_type="point_to_plane",
    point=[1, 2, 3],
    plane={
        "point": [0, 0, 0],
        "normal": [0, 0, 1]
    }
)
```

**Specialized Calculations**:
- Perpendicular distance
- Projection of point onto line/plane
- Closest point on geometry
- Intersection detection
- Containment tests (point in polygon, etc.)

**Applications**:
- Collision detection
- Path planning
- Tolerance checking
- Proximity sensors
- Clearance verification

## Technical Implementation

### File Structure

Add to `src/builtin/engineering_math_server/tools/geometry.py`:

```python
"""Geometry and spatial mathematics tools."""

import numpy as np
from typing import Dict, List, Tuple, Union, Optional
from mcp.types import Tool, TextContent, CallToolResult

def create_triangle_solver_tool() -> Tool:
    """Create triangle solver tool."""
    return Tool(
        name="triangle_solver",
        description="""Solve triangles given various combinations of sides and angles.
        
        Supported cases:
        - SSS: Three sides → all angles and properties
        - SAS: Two sides and included angle → third side and angles
        - ASA: Two angles and included side → other sides
        - AAS: Two angles and non-included side → other sides  
        - SSA: Two sides and non-included angle → ambiguous case handling
        
        Computes:
        - All sides and angles
        - Area (Heron's formula, trig formula)
        - Perimeter, inradius, circumradius
        - Heights, medians, angle bisectors
        
        Applications:
        - Force decomposition
        - Truss analysis
        - Surveying
        - Linkage kinematics
        """,
        inputSchema={
            "type": "object",
            "properties": {
                "case": {
                    "type": "string",
                    "enum": ["SSS", "SAS", "ASA", "AAS", "SSA"],
                    "description": "Triangle case type"
                },
                "side_a": {
                    "type": "number",
                    "description": "Side a length",
                    "minimum": 0,
                    "exclusiveMinimum": True
                },
                "side_b": {
                    "type": "number",
                    "description": "Side b length"
                },
                "side_c": {
                    "type": "number",
                    "description": "Side c length"
                },
                "angle_A": {
                    "type": "number",
                    "description": "Angle A (opposite side a)"
                },
                "angle_B": {
                    "type": "number",
                    "description": "Angle B (opposite side b)"
                },
                "angle_C": {
                    "type": "number",
                    "description": "Angle C (opposite side c)"
                },
                "angle_unit": {
                    "type": "string",
                    "enum": ["degrees", "radians"],
                    "default": "degrees"
                }
            },
            "required": ["case", "angle_unit"]
        }
    )

async def handle_triangle_solver(arguments: dict) -> CallToolResult:
    """Handle triangle solving."""
    try:
        case = arguments["case"]
        angle_unit = arguments.get("angle_unit", "degrees")
        
        # Extract provided values
        sides = {}
        angles = {}
        
        for key in ["side_a", "side_b", "side_c"]:
            if key in arguments:
                sides[key[-1]] = arguments[key]
        
        for key in ["angle_A", "angle_B", "angle_C"]:
            if key in arguments:
                angle_val = arguments[key]
                if angle_unit == "degrees":
                    angle_val = np.deg2rad(angle_val)
                angles[key[-1]] = angle_val
        
        # Solve triangle based on case
        if case == "SSS":
            result = solve_SSS(sides)
        elif case == "SAS":
            result = solve_SAS(sides, angles)
        elif case == "ASA":
            result = solve_ASA(sides, angles)
        elif case == "AAS":
            result = solve_AAS(sides, angles)
        elif case == "SSA":
            result = solve_SSA(sides, angles)
        
        # Calculate additional properties
        result["properties"] = calculate_triangle_properties(result)
        
        # Format output
        output = format_triangle_result(result, angle_unit)
        
        return CallToolResult(
            content=[TextContent(type="text", text=output)]
        )
        
    except Exception as e:
        return CallToolResult(
            content=[TextContent(type="text", text=f"Error: {str(e)}")],
            isError=True
        )

def solve_SSS(sides: Dict[str, float]) -> Dict:
    """Solve triangle given three sides using Law of Cosines."""
    a, b, c = sides['a'], sides['b'], sides['c']
    
    # Check triangle inequality
    if not (a + b > c and b + c > a and c + a > b):
        raise ValueError("Triangle inequality violated: sides cannot form a triangle")
    
    # Law of cosines: cos(A) = (b² + c² - a²) / (2bc)
    cos_A = (b**2 + c**2 - a**2) / (2 * b * c)
    cos_B = (a**2 + c**2 - b**2) / (2 * a * c)
    cos_C = (a**2 + b**2 - c**2) / (2 * a * b)
    
    angle_A = np.arccos(np.clip(cos_A, -1, 1))
    angle_B = np.arccos(np.clip(cos_B, -1, 1))
    angle_C = np.arccos(np.clip(cos_C, -1, 1))
    
    return {
        'sides': {'a': a, 'b': b, 'c': c},
        'angles': {'A': angle_A, 'B': angle_B, 'C': angle_C}
    }

def calculate_triangle_properties(triangle: Dict) -> Dict:
    """Calculate additional triangle properties."""
    sides = triangle['sides']
    a, b, c = sides['a'], sides['b'], sides['c']
    
    # Perimeter
    perimeter = a + b + c
    s = perimeter / 2  # semi-perimeter
    
    # Area using Heron's formula
    area = np.sqrt(s * (s - a) * (s - b) * (s - c))
    
    # Inradius
    inradius = area / s
    
    # Circumradius: R = abc / (4 * Area)
    circumradius = (a * b * c) / (4 * area)
    
    # Heights
    height_a = 2 * area / a
    height_b = 2 * area / b
    height_c = 2 * area / c
    
    return {
        'area': area,
        'perimeter': perimeter,
        'inradius': inradius,
        'circumradius': circumradius,
        'heights': {'ha': height_a, 'hb': height_b, 'hc': height_c}
    }
```

## Key Engineering Applications

### 1. Force Vector Decomposition
```python
triangle_solver(
    case="SAS",
    side_a=100,      # Force magnitude
    side_b=150,
    angle_C=60,      # Angle between forces
)
# Find resultant force and component angles
```

### 2. Robot Kinematics
```python
coordinate_transforms(
    from_system="cartesian",
    to_system="spherical",
    coordinates=[x, y, z]  # End effector position
)
# Convert to joint angles
```

### 3. Structural Beam Properties
```python
geometric_calculations(
    shape="I_beam",
    dimensions={"width": 200, "height": 400, ...},
    properties=["moment_of_inertia_x", "section_modulus"]
)
# Calculate bending stiffness
```

### 4. Proximity Detection
```python
distance_calculations(
    calculation_type="point_to_line",
    point=sensor_location,
    line={"point": p1, "direction": v}
)
# Check clearance requirements
```

## Testing Requirements

### Unit Tests
- Triangle solver for all cases
- Coordinate transform round-trip accuracy
- Geometric property calculations vs. hand calculations
- Rotation matrix orthogonality
- Distance calculation edge cases

### Validation Tests
- Triangle angle sum = 180°
- Transform composition identity
- Rotation matrix det = 1
- Moment of inertia parallel axis theorem

## Deliverables

- [ ] All 6 geometry tools implemented
- [ ] Comprehensive test suite
- [ ] Documentation with examples
- [ ] Standard shape library
- [ ] Integration with Engineering Math Server

## Success Criteria

- ✅ All geometry tools functional
- ✅ Numerical accuracy < 1e-10
- ✅ Support for 2D and 3D operations
- ✅ Standard shapes library complete
- ✅ Engineering examples working

## Timeline

**Days 1-2**: Triangle solver, angle calculations  
**Days 3-4**: Coordinate transforms, rotations  
**Day 5**: Geometric calculations, distances  
**Days 6-7**: Testing, documentation

## Related Issues

- Requires: #79 (Engineering Math Server)
- Blocks: #86 (Structural Analysis Server)

## References

- Engineering Mechanics (Hibbeler)
- Roark's Formulas for Stress and Strain
- Robot Modeling and Control (Spong)
- Computational Geometry (de Berg)
