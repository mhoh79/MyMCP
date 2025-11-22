# Issue #86: Implement Structural Analysis Server

**Priority**: High  
**Dependencies**: #79 (Engineering Math), #82 (Geometry)  
**Labels**: enhancement, builtin-server, structural-analysis, application-server  
**Estimated Effort**: 2-3 weeks

## Overview

Create a specialized MCP server for structural mechanics and analysis including beam calculations, truss analysis, stress analysis, section properties, and structural design. Essential for mechanical, civil, and aerospace engineering applications.

## MCP Server Architecture Mapping

**Server Name**: `structural_analysis_server` (Application Server)  
**Role**: Structural mechanics, stress analysis, and structural design  
**Tools**: 12 structural analysis functions  
**Dependencies**: Engineering Math (#79), Geometry (#82)

### Application Stack Coverage

This server is the **primary implementation** of:

1. **Structural/Mechanical Stack** - Complete implementation (100%)
   - Beams, trusses, frames, section properties, stress analysis, buckling, vibration
   
2. **Vibration Analysis & Diagnostics Stack** - Structural dynamics (40%)
   - Natural frequencies, mode shapes, resonance prediction
   - Works with Signal Processing (#85) for vibration signal analysis and Stats for statistics

### Tool Reuse from Foundation Server (#79)

- **Linear algebra** → FEA stiffness matrices (K·u = F), truss/frame global stiffness assembly
- **Matrix decomposition** → Eigenvalues (natural frequencies), eigenvectors (mode shapes), SVD for conditioning
- **Numerical methods** → Iterative solvers for large FEA systems, non-linear structural analysis
- **Optimization** → Minimum weight design, optimal cross-section selection, cost minimization
- **Integration** → Distributed load effects (shear/moment from loads), work-energy calculations
- **Calculus** → Deflection curves (double integration of moment), beam dynamics equations
- **Polynomial** → Deflection equations, characteristic equations for vibration frequencies
- **Root finding** → Buckling load iterations, intersection problems in geometry
- **Vector operations** → Force/moment resultants, cross products for torque calculations

### Tool Reuse from Geometry Server (#82)

- **Geometric calculations** → Complex section properties (centroid, moment of inertia) for custom shapes
- **Coordinate transforms** → 3D frame element transformations, global/local coordinate systems
- **Vector operations** → Force decomposition, spatial orientations

### Tool Reuse from Complex Analysis (#80)

- **Complex numbers** → Damped vibration (complex eigenvalues), frequency response with damping

### Cross-Server Workflows

**Example: Beam Design with Custom Section**
```python
# 1. Geometry Server (#82): Calculate section properties
section_properties(section_type="custom_polygon", vertices=[...])

# 2. Structural Analysis Server: Analyze beam
beam_analysis(beam_type="simply_supported", loads=[...], 
              section_properties={"I": ..., "A": ...})

# 3. Engineering Math Server (#79): Optimize for minimum weight
optimization(objective="minimize_weight", 
             constraints=["max_stress < yield", "max_deflection < L/360"])

# 4. Structural Analysis Server: Verify final design
stress_transformation(stress_state={...})  # Check combined stresses
```

**Example: Truss Bridge Analysis**
```python
# 1. Structural Analysis Server: Analyze truss
truss_analysis(nodes=[...], members=[...], loads=[...])

# 2. Engineering Math Server (#79): Solve global stiffness (K·u = F)
solve_linear_system(matrix=K_global, vector=F_loads)

# 3. Structural Analysis Server: Check member stresses
# (identifies members in tension vs. compression)

# 4. Engineering Math Server (#79): Optimize member sizes
optimization(objective="minimum_cost", constraints=["stress < allowable"])
```

**Example: Vibration Analysis**
```python
# 1. Structural Analysis Server: Calculate natural frequencies
vibration_analysis(system_type="beam", boundary="simply_supported", modes=5)

# 2. Engineering Math Server (#79): Eigenvalue problem
matrix_decomposition(matrix=K_mass_normalized, method="eigenvalue")

# 3. Signal Processing Server (#85): Analyze measured vibration
fft_analysis(accelerometer_data)  # Compare with predicted frequencies

# 4. Structural Analysis Server: Assess resonance risk
# (compare operating frequencies with natural frequencies)
```

## Objectives

- Enable structural design and verification workflows
- Provide beam deflection and stress calculations
- Support truss and frame analysis
- Calculate section properties
- Facilitate structural mechanics education

## Scope

### Structural Analysis Stack Tools (10-12 tools)

#### 1. `beam_analysis`

**Beam Deflection & Stress**:
- Simply supported, cantilever, fixed beams
- Point loads, distributed loads, moments
- Shear force diagrams
- Bending moment diagrams
- Deflection equations
- Support reactions

**Features**:
```python
beam_analysis(
    beam_type="simply_supported",
    length=5.0,                # m
    loads=[
        {"type": "point", "position": 2.0, "magnitude": 10000},  # N
        {"type": "distributed", "start": 0, "end": 5, "magnitude": 2000}  # N/m
    ],
    section_properties={"I": 8.33e-6, "A": 0.01},  # m⁴, m²
    material={"E": 200e9, "yield_strength": 250e6},  # Pa
    analysis_points=50
)
```

**Output**:
- Reaction forces
- Shear force diagram (V vs. x)
- Bending moment diagram (M vs. x)
- Deflection curve (y vs. x)
- Maximum deflection location & value
- Maximum stress & location
- Safety factor

**Beam Types**:
- Simply supported
- Cantilever
- Fixed-fixed
- Overhanging
- Continuous (multi-span)

#### 2. `section_properties`

**Cross-Section Calculations**:
- Area, centroid location
- Second moments of area (Ixx, Iyy, Ixy)
- Principal axes
- Section modulus (Sx, Sy)
- Radius of gyration
- Torsional constant (J)

**Features**:
```python
section_properties(
    section_type="I_beam",     # or custom polygon
    dimensions={
        "height": 0.3,         # m
        "width": 0.15,
        "web_thickness": 0.01,
        "flange_thickness": 0.015
    }
)
```

**Standard Sections**:
- Rectangle
- Circle, hollow circle
- I-beam (W, S, HP shapes)
- C-channel
- Angle (L-shape)
- T-section
- Custom polygon (defined by vertices)

**Composite Sections**:
- Parallel axis theorem
- Multiple materials (transformed sections)

**Output**:
- A (area)
- x̄, ȳ (centroid)
- Ixx, Iyy, Ixy (moments of inertia)
- I₁, I₂ (principal moments)
- θ (principal angle)
- Sx, Sy (section moduli)
- rx, ry (radii of gyration)
- J (torsional constant)

#### 3. `truss_analysis`

**2D Truss Systems**:
- Method of joints
- Method of sections
- Matrix stiffness method
- Member forces (tension/compression)
- Support reactions
- Joint displacements

**Features**:
```python
truss_analysis(
    nodes=[
        {"id": 1, "x": 0, "y": 0, "support": "pinned"},
        {"id": 2, "x": 4, "y": 0, "support": "roller"},
        {"id": 3, "x": 2, "y": 3, "support": "free"}
    ],
    members=[
        {"nodes": [1, 3], "area": 0.001, "E": 200e9},  # m², Pa
        {"nodes": [2, 3], "area": 0.001, "E": 200e9},
        {"nodes": [1, 2], "area": 0.001, "E": 200e9}
    ],
    loads=[
        {"node": 3, "Fx": 0, "Fy": -10000}  # N
    ]
)
```

**Output**:
- Member forces (+ tension, - compression)
- Support reactions
- Joint displacements
- Member stresses
- Zero-force members identification
- Visualization coordinates

**Applications**: Bridge trusses, roof trusses, tower structures

#### 4. `stress_transformation`

**2D & 3D Stress Analysis**:
- Stress transformation (rotation)
- Principal stresses
- Maximum shear stress
- Mohr's circle calculations
- Von Mises stress
- Factor of safety

**Features**:
```python
stress_transformation(
    stress_state={
        "sigma_x": 50e6,       # Pa
        "sigma_y": 30e6,
        "tau_xy": 20e6
    },
    angle=30,                  # degrees (optional, for rotation)
    yield_criterion="von_mises",
    yield_strength=250e6       # Pa
)
```

**Output**:
- σ₁, σ₂ (principal stresses)
- θp (principal angle)
- τmax (maximum shear stress)
- σ', τ' (stresses at specified angle)
- Mohr's circle data (center, radius, points)
- Von Mises equivalent stress
- Safety factor

#### 5. `column_buckling`

**Euler Buckling & Stability**:
- Critical buckling load (Pcr)
- Effective length factors (K)
- Slenderness ratio
- Buckling stress
- Design load capacity (with safety factor)

**Features**:
```python
column_buckling(
    length=3.0,                # m
    end_conditions="pinned_pinned",  # or "fixed_free", "fixed_fixed", etc.
    section_properties={"I": 8.33e-6, "A": 0.01},  # m⁴, m²
    material={"E": 200e9, "yield_strength": 250e6},
    safety_factor=2.0
)
```

**End Conditions**:
- Pinned-pinned (K=1.0)
- Fixed-free (K=2.0)
- Fixed-pinned (K=0.7)
- Fixed-fixed (K=0.5)

**Output**:
- Pcr (critical buckling load)
- Le (effective length)
- Slenderness ratio (Le/r)
- σcr (critical buckling stress)
- Pallow (allowable load with safety factor)
- Failure mode (buckling vs. yielding)

#### 6. `torsion_analysis`

**Shaft Torsion**:
- Shear stress due to torque
- Angle of twist
- Torsional stiffness
- Solid and hollow circular shafts
- Power transmission

**Features**:
```python
torsion_analysis(
    shaft_type="hollow_circular",
    outer_diameter=0.1,        # m
    inner_diameter=0.08,       # m (0 for solid)
    length=2.0,                # m
    torque=5000,               # N·m
    material={"G": 80e9},      # Shear modulus, Pa
    power=50000,               # W (optional, for speed calculation)
    rpm=300                    # (optional)
)
```

**Output**:
- J (polar moment of inertia)
- τmax (maximum shear stress)
- φ (angle of twist, radians)
- Torsional stiffness (T/φ)
- Power capacity
- Critical speed (if applicable)

#### 7. `pressure_vessel`

**Thin-Walled Pressure Vessels**:
- Hoop stress (circumferential)
- Longitudinal stress
- Radial stress (thick-walled)
- Required wall thickness
- Safety factor analysis

**Features**:
```python
pressure_vessel(
    geometry="cylindrical",    # or "spherical"
    pressure=2e6,              # Pa (internal)
    diameter=1.0,              # m (inner)
    wall_thickness=0.01,       # m
    material={"yield_strength": 250e6},
    end_condition="closed"     # for cylinders
)
```

**Calculations**:
- **Cylindrical**:
  - σ_hoop = P·r/t
  - σ_longitudinal = P·r/(2t)
- **Spherical**:
  - σ = P·r/(2t)

**Output**:
- Hoop stress
- Longitudinal stress (cylinders)
- Required thickness (with safety factor)
- Factor of safety
- Failure assessment

#### 8. `deflection_tables`

**Standard Beam Deflection Cases**:
- Pre-calculated formulas
- Common loading scenarios
- Quick reference database
- Superposition support

**Features**:
```python
deflection_tables(
    case="cantilever_point_load_tip",
    parameters={
        "length": 2.0,         # m
        "load": 1000,          # N
        "E": 200e9,            # Pa
        "I": 1e-6              # m⁴
    },
    query="max_deflection"     # or "deflection_at_x", "moment_at_x"
)
```

**Available Cases** (30+ standard cases):
- Cantilever beams (5+ load cases)
- Simply supported (8+ load cases)
- Fixed-fixed beams (5+ load cases)
- Propped cantilever
- Overhanging beams

**Applications**: Quick hand calculations, verification

#### 9. `frame_analysis`

**2D Rigid Frame Analysis**:
- Fixed/pinned joints
- Member forces and moments
- Joint rotations
- Sway deflections
- Matrix stiffness method

**Features**:
```python
frame_analysis(
    nodes=[
        {"id": 1, "x": 0, "y": 0, "support": ["Fx", "Fy", "Mz"]},  # Fixed
        {"id": 2, "x": 0, "y": 3},
        {"id": 3, "x": 4, "y": 3},
        {"id": 4, "x": 4, "y": 0, "support": ["Fy"]}  # Roller
    ],
    members=[
        {"nodes": [1, 2], "section": {...}, "E": 200e9},
        {"nodes": [2, 3], "section": {...}, "E": 200e9},
        {"nodes": [3, 4], "section": {...}, "E": 200e9}
    ],
    loads=[
        {"node": 2, "Fx": -10000},
        {"member": 1, "type": "distributed", "magnitude": 5000}  # N/m
    ]
)
```

**Output**:
- Member end forces (axial, shear, moment)
- Joint displacements and rotations
- Support reactions
- Member diagrams (M, V, N)

**Applications**: Building frames, portals, multi-story structures

#### 10. `combined_loading`

**Interaction of Multiple Stresses**:
- Axial + bending (beam-columns)
- Bending about two axes
- Torsion + bending
- Interaction diagrams
- Combined stress checks

**Features**:
```python
combined_loading(
    loads={
        "axial": 50000,        # N
        "moment_x": 10000,     # N·m
        "moment_y": 5000,
        "torque": 3000         # N·m
    },
    section_properties={
        "A": 0.01, "Ix": 8e-6, "Iy": 4e-6, "J": 10e-6  # m², m⁴
    },
    material={"yield_strength": 250e6}
)
```

**Output**:
- Combined stress at critical points
- Safety factor (interaction equation)
- Failure mode prediction
- Stress distribution visualization data

#### 11. `connection_design`

**Bolted & Welded Connections**:
- Bolt shear and bearing
- Bolt group analysis (shear center)
- Weld stress (fillet, groove)
- Connection capacity

**Features**:
```python
connection_design(
    connection_type="bolted",
    loading={"shear": 50000, "tension": 10000},  # N
    bolts={
        "diameter": 0.020,     # m (M20)
        "grade": "8.8",        # ISO grade
        "count": 4,
        "pattern": [[0, 0], [0.1, 0], [0, 0.1], [0.1, 0.1]]  # positions
    }
)
```

**Applications**: Connection verification, joint design

#### 12. `vibration_analysis`

**Simple Structural Vibration**:
- Natural frequencies (SDOF, MDOF)
- Mode shapes
- Free vibration (undamped, damped)
- Forced vibration response
- Resonance prediction

**Features**:
```python
vibration_analysis(
    system_type="beam",        # or "frame", "spring_mass"
    boundary_conditions="simply_supported",
    properties={
        "length": 5.0,         # m
        "E": 200e9, "I": 1e-6, "mass_per_length": 10  # Pa, m⁴, kg/m
    },
    modes=3                    # Number of modes to compute
)
```

**Output**:
- Natural frequencies (ωn, fn)
- Mode shapes
- Critical damping coefficient
- Resonance frequencies

## Technical Architecture

### Server Structure
```
src/builtin/structural_analysis_server/
├── __init__.py
├── __main__.py
├── server.py
├── tools/
│   ├── __init__.py
│   ├── beams.py             # Tools 1, 8
│   ├── sections.py          # Tool 2
│   ├── trusses.py           # Tool 3
│   ├── stress_analysis.py   # Tool 4
│   ├── stability.py         # Tools 5, 6
│   ├── pressure_vessels.py  # Tool 7
│   ├── frames.py            # Tool 9
│   ├── combined_loading.py  # Tool 10
│   ├── connections.py       # Tool 11
│   └── vibrations.py        # Tool 12
├── data/
│   ├── steel_sections.json  # Standard section database
│   ├── deflection_formulas.json
│   └── material_properties.json
└── README.md
```

### Dependencies
```python
# Additional requirements
sectionproperties>=2.0.0   # Section property calculations
anastruct>=1.3.0           # Structural analysis (optional)
```

### Tool Reuse from Engineering Math Server
- Matrix operations (stiffness matrices)
- Linear system solver (K·u = F)
- Eigenvalue problems (natural frequencies)
- Polynomial roots (characteristic equations)
- Numerical integration (distributed loads)

### Tool Reuse from Geometry Server
- Polygon area, centroid (#82)
- Moment of inertia for custom shapes
- Coordinate transformations

## Key Application Examples

### Example 1: Design a Simply Supported Beam
```python
# 1. Calculate section properties
section_properties(
    section_type="I_beam",
    dimensions={"height": 0.3, ...}
)

# 2. Analyze beam
beam_analysis(
    beam_type="simply_supported",
    length=6.0,
    loads=[...],
    section_properties={"I": ..., "A": ...},
    material={"E": 200e9, "yield_strength": 250e6}
)

# 3. Check: max stress < yield / safety_factor
# 4. Check: max deflection < L/360 (serviceability)
```

### Example 2: Analyze a Truss Bridge
```python
# Define nodes and members
truss_analysis(
    nodes=[...],
    members=[...],
    loads=[...]
)

# Identify max tension/compression members
# Size members based on forces
```

### Example 3: Check Column Buckling
```python
# 1. Section properties
section_properties(...)

# 2. Buckling analysis
column_buckling(
    length=4.0,
    end_conditions="fixed_pinned",
    section_properties={...},
    material={...}
)

# 3. Compare Pcr vs. applied load
```

### Example 4: Design Pressure Vessel
```python
# 1. Calculate required thickness
pressure_vessel(
    geometry="cylindrical",
    pressure=5e6,              # 5 MPa
    diameter=2.0,
    material={"yield_strength": 400e6}
)

# 2. Check hoop and longitudinal stresses
# 3. Add corrosion allowance
```

## Testing Requirements

### Unit Tests
- Beam deflection formulas vs. textbook
- Section property calculations (known shapes)
- Truss method of joints (simple cases)
- Mohr's circle calculations
- Euler buckling formula

### Validation Tests
- Compare with FEA software (ANSYS, Abaqus)
- Verify against AISC/Eurocode examples
- Cross-check deflection tables

### Integration Tests
- Complete structural design workflows
- Multi-member systems

## Deliverables

- [ ] StructuralAnalysisServer implementation
- [ ] All 12 structural tools functional
- [ ] Standard section database integrated
- [ ] Deflection formula library
- [ ] Comprehensive test suite
- [ ] Documentation with design examples
- [ ] Wrapper script: `start_structural_analysis_server.py`
- [ ] Claude Desktop configuration

## Success Criteria

- ✅ All structural tools working
- ✅ Accurate beam and truss analysis
- ✅ Section properties validated
- ✅ Standard sections database usable
- ✅ Example design workflows documented

## Timeline

**Week 1**: Beams, sections, trusses, stress transformation  
**Week 2**: Buckling, torsion, pressure vessels, frames  
**Week 3**: Combined loading, connections, vibrations, testing

## Related Issues

- Requires: #79 (Engineering Math), #82 (Geometry)
- Part of: Structural Engineering Stack

## References

- Mechanics of Materials (Beer, Johnston, DeWolf)
- Structural Analysis (Hibbeler)
- AISC Steel Construction Manual
- Design of Welded Structures (Blodgett)
- Roark's Formulas for Stress and Strain
