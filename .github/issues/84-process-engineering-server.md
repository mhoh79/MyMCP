# Issue #84: Implement Process Engineering Server

**Priority**: High  
**Dependencies**: #79 (Engineering Math), #81 (Transforms for FFT)  
**Labels**: enhancement, builtin-server, process-engineering, application-server  
**Estimated Effort**: 2-3 weeks

## Overview

Create a specialized MCP server for process engineering calculations including thermodynamics, fluid mechanics, heat transfer, reaction engineering, and equipment design. Essential for chemical, petroleum, and process industries.

## MCP Server Architecture Mapping

**Server Name**: `process_engineering_server` (Application Server)  
**Role**: Chemical/process engineering calculations and equipment design  
**Tools**: 15 process-specific functions  
**Dependencies**: Engineering Math (#79), Transforms (#81 for FFT in analysis)

### Application Stack Coverage

This server is the **primary implementation** of:

1. **Process Engineering Stack** - Complete implementation (100%)
   - Flash calculations, heat exchangers, pumps, distillation, reactors, material/energy balances
   
2. **Thermal Systems Stack** - Heat transfer components (60%)
   - Heat exchanger design, heat loss calculations, thermal properties
   - Works with Engineering Math (#79) for heat transfer ODEs
   
3. **Fluid Mechanics Stack** - Fluid flow and equipment (80%)
   - Pressure drop, pump sizing, pipe schedules, valve sizing
   - Works with Engineering Math (#79) for implicit friction factor equations
   
4. **Reliability & Risk Analysis Stack** - Equipment reliability (20%)
   - Works with Special Functions (#87) for Weibull analysis of equipment life

### Tool Reuse from Foundation Server (#79)

- **Root finding** → Flash calculations (bubble/dew point), implicit equations (friction factor, valve Cv)
- **Numerical methods** → Property iterations, equilibrium calculations
- **Optimization** → Economic pipe diameter, heat exchanger optimization, operating point optimization
- **Integration** → Energy balances, batch reactor residence time, flow accumulation
- **Interpolation** → Property table lookups (steam tables, fluid properties)
- **ODE solvers** → Reaction kinetics, batch reactor dynamics, transient heat transfer
- **Linear algebra** → Multi-component material balances, recycle stream calculations

### Tool Reuse from Transforms (#81)

- **FFT** → Vibration analysis of rotating equipment (pumps, compressors)
- **Convolution** → Dynamic response of process equipment

### Cross-Server Workflows

**Example: Heat Exchanger with Control Loop**
```python
# 1. Process Engineering Server: Design heat exchanger
heat_exchanger_design(type="shell_and_tube", duty=500000, ...)

# 2. Engineering Math Server (#79): Optimize for minimum cost
optimization(objective="minimize_cost", constraints=[...])

# 3. Control Systems Server (#83): Design temperature control
pid_tuner(plant_dynamics=first_order_lag, ...)

# 4. Process Engineering Server: Size control valve
valve_sizing(flow_rate=50, pressure_drop=100, ...)
```

**Example: Pump System Design**
```python
# 1. Process Engineering Server: Calculate pressure drop
pressure_drop(pipe=..., fittings=[...])

# 2. Engineering Math Server (#79): Solve implicit friction factor
root_finding(equation="colebrook", ...)

# 3. Process Engineering Server: Size pump
pump_sizing(flow=100, head=45, npsh_required=3)

# 4. Engineering Math Server (#79): Optimize pipe diameter
optimization(objective="minimum_lifecycle_cost", ...)
```

## Objectives

- Enable process design and analysis workflows
- Provide thermodynamic property calculations
- Support equipment sizing and design
- Enable process simulation calculations
- Facilitate material and energy balances

## Scope

### Process Engineering Stack Tools (12-15 tools)

#### 1. `flash_calculator`

**Vapor-Liquid Equilibrium (VLE)**:
- Flash calculations (PT, PH, TV)
- Bubble point, dew point
- Component splits
- Raoult's law, modified Raoult's
- Activity coefficient models (NRTL, UNIQUAC, Wilson)

**Features**:
```python
flash_calculator(
    calculation_type="PT_flash",  # Pressure-Temperature flash
    pressure=101.325,              # kPa
    temperature=80,                # °C
    composition={"ethanol": 0.4, "water": 0.6},
    thermodynamic_model="raoults_law"
)
```

**Applications**: Distillation, separations, phase equilibrium

#### 2. `heat_exchanger_design`

**Heat Exchanger Methods**:
- LMTD (Log Mean Temperature Difference)
- Effectiveness-NTU method
- Shell and tube design
- Plate heat exchangers
- Air coolers

**Features**:
```python
heat_exchanger_design(
    type="shell_and_tube",
    hot_fluid={"fluid": "water", "T_in": 90, "T_out": 60, "flow": 10},
    cold_fluid={"fluid": "water", "T_in": 20, "T_out": 50, "flow": 12},
    method="LMTD"
)
```

**Output**: Required area, number of tubes, LMTD correction factor, fouling margins

#### 3. `pressure_drop`

**Fluid Flow Calculations**:
- Pipe flow (Darcy-Weisbach)
- Fitting losses (K-factors)
- Control valve sizing (Cv)
- Two-phase flow
- Non-Newtonian fluids

**Features**:
```python
pressure_drop(
    pipe_diameter=0.1,     # m
    pipe_length=100,       # m
    flow_rate=0.05,        # m³/s
    fluid_properties={"density": 1000, "viscosity": 0.001},
    roughness=0.000045,    # m (steel pipe)
    fittings=[{"type": "elbow_90", "count": 3}, {"type": "gate_valve", "count": 1}]
)
```

**Applications**: Piping system design, pump sizing

#### 4. `pump_sizing`

**Pump Calculations**:
- Head requirements (static + dynamic + friction)
- Power calculations (brake HP, hydraulic HP)
- NPSH (Net Positive Suction Head)
- Pump curves and operating points
- Affinity laws

**Features**:
```python
pump_sizing(
    flow_rate=100,             # m³/h
    total_head=50,             # m
    fluid_density=1000,        # kg/m³
    fluid_viscosity=0.001,     # Pa·s
    suction_conditions={"pressure": 101.325, "elevation": -2},
    npsh_required=3.0          # m
)
```

**Output**: Required power, NPSH available, pump selection guidelines

#### 5. `tank_sizing`

**Storage Tank Design**:
- Volume calculations (cylindrical, spherical)
- Residence time
- Level-volume relationships
- Vapor space requirements
- Wall thickness (basic)

**Features**:
```python
tank_sizing(
    tank_type="vertical_cylindrical",
    required_volume=100,       # m³
    residence_time=30,         # minutes (optional)
    flow_rate=200,             # m³/h (optional)
    vapor_space_fraction=0.1,  # 10% vapor space
    aspect_ratio=2.0           # Height/Diameter
)
```

**Applications**: Storage tanks, process vessels, surge tanks

#### 6. `reaction_kinetics`

**Chemical Reaction Engineering**:
- Arrhenius equation (k = A·e^(-Ea/RT))
- Reaction rate calculations
- Conversion calculations
- Reactor sizing (CSTR, PFR, Batch)
- Adiabatic temperature rise

**Features**:
```python
reaction_kinetics(
    reaction_order=2,
    rate_constant=0.05,        # L/(mol·s) at reference T
    activation_energy=50000,   # J/mol
    temperature=350,           # K
    concentrations={"A": 2.0, "B": 1.5},  # mol/L
    reactor_type="CSTR"
)
```

**Applications**: Reactor design, batch time calculations, temperature control

#### 7. `distillation_design`

**Distillation Column Calculations**:
- McCabe-Thiele method
- Fenske-Underwood-Gilliland
- Minimum reflux ratio
- Minimum stages
- Feed stage location

**Features**:
```python
distillation_design(
    method="fenske_underwood_gilliland",
    feed={"composition": {"light": 0.5, "heavy": 0.5}, "condition": "saturated_liquid"},
    distillate_purity=0.95,
    bottoms_purity=0.05,
    reflux_ratio=1.5,
    relative_volatility=2.5
)
```

**Output**: Number of stages, feed stage, column diameter, reboiler/condenser duties

#### 8. `compressor_calculations`

**Compressor Design & Analysis**:
- Polytropic/isentropic compression
- Power requirements
- Discharge temperature
- Multi-stage compression
- Intercooling

**Features**:
```python
compressor_calculations(
    suction_pressure=100,      # kPa
    discharge_pressure=500,    # kPa
    flow_rate=1000,            # kg/h
    gas_properties={"molecular_weight": 29, "k": 1.4},
    efficiency=0.75,
    compression_type="polytropic"
)
```

**Applications**: Gas compression, refrigeration, air systems

#### 9. `fluid_properties`

**Thermophysical Properties**:
- Density (ideal gas, liquids, correlations)
- Viscosity (gas, liquid, temperature effects)
- Heat capacity (Cp, Cv)
- Thermal conductivity
- Vapor pressure (Antoine equation)
- Enthalpy, entropy

**Features**:
```python
fluid_properties(
    fluid="water",
    temperature=100,           # °C
    pressure=101.325,          # kPa
    properties=["density", "viscosity", "enthalpy", "vapor_pressure"]
)
```

**Database**: Common fluids (water, air, steam, hydrocarbons, refrigerants)

#### 10. `material_energy_balance`

**Process Balance Calculations**:
- Material balance (component, total)
- Energy balance (enthalpy method)
- Combined material & energy balance
- Recycle stream calculations
- Degree of freedom analysis

**Features**:
```python
material_energy_balance(
    streams={
        "feed": {"flow": 1000, "composition": {...}, "T": 25, "P": 101.325},
        "product": {"composition": {...}, "T": 80}  # Unknown flow
    },
    unit_operation="mixer",    # or "separator", "reactor", "heater"
    reactions=[...],           # optional for reactors
    duty=None                  # unknown, to be calculated
)
```

**Applications**: Process simulation, mass/energy balance verification

#### 11. `pipe_schedule`

**Pipe Sizing & Selection**:
- Standard pipe schedules (40, 80, etc.)
- Velocity limits (liquid, gas, steam)
- Erosional velocity
- Economic pipe diameter
- Pipe wall thickness

**Features**:
```python
pipe_schedule(
    flow_rate=100,             # m³/h
    fluid_type="liquid",
    service="water",
    max_velocity=3.0,          # m/s (optional)
    pressure_rating=150        # ANSI class
)
```

**Output**: Recommended pipe size, schedule, actual velocity

#### 12. `valve_sizing`

**Control Valve Calculations**:
- Cv coefficient calculation
- Liquid, gas, steam sizing
- Pressure drop across valve
- Cavitation/flashing check
- Noise prediction

**Features**:
```python
valve_sizing(
    fluid_type="liquid",
    flow_rate=50,              # m³/h
    inlet_pressure=500,        # kPa
    outlet_pressure=200,       # kPa
    fluid_properties={"density": 1000, "vapor_pressure": 3.2}
)
```

**Applications**: Control valve specification, process control

#### 13. `heat_loss_calculation`

**Thermal Insulation**:
- Heat loss through insulation
- Economic insulation thickness
- Surface temperature
- Multi-layer insulation
- Pipe/vessel heat loss

**Features**:
```python
heat_loss_calculation(
    geometry="pipe",
    dimensions={"diameter": 0.2, "length": 100},  # m
    process_temperature=200,   # °C
    ambient_temperature=20,    # °C
    insulation_layers=[
        {"material": "mineral_wool", "thickness": 0.05}  # m
    ],
    wind_speed=3.0             # m/s (for convection)
)
```

**Applications**: Energy efficiency, personnel protection, freeze protection

#### 14. `particle_settling`

**Solid-Fluid Separation**:
- Stokes' law (terminal velocity)
- Non-Stokes regime corrections
- Hindered settling
- Cyclone separator design
- Centrifuge calculations

**Features**:
```python
particle_settling(
    particle_diameter=100e-6,  # m (100 microns)
    particle_density=2500,     # kg/m³
    fluid_density=1000,        # kg/m³
    fluid_viscosity=0.001,     # Pa·s
    calculation="terminal_velocity"
)
```

**Applications**: Settling tanks, cyclones, filters, centrifuges

#### 15. `process_economics`

**Economic Analysis**:
- Equipment cost estimation (CEPCI indexed)
- Utility costs
- Operating cost estimation
- Payback period
- NPV for process alternatives

**Features**:
```python
process_economics(
    analysis_type="equipment_cost",
    equipment="heat_exchanger",
    parameters={"area": 50, "material": "stainless_steel", "pressure": 10},
    base_year=2020,
    target_year=2025
)
```

**Applications**: Cost estimation, project justification

## Technical Architecture

### Server Structure
```
src/builtin/process_engineering_server/
├── __init__.py
├── __main__.py
├── server.py
├── tools/
│   ├── __init__.py
│   ├── thermodynamics.py     # Tools 1, 9
│   ├── fluid_mechanics.py    # Tools 3, 4, 11, 12
│   ├── heat_transfer.py      # Tools 2, 13
│   ├── reaction_engineering.py  # Tool 6
│   ├── separations.py        # Tools 7, 14
│   ├── equipment_design.py   # Tools 5, 8
│   ├── process_integration.py   # Tool 10
│   └── economics.py          # Tool 15
├── data/
│   ├── fluid_properties.json
│   ├── pipe_schedules.csv
│   └── cost_correlations.json
└── README.md
```

### Dependencies
```python
# Additional requirements
CoolProp>=6.4.0         # Fluid property database
thermo>=0.2.0          # Chemical engineering thermodynamics
fluids>=1.0.0          # Fluid mechanics calculations
chemicals>=1.1.0       # Chemical property database
```

### Tool Reuse from Engineering Math Server
- Root finding (implicit equations like flash calculations)
- ODE solvers (reaction kinetics, batch reactors)
- Interpolation (property tables)
- Optimization (minimum cost, maximum efficiency)
- Numerical integration (heat exchanger NTU)

## Key Application Examples

### Example 1: Design Heat Exchanger
```python
# 1. Calculate duty
duty = flow * Cp * (T_hot_in - T_hot_out)

# 2. Design heat exchanger
heat_exchanger_design(
    type="shell_and_tube",
    duty=duty,
    hot_fluid={...},
    cold_fluid={...}
)

# 3. Check pressure drop
# 4. Verify fouling margins
```

### Example 2: Size Pump for Pipeline
```python
# 1. Calculate pressure drop
pressure_drop(
    pipe_diameter=0.15,
    pipe_length=500,
    flow_rate=0.02,
    fittings=[...]
)

# 2. Size pump
pump_sizing(
    flow_rate=72,  # m³/h
    total_head=45,  # m (static + friction)
    npsh_required=3.0
)

# 3. Check NPSH available > NPSH required
```

### Example 3: Distillation Column Design
```python
# 1. VLE data
flash_calculator(...)

# 2. Column stages
distillation_design(
    method="fenske_underwood_gilliland",
    feed={...},
    distillate_purity=0.95,
    reflux_ratio=1.5
)

# 3. Reboiler/condenser sizing
# 4. Economic optimization
```

## Testing Requirements

### Unit Tests
- Flash calculations vs. hand calculations
- Heat exchanger LMTD formulas
- Pressure drop correlations
- Pump affinity laws
- Arrhenius equation

### Validation Tests
- Compare with process simulation software (Aspen, HYSYS)
- Verify against textbook examples
- Cross-check with vendor sizing tools

### Integration Tests
- Complete process design workflows
- Multi-unit operations in series

## Deliverables

- [ ] ProcessEngineeringServer implementation
- [ ] All 15 process tools functional
- [ ] Fluid properties database integrated
- [ ] Comprehensive test suite
- [ ] Documentation with process examples
- [ ] Wrapper script: `start_process_engineering_server.py`
- [ ] Claude Desktop configuration

## Success Criteria

- ✅ All process engineering tools working
- ✅ Fluid property database accessible
- ✅ Accurate equipment sizing
- ✅ Example workflows documented
- ✅ Integration with foundation tools verified

## Timeline

**Week 1**: Thermodynamics, fluid properties, flash calculations  
**Week 2**: Heat transfer, fluid mechanics, equipment sizing  
**Week 3**: Reaction engineering, separations, economics, testing

## Related Issues

- Requires: #79 (Engineering Math)
- Related: Stats Server (data analysis for process data)
- Part of: Process Engineering Stack

## References

- Chemical Engineering Design (Coulson & Richardson)
- Perry's Chemical Engineers' Handbook
- Unit Operations of Chemical Engineering (McCabe, Smith, Harriott)
- Process Heat Transfer (Kern)
- CoolProp documentation
