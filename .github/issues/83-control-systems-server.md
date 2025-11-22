# Issue #83: Implement Control Systems Server

**Priority**: High  
**Dependencies**: #79 (Engineering Math), #80 (Complex Analysis), #81 (Transforms)  
**Labels**: enhancement, builtin-server, control-systems, application-server  
**Estimated Effort**: 2-3 weeks

## Overview

Create a specialized MCP server for control system design and analysis. This server provides tools for transfer function analysis, frequency response, state-space methods, stability analysis, and controller design - essential for industrial automation, process control, and feedback system design.

## MCP Server Architecture Mapping

**Server Name**: `control_systems_server` (Application Server)  
**Role**: Specialized control system design and analysis  
**Tools**: 12 control-specific functions  
**Dependencies**: Engineering Math (#79), Complex Analysis (#80), Transforms (#81)

### Application Stack Coverage

This server is the **primary implementation** of:

1. **Control System Design Stack** - Complete implementation (100%)
   - Transfer functions, poles/zeros, Bode/Nyquist plots, state-space, PID tuning
   
2. **Control & Instrumentation Stack** - Core control components (70%)
   - PID controller design, loop shaping, stability analysis
   - Works with Signal Processing (#85) for filtering and Stats for monitoring
   
3. **Electrical/Power Systems Stack** - Control aspects (30%)
   - Voltage regulation, frequency control loops
   - Works with Complex Analysis (#80) for phasor calculations

### Tool Reuse from Foundation Server (#79)

- **Polynomials** → Transfer function numerator/denominator, characteristic equations
- **Linear algebra** → State-space matrices (A, B, C, D), controllability/observability
- **Matrix decomposition** → Eigenvalues for pole placement, modal decomposition
- **Root finding** → Crossover frequencies, gain calculations
- **Optimization** → Controller parameter tuning
- **ODE solvers** → Step response, impulse response simulation

### Tool Reuse from Complex Analysis (#80)

- **Complex operations** → Poles/zeros in s-plane, frequency response calculations
- **Complex functions** → Exponential for Laplace transforms

### Tool Reuse from Transforms (#81)

- **Laplace transform** → Transfer function domain conversions
- **Fourier analysis** → Frequency response (Bode plots)

### Cross-Server Workflows

**Example: Complete PID Loop Design**
```python
# 1. Control Systems Server: Identify plant transfer function
transfer_function_analyzer(num=[1], den=[1, 2, 1])

# 2. Control Systems Server: Design PID controller
pid_tuner(method="ziegler_nichols", plant_tf=...)

# 3. Signal Processing Server (#85): Design anti-aliasing filter
filter_design(type="butterworth", cutoff=100)

# 4. Engineering Math Server (#79): Optimize for robustness
optimization(objective="maximize_phase_margin", ...)
```

## Objectives

- Enable complete control system analysis workflow
- Support both classical and modern control methods
- Provide transfer function and state-space representations
- Implement frequency and time domain analysis
- Support PID and advanced controller design

## Scope

### Control System Design Stack Tools

This server implements the complete Control System Design Stack identified in the requirements:
1. Polynomials (transfer functions)
2. Complex analysis (poles/zeros)
3. Transforms (Laplace, frequency response)
4. Calculus (ODE simulation)
5. Linear algebra (state-space)

### Core Tools (10-12 tools)

#### 1. `transfer_function_analyzer`

**Transfer Function Analysis**:
- Create transfer function from numerator/denominator polynomials
- Extract poles and zeros
- Calculate DC gain, steady-state gain
- Determine system type (0, 1, 2, ...)
- Check stability (all poles in LHP)

**Features**:
```python
transfer_function_analyzer(
    numerator=[1, 2],          # s + 2
    denominator=[1, 3, 2],     # s² + 3s + 2
    analysis=["poles_zeros", "stability", "gain", "time_constants"]
)
```

**Output**:
- Poles (location, damping, natural frequency)
- Zeros (location)
- DC gain
- System type
- Stability status
- Dominant poles
- Time constants (τ = 1/|pole|)
- Factored form display

**Transfer Function Forms**:
- Polynomial form: N(s)/D(s)
- Factored form: K(s-z₁)(s-z₂)/[(s-p₁)(s-p₂)]
- Time constant form: K/(τs+1)
- Standard second-order: ωₙ²/(s² + 2ζωₙs + ωₙ²)

#### 2. `frequency_response`

**Frequency Domain Analysis**:
- Bode plots (magnitude and phase)
- Nyquist plot data
- Nichols chart data
- Gain margin, phase margin
- Crossover frequencies
- Bandwidth calculation

**Features**:
```python
frequency_response(
    transfer_function={
        "numerator": [10],
        "denominator": [1, 5, 6]
    },
    frequency_range=[0.01, 100],  # rad/s
    num_points=1000,
    plot_type="bode"              # or "nyquist", "nichols"
)
```

**Bode Plot Analysis**:
- Magnitude plot (dB vs. frequency)
- Phase plot (degrees vs. frequency)
- -3dB bandwidth
- Resonant peak
- Gain margin (GM)
- Phase margin (PM)
- Gain crossover frequency (ωgc)
- Phase crossover frequency (ωpc)

**Nyquist Plot Analysis**:
- Nyquist contour
- Encirclements of -1 point
- Stability from Nyquist criterion
- Relative stability metrics

**Stability Margins**:
- Gain margin: GM = 1/|G(jωpc)| (in dB: 20log₁₀(GM))
- Phase margin: PM = 180° + ∠G(jωgc)
- Delay margin: DM = PM/ωgc

#### 3. `state_space_converter`

**State-Space Representation**:
- Convert transfer function ↔ state-space
- Multiple canonical forms:
  - Controllable canonical form
  - Observable canonical form
  - Diagonal form (if possible)
  - Jordan canonical form

**Features**:
```python
state_space_converter(
    input_type="transfer_function",
    numerator=[1, 2],
    denominator=[1, 5, 6],
    output_form="controllable_canonical"
)
```

**State-Space Model**:
```
ẋ = Ax + Bu  (state equation)
y = Cx + Du  (output equation)

where:
- x: state vector (n×1)
- u: input vector (m×1)
- y: output vector (p×1)
- A: system matrix (n×n)
- B: input matrix (n×m)
- C: output matrix (p×n)
- D: feedthrough matrix (p×m)
```

**Operations**:
- Transfer function → state-space (tf2ss)
- State-space → transfer function (ss2tf)
- Similarity transformations
- Canonical form conversions
- MIMO system handling

**Properties Computed**:
- Controllability matrix rank
- Observability matrix rank
- Eigenvalues (poles)
- Eigenvectors (modes)

#### 4. `stability_analyzer`

**Stability Analysis Methods**:
- Routh-Hurwitz criterion
- Root locus analysis
- Lyapunov stability
- Gain range for stability

**Routh-Hurwitz**:
```python
stability_analyzer(
    method="routh_hurwitz",
    characteristic_polynomial=[1, 3, 3, 1],  # s³+3s²+3s+1
)
```

**Output**:
- Routh array
- Number of RHP roots
- Stability conclusion
- Critical gain (if parametric)

**Root Locus**:
```python
stability_analyzer(
    method="root_locus",
    open_loop_tf={
        "numerator": [1],
        "denominator": [1, 2, 1]
    },
    gain_range=[0, 100],
    num_points=500
)
```

**Output**:
- Root locus branches
- Breakaway/break-in points
- Asymptotes (angles and centroid)
- Departure/arrival angles
- Stability boundary crossings
- Critical gain for stability

#### 5. `time_response`

**Time Domain Analysis**:
- Step response
- Impulse response
- Ramp response
- Initial condition response
- Arbitrary input response

**Features**:
```python
time_response(
    system={
        "numerator": [1],
        "denominator": [1, 2, 1]
    },
    input_type="step",
    time_span=[0, 10],
    num_points=1000
)
```

**Performance Metrics**:
- Rise time (tr): 10% to 90%
- Peak time (tp): time to first peak
- Settling time (ts): within ±2% of final value
- Percent overshoot (PO): (peak - final)/final × 100%
- Steady-state error (ess)
- Decay ratio

**Second-Order System Metrics**:
- Natural frequency (ωₙ)
- Damping ratio (ζ)
- Damped natural frequency (ωd = ωₙ√(1-ζ²))
- Classification (underdamped, critically damped, overdamped)

#### 6. `pid_tuner`

**PID Controller Design**:
- Ziegler-Nichols methods (open-loop and closed-loop)
- Cohen-Coon method
- IMC (Internal Model Control) tuning
- ITAE (Integral Time Absolute Error) tuning
- Lambda tuning

**Features**:
```python
pid_tuner(
    process_model={
        "type": "FOPDT",  # First-Order Plus Dead Time
        "K": 2.0,         # Process gain
        "tau": 5.0,       # Time constant
        "theta": 1.0      # Dead time
    },
    tuning_method="ziegler_nichols_open_loop",
    controller_type="PID"  # or "PI", "PD"
)
```

**Output**:
- Proportional gain (Kp)
- Integral time (Ti) or Ki = Kp/Ti
- Derivative time (Td) or Kd = Kp*Td
- Controller transfer function
- Predicted closed-loop response
- Performance metrics
- Robustness measures

**Tuning Methods**:
- Ziegler-Nichols reaction curve
- Ziegler-Nichols ultimate sensitivity
- Cohen-Coon
- Tyreus-Luyben
- IMC-PID
- Direct synthesis

#### 7. `controllability_observability`

**State-Space System Properties**:
- Controllability analysis
- Observability analysis
- Pole placement feasibility
- Observer design feasibility

**Features**:
```python
controllability_observability(
    A=[[0, 1], [-2, -3]],  # System matrix
    B=[[0], [1]],          # Input matrix
    C=[[1, 0]],            # Output matrix
    analysis="both"         # or "controllability", "observability"
)
```

**Controllability**:
- Controllability matrix: [B AB A²B ... Aⁿ⁻¹B]
- Rank of controllability matrix
- Controllable/uncontrollable subspaces
- PBH test for specific modes

**Observability**:
- Observability matrix: [C; CA; CA²; ...; CAⁿ⁻¹]
- Rank of observability matrix
- Observable/unobservable subspaces
- PBH test for specific modes

#### 8. `pole_placement`

**State Feedback Design**:
- Pole placement via state feedback
- Ackermann's formula
- Bass-Gura formula
- Robust pole placement

**Features**:
```python
pole_placement(
    A=[[0, 1], [-2, -3]],
    B=[[0], [1]],
    desired_poles=[-5, -6],  # Desired closed-loop poles
    method="ackermann"
)
```

**Output**:
- State feedback gain matrix K
- Closed-loop system matrix (A-BK)
- Closed-loop poles (verification)
- Sensitivity to parameter variations

#### 9. `observer_design`

**State Observer Design**:
- Full-order observer (Luenberger observer)
- Reduced-order observer
- Kalman filter (basic)
- Observer pole placement

**Features**:
```python
observer_design(
    A=[[0, 1], [-2, -3]],
    B=[[0], [1]],
    C=[[1, 0]],
    desired_observer_poles=[-10, -12],  # Faster than system poles
    observer_type="full_order"
)
```

**Output**:
- Observer gain matrix L
- Observer system matrices
- Estimation error dynamics
- Separation principle verification

#### 10. `compensator_design`

**Classical Compensation**:
- Lead compensator
- Lag compensator
- Lead-lag compensator
- Notch filter

**Features**:
```python
compensator_design(
    plant_tf={
        "numerator": [1],
        "denominator": [1, 1, 0]  # Type 1 system
    },
    design_specs={
        "phase_margin": 45,        # degrees
        "gain_crossover": 10,      # rad/s
        "steady_state_error": 0.1  # for ramp input
    },
    compensator_type="lead"
)
```

**Lead Compensator**:
- Adds phase lead for stability
- Transfer function: Gc(s) = K(τs+1)/(ατs+1), α<1
- Design for phase margin improvement

**Lag Compensator**:
- Improves steady-state error
- Transfer function: Gc(s) = K(τs+1)/(βτs+1), β>1
- Design for error reduction

#### 11. `digital_control_design`

**Discrete-Time Control**:
- Continuous → discrete (Z-transform, ZOH, Tustin)
- Digital PID implementation
- Deadbeat control
- Discrete pole placement

**Features**:
```python
digital_control_design(
    continuous_tf={
        "numerator": [1],
        "denominator": [1, 1]
    },
    sampling_time=0.1,  # seconds
    discretization_method="tustin"  # or "zoh", "backward_euler"
)
```

**Output**:
- Discrete transfer function H(z)
- Difference equation
- Stability analysis in z-plane
- Implementation code (pseudocode)

#### 12. `system_identification`

**Identify System from Data**:
- Transfer function estimation
- State-space model estimation
- ARX, ARMAX models
- Frequency response estimation

**Features**:
```python
system_identification(
    input_data=[...],      # System input
    output_data=[...],     # System output
    sampling_time=0.1,
    model_type="transfer_function",
    model_order=2          # First or second order
)
```

**Methods**:
- Least squares
- Subspace identification
- Frequency response fitting
- Step/impulse response fitting

## Technical Architecture

### Server Structure
```
src/builtin/control_systems_server/
├── __init__.py
├── __main__.py
├── server.py              # ControlSystemsServer class
├── tools/
│   ├── __init__.py
│   ├── transfer_functions.py    # Tools 1-2
│   ├── state_space.py           # Tools 3, 7-9
│   ├── stability.py             # Tools 4
│   ├── time_analysis.py         # Tool 5
│   ├── controllers.py           # Tools 6, 10-11
│   └── identification.py        # Tool 12
├── utils/
│   ├── __init__.py
│   ├── conversions.py           # TF ↔ SS conversions
│   ├── canonical_forms.py       # State-space forms
│   └── plotting_data.py         # Generate plot data
└── README.md
```

### Dependencies
```python
# Additional requirements
control>=0.9.0          # Python Control Systems Library
slycot>=0.5.0          # System and control library (optional)
```

### Tool Reuse from Engineering Math Server
- Polynomial arithmetic and roots (transfer functions)
- Complex analysis (poles/zeros in s-plane)
- Laplace transforms
- Matrix operations (state-space)
- Linear system solvers
- Eigenvalue computation
- ODE solvers (time response)

## Key Application Examples

### Example 1: Design PI Controller for Tank Level
```python
# 1. Model the process
process = {
    "type": "FOPDT",
    "K": 1.5,      # m/(%valve opening)
    "tau": 120,    # seconds
    "theta": 15    # seconds dead time
}

# 2. Tune PI controller
pid_tuner(
    process_model=process,
    tuning_method="IMC",
    controller_type="PI",
    desired_closed_loop_time_constant=60
)

# 3. Analyze closed-loop response
# 4. Implement in DCS
```

### Example 2: Motor Speed Control Analysis
```python
# 1. Transfer function
motor_tf = {
    "numerator": [10],
    "denominator": [1, 2, 10]  # Second-order
}

# 2. Frequency response
frequency_response(
    transfer_function=motor_tf,
    frequency_range=[0.1, 100]
)

# 3. Check stability margins
# GM > 6 dB, PM > 45° for good robustness
```

### Example 3: State Feedback for Inverted Pendulum
```python
# 1. Linearized state-space model
A = [[0, 1, 0, 0],
     [0, 0, -m*g/M, 0],
     [0, 0, 0, 1],
     [0, 0, (M+m)*g/(M*L), 0]]

B = [[0], [1/M], [0], [-1/(M*L)]]

# 2. Check controllability
controllability_observability(A=A, B=B, analysis="controllability")

# 3. Place poles for fast stabilization
pole_placement(
    A=A, B=B,
    desired_poles=[-2, -2.1, -10, -10.1]
)
```

## Testing Requirements

### Unit Tests
- Transfer function operations
- Pole/zero extraction
- State-space conversions
- Stability criteria
- PID tuning formulas

### Integration Tests
- Complete design workflows
- Multi-tool pipelines
- Numerical accuracy of responses

### Validation Tests
- Known analytical solutions
- Standard test systems (canonical forms)
- Published design examples

## Deliverables

- [ ] ControlSystemsServer implementation
- [ ] All 12 control tools functional
- [ ] Comprehensive test suite
- [ ] Documentation with control examples
- [ ] Integration with Engineering Math Server
- [ ] Wrapper script: `start_control_systems_server.py`
- [ ] Claude Desktop configuration example

## Success Criteria

- ✅ All control system tools working
- ✅ Accurate frequency/time response
- ✅ PID tuning methods validated
- ✅ State-space tools functional
- ✅ Example workflows documented
- ✅ Integration with foundation tools verified

## Timeline

**Week 1**: Transfer functions, frequency response, stability  
**Week 2**: State-space, time response, controllability  
**Week 3**: PID tuning, compensators, digital control, identification

## Related Issues

- Requires: #79, #80, #81
- Related: #85 (Signal Processing - frequency analysis)
- Part of: Control System Design Stack

## References

- Modern Control Engineering (Ogata)
- Control Systems Engineering (Nise)
- Feedback Control of Dynamic Systems (Franklin, Powell, Emami-Naeini)
- Python Control Systems Library documentation
