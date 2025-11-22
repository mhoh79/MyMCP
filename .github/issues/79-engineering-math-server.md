# Issue #79: Implement Engineering Math Server (Foundation)

**Priority**: High (required by all other servers)  
**Dependencies**: None  
**Labels**: enhancement, builtin-server, foundation  
**Estimated Effort**: 2-3 weeks

## Overview

Create the foundational Engineering Math Server that provides core mathematical tools used by all specialized application servers. This server implements Groups 1-4 (Linear Algebra, Calculus, Numerical Methods, Polynomials) plus cross-cutting utilities.

## MCP Server Architecture Mapping

**Server Name**: `engineering_math_server` (Foundation Server)  
**Role**: Provides mathematical primitives reused by all application servers  
**Tools**: 31 core mathematical functions across 5 categories  
**Enhancement**: Issue #87 adds 10 more advanced tools (total: 41 tools)

### Tool Distribution Across Application Stacks

This foundation server provides tools used in ALL 12 application stacks:

1. **Control System Design** - Linear algebra (state-space), polynomials (transfer functions), calculus (ODE simulation)
2. **Process Engineering** - Numerical methods (implicit equations), calculus (balances), optimization
3. **Signal Processing** - Linear algebra (filter banks), polynomials (IIR filters), calculus (convolution)
4. **Structural/Mechanical** - Linear algebra (FEA matrices), matrix decomposition (modal analysis), calculus (dynamics)
5. **Vibration Analysis** - Calculus (integration for displacement), root finding (resonances)
6. **Electrical/Power** - Linear algebra (network equations), numerical methods (load flow)
7. **Thermal Systems** - Calculus (heat transfer ODEs), numerical methods (implicit schemes)
8. **Fluid Mechanics** - Root finding (friction factor), calculus (flow integration)
9. **Control & Instrumentation** - Calculus (PID derivatives/integrals), optimization (tuning)
10. **Reliability & Risk** - Probability tools (#87), Monte Carlo (#87)
11. **Data Analysis & ML** - Linear algebra (PCA, regression), matrix decomposition (SVD)
12. **Financial Engineering** - Numerical methods (IRR, YTM), optimization (portfolio), calculus (Greeks)

### Integration Pattern

```
Engineering Math Server (#79) ← FOUNDATION
├── Used by: Control Systems Server (#83)
├── Used by: Process Engineering Server (#84)
├── Used by: Signal Processing Server (#85)
├── Used by: Structural Analysis Server (#86)
├── Used by: Financial Engineering Server (#88)
└── Enhanced by: Special Functions & Probability (#87)
```

## Objectives

- Provide mathematical foundation for Control Systems, Process Engineering, Signal Processing, and Structural Analysis servers
- Implement 31+ core mathematical tools across 5 categories
- Establish patterns for tool organization and code reuse
- Create comprehensive test suite for numerical accuracy

## Scope

### Group 1: Linear Algebra & Matrix Mathematics (6 tools)

#### 1. `matrix_operations`
- Add, subtract, multiply matrices
- Transpose, trace, determinant
- Input validation for dimension compatibility
- Support for both dense and sparse representations

#### 2. `matrix_inverse`
- Multiple inversion methods (LU, Cholesky, SVD-based)
- Conditioning number calculation
- Singularity detection
- Regularization options for ill-conditioned matrices

#### 3. `matrix_decomposition`
- LU decomposition (with partial pivoting)
- QR decomposition (Gram-Schmidt, Householder)
- Cholesky decomposition (for positive definite)
- SVD (Singular Value Decomposition)
- Eigenvalue/eigenvector decomposition

#### 4. `solve_linear_system`
- Direct solvers (Gaussian elimination, LU)
- Iterative solvers (Jacobi, Gauss-Seidel, Conjugate Gradient)
- Support for sparse matrices
- Solution uniqueness checking

#### 5. `vector_operations`
- Dot product, cross product
- Vector norms (L1, L2, Linf)
- Projections, angles between vectors
- Normalization

#### 6. `least_squares_fit`
- Ordinary least squares
- Weighted least squares
- Ridge regression (L2 regularization)
- Lasso regression (L1 regularization)
- Model diagnostics (R², residuals, condition number)

### Group 2: Calculus & Analysis (9 tools)

#### 7. `numerical_derivative`
- Finite difference methods:
  - Forward difference (O(h))
  - Backward difference (O(h))
  - Central difference (O(h²))
  - Five-point stencil (O(h⁴))
- Adaptive step size selection

#### 8. `numerical_differentiation_advanced`
- Richardson extrapolation for higher accuracy
- Adaptive methods with error estimation
- Multipoint formulas

#### 9. `numerical_integration`
- Trapezoidal rule
- Simpson's rule (1/3 and 3/8)
- Romberg integration
- Gaussian quadrature (adaptive)
- Error estimation

#### 10. `symbolic_derivative`
- Symbolic differentiation with simplification
- Support for elementary functions
- Chain rule, product rule, quotient rule
- Expression parsing and simplification

#### 11. `partial_derivatives`
- Gradient computation
- Hessian matrix
- Jacobian matrix
- Directional derivatives

#### 12. `limit_calculator`
- One-sided and two-sided limits
- Limits at infinity
- L'Hôpital's rule for indeterminate forms
- Symbolic limit evaluation

#### 13. `taylor_series`
- Taylor series expansion
- Maclaurin series (Taylor at x=0)
- Remainder term estimation
- Convergence radius

#### 14. `ode_solver`
- Initial value problem solvers:
  - Euler method
  - Runge-Kutta 4th order (RK4)
  - Adaptive step size (RK45)
  - Implicit methods for stiff equations
- Event detection

#### 15. `ode_boundary_value`
- Boundary value problem solvers:
  - Shooting method
  - Finite difference method
  - Collocation methods

### Group 3: Numerical Methods & Equation Solving (5 tools)

#### 16. `root_finding`
- Bisection method (guaranteed convergence)
- Newton-Raphson method (fast convergence)
- Secant method (no derivative needed)
- Brent's method (hybrid approach)
- Convergence criteria and iteration limits

#### 17. `system_of_equations_solver`
- Non-linear system solvers:
  - Newton's method with Jacobian
  - Broyden's method (quasi-Newton)
  - Powell's hybrid method
- Convergence diagnostics

#### 18. `interpolation`
- Linear interpolation
- Polynomial interpolation (Lagrange, Newton)
- Cubic spline interpolation
- Akima spline (overshooting prevention)
- Extrapolation warnings

#### 19. `optimization_1d`
- Golden section search
- Brent's method
- Parabolic interpolation
- Constrained optimization with bounds

#### 20. `curve_fitting_advanced`
- Non-linear least squares (Levenberg-Marquardt)
- Trust region methods
- Parameter confidence intervals
- Goodness of fit metrics

### Group 4: Polynomial & Algebraic Structures (4 tools)

#### 21. `polynomial_arithmetic`
- Addition, subtraction, multiplication
- Division with remainder
- Composition
- Coefficient manipulation

#### 22. `polynomial_roots`
- Analytical solutions (quadratic, cubic, quartic)
- Numerical root finding (companion matrix eigenvalues)
- Root multiplicity detection
- Complex root handling

#### 23. `polynomial_interpolation`
- Construct polynomial through points
- Lagrange interpolation
- Newton divided differences
- Runge phenomenon warning for high degrees

#### 24. `polynomial_analysis`
- Critical points (local min/max)
- Inflection points
- End behavior analysis
- Turning points

### Cross-Cutting Utilities (7 tools)

#### 25. `unit_converter` (Enhanced)
- Current categories + additions:
  - Angles (rad, deg, grad)
  - Pressure (Pa, bar, psi, atm, mmHg)
  - Energy (J, kWh, BTU, cal)
  - Power (W, hp, BTU/hr)
  - Force (N, lbf, kgf)
- Complex unit conversion (e.g., N·m to lbf·ft)

#### 26. `precision_calculator`
- Arbitrary precision arithmetic
- Decimal precision control
- Rounding modes
- Prevent floating point errors

#### 27. `expression_parser`
- Parse mathematical expressions
- Support for variables
- Function evaluation
- Syntax error reporting

#### 28. `symbolic_simplification`
- Algebraic simplification
- Trigonometric identities
- Combining like terms
- Factoring

#### 29. `equation_balancer`
- Balance chemical equations
- Balance physical equations
- Mass/energy balance validation

#### 30. `significant_figures`
- Proper sig fig handling in calculations
- Rounding to significant figures
- Precision propagation

#### 31. `error_propagation`
- Uncertainty through calculations
- First-order error propagation
- Monte Carlo uncertainty estimation
- Correlation handling

## Technical Architecture

### Server Structure
```
src/builtin/engineering_math_server/
├── __init__.py              # Export server class
├── __main__.py              # Module entry point
├── server.py                # EngineeringMathServer class
├── tools/                   # Tool implementations
│   ├── __init__.py
│   ├── linear_algebra.py    # Group 1: 6 tools
│   ├── calculus.py          # Group 2: 9 tools
│   ├── numerical_methods.py # Group 3: 5 tools
│   ├── polynomials.py       # Group 4: 4 tools
│   └── utilities.py         # Utilities: 7 tools
└── README.md                # Server documentation
```

### Dependencies
```python
# requirements.txt additions
numpy>=1.24.0           # Matrix operations, linear algebra
scipy>=1.11.0           # Advanced numerical methods
sympy>=1.12             # Symbolic mathematics
mpmath>=1.3.0           # Arbitrary precision arithmetic
```

### Server Class Implementation
```python
class EngineeringMathServer(BaseMCPServer):
    """Engineering Math MCP Server - Foundation for specialized servers."""
    
    def register_tools(self) -> None:
        """Register all 31 engineering math tools."""
        from .tools import linear_algebra, calculus, numerical_methods, polynomials, utilities
        
        # Register all tools from each module
        
    def get_server_name(self) -> str:
        return "engineering-math-server"
    
    def get_server_version(self) -> str:
        return "1.0.0"
```

## Testing Requirements

### Unit Tests
- Test each tool with known analytical solutions
- Validate numerical accuracy (tolerance checking)
- Edge case handling (singular matrices, convergence failures)
- Input validation and error messages

### Integration Tests
- Tool chaining (e.g., matrix decomposition → linear solver)
- Large-scale problems (performance testing)
- Memory usage profiling

### Test Coverage Target
- Minimum 85% code coverage
- 100% coverage for critical numerical algorithms

## Documentation Requirements

1. **Tool Documentation**
   - Mathematical background for each tool
   - Input/output specifications
   - Example usage with explanations
   - Performance characteristics (Big-O)
   - Numerical stability notes

2. **API Reference**
   - Complete tool schema definitions
   - Parameter descriptions with valid ranges
   - Return value structures
   - Error conditions

3. **User Guide**
   - Getting started examples
   - Common workflows
   - Integration with other servers
   - Troubleshooting guide

## Deliverables

- [ ] Server implementation with all 31 tools
- [ ] Comprehensive test suite (85%+ coverage)
- [ ] Server documentation (README.md)
- [ ] Individual tool documentation
- [ ] Example notebooks demonstrating usage
- [ ] Wrapper script: `start_engineering_math_server.py`
- [ ] Claude Desktop configuration example
- [ ] Performance benchmarks

## Success Criteria

- ✅ All 31 tools implemented and functional
- ✅ Tests pass with >85% coverage
- ✅ Numerical accuracy validated against known solutions
- ✅ Server runs in both stdio and HTTP modes
- ✅ Claude Desktop integration working
- ✅ Documentation complete and clear
- ✅ Tools can be imported by other servers
- ✅ Performance acceptable (<1s for typical operations)

## Implementation Notes

### Numerical Stability
- Use stable algorithms (e.g., QR for least squares, not normal equations)
- Check condition numbers for linear systems
- Warn users about ill-conditioned problems
- Provide regularization options

### Error Handling
- Validate inputs (dimensions, domains, convergence criteria)
- Provide informative error messages
- Graceful degradation when possible
- Return partial results with warnings when appropriate

### Code Organization
- One tool per function in appropriate module
- Share common utilities (e.g., validation functions)
- Keep tool handlers focused and readable
- Use type hints throughout

### Performance Considerations
- Leverage NumPy vectorization
- Use SciPy optimized routines
- Profile critical paths
- Consider caching for expensive operations

## Timeline

**Week 1**:
- Set up server structure
- Implement Linear Algebra tools (Group 1)
- Initial testing framework

**Week 2**:
- Implement Calculus tools (Group 2)
- Implement Numerical Methods (Group 3)
- Unit tests for Groups 1-3

**Week 3**:
- Implement Polynomial tools (Group 4)
- Implement Utilities
- Complete testing
- Documentation

## Related Issues

- Blocks: #80 (Complex Analysis), #81 (Transforms), #82 (Geometry)
- Blocks: #83 (Control Systems), #84 (Process Engineering)
- Blocks: #85 (Signal Processing), #86 (Structural Analysis)
- Related: #89 (Integration Testing)

## References

- Numerical Recipes in Python
- SciPy documentation
- NumPy Linear Algebra reference
- SymPy symbolic mathematics guide
