"""
Engineering Math Server Tools.

This module exports all tools and handlers for the Engineering Math Server.
"""

from .linear_algebra import (
    LINEAR_ALGEBRA_TOOLS,
    handle_matrix_operations,
    handle_matrix_inverse,
    handle_matrix_decomposition,
    handle_solve_linear_system,
    handle_vector_operations,
    handle_least_squares_fit,
)

from .calculus import (
    CALCULUS_TOOLS,
    handle_numerical_derivative,
    handle_numerical_differentiation_advanced,
    handle_numerical_integration,
    handle_symbolic_derivative,
    handle_partial_derivatives,
    handle_limit_calculator,
    handle_taylor_series,
    handle_ode_solver,
    handle_ode_boundary_value,
)

from .numerical_methods import (
    NUMERICAL_METHODS_TOOLS,
    handle_root_finding,
    handle_system_of_equations_solver,
    handle_interpolation,
    handle_optimization_1d,
    handle_curve_fitting_advanced,
)

from .polynomials import (
    POLYNOMIAL_TOOLS,
    handle_polynomial_arithmetic,
    handle_polynomial_roots,
    handle_polynomial_interpolation,
    handle_polynomial_analysis,
)

from .utilities import (
    UTILITY_TOOLS,
    handle_unit_converter,
    handle_precision_calculator,
    handle_expression_parser,
    handle_symbolic_simplification,
    handle_equation_balancer,
    handle_significant_figures,
    handle_error_propagation,
)

from .complex_analysis import (
    COMPLEX_ANALYSIS_TOOLS,
    handle_complex_operations,
    handle_complex_functions,
    handle_roots_of_unity,
    handle_complex_conjugate_operations,
)

# Combine all tools
ALL_TOOLS = (
    LINEAR_ALGEBRA_TOOLS +
    CALCULUS_TOOLS +
    NUMERICAL_METHODS_TOOLS +
    POLYNOMIAL_TOOLS +
    UTILITY_TOOLS +
    COMPLEX_ANALYSIS_TOOLS
)

__all__ = [
    # Tool lists
    "ALL_TOOLS",
    "LINEAR_ALGEBRA_TOOLS",
    "CALCULUS_TOOLS",
    "NUMERICAL_METHODS_TOOLS",
    "POLYNOMIAL_TOOLS",
    "UTILITY_TOOLS",
    "COMPLEX_ANALYSIS_TOOLS",
    
    # Linear algebra handlers
    "handle_matrix_operations",
    "handle_matrix_inverse",
    "handle_matrix_decomposition",
    "handle_solve_linear_system",
    "handle_vector_operations",
    "handle_least_squares_fit",
    
    # Calculus handlers
    "handle_numerical_derivative",
    "handle_numerical_differentiation_advanced",
    "handle_numerical_integration",
    "handle_symbolic_derivative",
    "handle_partial_derivatives",
    "handle_limit_calculator",
    "handle_taylor_series",
    "handle_ode_solver",
    "handle_ode_boundary_value",
    
    # Numerical methods handlers
    "handle_root_finding",
    "handle_system_of_equations_solver",
    "handle_interpolation",
    "handle_optimization_1d",
    "handle_curve_fitting_advanced",
    
    # Polynomial handlers
    "handle_polynomial_arithmetic",
    "handle_polynomial_roots",
    "handle_polynomial_interpolation",
    "handle_polynomial_analysis",
    
    # Utility handlers
    "handle_unit_converter",
    "handle_precision_calculator",
    "handle_expression_parser",
    "handle_symbolic_simplification",
    "handle_equation_balancer",
    "handle_significant_figures",
    "handle_error_propagation",
    
    # Complex analysis handlers
    "handle_complex_operations",
    "handle_complex_functions",
    "handle_roots_of_unity",
    "handle_complex_conjugate_operations",
]
