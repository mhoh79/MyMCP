"""
Engineering Math MCP Server - Foundation for specialized engineering servers.

This server provides 31+ core mathematical tools across 5 categories:
- Linear Algebra & Matrix Mathematics (6 tools)
- Calculus & Analysis (9 tools)
- Numerical Methods & Equation Solving (5 tools)
- Polynomial & Algebraic Structures (4 tools)
- Cross-Cutting Utilities (7 tools)
"""

import sys
from pathlib import Path
from typing import Optional

# Add parent directory to path
_parent_dir = Path(__file__).parent.parent.parent
if str(_parent_dir) not in sys.path:
    sys.path.insert(0, str(_parent_dir))

from core import BaseMCPServer
from .tools import (
    ALL_TOOLS,
    # Linear algebra handlers
    handle_matrix_operations,
    handle_matrix_inverse,
    handle_matrix_decomposition,
    handle_solve_linear_system,
    handle_vector_operations,
    handle_least_squares_fit,
    # Calculus handlers
    handle_numerical_derivative,
    handle_numerical_differentiation_advanced,
    handle_numerical_integration,
    handle_symbolic_derivative,
    handle_partial_derivatives,
    handle_limit_calculator,
    handle_taylor_series,
    handle_ode_solver,
    handle_ode_boundary_value,
    # Numerical methods handlers
    handle_root_finding,
    handle_system_of_equations_solver,
    handle_interpolation,
    handle_optimization_1d,
    handle_curve_fitting_advanced,
    # Polynomial handlers
    handle_polynomial_arithmetic,
    handle_polynomial_roots,
    handle_polynomial_interpolation,
    handle_polynomial_analysis,
    # Utility handlers
    handle_unit_converter,
    handle_precision_calculator,
    handle_expression_parser,
    handle_symbolic_simplification,
    handle_equation_balancer,
    handle_significant_figures,
    handle_error_propagation,
)


class EngineeringMathServer(BaseMCPServer):
    """
    Engineering Math MCP Server - Foundation for specialized servers.
    
    Provides core mathematical tools used by Control Systems, Process Engineering,
    Signal Processing, Structural Analysis, and other specialized application servers.
    
    Inherits from BaseMCPServer to leverage:
    - Dual transport support (stdio/HTTP)
    - Automatic endpoint management
    - Built-in health/ready/metrics endpoints
    - Graceful shutdown handling
    - Middleware integration (CORS, auth, rate limiting)
    """
    
    def register_tools(self) -> None:
        """Register all 31+ engineering math tools."""
        # Tool name to handler mapping
        tool_handlers = {
            # Linear Algebra (6 tools)
            "matrix_operations": handle_matrix_operations,
            "matrix_inverse": handle_matrix_inverse,
            "matrix_decomposition": handle_matrix_decomposition,
            "solve_linear_system": handle_solve_linear_system,
            "vector_operations": handle_vector_operations,
            "least_squares_fit": handle_least_squares_fit,
            
            # Calculus (9 tools)
            "numerical_derivative": handle_numerical_derivative,
            "numerical_differentiation_advanced": handle_numerical_differentiation_advanced,
            "numerical_integration": handle_numerical_integration,
            "symbolic_derivative": handle_symbolic_derivative,
            "partial_derivatives": handle_partial_derivatives,
            "limit_calculator": handle_limit_calculator,
            "taylor_series": handle_taylor_series,
            "ode_solver": handle_ode_solver,
            "ode_boundary_value": handle_ode_boundary_value,
            
            # Numerical Methods (5 tools)
            "root_finding": handle_root_finding,
            "system_of_equations_solver": handle_system_of_equations_solver,
            "interpolation": handle_interpolation,
            "optimization_1d": handle_optimization_1d,
            "curve_fitting_advanced": handle_curve_fitting_advanced,
            
            # Polynomials (4 tools)
            "polynomial_arithmetic": handle_polynomial_arithmetic,
            "polynomial_roots": handle_polynomial_roots,
            "polynomial_interpolation": handle_polynomial_interpolation,
            "polynomial_analysis": handle_polynomial_analysis,
            
            # Utilities (7 tools)
            "unit_converter": handle_unit_converter,
            "precision_calculator": handle_precision_calculator,
            "expression_parser": handle_expression_parser,
            "symbolic_simplification": handle_symbolic_simplification,
            "equation_balancer": handle_equation_balancer,
            "significant_figures": handle_significant_figures,
            "error_propagation": handle_error_propagation,
        }
        
        # Register each tool with its handler
        for tool in ALL_TOOLS:
            handler = tool_handlers.get(tool.name)
            if handler:
                self.tool_registry.register_tool(tool, handler)
            else:
                self.logger.warning(f"No handler found for tool: {tool.name}")
    
    def get_server_name(self) -> str:
        """Return the server name."""
        return "engineering-math-server"
    
    def get_server_version(self) -> str:
        """Return the server version."""
        return "1.0.0"


def main():
    """Entry point for the Engineering Math MCP Server."""
    # Create argument parser with server description
    parser = BaseMCPServer.create_argument_parser(
        description="Engineering Math MCP Server - Provides 31+ core mathematical tools"
    )
    args = parser.parse_args()
    
    # Create and run the server
    server = EngineeringMathServer(config_path=args.config)
    server.run(
        transport=args.transport,
        host=args.host,
        port=args.port,
        dev_mode=args.dev
    )


if __name__ == "__main__":
    main()
