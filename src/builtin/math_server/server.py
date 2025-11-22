"""
Math Calculator MCP Server - Refactored using BaseMCPServer.

This server provides tools for Fibonacci, prime numbers, number theory,
sequence generation, cryptographic hashing, unit conversion, date calculations,
and text processing.
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
    MATH_TOOLS,
    handle_fibonacci,
    handle_is_prime,
    handle_generate_primes,
    handle_nth_prime,
    handle_prime_factorization,
    handle_gcd,
    handle_lcm,
    handle_factorial,
    handle_combinations,
    handle_permutations,
    handle_pascal_triangle,
    handle_triangular_numbers,
    handle_perfect_numbers,
    handle_collatz_sequence,
    handle_generate_hash,
    handle_unit_convert,
    handle_date_diff,
    handle_date_add,
    handle_business_days,
    handle_age_calculator,
    handle_day_of_week,
    handle_text_stats,
    handle_word_frequency,
    handle_text_transform,
    handle_encode_decode,
)


class MathServer(BaseMCPServer):
    """
    Math Calculator MCP Server with 25+ mathematical tools.
    
    Inherits from BaseMCPServer to leverage:
    - Dual transport support (stdio/HTTP)
    - Automatic endpoint management
    - Built-in health/ready/metrics endpoints
    - Graceful shutdown handling
    - Middleware integration (CORS, auth, rate limiting)
    """
    
    def register_tools(self) -> None:
        """Register all mathematical tools."""
        # Tool name to handler mapping
        tool_handlers = {
            "calculate_fibonacci": handle_fibonacci,
            "is_prime": handle_is_prime,
            "generate_primes": handle_generate_primes,
            "nth_prime": handle_nth_prime,
            "prime_factorization": handle_prime_factorization,
            "gcd": handle_gcd,
            "lcm": handle_lcm,
            "factorial": handle_factorial,
            "combinations": handle_combinations,
            "permutations": handle_permutations,
            "pascal_triangle": handle_pascal_triangle,
            "triangular_numbers": handle_triangular_numbers,
            "perfect_numbers": handle_perfect_numbers,
            "collatz_sequence": handle_collatz_sequence,
            "generate_hash": handle_generate_hash,
            "unit_convert": handle_unit_convert,
            "date_diff": handle_date_diff,
            "date_add": handle_date_add,
            "business_days": handle_business_days,
            "age_calculator": handle_age_calculator,
            "day_of_week": handle_day_of_week,
            "text_stats": handle_text_stats,
            "word_frequency": handle_word_frequency,
            "text_transform": handle_text_transform,
            "encode_decode": handle_encode_decode,
        }
        
        # Register each tool with its handler
        for tool in MATH_TOOLS:
            handler = tool_handlers.get(tool.name)
            if handler:
                self.tool_registry.register_tool(tool, handler)
            else:
                self.logger.warning(f"No handler found for tool: {tool.name}")
    
    def get_server_name(self) -> str:
        """Return the server name."""
        return "math-calculator"
    
    def get_server_version(self) -> str:
        """Return the server version."""
        return "2.0.0"


def main():
    """Entry point for the Math Calculator MCP Server."""
    # Create argument parser with server description
    parser = BaseMCPServer.create_argument_parser(
        description="Math Calculator MCP Server - Provides 25+ mathematical tools"
    )
    args = parser.parse_args()
    
    # Create and run the server
    server = MathServer(config_path=args.config)
    server.run(
        transport=args.transport,
        host=args.host,
        port=args.port,
        dev_mode=args.dev
    )


if __name__ == "__main__":
    main()
