"""
Main MCP server implementation for Fibonacci calculations.
This server exposes a tool that calculates Fibonacci numbers.
"""

import asyncio
import logging
from typing import Any

# MCP SDK imports for building the server
from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.types import (
    Tool,
    TextContent,
    CallToolResult,
    INVALID_PARAMS,
    INTERNAL_ERROR,
)

# Configure logging to stderr (stdout is reserved for MCP protocol messages)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler()],  # Logs go to stderr by default
)
logger = logging.getLogger("fibonacci-server")


def calculate_fibonacci(n: int) -> int:
    """
    Calculate the nth Fibonacci number using iterative approach.
    
    Args:
        n: The position in the Fibonacci sequence (0-indexed)
        
    Returns:
        The Fibonacci number at position n
        
    Raises:
        ValueError: If n is negative
        
    Examples:
        fib(0) = 0
        fib(1) = 1
        fib(5) = 5
        fib(10) = 55
    """
    if n < 0:
        raise ValueError("Fibonacci is not defined for negative numbers")
    
    if n <= 1:
        return n
    
    # Iterative approach is more efficient than recursive for large n
    # Avoids stack overflow and has O(n) time complexity
    prev, curr = 0, 1
    for _ in range(2, n + 1):
        prev, curr = curr, prev + curr
    
    return curr


def calculate_fibonacci_sequence(n: int) -> list[int]:
    """
    Calculate the Fibonacci sequence up to the nth number.
    
    Args:
        n: The number of Fibonacci numbers to generate
        
    Returns:
        List containing the first n Fibonacci numbers
        
    Examples:
        fib_sequence(5) = [0, 1, 1, 2, 3]
        fib_sequence(10) = [0, 1, 1, 2, 3, 5, 8, 13, 21, 34]
    """
    if n <= 0:
        return []
    
    sequence = [0]
    if n == 1:
        return sequence
    
    sequence.append(1)
    for i in range(2, n):
        sequence.append(sequence[i-1] + sequence[i-2])
    
    return sequence


# ============================================================================
# Prime Number Functions
# ============================================================================


def is_prime(n: int) -> bool:
    """
    Check if a number is prime using trial division with optimization.
    
    This function uses the trial division algorithm, checking divisibility
    only up to the square root of n. This is efficient because if n has
    a divisor greater than sqrt(n), it must also have a divisor less than sqrt(n).
    
    Args:
        n: The number to check for primality (must be >= 2)
        
    Returns:
        True if n is prime, False otherwise
        
    Raises:
        ValueError: If n is less than 2
        
    Examples:
        is_prime(2) = True   # 2 is the smallest prime
        is_prime(17) = True  # 17 is prime
        is_prime(20) = False # 20 = 2 × 2 × 5
        is_prime(97) = True  # 97 is prime
        
    Algorithm Explanation:
        1. Numbers less than 2 are not prime by definition
        2. 2 and 3 are prime numbers
        3. Even numbers (except 2) are not prime
        4. For odd numbers, test divisibility from 3 to sqrt(n) by odd numbers only
        5. If no divisor is found, the number is prime
    """
    if n < 2:
        raise ValueError("Prime checking is only defined for integers >= 2")
    
    # 2 and 3 are prime
    if n <= 3:
        return True
    
    # Even numbers (except 2) are not prime
    if n % 2 == 0:
        return False
    
    # Check odd divisors up to sqrt(n)
    # We only need to check up to sqrt(n) because if n = a × b,
    # one of the factors must be <= sqrt(n)
    i = 3
    while i * i <= n:
        if n % i == 0:
            return False
        i += 2  # Only check odd numbers
    
    return True


def generate_primes(limit: int) -> list[int]:
    """
    Generate all prime numbers up to the given limit using the Sieve of Eratosthenes.
    
    The Sieve of Eratosthenes is an ancient and efficient algorithm for finding
    all primes up to a given limit. It works by iteratively marking the multiples
    of each prime as composite (not prime).
    
    Args:
        limit: The upper bound (inclusive) for prime generation
        
    Returns:
        List of all prime numbers from 2 to limit (inclusive)
        
    Raises:
        ValueError: If limit is less than 2
        
    Examples:
        generate_primes(10) = [2, 3, 5, 7]
        generate_primes(20) = [2, 3, 5, 7, 11, 13, 17, 19]
        generate_primes(50) = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47]
        
    Algorithm Explanation:
        1. Create a boolean array "is_prime" of size limit+1, initially all True
        2. Mark 0 and 1 as not prime
        3. Starting from 2, for each number i that is still marked as prime:
           a. Add i to the result list
           b. Mark all multiples of i (i², i²+i, i²+2i, ...) as not prime
        4. Continue until we've processed all numbers up to sqrt(limit)
        
    Time Complexity: O(n log log n) where n is the limit
    Space Complexity: O(n) for the boolean array
    """
    if limit < 2:
        raise ValueError("Limit must be at least 2 to generate primes")
    
    # Create a boolean array "is_prime" and initialize all entries as True
    # Index i represents the number i
    is_prime_arr = [True] * (limit + 1)
    is_prime_arr[0] = is_prime_arr[1] = False  # 0 and 1 are not prime
    
    # Start with the smallest prime number, 2
    p = 2
    while p * p <= limit:
        # If is_prime_arr[p] is still True, then it's a prime
        if is_prime_arr[p]:
            # Mark all multiples of p as not prime
            # Start from p² because smaller multiples have already been marked
            # by smaller primes
            for i in range(p * p, limit + 1, p):
                is_prime_arr[i] = False
        p += 1
    
    # Collect all numbers that are still marked as prime
    primes = [i for i in range(2, limit + 1) if is_prime_arr[i]]
    return primes


def nth_prime(n: int) -> int:
    """
    Find the nth prime number (1-indexed).
    
    This function generates primes sequentially until we reach the nth one.
    It uses a growing sieve approach, expanding the search space as needed.
    
    Args:
        n: The position of the desired prime (1-indexed, must be >= 1)
        
    Returns:
        The nth prime number
        
    Raises:
        ValueError: If n is less than 1
        
    Examples:
        nth_prime(1) = 2    # 1st prime is 2
        nth_prime(5) = 11   # 5th prime is 11
        nth_prime(10) = 29  # 10th prime is 29
        nth_prime(100) = 541 # 100th prime is 541
        
    Algorithm Explanation:
        1. Use an estimate for the upper bound based on prime number theorem
        2. Generate primes up to that limit using Sieve of Eratosthenes
        3. If we don't have enough primes, expand the limit and try again
        4. Return the nth prime from the generated list
        
    Note: For very large n, this could be memory-intensive. For n <= 10000,
    it performs well with reasonable memory usage.
    """
    if n < 1:
        raise ValueError("n must be at least 1")
    
    # For small values of n, use a simple approach
    if n == 1:
        return 2
    if n == 2:
        return 3
    
    # Estimate upper bound for the nth prime using approximation:
    # p_n ≈ n * (ln(n) + ln(ln(n))) for n >= 6
    # For smaller n, use a safe default
    import math
    if n < 6:
        limit = 15
    else:
        # Prime number theorem approximation with a safety margin
        limit = int(n * (math.log(n) + math.log(math.log(n)) + 2))
    
    # Generate primes up to the estimated limit
    primes = generate_primes(limit)
    
    # If we don't have enough primes, expand the limit
    while len(primes) < n:
        limit *= 2
        primes = generate_primes(limit)
    
    # Return the nth prime (1-indexed)
    return primes[n - 1]


def prime_factorization(n: int) -> list[list[int]]:
    """
    Find the prime factorization of a number with exponents.
    
    Returns the prime factors as a list of [prime, exponent] pairs.
    For example, 24 = 2³ × 3¹ is returned as [[2, 3], [3, 1]].
    
    Args:
        n: The number to factorize (must be >= 2)
        
    Returns:
        List of [prime, exponent] pairs representing the prime factorization
        Empty list if n is 1
        
    Raises:
        ValueError: If n is less than 2
        
    Examples:
        prime_factorization(2) = [[2, 1]]           # 2¹
        prime_factorization(24) = [[2, 3], [3, 1]]  # 2³ × 3¹
        prime_factorization(60) = [[2, 2], [3, 1], [5, 1]]  # 2² × 3¹ × 5¹
        prime_factorization(97) = [[97, 1]]         # 97 is prime
        prime_factorization(100) = [[2, 2], [5, 2]] # 2² × 5²
        
    Algorithm Explanation:
        1. Divide n by 2 repeatedly, counting occurrences
        2. For odd numbers from 3 to sqrt(n), divide repeatedly if divisible
        3. If n > 1 after all divisions, n itself is a prime factor
        4. Return factors with their exponents as [prime, exponent] pairs
        
    Time Complexity: O(sqrt(n)) in the worst case
    """
    if n < 2:
        raise ValueError("Prime factorization is only defined for integers >= 2")
    
    factors = []
    
    # Check for factor 2
    if n % 2 == 0:
        count = 0
        while n % 2 == 0:
            count += 1
            n //= 2
        factors.append([2, count])
    
    # Check for odd factors from 3 onwards
    # Only need to check up to sqrt(n)
    i = 3
    while i * i <= n:
        if n % i == 0:
            count = 0
            while n % i == 0:
                count += 1
                n //= i
            factors.append([i, count])
        i += 2  # Only check odd numbers
    
    # If n is still greater than 1, then it's a prime factor
    if n > 1:
        factors.append([n, 1])
    
    return factors


# Initialize the MCP server with a descriptive name
app = Server("fibonacci-calculator")


@app.list_tools()
async def list_tools() -> list[Tool]:
    """
    Register and list all available tools that this MCP server provides.
    
    This function is called by MCP clients to discover what tools are available.
    Each tool has a name, description, and input schema defined using JSON Schema.
    
    Returns:
        List of Tool objects describing available tools
    """
    logger.info("Client requested tool list")
    
    return [
        Tool(
            name="calculate_fibonacci",
            description=(
                "Calculate the nth Fibonacci number or generate a Fibonacci sequence. "
                "Supports both single value calculation and sequence generation. "
                "The Fibonacci sequence starts: 0, 1, 1, 2, 3, 5, 8, 13, 21, 34..."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "n": {
                        "type": "integer",
                        "description": "The position in the Fibonacci sequence (0-indexed) or count of numbers to generate",
                        "minimum": 0,
                        "maximum": 1000,  # Reasonable limit to prevent performance issues
                    },
                    "return_sequence": {
                        "type": "boolean",
                        "description": "If true, returns the entire sequence up to n. If false, returns only the nth number.",
                        "default": False,
                    },
                },
                "required": ["n"],
            },
        ),
        Tool(
            name="is_prime",
            description=(
                "Check if a number is prime. Uses optimized trial division algorithm "
                "that checks divisibility up to the square root of n. "
                "Returns a boolean result with explanation."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "n": {
                        "type": "integer",
                        "description": "The number to check for primality",
                        "minimum": 2,
                        "maximum": 1000000,
                    },
                },
                "required": ["n"],
            },
        ),
        Tool(
            name="generate_primes",
            description=(
                "Generate all prime numbers up to a given limit using the Sieve of Eratosthenes algorithm. "
                "This is an efficient method for finding all primes up to a specified number. "
                "Returns a list of all prime numbers from 2 to the limit."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "limit": {
                        "type": "integer",
                        "description": "The upper bound (inclusive) for prime generation",
                        "minimum": 2,
                        "maximum": 10000,
                    },
                },
                "required": ["limit"],
            },
        ),
        Tool(
            name="nth_prime",
            description=(
                "Find the nth prime number (1-indexed). "
                "For example, the 1st prime is 2, the 5th prime is 11, and the 10th prime is 29. "
                "Uses efficient prime generation to find the requested prime."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "n": {
                        "type": "integer",
                        "description": "The position of the desired prime number (1-indexed)",
                        "minimum": 1,
                        "maximum": 10000,
                    },
                },
                "required": ["n"],
            },
        ),
        Tool(
            name="prime_factorization",
            description=(
                "Find the prime factorization of a number with exponents. "
                "Returns a list of [prime, exponent] pairs. "
                "For example, 24 = 2³ × 3¹ is returned as [[2, 3], [3, 1]]."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "n": {
                        "type": "integer",
                        "description": "The number to factorize",
                        "minimum": 2,
                        "maximum": 1000000,
                    },
                },
                "required": ["n"],
            },
        ),
    ]


@app.call_tool()
async def call_tool(name: str, arguments: Any) -> CallToolResult:
    """
    Handle tool execution requests from MCP clients.
    
    This function is called when an MCP client (like Claude) wants to use one of
    the tools that this server provides. It validates inputs, executes the tool,
    and returns results in the proper MCP format.
    
    Args:
        name: The name of the tool to execute
        arguments: Dictionary containing the tool's input parameters
        
    Returns:
        CallToolResult containing the tool's output or error information
    """
    logger.info(f"Tool call requested: {name} with arguments: {arguments}")
    
    try:
        # Route to appropriate tool handler
        if name == "calculate_fibonacci":
            return await handle_fibonacci(arguments)
        elif name == "is_prime":
            return await handle_is_prime(arguments)
        elif name == "generate_primes":
            return await handle_generate_primes(arguments)
        elif name == "nth_prime":
            return await handle_nth_prime(arguments)
        elif name == "prime_factorization":
            return await handle_prime_factorization(arguments)
        else:
            logger.error(f"Unknown tool requested: {name}")
            return CallToolResult(
                content=[TextContent(type="text", text=f"Unknown tool: {name}")],
                isError=True,
            )
        
    except ValueError as e:
        # Handle calculation errors (e.g., invalid inputs)
        logger.error(f"Calculation error: {e}")
        return CallToolResult(
            content=[TextContent(type="text", text=f"Calculation error: {str(e)}")],
            isError=True,
        )
    except Exception as e:
        # Handle unexpected errors
        logger.exception(f"Unexpected error during tool execution: {e}")
        return CallToolResult(
            content=[TextContent(type="text", text=f"Internal error: {str(e)}")],
            isError=True,
        )


async def handle_fibonacci(arguments: Any) -> CallToolResult:
    """Handle calculate_fibonacci tool calls."""
    # Extract and validate parameters
    n = arguments.get("n")
    return_sequence = arguments.get("return_sequence", False)
    
    # Validate required parameter
    if n is None:
        logger.error("Missing required parameter: n")
        return CallToolResult(
            content=[TextContent(
                type="text",
                text="Missing required parameter 'n'"
            )],
            isError=True,
        )
    
    # Validate parameter type and range
    if not isinstance(n, int):
        logger.error(f"Invalid parameter type for n: {type(n)}")
        return CallToolResult(
            content=[TextContent(
                type="text",
                text=f"Parameter 'n' must be an integer, got {type(n).__name__}"
            )],
            isError=True,
        )
    
    if n < 0 or n > 1000:
        logger.error(f"Parameter n out of range: {n}")
        return CallToolResult(
            content=[TextContent(
                type="text",
                text="Parameter 'n' must be between 0 and 1000"
            )],
            isError=True,
        )
    
    # Execute the appropriate calculation
    if return_sequence:
        logger.info(f"Calculating Fibonacci sequence up to position {n}")
        result = calculate_fibonacci_sequence(n)
        result_text = (
            f"Fibonacci sequence (first {n} numbers):\n"
            f"{result}\n\n"
            f"Last number: {result[-1] if result else 'N/A'}"
        )
    else:
        logger.info(f"Calculating Fibonacci number at position {n}")
        result = calculate_fibonacci(n)
        result_text = f"Fibonacci number at position {n}: {result}"
    
    logger.info(f"Calculation successful: {result}")
    
    # Return successful result in MCP format
    return CallToolResult(
        content=[TextContent(type="text", text=result_text)],
        isError=False,
    )


async def handle_is_prime(arguments: Any) -> CallToolResult:
    """Handle is_prime tool calls."""
    # Extract and validate parameters
    n = arguments.get("n")
    
    # Validate required parameter
    if n is None:
        logger.error("Missing required parameter: n")
        return CallToolResult(
            content=[TextContent(
                type="text",
                text="Missing required parameter 'n'"
            )],
            isError=True,
        )
    
    # Validate parameter type and range
    if not isinstance(n, int):
        logger.error(f"Invalid parameter type for n: {type(n)}")
        return CallToolResult(
            content=[TextContent(
                type="text",
                text=f"Parameter 'n' must be an integer, got {type(n).__name__}"
            )],
            isError=True,
        )
    
    if n < 2 or n > 1000000:
        logger.error(f"Parameter n out of range: {n}")
        return CallToolResult(
            content=[TextContent(
                type="text",
                text="Parameter 'n' must be between 2 and 1000000"
            )],
            isError=True,
        )
    
    # Check if the number is prime
    logger.info(f"Checking if {n} is prime")
    result = is_prime(n)
    
    if result:
        result_text = f"Yes, {n} is a prime number."
    else:
        result_text = f"No, {n} is not a prime number."
    
    logger.info(f"Prime check result for {n}: {result}")
    
    return CallToolResult(
        content=[TextContent(type="text", text=result_text)],
        isError=False,
    )


async def handle_generate_primes(arguments: Any) -> CallToolResult:
    """Handle generate_primes tool calls."""
    # Extract and validate parameters
    limit = arguments.get("limit")
    
    # Validate required parameter
    if limit is None:
        logger.error("Missing required parameter: limit")
        return CallToolResult(
            content=[TextContent(
                type="text",
                text="Missing required parameter 'limit'"
            )],
            isError=True,
        )
    
    # Validate parameter type and range
    if not isinstance(limit, int):
        logger.error(f"Invalid parameter type for limit: {type(limit)}")
        return CallToolResult(
            content=[TextContent(
                type="text",
                text=f"Parameter 'limit' must be an integer, got {type(limit).__name__}"
            )],
            isError=True,
        )
    
    if limit < 2 or limit > 10000:
        logger.error(f"Parameter limit out of range: {limit}")
        return CallToolResult(
            content=[TextContent(
                type="text",
                text="Parameter 'limit' must be between 2 and 10000"
            )],
            isError=True,
        )
    
    # Generate primes up to the limit
    logger.info(f"Generating primes up to {limit}")
    primes = generate_primes(limit)
    
    result_text = (
        f"Prime numbers up to {limit}:\n"
        f"{primes}\n\n"
        f"Count: {len(primes)} prime numbers found"
    )
    
    logger.info(f"Generated {len(primes)} primes up to {limit}")
    
    return CallToolResult(
        content=[TextContent(type="text", text=result_text)],
        isError=False,
    )


async def handle_nth_prime(arguments: Any) -> CallToolResult:
    """Handle nth_prime tool calls."""
    # Extract and validate parameters
    n = arguments.get("n")
    
    # Validate required parameter
    if n is None:
        logger.error("Missing required parameter: n")
        return CallToolResult(
            content=[TextContent(
                type="text",
                text="Missing required parameter 'n'"
            )],
            isError=True,
        )
    
    # Validate parameter type and range
    if not isinstance(n, int):
        logger.error(f"Invalid parameter type for n: {type(n)}")
        return CallToolResult(
            content=[TextContent(
                type="text",
                text=f"Parameter 'n' must be an integer, got {type(n).__name__}"
            )],
            isError=True,
        )
    
    if n < 1 or n > 10000:
        logger.error(f"Parameter n out of range: {n}")
        return CallToolResult(
            content=[TextContent(
                type="text",
                text="Parameter 'n' must be between 1 and 10000"
            )],
            isError=True,
        )
    
    # Find the nth prime
    logger.info(f"Finding the {n}th prime number")
    prime = nth_prime(n)
    
    result_text = f"The {n}th prime number is: {prime}"
    
    logger.info(f"The {n}th prime is {prime}")
    
    return CallToolResult(
        content=[TextContent(type="text", text=result_text)],
        isError=False,
    )


async def handle_prime_factorization(arguments: Any) -> CallToolResult:
    """Handle prime_factorization tool calls."""
    # Extract and validate parameters
    n = arguments.get("n")
    
    # Validate required parameter
    if n is None:
        logger.error("Missing required parameter: n")
        return CallToolResult(
            content=[TextContent(
                type="text",
                text="Missing required parameter 'n'"
            )],
            isError=True,
        )
    
    # Validate parameter type and range
    if not isinstance(n, int):
        logger.error(f"Invalid parameter type for n: {type(n)}")
        return CallToolResult(
            content=[TextContent(
                type="text",
                text=f"Parameter 'n' must be an integer, got {type(n).__name__}"
            )],
            isError=True,
        )
    
    if n < 2 or n > 1000000:
        logger.error(f"Parameter n out of range: {n}")
        return CallToolResult(
            content=[TextContent(
                type="text",
                text="Parameter 'n' must be between 2 and 1000000"
            )],
            isError=True,
        )
    
    # Find the prime factorization
    logger.info(f"Finding prime factorization of {n}")
    factors = prime_factorization(n)
    
    # Format the result nicely
    if not factors:
        result_text = f"{n} = 1 (no prime factors)"
    else:
        # Build the factorization string
        factor_strings = [f"{p}^{e}" if e > 1 else str(p) for p, e in factors]
        factorization_str = " × ".join(factor_strings)
        
        result_text = (
            f"Prime factorization of {n}:\n"
            f"{n} = {factorization_str}\n\n"
            f"Factors: {factors}"
        )
    
    logger.info(f"Prime factorization of {n}: {factors}")
    
    return CallToolResult(
        content=[TextContent(type="text", text=result_text)],
        isError=False,
    )


async def main():
    """
    Main entry point for the MCP server.
    
    This function starts the server using stdio transport, which means:
    - The server reads MCP protocol messages from stdin
    - The server writes MCP protocol messages to stdout
    - All logging and debugging output goes to stderr
    
    This is the standard transport for MCP servers that will be launched
    by client applications like Claude Desktop.
    """
    logger.info("Starting Fibonacci MCP server")
    
    # Run the server using stdio transport
    # This will block until the server receives a shutdown signal
    async with stdio_server() as (read_stream, write_stream):
        logger.info("Server started, waiting for requests...")
        await app.run(
            read_stream,
            write_stream,
            app.create_initialization_options()
        )
    
    logger.info("Server shut down successfully")


# Entry point when running as a script
if __name__ == "__main__":
    # Run the async main function
    asyncio.run(main())
