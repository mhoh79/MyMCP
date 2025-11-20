"""
Main MCP server implementation for Fibonacci calculations.
This server exposes a tool that calculates Fibonacci numbers.
"""

import asyncio
import base64
import hashlib
import logging
from datetime import datetime, timedelta
from typing import Any

# Import dateutil for advanced date operations
from dateutil.relativedelta import relativedelta

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


# ============================================================================
# Sequence Generator Functions
# ============================================================================


# Memoization cache for Pascal's triangle to improve performance
_pascal_cache: dict[int, list[list[int]]] = {}


def pascal_triangle(rows: int) -> list[list[int]]:
    """
    Generate Pascal's triangle up to n rows with memoization.
    
    Pascal's triangle is a triangular array where each number is the sum of the
    two numbers directly above it. The triangle starts with 1 at the top, and
    each row begins and ends with 1.
    
    Args:
        rows: Number of rows to generate (1-30)
        
    Returns:
        2D array representing Pascal's triangle, where each inner list is a row
        
    Raises:
        ValueError: If rows is less than 1 or greater than 30
        
    Examples:
        pascal_triangle(1) = [[1]]
        pascal_triangle(3) = [[1], [1, 1], [1, 2, 1]]
        pascal_triangle(5) = [[1], [1, 1], [1, 2, 1], [1, 3, 3, 1], [1, 4, 6, 4, 1]]
        
    Mathematical Background:
        - Each entry is a binomial coefficient C(n,k)
        - Entry in row n, position k is: C(n,k) = n! / (k! × (n-k)!)
        - Used in probability, combinatorics, and algebra
        - The sum of row n equals 2^n
        
    Algorithm Explanation:
        1. Start with row 0: [1]
        2. For each new row:
           a. Start and end with 1
           b. Each interior element is the sum of two elements from previous row
        3. Use memoization to avoid recalculating previously computed rows
        
    Time Complexity: O(rows²) without memoization, O(new_rows²) with memoization
    Space Complexity: O(rows²) for storing the triangle
    """
    if rows < 1 or rows > 30:
        raise ValueError("Number of rows must be between 1 and 30")
    
    # Check cache for previously computed results
    if rows in _pascal_cache:
        return _pascal_cache[rows]
    
    # Initialize the triangle with the first row
    triangle = [[1]]
    
    # Generate each subsequent row
    for i in range(1, rows):
        # Each row starts with 1
        row = [1]
        
        # Calculate interior elements by summing pairs from previous row
        prev_row = triangle[i - 1]
        for j in range(len(prev_row) - 1):
            row.append(prev_row[j] + prev_row[j + 1])
        
        # Each row ends with 1
        row.append(1)
        triangle.append(row)
    
    # Cache the result for future use
    _pascal_cache[rows] = triangle
    
    return triangle


def triangular_numbers(n: int = None, limit: int = None) -> int | list[int]:
    """
    Generate or calculate triangular numbers.
    
    Triangular numbers represent the number of dots that can form an equilateral
    triangle. They follow the pattern: 1, 3, 6, 10, 15, 21, ...
    
    Args:
        n: Position of triangular number to calculate (1-1000), mutually exclusive with limit
        limit: Generate sequence up to this many numbers, mutually exclusive with n
        
    Returns:
        If n is provided: the nth triangular number (integer)
        If limit is provided: sequence of first 'limit' triangular numbers (list)
        
    Raises:
        ValueError: If neither or both n and limit are provided
        ValueError: If n or limit is out of valid range
        
    Examples:
        triangular_numbers(n=5) = 15
        triangular_numbers(n=10) = 55
        triangular_numbers(limit=5) = [1, 3, 6, 10, 15]
        triangular_numbers(limit=10) = [1, 3, 6, 10, 15, 21, 28, 36, 45, 55]
        
    Formula: T(n) = n × (n + 1) / 2
    
    Mathematical Background:
        - Also equals the sum of first n natural numbers: 1 + 2 + 3 + ... + n
        - Can be visualized as triangular arrangements of dots
        - Used in combinatorics and number theory
        - The sequence appears in many mathematical contexts
        
    Algorithm Explanation:
        1. For single value: Use direct formula T(n) = n × (n + 1) / 2
        2. For sequence: Calculate each term iteratively for efficiency
        3. Iterative approach avoids redundant calculations
        
    Time Complexity: O(1) for single value, O(limit) for sequence
    Space Complexity: O(1) for single value, O(limit) for sequence
    """
    # Validate that exactly one parameter is provided
    if n is None and limit is None:
        raise ValueError("Either 'n' or 'limit' must be provided")
    
    if n is not None and limit is not None:
        raise ValueError("Cannot provide both 'n' and 'limit', use only one")
    
    # Calculate single triangular number
    if n is not None:
        if n < 1 or n > 1000:
            raise ValueError("n must be between 1 and 1000")
        
        # T(n) = n × (n + 1) / 2
        return n * (n + 1) // 2
    
    # Generate sequence of triangular numbers
    if limit < 1 or limit > 1000:
        raise ValueError("limit must be between 1 and 1000")
    
    sequence = []
    for i in range(1, limit + 1):
        sequence.append(i * (i + 1) // 2)
    
    return sequence


def perfect_numbers(limit: int) -> list[int]:
    """
    Find all perfect numbers up to a given limit.
    
    A perfect number is a positive integer that equals the sum of its proper
    divisors (divisors excluding the number itself). For example, 6 = 1 + 2 + 3.
    Perfect numbers are extremely rare.
    
    Args:
        limit: Upper bound for searching perfect numbers (1-10000)
        
    Returns:
        List of all perfect numbers up to and including the limit
        
    Raises:
        ValueError: If limit is less than 1 or greater than 10000
        
    Examples:
        perfect_numbers(10) = [6]
        perfect_numbers(30) = [6, 28]
        perfect_numbers(10000) = [6, 28, 496, 8128]
        
    Mathematical Background:
        - First discovered by ancient Greek mathematicians
        - Only 51 perfect numbers are known (as of 2024)
        - All known perfect numbers are even (odd perfect numbers unknown)
        - Related to Mersenne primes via Euclid-Euler theorem
        - Form: 2^(p-1) × (2^p - 1) where 2^p - 1 is a Mersenne prime
        
    Known perfect numbers:
        - 6 = 1 + 2 + 3
        - 28 = 1 + 2 + 4 + 7 + 14
        - 496 = 1 + 2 + 4 + 8 + 16 + 31 + 62 + 124 + 248
        - 8128 = sum of its proper divisors
        - Next one is 33,550,336 (outside typical limits)
        
    Algorithm Explanation:
        1. For each number n from 2 to limit:
           a. Find all divisors up to n/2 (proper divisors)
           b. Sum the divisors
           c. If sum equals n, it's a perfect number
        2. Optimization: Only check divisors up to sqrt(n)
        
    Time Complexity: O(limit × sqrt(limit)) with optimization
    Space Complexity: O(k) where k is the count of perfect numbers found
    """
    if limit < 1 or limit > 10000:
        raise ValueError("limit must be between 1 and 10000")
    
    perfect = []
    
    # Check each number from 2 to limit
    for n in range(2, limit + 1):
        # Calculate sum of proper divisors
        divisor_sum = 1  # 1 is always a proper divisor
        
        # Find divisors up to sqrt(n) for efficiency
        # If i divides n, both i and n/i are divisors
        i = 2
        while i * i <= n:
            if n % i == 0:
                divisor_sum += i
                # Add the complementary divisor if it's different
                if i != n // i and i * i != n:
                    divisor_sum += n // i
            i += 1
        
        # Check if sum of proper divisors equals the number
        if divisor_sum == n:
            perfect.append(n)
    
    return perfect


def collatz_sequence(n: int) -> dict[str, any]:
    """
    Generate the Collatz sequence (3n+1 problem) for a given starting number.
    
    The Collatz conjecture states that for any positive integer:
    - If even: divide by 2
    - If odd: multiply by 3 and add 1
    The sequence eventually reaches 1 (unproven but holds for all tested values).
    
    Args:
        n: Starting number for the sequence (1-100000)
        
    Returns:
        Dictionary containing:
        - 'sequence': List of numbers in the sequence
        - 'steps': Number of steps to reach 1
        - 'max_value': Maximum value reached in the sequence
        
    Raises:
        ValueError: If n is less than 1 or greater than 100000
        
    Examples:
        collatz_sequence(1) = {'sequence': [1], 'steps': 0, 'max_value': 1}
        collatz_sequence(5) = {'sequence': [5, 16, 8, 4, 2, 1], 'steps': 5, 'max_value': 16}
        collatz_sequence(13) = {'sequence': [13, 40, 20, 10, 5, 16, 8, 4, 2, 1], 'steps': 9, 'max_value': 40}
        collatz_sequence(27) has 111 steps with max value 9232
        
    Mathematical Background:
        - Also known as the 3n+1 problem, Ulam conjecture, or Syracuse problem
        - One of the most famous unsolved problems in mathematics
        - Despite its simplicity, no one has proven it works for all numbers
        - Tested for all numbers up to 2^68 (as of 2024)
        - Some sequences can grow very large before returning to 1
        
    Algorithm Explanation:
        1. Start with the given number n
        2. Apply the rules repeatedly:
           - If n is even: n = n / 2
           - If n is odd: n = 3n + 1
        3. Continue until n = 1
        4. Track the sequence, step count, and maximum value
        
    Time Complexity: Unknown (depends on sequence length, which varies)
    Space Complexity: O(steps) for storing the sequence
    """
    if n < 1 or n > 100000:
        raise ValueError("n must be between 1 and 100000")
    
    sequence = [n]
    max_value = n
    steps = 0
    
    # Generate sequence until we reach 1
    while n != 1:
        if n % 2 == 0:
            # Even: divide by 2
            n = n // 2
        else:
            # Odd: multiply by 3 and add 1
            n = 3 * n + 1
        
        sequence.append(n)
        max_value = max(max_value, n)
        steps += 1
    
    return {
        'sequence': sequence,
        'steps': steps,
        'max_value': max_value
    }


# ============================================================================
# Number Theory Functions
# ============================================================================


def gcd(numbers: list[int]) -> int:
    """
    Calculate the greatest common divisor (GCD) of two or more numbers.
    
    Uses the Euclidean algorithm iteratively to find the GCD of multiple numbers.
    The GCD is the largest positive integer that divides all the given numbers
    without a remainder.
    
    Args:
        numbers: List of integers (2-10 numbers) to find GCD of
        
    Returns:
        The greatest common divisor of all numbers
        
    Raises:
        ValueError: If fewer than 2 or more than 10 numbers provided
        
    Examples:
        gcd([48, 18]) = 6          # 48 = 6×8, 18 = 6×3
        gcd([12, 18, 24]) = 6      # Common divisor is 6
        gcd([17, 19]) = 1          # Coprime numbers
        gcd([100, 50, 25]) = 25    # All divisible by 25
        
    Algorithm Explanation (Euclidean Algorithm):
        1. For two numbers a and b (a >= b):
           - If b = 0, then GCD(a, b) = a
           - Otherwise, GCD(a, b) = GCD(b, a mod b)
        2. For multiple numbers, apply the algorithm pairwise:
           - GCD(a, b, c) = GCD(GCD(a, b), c)
        3. Continue until all numbers are processed
        
    Time Complexity: O(n log m) where n is the count and m is the largest number
    """
    if len(numbers) < 2 or len(numbers) > 10:
        raise ValueError("GCD requires between 2 and 10 numbers")
    
    # Helper function to compute GCD of two numbers using Euclidean algorithm
    def gcd_two(a: int, b: int) -> int:
        """Compute GCD of two numbers using Euclidean algorithm."""
        # Convert to absolute values to handle negative numbers
        a, b = abs(a), abs(b)
        
        # Euclidean algorithm: repeatedly replace (a, b) with (b, a mod b)
        while b != 0:
            a, b = b, a % b
        
        return a
    
    # Start with the first number and apply GCD pairwise
    result = numbers[0]
    for num in numbers[1:]:
        result = gcd_two(result, num)
        # Early termination: if GCD becomes 1, it won't change further
        if result == 1:
            break
    
    return result


def lcm(numbers: list[int]) -> int:
    """
    Calculate the least common multiple (LCM) of two or more numbers.
    
    The LCM is the smallest positive integer that is divisible by all the
    given numbers. Uses the relationship: LCM(a,b) = (a × b) / GCD(a,b)
    
    Args:
        numbers: List of integers (2-10 numbers) to find LCM of
        
    Returns:
        The least common multiple of all numbers
        
    Raises:
        ValueError: If fewer than 2 or more than 10 numbers provided
        ValueError: If any number is zero (LCM undefined for zero)
        
    Examples:
        lcm([12, 18]) = 36         # 12 = 2²×3, 18 = 2×3², LCM = 2²×3² = 36
        lcm([4, 6, 8]) = 24        # Smallest number divisible by 4, 6, and 8
        lcm([5, 7]) = 35           # For coprime numbers, LCM = product
        lcm([10, 15, 20]) = 60     # Common multiple is 60
        
    Algorithm Explanation:
        1. For two numbers a and b:
           - LCM(a, b) = |a × b| / GCD(a, b)
        2. For multiple numbers, apply the formula pairwise:
           - LCM(a, b, c) = LCM(LCM(a, b), c)
        3. Continue until all numbers are processed
        
    Note: Results can grow very large, so overflow protection is important
    """
    if len(numbers) < 2 or len(numbers) > 10:
        raise ValueError("LCM requires between 2 and 10 numbers")
    
    # Check for zero in the list (LCM with 0 is undefined)
    if 0 in numbers:
        raise ValueError("LCM is undefined when any number is zero")
    
    # Helper function to compute LCM of two numbers
    def lcm_two(a: int, b: int) -> int:
        """Compute LCM of two numbers using GCD."""
        # Convert to absolute values
        a, b = abs(a), abs(b)
        
        # LCM(a, b) = (a * b) / GCD(a, b)
        # We compute GCD first to avoid overflow
        from math import gcd as math_gcd
        return (a * b) // math_gcd(a, b)
    
    # Start with the first number and apply LCM pairwise
    result = abs(numbers[0])
    for num in numbers[1:]:
        result = lcm_two(result, num)
    
    return result


def factorial(n: int) -> int:
    """
    Calculate the factorial of a number (n!).
    
    The factorial of a non-negative integer n is the product of all positive
    integers less than or equal to n. By definition, 0! = 1.
    
    Args:
        n: The number to calculate factorial of (0-170)
        
    Returns:
        The factorial of n (n!)
        
    Raises:
        ValueError: If n is negative or greater than 170
        
    Examples:
        factorial(0) = 1           # By definition
        factorial(1) = 1           # 1
        factorial(5) = 120         # 5 × 4 × 3 × 2 × 1
        factorial(10) = 3628800    # 10!
        
    Algorithm Explanation:
        1. Base case: 0! = 1 and 1! = 1
        2. For n > 1: n! = n × (n-1) × (n-2) × ... × 2 × 1
        3. Iterative approach is used for efficiency
        
    Note: Limited to 170 because 171! exceeds Python's float representation
    and causes overflow issues in many contexts. For exact integer computation,
    Python can handle larger values, but we enforce this limit for safety.
    
    Time Complexity: O(n)
    Space Complexity: O(1)
    """
    if n < 0:
        raise ValueError("Factorial is not defined for negative numbers")
    
    if n > 170:
        raise ValueError("Factorial is limited to n <= 170 to avoid overflow")
    
    # Base cases: 0! = 1, 1! = 1
    if n <= 1:
        return 1
    
    # Iterative calculation for efficiency
    result = 1
    for i in range(2, n + 1):
        result *= i
    
    return result


# Memoization cache for factorial to improve performance
_factorial_cache: dict[int, int] = {0: 1, 1: 1}


def factorial_memoized(n: int) -> int:
    """
    Calculate factorial with memoization for better performance in repeated calls.
    
    This is used internally by combinations and permutations functions to avoid
    redundant factorial calculations.
    
    Args:
        n: The number to calculate factorial of (0-170)
        
    Returns:
        The factorial of n (n!)
    """
    if n in _factorial_cache:
        return _factorial_cache[n]
    
    if n < 0:
        raise ValueError("Factorial is not defined for negative numbers")
    
    if n > 170:
        raise ValueError("Factorial is limited to n <= 170 to avoid overflow")
    
    # Calculate and cache
    result = n * factorial_memoized(n - 1)
    _factorial_cache[n] = result
    
    return result


def combinations(n: int, r: int) -> int:
    """
    Calculate combinations (nCr) - ways to choose r items from n items.
    
    Combinations represent the number of ways to select r items from n items
    where order does not matter. Also written as C(n,r) or "n choose r".
    
    Args:
        n: Total number of items (0-1000)
        r: Number of items to choose (0 to n)
        
    Returns:
        The number of combinations C(n,r) = n! / (r! × (n-r)!)
        
    Raises:
        ValueError: If n or r are negative, r > n, or n > 1000
        
    Examples:
        combinations(5, 2) = 10    # Choose 2 from 5: {1,2}, {1,3}, {1,4}, {1,5}, {2,3}, {2,4}, {2,5}, {3,4}, {3,5}, {4,5}
        combinations(10, 3) = 120  # Choose 3 from 10
        combinations(52, 5) = 2,598,960  # Poker hands (5 cards from 52)
        combinations(5, 0) = 1     # One way to choose nothing
        combinations(5, 5) = 1     # One way to choose everything
        
    Formula: C(n,r) = n! / (r! × (n-r)!)
    
    Algorithm Explanation:
        1. Use the multiplicative formula to avoid computing large factorials:
           C(n,r) = n × (n-1) × ... × (n-r+1) / (r × (r-1) × ... × 1)
        2. This is more efficient and avoids overflow for large n
        3. Optimization: C(n,r) = C(n, n-r), so use the smaller value
        
    Time Complexity: O(min(r, n-r))
    Space Complexity: O(1)
    """
    if n < 0 or r < 0:
        raise ValueError("n and r must be non-negative")
    
    if r > n:
        raise ValueError("r cannot be greater than n")
    
    if n > 1000:
        raise ValueError("n is limited to 1000 to prevent overflow")
    
    # Base cases
    if r == 0 or r == n:
        return 1
    
    # Optimization: C(n,r) = C(n, n-r), so calculate with smaller r
    r = min(r, n - r)
    
    # Use multiplicative formula: C(n,r) = n! / (r! * (n-r)!)
    # But calculate as: (n * (n-1) * ... * (n-r+1)) / (r * (r-1) * ... * 1)
    # This avoids computing large factorials
    result = 1
    for i in range(r):
        result = result * (n - i) // (i + 1)
    
    return result


def permutations(n: int, r: int) -> int:
    """
    Calculate permutations (nPr) - ordered ways to choose r items from n items.
    
    Permutations represent the number of ways to select and arrange r items
    from n items where order matters. Also written as P(n,r) or "n permute r".
    
    Args:
        n: Total number of items (0-1000)
        r: Number of items to arrange (0 to n)
        
    Returns:
        The number of permutations P(n,r) = n! / (n-r)!
        
    Raises:
        ValueError: If n or r are negative, r > n, or n > 1000
        
    Examples:
        permutations(5, 2) = 20    # Arrange 2 from 5: 5 choices for 1st, 4 for 2nd
        permutations(10, 3) = 720  # Arrange 3 from 10
        permutations(5, 5) = 120   # All arrangements of 5 items = 5!
        permutations(5, 0) = 1     # One way to arrange nothing
        
    Formula: P(n,r) = n! / (n-r)!
    
    Algorithm Explanation:
        1. P(n,r) = n × (n-1) × (n-2) × ... × (n-r+1)
        2. This is equivalent to n! / (n-r)! but more efficient
        3. Multiply r consecutive descending integers starting from n
        
    Time Complexity: O(r)
    Space Complexity: O(1)
    """
    if n < 0 or r < 0:
        raise ValueError("n and r must be non-negative")
    
    if r > n:
        raise ValueError("r cannot be greater than n")
    
    if n > 1000:
        raise ValueError("n is limited to 1000 to prevent overflow")
    
    # Base case
    if r == 0:
        return 1
    
    # Calculate P(n,r) = n × (n-1) × (n-2) × ... × (n-r+1)
    result = 1
    for i in range(n, n - r, -1):
        result *= i
    
    return result


# ============================================================================
# Statistical Analysis Functions
# ============================================================================


def descriptive_stats(data: list[float]) -> dict[str, Any]:
    """
    Calculate comprehensive descriptive statistics for a dataset.
    
    This function computes key statistical measures that describe the central
    tendency, dispersion, and distribution of a dataset.
    
    Args:
        data: List of numeric values (1-10000 items)
        
    Returns:
        Dictionary containing:
        - 'mean': Arithmetic average of the data
        - 'median': Middle value when data is sorted
        - 'mode': Most frequently occurring value(s)
        - 'range': Difference between max and min values
        - 'variance': Average of squared deviations from mean
        - 'std_dev': Square root of variance (standard deviation)
        - 'count': Number of data points
        - 'min': Minimum value
        - 'max': Maximum value
        
    Raises:
        ValueError: If data is empty, not a list, or contains non-numeric values
        ValueError: If data contains fewer than 1 or more than 10000 items
        
    Examples:
        descriptive_stats([23, 45, 12, 67, 34, 89, 23, 56])
        >>> {
        ...   'mean': 43.625,
        ...   'median': 39.5,
        ...   'mode': [23],
        ...   'range': 77,
        ...   'variance': 642.734375,
        ...   'std_dev': 25.352...,
        ...   'count': 8,
        ...   'min': 12,
        ...   'max': 89
        ... }
        
    Formula Explanations:
        - Mean: μ = Σx / n (sum of all values divided by count)
        - Median: Middle value of sorted data (or average of two middle values)
        - Mode: Value(s) with highest frequency
        - Range: max - min
        - Variance: σ² = Σ(x - μ)² / n (population variance)
        - Standard Deviation: σ = √(variance)
        
    Algorithm Explanation:
        1. Validate input data (type, size, numeric values)
        2. Calculate mean by summing all values and dividing by count
        3. Calculate median by sorting data and finding middle value(s)
        4. Calculate mode by counting frequencies and finding maximum
        5. Calculate variance as average of squared deviations from mean
        6. Calculate standard deviation as square root of variance
        7. Find min, max, and range
        
    Time Complexity: O(n log n) due to sorting for median
    Space Complexity: O(n) for storing sorted data and frequency map
    """
    # Validate input type
    if not isinstance(data, list):
        raise ValueError(f"Data must be a list, got {type(data).__name__}")
    
    # Validate data size
    if len(data) < 1 or len(data) > 10000:
        raise ValueError("Data must contain between 1 and 10000 items")
    
    # Validate all elements are numeric
    for i, value in enumerate(data):
        if not isinstance(value, (int, float)):
            raise ValueError(f"All data values must be numeric. Item at index {i} is {type(value).__name__}: {value}")
    
    n = len(data)
    
    # Calculate mean
    # μ = Σx / n
    mean = sum(data) / n
    
    # Calculate median
    # Sort data and find middle value(s)
    sorted_data = sorted(data)
    if n % 2 == 0:
        # Even number of elements: average of two middle values
        median = (sorted_data[n // 2 - 1] + sorted_data[n // 2]) / 2
    else:
        # Odd number of elements: middle value
        median = sorted_data[n // 2]
    
    # Calculate mode (most frequent value(s))
    # Build frequency map
    frequency = {}
    for value in data:
        frequency[value] = frequency.get(value, 0) + 1
    
    # Find maximum frequency
    max_freq = max(frequency.values())
    
    # Find all values with maximum frequency
    mode = [value for value, freq in frequency.items() if freq == max_freq]
    mode.sort()  # Sort for consistent output
    
    # Calculate variance
    # σ² = Σ(x - μ)² / n (population variance)
    variance = sum((x - mean) ** 2 for x in data) / n
    
    # Calculate standard deviation
    # σ = √(variance)
    std_dev = variance ** 0.5
    
    # Calculate min, max, and range
    min_value = min(data)
    max_value = max(data)
    data_range = max_value - min_value
    
    return {
        'mean': mean,
        'median': median,
        'mode': mode,
        'range': data_range,
        'variance': variance,
        'std_dev': std_dev,
        'count': n,
        'min': min_value,
        'max': max_value
    }


def correlation(x: list[float], y: list[float]) -> dict[str, Any]:
    """
    Calculate Pearson correlation coefficient between two datasets.
    
    The Pearson correlation coefficient measures the linear relationship between
    two variables. It ranges from -1 (perfect negative correlation) to +1
    (perfect positive correlation), with 0 indicating no linear correlation.
    
    Args:
        x: First dataset (2-1000 items)
        y: Second dataset (must be same length as x)
        
    Returns:
        Dictionary containing:
        - 'coefficient': Pearson correlation coefficient (r)
        - 'interpretation': Human-readable interpretation of the correlation
        
    Raises:
        ValueError: If x or y are not lists, have different lengths,
                   contain non-numeric values, or are outside valid size range
        
    Examples:
        correlation([1, 2, 3, 4, 5], [2, 4, 6, 8, 10])
        >>> {'coefficient': 1.0, 'interpretation': 'Perfect positive correlation'}
        
        correlation([1, 2, 3, 4, 5], [5, 4, 3, 2, 1])
        >>> {'coefficient': -1.0, 'interpretation': 'Perfect negative correlation'}
        
        correlation([1, 2, 3, 4, 5], [3, 5, 2, 7, 4])
        >>> {'coefficient': 0.5..., 'interpretation': 'Moderate positive correlation'}
        
    Formula:
        r = Σ[(x - x̄)(y - ȳ)] / √[Σ(x - x̄)² × Σ(y - ȳ)²]
        
        Where:
        - x̄ is the mean of x
        - ȳ is the mean of y
        - Numerator: covariance between x and y
        - Denominator: product of standard deviations
        
    Interpretation Guide:
        - r = 1.0: Perfect positive correlation
        - 0.7 ≤ r < 1.0: Strong positive correlation
        - 0.4 ≤ r < 0.7: Moderate positive correlation
        - 0.1 ≤ r < 0.4: Weak positive correlation
        - -0.1 < r < 0.1: No correlation
        - -0.4 < r ≤ -0.1: Weak negative correlation
        - -0.7 < r ≤ -0.4: Moderate negative correlation
        - -1.0 < r ≤ -0.7: Strong negative correlation
        - r = -1.0: Perfect negative correlation
        
    Algorithm Explanation:
        1. Validate inputs (type, size, equal length, numeric values)
        2. Calculate means of both datasets
        3. Calculate covariance: Σ[(x - x̄)(y - ȳ)]
        4. Calculate standard deviations for both datasets
        5. Compute correlation coefficient: covariance / (std_x × std_y)
        6. Provide human-readable interpretation
        
    Time Complexity: O(n) where n is the length of the datasets
    Space Complexity: O(1) not counting the input
    """
    # Validate input types
    if not isinstance(x, list):
        raise ValueError(f"Parameter 'x' must be a list, got {type(x).__name__}")
    if not isinstance(y, list):
        raise ValueError(f"Parameter 'y' must be a list, got {type(y).__name__}")
    
    # Validate sizes
    if len(x) < 2 or len(x) > 1000:
        raise ValueError("Parameter 'x' must contain between 2 and 1000 items")
    if len(y) < 2 or len(y) > 1000:
        raise ValueError("Parameter 'y' must contain between 2 and 1000 items")
    
    # Validate equal length
    if len(x) != len(y):
        raise ValueError(f"Arrays must have equal length. x has {len(x)} items, y has {len(y)} items")
    
    # Validate all elements are numeric
    for i, value in enumerate(x):
        if not isinstance(value, (int, float)):
            raise ValueError(f"All values in 'x' must be numeric. Item at index {i} is {type(value).__name__}: {value}")
    
    for i, value in enumerate(y):
        if not isinstance(value, (int, float)):
            raise ValueError(f"All values in 'y' must be numeric. Item at index {i} is {type(value).__name__}: {value}")
    
    n = len(x)
    
    # Calculate means
    mean_x = sum(x) / n
    mean_y = sum(y) / n
    
    # Calculate covariance: Σ[(x - x̄)(y - ȳ)]
    covariance = sum((x[i] - mean_x) * (y[i] - mean_y) for i in range(n))
    
    # Calculate sum of squared deviations for both variables
    sum_sq_x = sum((xi - mean_x) ** 2 for xi in x)
    sum_sq_y = sum((yi - mean_y) ** 2 for yi in y)
    
    # Handle edge case: all values are the same (no variance)
    if sum_sq_x == 0 or sum_sq_y == 0:
        raise ValueError("Cannot calculate correlation when all values in a dataset are identical (zero variance)")
    
    # Calculate Pearson correlation coefficient
    # r = covariance / (std_x × std_y)
    # = Σ[(x - x̄)(y - ȳ)] / √[Σ(x - x̄)² × Σ(y - ȳ)²]
    r = covariance / (sum_sq_x * sum_sq_y) ** 0.5
    
    # Interpret the correlation
    if r == 1.0:
        interpretation = "Perfect positive correlation"
    elif r >= 0.7:
        interpretation = "Strong positive correlation"
    elif r >= 0.4:
        interpretation = "Moderate positive correlation"
    elif r >= 0.1:
        interpretation = "Weak positive correlation"
    elif r > -0.1:
        interpretation = "No correlation"
    elif r > -0.4:
        interpretation = "Weak negative correlation"
    elif r > -0.7:
        interpretation = "Moderate negative correlation"
    elif r > -1.0:
        interpretation = "Strong negative correlation"
    else:  # r == -1.0
        interpretation = "Perfect negative correlation"
    
    return {
        'coefficient': r,
        'interpretation': interpretation
    }


def percentile(data: list[float], p: float) -> float:
    """
    Calculate the value at a specific percentile in a dataset.
    
    A percentile is a value below which a given percentage of observations fall.
    For example, the 50th percentile (median) is the value below which 50% of
    the data falls.
    
    Args:
        data: List of numeric values (1-10000 items)
        p: Percentile to calculate (0-100)
        
    Returns:
        The value at the given percentile
        
    Raises:
        ValueError: If data is empty, not a list, contains non-numeric values,
                   or percentile is outside 0-100 range
        
    Examples:
        percentile([15, 20, 35, 40, 50], 50)
        >>> 35  # Median (50th percentile)
        
        percentile([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], 90)
        >>> 9.1  # 90th percentile
        
        percentile([23, 45, 12, 67, 34, 89, 23, 56], 25)
        >>> 23.0  # First quartile (Q1)
        
    Formula (Linear Interpolation Method):
        1. Sort the data in ascending order
        2. Calculate rank: k = (p / 100) × (n - 1)
        3. If k is an integer, return data[k]
        4. Otherwise, interpolate between data[floor(k)] and data[ceil(k)]
        
    Special Cases:
        - p = 0: minimum value
        - p = 25: first quartile (Q1)
        - p = 50: median (second quartile, Q2)
        - p = 75: third quartile (Q3)
        - p = 100: maximum value
        
    Algorithm Explanation:
        1. Validate input data and percentile value
        2. Sort the data
        3. Calculate the position (rank) in the sorted data
        4. If position is exact, return that value
        5. If position is between two values, interpolate linearly
        
    Time Complexity: O(n log n) due to sorting
    Space Complexity: O(n) for storing sorted data
    """
    # Validate input type
    if not isinstance(data, list):
        raise ValueError(f"Data must be a list, got {type(data).__name__}")
    
    # Validate data size
    if len(data) < 1 or len(data) > 10000:
        raise ValueError("Data must contain between 1 and 10000 items")
    
    # Validate all elements are numeric
    for i, value in enumerate(data):
        if not isinstance(value, (int, float)):
            raise ValueError(f"All data values must be numeric. Item at index {i} is {type(value).__name__}: {value}")
    
    # Validate percentile range
    if not isinstance(p, (int, float)):
        raise ValueError(f"Percentile must be numeric, got {type(p).__name__}")
    
    if p < 0 or p > 100:
        raise ValueError(f"Percentile must be between 0 and 100, got {p}")
    
    # Sort the data
    sorted_data = sorted(data)
    n = len(sorted_data)
    
    # Handle edge cases
    if p == 0:
        return float(sorted_data[0])
    if p == 100:
        return float(sorted_data[-1])
    
    # Calculate rank using linear interpolation method
    # k = (p / 100) × (n - 1)
    k = (p / 100) * (n - 1)
    
    # Get lower and upper indices
    lower_index = int(k)
    upper_index = lower_index + 1
    
    # If k is exactly an integer, return that element
    if k == lower_index:
        return float(sorted_data[lower_index])
    
    # Otherwise, interpolate between two values
    # Interpolation formula: value = lower + (upper - lower) × fraction
    fraction = k - lower_index
    lower_value = sorted_data[lower_index]
    upper_value = sorted_data[upper_index]
    
    result = lower_value + (upper_value - lower_value) * fraction
    
    return float(result)


def outliers(data: list[float]) -> dict[str, Any]:
    """
    Identify outliers in a dataset using the Interquartile Range (IQR) method.
    
    The IQR method is a robust statistical technique for detecting outliers.
    It uses the spread of the middle 50% of the data to identify values that
    are unusually far from the central tendency.
    
    Args:
        data: List of numeric values (1-10000 items)
        
    Returns:
        Dictionary containing:
        - 'outliers': List of outlier values
        - 'indices': List of indices where outliers occur in original data
        - 'count': Number of outliers found
        - 'lower_bound': Lower threshold (Q1 - 1.5×IQR)
        - 'upper_bound': Upper threshold (Q3 + 1.5×IQR)
        - 'q1': First quartile (25th percentile)
        - 'q3': Third quartile (75th percentile)
        - 'iqr': Interquartile range (Q3 - Q1)
        
    Raises:
        ValueError: If data is empty, not a list, or contains non-numeric values
        
    Examples:
        outliers([10, 12, 14, 13, 15, 100, 11, 13, 14])
        >>> {
        ...   'outliers': [100],
        ...   'indices': [5],
        ...   'count': 1,
        ...   'lower_bound': 5.5,
        ...   'upper_bound': 20.5,
        ...   'q1': 11.5,
        ...   'q3': 14.5,
        ...   'iqr': 3.0
        ... }
        
    IQR Method Explanation:
        1. Calculate Q1 (25th percentile) and Q3 (75th percentile)
        2. Calculate IQR = Q3 - Q1 (range of middle 50% of data)
        3. Define outlier boundaries:
           - Lower bound = Q1 - 1.5 × IQR
           - Upper bound = Q3 + 1.5 × IQR
        4. Any value < lower_bound or > upper_bound is an outlier
        
    Why 1.5 × IQR?
        - This is a standard convention in statistics
        - Balances between detecting true outliers and false positives
        - Used in box plots and many statistical packages
        - Values beyond 3 × IQR are considered "extreme outliers"
        
    Advantages of IQR Method:
        - Robust: not affected by extreme values
        - Works well for skewed distributions
        - Does not assume normal distribution
        - Easy to interpret and implement
        
    Algorithm Explanation:
        1. Validate input data
        2. Calculate Q1 (25th percentile) and Q3 (75th percentile)
        3. Calculate IQR = Q3 - Q1
        4. Calculate lower and upper bounds
        5. Identify values outside the bounds
        6. Return outliers with their indices and statistics
        
    Time Complexity: O(n log n) due to percentile calculations (sorting)
    Space Complexity: O(k) where k is the number of outliers
    """
    # Validate input type
    if not isinstance(data, list):
        raise ValueError(f"Data must be a list, got {type(data).__name__}")
    
    # Validate data size
    if len(data) < 1 or len(data) > 10000:
        raise ValueError("Data must contain between 1 and 10000 items")
    
    # Validate all elements are numeric
    for i, value in enumerate(data):
        if not isinstance(value, (int, float)):
            raise ValueError(f"All data values must be numeric. Item at index {i} is {type(value).__name__}: {value}")
    
    # Handle edge case: single value (no outliers possible)
    if len(data) == 1:
        return {
            'outliers': [],
            'indices': [],
            'count': 0,
            'lower_bound': data[0],
            'upper_bound': data[0],
            'q1': data[0],
            'q3': data[0],
            'iqr': 0
        }
    
    # Calculate quartiles using the percentile function
    q1 = percentile(data, 25)
    q3 = percentile(data, 75)
    
    # Calculate IQR (Interquartile Range)
    iqr = q3 - q1
    
    # Calculate outlier boundaries
    # Standard method: 1.5 × IQR
    lower_bound = q1 - 1.5 * iqr
    upper_bound = q3 + 1.5 * iqr
    
    # Identify outliers
    outlier_values = []
    outlier_indices = []
    
    for i, value in enumerate(data):
        if value < lower_bound or value > upper_bound:
            outlier_values.append(value)
            outlier_indices.append(i)
    
    return {
        'outliers': outlier_values,
        'indices': outlier_indices,
        'count': len(outlier_values),
        'lower_bound': lower_bound,
        'upper_bound': upper_bound,
        'q1': q1,
        'q3': q3,
        'iqr': iqr
    }


# ============================================================================
# Cryptographic Hash Functions
# ============================================================================


def generate_hash(data: str, algorithm: str, output_format: str = "hex") -> dict[str, Any]:
    """
    Generate cryptographic hash of input data using various algorithms.
    
    This function supports multiple hashing algorithms with different security
    characteristics. Each algorithm has specific use cases and security implications.
    
    Args:
        data: Input string to hash (max 1MB = 1,048,576 bytes)
        algorithm: Hash algorithm to use (md5, sha1, sha256, sha512, blake2b)
        output_format: Output format (hex or base64, default: hex)
        
    Returns:
        Dictionary containing:
        - 'hash': The computed hash string in requested format
        - 'algorithm': The algorithm used
        - 'output_format': The output format used
        - 'input_size': Size of input data in bytes
        - 'hash_size': Size of hash output in bits
        - 'security_note': Security warning/recommendation for the algorithm
        
    Raises:
        ValueError: If algorithm is not supported
        ValueError: If output_format is not supported
        ValueError: If data exceeds 1MB size limit
        
    Examples:
        generate_hash("Hello World", "sha256", "hex")
        >>> {'hash': 'a591a6d40bf420404a011733cfb7b190d62c65bf0bcda32b57b277d9ad9f146e', ...}
        
        generate_hash("password123", "md5", "hex")
        >>> {'hash': '482c811da5d5b4bc6d497ffa98491e38', ...}
        
    Security Notes:
        **MD5 (128-bit)**:
        - BROKEN for security purposes - collision attacks are practical
        - Use ONLY for non-security purposes (checksums, cache keys)
        - NEVER use for passwords, digital signatures, or certificates
        
        **SHA-1 (160-bit)**:
        - DEPRECATED - collision attacks demonstrated (SHAttered, 2017)
        - Still used in git commits but NOT recommended for new applications
        - Acceptable for HMAC in legacy systems but migrate to SHA-256
        
        **SHA-256 (256-bit)**:
        - SECURE and widely used (part of SHA-2 family)
        - Recommended for most security applications
        - Good for password hashing WITH proper salt and key derivation
        - Used in Bitcoin, SSL/TLS certificates, code signing
        
        **SHA-512 (512-bit)**:
        - MORE SECURE than SHA-256 with larger output
        - Slower but more resistant to collision attacks
        - Preferred for high-security applications
        - Good choice when hash length truncation is needed
        
        **BLAKE2b (512-bit)**:
        - MODERN, fast alternative to SHA-2
        - Comparable security to SHA-3 but faster than MD5
        - Not yet as widely adopted but excellent choice
        - Used in file integrity systems and cryptocurrencies (Zcash)
        
    Password Hashing Best Practices:
        - NEVER store passwords as plain hashes
        - Always use salt (unique random value per password)
        - Use key derivation functions (PBKDF2, bcrypt, Argon2)
        - Minimum: SHA-256 with salt, better: use bcrypt/Argon2
        
    Algorithm Characteristics:
        - MD5: Fast, 128-bit, INSECURE for security
        - SHA-1: Medium, 160-bit, DEPRECATED for security
        - SHA-256: Medium, 256-bit, SECURE and recommended
        - SHA-512: Slower, 512-bit, HIGH security
        - BLAKE2b: Fast, 512-bit, MODERN and secure
        
    Time Complexity: O(n) where n is the length of input data
    Space Complexity: O(n) for storing input + O(1) for hash computation
    """
    # Input size validation - prevent DoS with huge inputs
    MAX_SIZE = 1024 * 1024  # 1MB limit
    data_bytes = data.encode('utf-8')
    data_size = len(data_bytes)
    
    if data_size > MAX_SIZE:
        raise ValueError(
            f"Input data size ({data_size:,} bytes) exceeds maximum allowed size "
            f"({MAX_SIZE:,} bytes = 1MB). Please use smaller input."
        )
    
    # Validate algorithm
    supported_algorithms = {
        'md5': (hashlib.md5, 128, 
                "⚠️ WARNING: MD5 is CRYPTOGRAPHICALLY BROKEN. Use only for non-security purposes like checksums."),
        'sha1': (hashlib.sha1, 160, 
                 "⚠️ CAUTION: SHA-1 is DEPRECATED for security. Avoid for new applications."),
        'sha256': (hashlib.sha256, 256, 
                   "✓ SECURE: SHA-256 is recommended for most security applications."),
        'sha512': (hashlib.sha512, 512, 
                   "✓ HIGH SECURITY: SHA-512 provides enhanced security with larger output."),
        'blake2b': (hashlib.blake2b, 512, 
                    "✓ MODERN: BLAKE2b is fast and secure, comparable to SHA-3.")
    }
    
    algorithm_lower = algorithm.lower()
    if algorithm_lower not in supported_algorithms:
        raise ValueError(
            f"Unsupported algorithm: '{algorithm}'. "
            f"Supported algorithms are: {', '.join(supported_algorithms.keys())}"
        )
    
    # Validate output format
    supported_formats = ['hex', 'base64']
    output_format_lower = output_format.lower()
    if output_format_lower not in supported_formats:
        raise ValueError(
            f"Unsupported output format: '{output_format}'. "
            f"Supported formats are: {', '.join(supported_formats)}"
        )
    
    # Get hash function and metadata
    hash_func, hash_size_bits, security_note = supported_algorithms[algorithm_lower]
    
    # Compute hash
    hash_obj = hash_func()
    hash_obj.update(data_bytes)
    
    # Format output
    if output_format_lower == 'hex':
        hash_value = hash_obj.hexdigest()
    else:  # base64
        hash_value = base64.b64encode(hash_obj.digest()).decode('utf-8')
    
    # Return comprehensive result with metadata
    return {
        'hash': hash_value,
        'algorithm': algorithm_lower,
        'output_format': output_format_lower,
        'input_size': data_size,
        'hash_size': hash_size_bits,
        'security_note': security_note
    }


# ============================================================================
# Unit Conversion Functions
# ============================================================================


def unit_convert(value: float, from_unit: str, to_unit: str) -> dict[str, Any]:
    """
    Convert between different units of measurement across multiple categories.
    
    Supports comprehensive unit conversions for:
    - Length (metric and imperial)
    - Weight/Mass (metric and imperial)
    - Temperature (with formula display)
    - Volume (metric and imperial)
    - Time (milliseconds to years)
    - Digital Storage (binary, 1024-based)
    - Speed (m/s, km/h, mph)
    
    Args:
        value: The numeric value to convert
        from_unit: The source unit (case-insensitive, accepts abbreviations)
        to_unit: The target unit (case-insensitive, accepts abbreviations)
        
    Returns:
        Dictionary containing:
        - 'value': The original value
        - 'from_unit': The source unit (normalized)
        - 'to_unit': The target unit (normalized)
        - 'result': The converted value
        - 'formatted': Human-readable result string
        - 'formula': Conversion formula (for temperature) or factor used
        
    Raises:
        ValueError: If units are invalid, not in same category, or conversion fails
        
    Examples:
        unit_convert(100, "km", "mi")
        >>> {'result': 62.137, 'formatted': '100 km = 62.137 mi', ...}
        
        unit_convert(75, "F", "C")
        >>> {'result': 23.89, 'formatted': '75°F = 23.89°C', 'formula': '(F - 32) × 5/9', ...}
        
        unit_convert(2, "GB", "MB")
        >>> {'result': 2048, 'formatted': '2 GB = 2048 MB', ...}
        
    Algorithm Explanation:
        1. Normalize unit names to lowercase and map aliases
        2. Identify the category of units (length, weight, etc.)
        3. Validate both units are in the same category
        4. For temperature: use conversion formulas
        5. For other units: convert to base unit, then to target unit
        6. Round result to appropriate precision
        7. Return formatted result with formula/factor information
        
    Time Complexity: O(1) - all conversions are direct calculations
    Space Complexity: O(1) - fixed size conversion tables
    """
    # Normalize unit names to lowercase for case-insensitive matching
    from_unit_lower = from_unit.lower().strip()
    to_unit_lower = to_unit.lower().strip()
    
    # Define unit categories with conversion factors to base units
    # Base units: meter (length), gram (weight), celsius (temp), liter (volume),
    #             second (time), byte (storage), m/s (speed)
    
    # ===== LENGTH CONVERSIONS =====
    # Base unit: meter (m)
    length_units = {
        'millimeter': 0.001, 'mm': 0.001,
        'centimeter': 0.01, 'cm': 0.01,
        'meter': 1.0, 'm': 1.0,
        'kilometer': 1000.0, 'km': 1000.0,
        'inch': 0.0254, 'in': 0.0254,  # Exactly 2.54 cm
        'foot': 0.3048, 'ft': 0.3048,  # Exactly 0.3048 m
        'yard': 0.9144, 'yd': 0.9144,  # Exactly 0.9144 m
        'mile': 1609.344, 'mi': 1609.344,  # Exactly 1609.344 m
    }
    
    # ===== WEIGHT/MASS CONVERSIONS =====
    # Base unit: gram (g)
    weight_units = {
        'milligram': 0.001, 'mg': 0.001,
        'gram': 1.0, 'g': 1.0,
        'kilogram': 1000.0, 'kg': 1000.0,
        'tonne': 1000000.0, 't': 1000000.0, 'metric ton': 1000000.0,
        'ounce': 28.349523125, 'oz': 28.349523125,  # Exactly
        'pound': 453.59237, 'lb': 453.59237,  # Exactly
        'ton': 907184.74, 'us ton': 907184.74,  # US short ton
    }
    
    # ===== TEMPERATURE CONVERSIONS =====
    # Special handling - not factor-based, uses formulas
    temperature_units = ['celsius', 'c', 'fahrenheit', 'f', 'kelvin', 'k']
    
    # ===== VOLUME CONVERSIONS =====
    # Base unit: liter (l)
    volume_units = {
        'milliliter': 0.001, 'ml': 0.001,
        'liter': 1.0, 'l': 1.0, 'litre': 1.0,
        'cubic meter': 1000.0, 'm3': 1000.0, 'm³': 1000.0,
        'fluid ounce': 0.0295735, 'fl oz': 0.0295735, 'floz': 0.0295735,  # US fluid ounce
        'cup': 0.236588, 'cups': 0.236588,  # US cup
        'pint': 0.473176, 'pt': 0.473176, 'pints': 0.473176,  # US liquid pint
        'quart': 0.946353, 'qt': 0.946353, 'quarts': 0.946353,  # US liquid quart
        'gallon': 3.78541, 'gal': 3.78541, 'gallons': 3.78541,  # US gallon
    }
    
    # ===== TIME CONVERSIONS =====
    # Base unit: second (s)
    time_units = {
        'millisecond': 0.001, 'ms': 0.001,
        'second': 1.0, 's': 1.0, 'sec': 1.0,
        'minute': 60.0, 'min': 60.0, 'minutes': 60.0,
        'hour': 3600.0, 'h': 3600.0, 'hr': 3600.0, 'hours': 3600.0,
        'day': 86400.0, 'd': 86400.0, 'days': 86400.0,
        'week': 604800.0, 'weeks': 604800.0, 'wk': 604800.0,
        'year': 31536000.0, 'years': 31536000.0, 'yr': 31536000.0,  # 365 days
    }
    
    # ===== DIGITAL STORAGE CONVERSIONS =====
    # Base unit: byte (B)
    # Using binary (1024) not decimal (1000) as per standard computer storage
    storage_units = {
        'bit': 0.125, 'bits': 0.125,
        'byte': 1.0, 'b': 1.0, 'bytes': 1.0,
        'kilobyte': 1024.0, 'kb': 1024.0,  # Binary kilobyte
        'megabyte': 1048576.0, 'mb': 1048576.0,  # 1024^2
        'gigabyte': 1073741824.0, 'gb': 1073741824.0,  # 1024^3
        'terabyte': 1099511627776.0, 'tb': 1099511627776.0,  # 1024^4
    }
    
    # ===== SPEED CONVERSIONS =====
    # Base unit: meters per second (m/s)
    speed_units = {
        'meters per second': 1.0, 'm/s': 1.0, 'mps': 1.0,
        'kilometers per hour': 0.277778, 'km/h': 0.277778, 'kph': 0.277778, 'kmh': 0.277778,
        'miles per hour': 0.44704, 'mph': 0.44704,
    }
    
    # Determine which category the units belong to
    def get_unit_category(unit: str) -> tuple[str, dict]:
        """Return (category_name, conversion_dict) for a unit."""
        if unit in length_units:
            return ('length', length_units)
        elif unit in weight_units:
            return ('weight', weight_units)
        elif unit in temperature_units:
            return ('temperature', {})
        elif unit in volume_units:
            return ('volume', volume_units)
        elif unit in time_units:
            return ('time', time_units)
        elif unit in storage_units:
            return ('storage', storage_units)
        elif unit in speed_units:
            return ('speed', speed_units)
        else:
            return (None, {})
    
    # Get categories for both units
    from_category, from_dict = get_unit_category(from_unit_lower)
    to_category, to_dict = get_unit_category(to_unit_lower)
    
    # Validate units are recognized
    if from_category is None:
        raise ValueError(
            f"Unknown unit: '{from_unit}'. Please check the unit name or abbreviation."
        )
    
    if to_category is None:
        raise ValueError(
            f"Unknown unit: '{to_unit}'. Please check the unit name or abbreviation."
        )
    
    # Validate units are in the same category
    if from_category != to_category:
        raise ValueError(
            f"Cannot convert between different unit categories: "
            f"'{from_unit}' is a {from_category} unit, "
            f"but '{to_unit}' is a {to_category} unit."
        )
    
    # Perform conversion based on category
    result = 0.0
    formula = None
    
    if from_category == 'temperature':
        # Temperature conversions use formulas, not factors
        # Normalize temperature unit names
        from_temp = from_unit_lower
        to_temp = to_unit_lower
        
        # Map to standard names
        temp_map = {'c': 'celsius', 'f': 'fahrenheit', 'k': 'kelvin'}
        from_temp = temp_map.get(from_temp, from_temp)
        to_temp = temp_map.get(to_temp, to_temp)
        
        # Convert from source to Celsius first (as intermediate)
        if from_temp == 'celsius':
            celsius_value = value
        elif from_temp == 'fahrenheit':
            celsius_value = (value - 32) * 5/9
            formula = "(F - 32) × 5/9"
        elif from_temp == 'kelvin':
            celsius_value = value - 273.15
            formula = "K - 273.15"
        else:
            raise ValueError(f"Unknown temperature unit: {from_unit}")
        
        # Convert from Celsius to target
        if to_temp == 'celsius':
            result = celsius_value
            if formula is None:
                formula = "Same unit"
        elif to_temp == 'fahrenheit':
            result = celsius_value * 9/5 + 32
            if from_temp == 'celsius':
                formula = "C × 9/5 + 32"
            elif from_temp == 'fahrenheit':
                formula = "Same unit"
            else:
                formula = f"({formula}) × 9/5 + 32"
        elif to_temp == 'kelvin':
            result = celsius_value + 273.15
            if from_temp == 'celsius':
                formula = "C + 273.15"
            elif from_temp == 'kelvin':
                formula = "Same unit"
            else:
                formula = f"({formula}) + 273.15"
        else:
            raise ValueError(f"Unknown temperature unit: {to_unit}")
        
        # Temperature-specific formatting
        temp_symbols = {'celsius': '°C', 'fahrenheit': '°F', 'kelvin': 'K'}
        from_symbol = temp_symbols.get(from_temp, from_unit)
        to_symbol = temp_symbols.get(to_temp, to_unit)
        
        formatted = f"{value:.2f}{from_symbol} = {result:.2f}{to_symbol}"
        if formula and formula != "Same unit":
            formatted += f" (Formula: {formula})"
            
    else:
        # Factor-based conversion for all other categories
        # Convert: value in from_unit -> base unit -> to_unit
        # result = value * (from_factor / to_factor)
        from_factor = from_dict[from_unit_lower]
        to_factor = to_dict[to_unit_lower]
        
        # Convert to base unit, then to target unit
        base_value = value * from_factor
        result = base_value / to_factor
        
        # Calculate the effective conversion factor
        conversion_factor = from_factor / to_factor
        formula = f"× {conversion_factor:.6g}"
        
        # Format with appropriate precision
        # Use more decimals for very small or very large results
        if abs(result) < 0.01 or abs(result) > 10000:
            formatted = f"{value:g} {from_unit} = {result:.6g} {to_unit}"
        else:
            formatted = f"{value:g} {from_unit} = {result:.4g} {to_unit}"
    
    return {
        'value': value,
        'from_unit': from_unit,
        'to_unit': to_unit,
        'result': round(result, 10),  # Round to avoid floating point artifacts
        'formatted': formatted,
        'formula': formula,
        'category': from_category
    }


# ============================================================================
# Date Calculator Functions
# ============================================================================


def date_diff(date1: str, date2: str, unit: str = "all") -> dict[str, Any]:
    """
    Calculate the difference between two dates.
    
    Supports multiple units of time difference calculation with proper handling
    of leap years and varying month lengths using the dateutil library.
    
    Args:
        date1: First date in ISO format (YYYY-MM-DD)
        date2: Second date in ISO format (YYYY-MM-DD)
        unit: Unit of difference ('days', 'weeks', 'months', 'years', 'all')
              Default is 'all' which returns all units
    
    Returns:
        Dictionary containing the difference in requested unit(s)
        
    Raises:
        ValueError: If date format is invalid or unit is not recognized
        
    Examples:
        date_diff("2025-01-01", "2025-12-31", "days") → 364 days
        date_diff("2025-01-15", "2025-02-14", "weeks") → 4 weeks and 2 days
        date_diff("2020-01-01", "2025-01-01", "years") → 5 years
        
    Algorithm Explanation:
        1. Parse dates using ISO 8601 format
        2. Calculate the timedelta between dates
        3. Convert to requested unit(s) with proper rounding
        4. Handle negative differences (date1 > date2)
    """
    # Validate and parse dates
    try:
        dt1 = datetime.fromisoformat(date1)
        dt2 = datetime.fromisoformat(date2)
    except ValueError as e:
        raise ValueError(f"Invalid date format. Expected YYYY-MM-DD, got date1='{date1}', date2='{date2}'. Error: {e}")
    
    # Validate unit
    valid_units = ["days", "weeks", "months", "years", "all"]
    if unit not in valid_units:
        raise ValueError(f"Invalid unit '{unit}'. Must be one of: {', '.join(valid_units)}")
    
    # Calculate basic difference
    delta = dt2 - dt1
    days_diff = delta.days
    
    # Calculate using relativedelta for accurate month/year differences
    rel_delta = relativedelta(dt2, dt1)
    
    # Build result based on requested unit
    if unit == "days":
        return {
            "days": days_diff,
            "formatted": f"{abs(days_diff)} day{'s' if abs(days_diff) != 1 else ''}"
        }
    elif unit == "weeks":
        weeks = days_diff // 7
        remaining_days = days_diff % 7
        return {
            "weeks": weeks,
            "days": remaining_days,
            "total_days": days_diff,
            "formatted": f"{abs(weeks)} week{'s' if abs(weeks) != 1 else ''}, {abs(remaining_days)} day{'s' if abs(remaining_days) != 1 else ''}"
        }
    elif unit == "months":
        total_months = rel_delta.years * 12 + rel_delta.months
        return {
            "months": total_months,
            "days": rel_delta.days,
            "total_days": days_diff,
            "formatted": f"{abs(total_months)} month{'s' if abs(total_months) != 1 else ''}"
        }
    elif unit == "years":
        return {
            "years": rel_delta.years,
            "months": rel_delta.months,
            "days": rel_delta.days,
            "total_days": days_diff,
            "formatted": f"{abs(rel_delta.years)} year{'s' if abs(rel_delta.years) != 1 else ''}"
        }
    else:  # unit == "all"
        return {
            "years": rel_delta.years,
            "months": rel_delta.months,
            "days": rel_delta.days,
            "total_days": days_diff,
            "total_weeks": days_diff // 7,
            "formatted": (
                f"{abs(rel_delta.years)} year{'s' if abs(rel_delta.years) != 1 else ''}, "
                f"{abs(rel_delta.months)} month{'s' if abs(rel_delta.months) != 1 else ''}, "
                f"{abs(rel_delta.days)} day{'s' if abs(rel_delta.days) != 1 else ''}"
            )
        }


def date_add(date: str, amount: int, unit: str) -> dict[str, Any]:
    """
    Add or subtract time from a date.
    
    Properly handles month-end dates, leap years, and DST transitions.
    Negative amounts subtract time from the date.
    
    Args:
        date: Starting date in ISO format (YYYY-MM-DD)
        amount: Amount to add (can be negative to subtract)
        unit: Time unit ('days', 'weeks', 'months', 'years')
    
    Returns:
        Dictionary with new date and formatted information
        
    Raises:
        ValueError: If date format is invalid or unit is not recognized
        
    Examples:
        date_add("2025-01-15", 30, "days") → "2025-02-14"
        date_add("2025-01-31", 1, "months") → "2025-02-28"
        date_add("2024-02-29", 1, "years") → "2025-02-28" (leap year handling)
        date_add("2025-06-15", -90, "days") → "2025-03-17"
        
    Algorithm Explanation:
        1. Parse the input date
        2. Use relativedelta for accurate month/year arithmetic
        3. Use timedelta for day/week arithmetic
        4. Handle edge cases (month-end dates, leap years)
    """
    # Validate and parse date
    try:
        dt = datetime.fromisoformat(date)
    except ValueError as e:
        raise ValueError(f"Invalid date format. Expected YYYY-MM-DD, got '{date}'. Error: {e}")
    
    # Validate unit
    valid_units = ["days", "weeks", "months", "years"]
    if unit not in valid_units:
        raise ValueError(f"Invalid unit '{unit}'. Must be one of: {', '.join(valid_units)}")
    
    # Calculate new date based on unit
    if unit == "days":
        new_dt = dt + timedelta(days=amount)
    elif unit == "weeks":
        new_dt = dt + timedelta(weeks=amount)
    elif unit == "months":
        new_dt = dt + relativedelta(months=amount)
    elif unit == "years":
        new_dt = dt + relativedelta(years=amount)
    
    return {
        "original_date": date,
        "new_date": new_dt.strftime("%Y-%m-%d"),
        "amount": amount,
        "unit": unit,
        "formatted": f"{date} + {amount} {unit} = {new_dt.strftime('%Y-%m-%d')}"
    }


def business_days(start_date: str, end_date: str, exclude_holidays: list[str] = None) -> dict[str, Any]:
    """
    Calculate business days between two dates (excluding weekends).
    
    Business days are Monday through Friday. Optionally excludes custom holidays.
    Weekends (Saturday and Sunday) are always excluded.
    
    Args:
        start_date: Start date in ISO format (YYYY-MM-DD)
        end_date: End date in ISO format (YYYY-MM-DD)
        exclude_holidays: Optional list of holiday dates in ISO format to exclude
    
    Returns:
        Dictionary with business day count and breakdown
        
    Raises:
        ValueError: If date format is invalid
        
    Examples:
        business_days("2025-01-06", "2025-01-10") → 5 business days (Mon-Fri)
        business_days("2025-01-04", "2025-01-11") → 6 business days (Sat-Sat, excludes weekends)
        business_days("2025-12-22", "2025-12-26", ["2025-12-25"]) → 3 business days (excluding Christmas)
        
    Algorithm Explanation:
        1. Parse start and end dates
        2. Iterate through all dates in range
        3. Count weekdays (Monday=0 to Friday=4)
        4. Exclude any dates in the holidays list
        5. Return count and breakdown
    """
    # Validate and parse dates
    try:
        start_dt = datetime.fromisoformat(start_date)
        end_dt = datetime.fromisoformat(end_date)
    except ValueError as e:
        raise ValueError(f"Invalid date format. Expected YYYY-MM-DD. Error: {e}")
    
    # Ensure start_date is before end_date
    if start_dt > end_dt:
        start_dt, end_dt = end_dt, start_dt
    
    # Parse holidays if provided
    holiday_set = set()
    if exclude_holidays:
        for holiday in exclude_holidays:
            try:
                holiday_dt = datetime.fromisoformat(holiday)
                holiday_set.add(holiday_dt.date())
            except ValueError:
                raise ValueError(f"Invalid holiday date format: {holiday}")
    
    # Count business days
    business_day_count = 0
    weekend_count = 0
    holiday_count = 0
    current_dt = start_dt
    
    while current_dt <= end_dt:
        # Check if it's a weekday (Monday=0, Friday=4, Saturday=5, Sunday=6)
        if current_dt.weekday() < 5:  # Monday to Friday
            if current_dt.date() not in holiday_set:
                business_day_count += 1
            else:
                holiday_count += 1
        else:
            weekend_count += 1
        
        current_dt += timedelta(days=1)
    
    total_days = (end_dt - start_dt).days + 1
    
    return {
        "business_days": business_day_count,
        "total_days": total_days,
        "weekend_days": weekend_count,
        "holidays_excluded": holiday_count,
        "start_date": start_date,
        "end_date": end_date,
        "formatted": (
            f"{business_day_count} business day{'s' if business_day_count != 1 else ''} "
            f"between {start_date} and {end_date}"
        )
    }


def age_calculator(birthdate: str, reference_date: str = None) -> dict[str, Any]:
    """
    Calculate age from birthdate.
    
    Computes precise age in years, months, and days. Handles leap years correctly.
    Can calculate age as of any reference date (defaults to today).
    
    Args:
        birthdate: Birth date in ISO format (YYYY-MM-DD)
        reference_date: Reference date for age calculation (default: today)
    
    Returns:
        Dictionary with age in years, months, days and additional info
        
    Raises:
        ValueError: If date format is invalid or birthdate is in the future
        
    Examples:
        age_calculator("1990-05-15") → "35 years, 6 months, 4 days old"
        age_calculator("2000-01-01", "2025-01-01") → "25 years, 0 months, 0 days old"
        age_calculator("2020-02-29", "2021-03-01") → "1 year, 0 months, 1 day old"
        
    Algorithm Explanation:
        1. Parse birthdate and reference date
        2. Validate birthdate is not in the future
        3. Use relativedelta for accurate age calculation
        4. Handle leap year birthdates properly
        5. Calculate additional metadata (days lived, etc.)
    """
    # Parse birthdate
    try:
        birth_dt = datetime.fromisoformat(birthdate)
    except ValueError as e:
        raise ValueError(f"Invalid birthdate format. Expected YYYY-MM-DD, got '{birthdate}'. Error: {e}")
    
    # Parse or default reference date
    if reference_date:
        try:
            ref_dt = datetime.fromisoformat(reference_date)
        except ValueError as e:
            raise ValueError(f"Invalid reference_date format. Expected YYYY-MM-DD, got '{reference_date}'. Error: {e}")
    else:
        ref_dt = datetime.now()
        reference_date = ref_dt.strftime("%Y-%m-%d")
    
    # Validate birthdate is not in the future
    if birth_dt > ref_dt:
        raise ValueError(f"Birthdate {birthdate} is in the future relative to reference date {reference_date}")
    
    # Calculate age using relativedelta
    age = relativedelta(ref_dt, birth_dt)
    
    # Calculate total days lived
    total_days = (ref_dt - birth_dt).days
    
    return {
        "years": age.years,
        "months": age.months,
        "days": age.days,
        "total_days": total_days,
        "birthdate": birthdate,
        "reference_date": reference_date,
        "formatted": (
            f"{age.years} year{'s' if age.years != 1 else ''}, "
            f"{age.months} month{'s' if age.months != 1 else ''}, "
            f"{age.days} day{'s' if age.days != 1 else ''} old"
        )
    }


def day_of_week(date: str) -> dict[str, Any]:
    """
    Determine day of week and additional calendar information for any date.
    
    Provides comprehensive calendar information including day name, week number,
    day of year, and whether it's a weekend.
    
    Args:
        date: Date in ISO format (YYYY-MM-DD)
    
    Returns:
        Dictionary with day name, week number, day of year, and more
        
    Raises:
        ValueError: If date format is invalid
        
    Examples:
        day_of_week("2025-11-19") → "Wednesday, Week 47, Day 323"
        day_of_week("2000-01-01") → "Saturday, Week 1, Day 1"
        day_of_week("2024-12-25") → "Wednesday, Week 52, Day 360"
        
    Algorithm Explanation:
        1. Parse the date
        2. Use datetime methods to extract calendar information
        3. Calculate ISO week number
        4. Determine if it's a weekend
        5. Format comprehensive output
    """
    # Validate and parse date
    try:
        dt = datetime.fromisoformat(date)
    except ValueError as e:
        raise ValueError(f"Invalid date format. Expected YYYY-MM-DD, got '{date}'. Error: {e}")
    
    # Get day of week information
    day_name = dt.strftime("%A")  # Full day name (e.g., "Monday")
    day_abbr = dt.strftime("%a")  # Abbreviated day name (e.g., "Mon")
    day_number = dt.weekday()  # 0=Monday, 6=Sunday
    
    # Get ISO calendar information
    iso_calendar = dt.isocalendar()
    iso_year = iso_calendar[0]
    iso_week = iso_calendar[1]
    iso_weekday = iso_calendar[2]  # 1=Monday, 7=Sunday
    
    # Get day of year
    day_of_year = dt.timetuple().tm_yday
    
    # Check if weekend
    is_weekend = day_number >= 5  # Saturday=5, Sunday=6
    
    return {
        "date": date,
        "day_name": day_name,
        "day_abbr": day_abbr,
        "day_number": day_number,  # 0=Monday, 6=Sunday
        "iso_weekday": iso_weekday,  # 1=Monday, 7=Sunday
        "week_number": iso_week,
        "iso_year": iso_year,
        "day_of_year": day_of_year,
        "is_weekend": is_weekend,
        "formatted": f"{day_name}, Week {iso_week}, Day {day_of_year}"
    }


# ============================================================================
# MCP Server Implementation
# ============================================================================


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
        Tool(
            name="gcd",
            description=(
                "Calculate the greatest common divisor (GCD) of two or more numbers using the Euclidean algorithm. "
                "The GCD is the largest positive integer that divides all given numbers without a remainder. "
                "For example, GCD(48, 18) = 6."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "numbers": {
                        "type": "array",
                        "description": "Array of integers to find GCD of",
                        "items": {
                            "type": "integer"
                        },
                        "minItems": 2,
                        "maxItems": 10,
                    },
                },
                "required": ["numbers"],
            },
        ),
        Tool(
            name="lcm",
            description=(
                "Calculate the least common multiple (LCM) of two or more numbers. "
                "The LCM is the smallest positive integer divisible by all given numbers. "
                "Uses the formula LCM(a,b) = (a × b) / GCD(a,b). "
                "For example, LCM(12, 18) = 36."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "numbers": {
                        "type": "array",
                        "description": "Array of integers to find LCM of",
                        "items": {
                            "type": "integer"
                        },
                        "minItems": 2,
                        "maxItems": 10,
                    },
                },
                "required": ["numbers"],
            },
        ),
        Tool(
            name="factorial",
            description=(
                "Calculate the factorial of a number (n!). "
                "The factorial is the product of all positive integers less than or equal to n. "
                "By definition, 0! = 1. Limited to n ≤ 170 to avoid overflow. "
                "For example, factorial(5) = 120."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "n": {
                        "type": "integer",
                        "description": "The number to calculate factorial of",
                        "minimum": 0,
                        "maximum": 170,
                    },
                },
                "required": ["n"],
            },
        ),
        Tool(
            name="combinations",
            description=(
                "Calculate combinations (nCr) - the number of ways to choose r items from n items "
                "where order does not matter. Also written as 'n choose r' or C(n,r). "
                "Formula: C(n,r) = n! / (r! × (n-r)!). "
                "For example, C(5,2) = 10."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "n": {
                        "type": "integer",
                        "description": "Total number of items",
                        "minimum": 0,
                        "maximum": 1000,
                    },
                    "r": {
                        "type": "integer",
                        "description": "Number of items to choose (must be ≤ n)",
                        "minimum": 0,
                    },
                },
                "required": ["n", "r"],
            },
        ),
        Tool(
            name="permutations",
            description=(
                "Calculate permutations (nPr) - the number of ordered ways to choose r items from n items "
                "where order matters. Also written as 'n permute r' or P(n,r). "
                "Formula: P(n,r) = n! / (n-r)!. "
                "For example, P(5,2) = 20."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "n": {
                        "type": "integer",
                        "description": "Total number of items",
                        "minimum": 0,
                        "maximum": 1000,
                    },
                    "r": {
                        "type": "integer",
                        "description": "Number of items to arrange (must be ≤ n)",
                        "minimum": 0,
                    },
                },
                "required": ["n", "r"],
            },
        ),
        Tool(
            name="pascal_triangle",
            description=(
                "Generate Pascal's triangle up to n rows. "
                "Each number in the triangle is the sum of the two numbers directly above it. "
                "The triangle has applications in combinatorics, probability, and algebra. "
                "Uses memoization for efficient repeated calculations."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "rows": {
                        "type": "integer",
                        "description": "Number of rows to generate in Pascal's triangle",
                        "minimum": 1,
                        "maximum": 30,
                    },
                },
                "required": ["rows"],
            },
        ),
        Tool(
            name="triangular_numbers",
            description=(
                "Calculate triangular numbers - numbers that can form equilateral triangles. "
                "Can either calculate the nth triangular number or generate a sequence. "
                "Formula: T(n) = n × (n + 1) / 2. "
                "Examples: T(5) = 15, sequence: [1, 3, 6, 10, 15]"
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "n": {
                        "type": "integer",
                        "description": "Position of the triangular number to calculate (1-1000). Mutually exclusive with 'limit'.",
                        "minimum": 1,
                        "maximum": 1000,
                    },
                    "limit": {
                        "type": "integer",
                        "description": "Generate sequence of first 'limit' triangular numbers (1-1000). Mutually exclusive with 'n'.",
                        "minimum": 1,
                        "maximum": 1000,
                    },
                },
            },
        ),
        Tool(
            name="perfect_numbers",
            description=(
                "Find perfect numbers up to a given limit. "
                "A perfect number equals the sum of its proper divisors. "
                "Examples: 6 (1+2+3=6), 28 (1+2+4+7+14=28). "
                "Perfect numbers are extremely rare - only 4 exist below 10000."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "limit": {
                        "type": "integer",
                        "description": "Upper bound for searching perfect numbers",
                        "minimum": 1,
                        "maximum": 10000,
                    },
                },
                "required": ["limit"],
            },
        ),
        Tool(
            name="collatz_sequence",
            description=(
                "Generate the Collatz sequence (3n+1 problem) for a starting number. "
                "Rules: If even, divide by 2; if odd, multiply by 3 and add 1. "
                "The sequence continues until reaching 1. "
                "Returns the complete sequence, step count, and maximum value reached. "
                "Example: 13 → [13, 40, 20, 10, 5, 16, 8, 4, 2, 1] (9 steps)"
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "n": {
                        "type": "integer",
                        "description": "Starting number for the Collatz sequence",
                        "minimum": 1,
                        "maximum": 100000,
                    },
                },
                "required": ["n"],
            },
        ),
        Tool(
            name="generate_hash",
            description=(
                "Generate cryptographic hash of text or data using various algorithms. "
                "Supports MD5, SHA-1, SHA-256, SHA-512, and BLAKE2b with hex or base64 output. "
                "Includes security notes and recommendations for each algorithm. "
                "WARNING: MD5 and SHA-1 are not cryptographically secure - use SHA-256+ for security."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "data": {
                        "type": "string",
                        "description": "Input text or data to hash (max 1MB)",
                    },
                    "algorithm": {
                        "type": "string",
                        "description": "Hash algorithm to use",
                        "enum": ["md5", "sha1", "sha256", "sha512", "blake2b"],
                    },
                    "output_format": {
                        "type": "string",
                        "description": "Output format for the hash (default: hex)",
                        "enum": ["hex", "base64"],
                        "default": "hex",
                    },
                },
                "required": ["data", "algorithm"],
            },
        ),
        Tool(
            name="descriptive_stats",
            description=(
                "Calculate comprehensive descriptive statistics for a dataset. "
                "Computes mean, median, mode, range, variance, and standard deviation. "
                "Provides a complete statistical summary of the data distribution."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "data": {
                        "type": "array",
                        "description": "Array of numeric values to analyze (1-10000 items)",
                        "items": {
                            "type": "number"
                        },
                        "minItems": 1,
                        "maxItems": 10000,
                    },
                },
                "required": ["data"],
            },
        ),
        Tool(
            name="correlation",
            description=(
                "Calculate Pearson correlation coefficient between two datasets. "
                "Measures the linear relationship between two variables. "
                "Returns coefficient (-1 to 1) with human-readable interpretation. "
                "Values near 1 indicate strong positive correlation, near -1 strong negative correlation, "
                "and near 0 indicate no linear relationship."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "x": {
                        "type": "array",
                        "description": "First dataset (2-1000 items)",
                        "items": {
                            "type": "number"
                        },
                        "minItems": 2,
                        "maxItems": 1000,
                    },
                    "y": {
                        "type": "array",
                        "description": "Second dataset (must be same length as x)",
                        "items": {
                            "type": "number"
                        },
                        "minItems": 2,
                        "maxItems": 1000,
                    },
                },
                "required": ["x", "y"],
            },
        ),
        Tool(
            name="percentile",
            description=(
                "Calculate the value at a specific percentile in a dataset. "
                "A percentile indicates the value below which a given percentage of data falls. "
                "For example, the 50th percentile is the median, and the 25th/75th percentiles are quartiles. "
                "Uses linear interpolation method for accurate results."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "data": {
                        "type": "array",
                        "description": "Array of numeric values (1-10000 items)",
                        "items": {
                            "type": "number"
                        },
                        "minItems": 1,
                        "maxItems": 10000,
                    },
                    "percentile": {
                        "type": "number",
                        "description": "Percentile to calculate (0-100)",
                        "minimum": 0,
                        "maximum": 100,
                    },
                },
                "required": ["data", "percentile"],
            },
        ),
        Tool(
            name="outliers",
            description=(
                "Identify outliers in a dataset using the IQR (Interquartile Range) method. "
                "Detects values that fall outside the range [Q1 - 1.5×IQR, Q3 + 1.5×IQR]. "
                "Returns outlier values, their indices, and statistical boundaries. "
                "This is a robust method that works well for various data distributions."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "data": {
                        "type": "array",
                        "description": "Array of numeric values to analyze (1-10000 items)",
                        "items": {
                            "type": "number"
                        },
                        "minItems": 1,
                        "maxItems": 10000,
                    },
                },
                "required": ["data"],
            },
        ),
        Tool(
            name="unit_convert",
            description=(
                "Convert between different units of measurement across multiple categories. "
                "Supports: Length (mm, cm, m, km, in, ft, yd, mi), "
                "Weight (mg, g, kg, t, oz, lb, ton), "
                "Temperature (C, F, K with formulas), "
                "Volume (ml, l, m3, fl oz, cup, pt, qt, gal), "
                "Time (ms, s, min, h, d, week, year), "
                "Digital Storage (bit, B, KB, MB, GB, TB using binary 1024), "
                "Speed (m/s, km/h, mph). "
                "Uses precise conversion factors and handles both abbreviated and full unit names (case-insensitive)."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "value": {
                        "type": "number",
                        "description": "The numeric value to convert",
                    },
                    "from_unit": {
                        "type": "string",
                        "description": "The source unit (e.g., 'km', 'kilometer', 'F', 'fahrenheit', 'GB')",
                    },
                    "to_unit": {
                        "type": "string",
                        "description": "The target unit (e.g., 'mi', 'mile', 'C', 'celsius', 'MB')",
                    },
                },
                "required": ["value", "from_unit", "to_unit"],
            },
        ),
        Tool(
            name="date_diff",
            description=(
                "Calculate the difference between two dates in various units. "
                "Returns the time span in days, weeks, months, years, or all units. "
                "Properly handles leap years and varying month lengths. "
                "Example: '2025-01-01' to '2025-12-31' = 364 days."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "date1": {
                        "type": "string",
                        "description": "First date in ISO format (YYYY-MM-DD)",
                    },
                    "date2": {
                        "type": "string",
                        "description": "Second date in ISO format (YYYY-MM-DD)",
                    },
                    "unit": {
                        "type": "string",
                        "description": "Unit of difference: 'days', 'weeks', 'months', 'years', or 'all' (default)",
                        "enum": ["days", "weeks", "months", "years", "all"],
                        "default": "all",
                    },
                },
                "required": ["date1", "date2"],
            },
        ),
        Tool(
            name="date_add",
            description=(
                "Add or subtract time from a date. "
                "Properly handles month-end dates, leap years, and negative amounts. "
                "Example: '2025-01-15' + 30 days = '2025-02-14'."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "date": {
                        "type": "string",
                        "description": "Starting date in ISO format (YYYY-MM-DD)",
                    },
                    "amount": {
                        "type": "integer",
                        "description": "Amount to add (can be negative to subtract)",
                    },
                    "unit": {
                        "type": "string",
                        "description": "Time unit: 'days', 'weeks', 'months', or 'years'",
                        "enum": ["days", "weeks", "months", "years"],
                    },
                },
                "required": ["date", "amount", "unit"],
            },
        ),
        Tool(
            name="business_days",
            description=(
                "Calculate business days between two dates, excluding weekends (Saturday and Sunday). "
                "Optionally exclude custom holidays. "
                "Business days are Monday through Friday only. "
                "Example: Jan 6-10, 2025 (Mon-Fri) = 5 business days."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "start_date": {
                        "type": "string",
                        "description": "Start date in ISO format (YYYY-MM-DD)",
                    },
                    "end_date": {
                        "type": "string",
                        "description": "End date in ISO format (YYYY-MM-DD)",
                    },
                    "exclude_holidays": {
                        "type": "array",
                        "description": "Optional array of holiday dates in ISO format to exclude",
                        "items": {
                            "type": "string",
                        },
                        "default": [],
                    },
                },
                "required": ["start_date", "end_date"],
            },
        ),
        Tool(
            name="age_calculator",
            description=(
                "Calculate age from birthdate with precise years, months, and days. "
                "Can calculate age as of any reference date (defaults to today). "
                "Handles leap years correctly. "
                "Example: Born 1990-05-15 → '35 years, 6 months, 4 days old'."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "birthdate": {
                        "type": "string",
                        "description": "Birth date in ISO format (YYYY-MM-DD)",
                    },
                    "reference_date": {
                        "type": "string",
                        "description": "Optional reference date for age calculation (default: today)",
                    },
                },
                "required": ["birthdate"],
            },
        ),
        Tool(
            name="day_of_week",
            description=(
                "Determine the day of week and additional calendar information for any date. "
                "Returns day name, week number, day of year, and whether it's a weekend. "
                "Example: '2025-11-19' → 'Wednesday, Week 47, Day 323'."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "date": {
                        "type": "string",
                        "description": "Date in ISO format (YYYY-MM-DD)",
                    },
                },
                "required": ["date"],
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
        elif name == "gcd":
            return await handle_gcd(arguments)
        elif name == "lcm":
            return await handle_lcm(arguments)
        elif name == "factorial":
            return await handle_factorial(arguments)
        elif name == "combinations":
            return await handle_combinations(arguments)
        elif name == "permutations":
            return await handle_permutations(arguments)
        elif name == "pascal_triangle":
            return await handle_pascal_triangle(arguments)
        elif name == "triangular_numbers":
            return await handle_triangular_numbers(arguments)
        elif name == "perfect_numbers":
            return await handle_perfect_numbers(arguments)
        elif name == "collatz_sequence":
            return await handle_collatz_sequence(arguments)
        elif name == "generate_hash":
            return await handle_generate_hash(arguments)
        elif name == "descriptive_stats":
            return await handle_descriptive_stats(arguments)
        elif name == "correlation":
            return await handle_correlation(arguments)
        elif name == "percentile":
            return await handle_percentile(arguments)
        elif name == "outliers":
            return await handle_outliers(arguments)
        elif name == "unit_convert":
            return await handle_unit_convert(arguments)
        elif name == "date_diff":
            return await handle_date_diff(arguments)
        elif name == "date_add":
            return await handle_date_add(arguments)
        elif name == "business_days":
            return await handle_business_days(arguments)
        elif name == "age_calculator":
            return await handle_age_calculator(arguments)
        elif name == "day_of_week":
            return await handle_day_of_week(arguments)
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


async def handle_gcd(arguments: Any) -> CallToolResult:
    """Handle gcd tool calls."""
    # Extract and validate parameters
    numbers = arguments.get("numbers")
    
    # Validate required parameter
    if numbers is None:
        logger.error("Missing required parameter: numbers")
        return CallToolResult(
            content=[TextContent(
                type="text",
                text="Missing required parameter 'numbers'"
            )],
            isError=True,
        )
    
    # Validate parameter type
    if not isinstance(numbers, list):
        logger.error(f"Invalid parameter type for numbers: {type(numbers)}")
        return CallToolResult(
            content=[TextContent(
                type="text",
                text=f"Parameter 'numbers' must be an array, got {type(numbers).__name__}"
            )],
            isError=True,
        )
    
    # Validate array length
    if len(numbers) < 2 or len(numbers) > 10:
        logger.error(f"Invalid number count: {len(numbers)}")
        return CallToolResult(
            content=[TextContent(
                type="text",
                text="Parameter 'numbers' must contain between 2 and 10 integers"
            )],
            isError=True,
        )
    
    # Validate all elements are integers
    if not all(isinstance(n, int) for n in numbers):
        logger.error("Not all elements in numbers are integers")
        return CallToolResult(
            content=[TextContent(
                type="text",
                text="All elements in 'numbers' must be integers"
            )],
            isError=True,
        )
    
    # Calculate GCD
    logger.info(f"Calculating GCD of {numbers}")
    result = gcd(numbers)
    
    result_text = (
        f"Greatest Common Divisor (GCD) of {numbers}:\n"
        f"GCD = {result}"
    )
    
    logger.info(f"GCD of {numbers} = {result}")
    
    return CallToolResult(
        content=[TextContent(type="text", text=result_text)],
        isError=False,
    )


async def handle_lcm(arguments: Any) -> CallToolResult:
    """Handle lcm tool calls."""
    # Extract and validate parameters
    numbers = arguments.get("numbers")
    
    # Validate required parameter
    if numbers is None:
        logger.error("Missing required parameter: numbers")
        return CallToolResult(
            content=[TextContent(
                type="text",
                text="Missing required parameter 'numbers'"
            )],
            isError=True,
        )
    
    # Validate parameter type
    if not isinstance(numbers, list):
        logger.error(f"Invalid parameter type for numbers: {type(numbers)}")
        return CallToolResult(
            content=[TextContent(
                type="text",
                text=f"Parameter 'numbers' must be an array, got {type(numbers).__name__}"
            )],
            isError=True,
        )
    
    # Validate array length
    if len(numbers) < 2 or len(numbers) > 10:
        logger.error(f"Invalid number count: {len(numbers)}")
        return CallToolResult(
            content=[TextContent(
                type="text",
                text="Parameter 'numbers' must contain between 2 and 10 integers"
            )],
            isError=True,
        )
    
    # Validate all elements are integers
    if not all(isinstance(n, int) for n in numbers):
        logger.error("Not all elements in numbers are integers")
        return CallToolResult(
            content=[TextContent(
                type="text",
                text="All elements in 'numbers' must be integers"
            )],
            isError=True,
        )
    
    # Check for zeros
    if 0 in numbers:
        logger.error("LCM is undefined for zero")
        return CallToolResult(
            content=[TextContent(
                type="text",
                text="LCM is undefined when any number is zero"
            )],
            isError=True,
        )
    
    # Calculate LCM
    logger.info(f"Calculating LCM of {numbers}")
    result = lcm(numbers)
    
    result_text = (
        f"Least Common Multiple (LCM) of {numbers}:\n"
        f"LCM = {result}"
    )
    
    logger.info(f"LCM of {numbers} = {result}")
    
    return CallToolResult(
        content=[TextContent(type="text", text=result_text)],
        isError=False,
    )


async def handle_factorial(arguments: Any) -> CallToolResult:
    """Handle factorial tool calls."""
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
    
    # Validate parameter type
    if not isinstance(n, int):
        logger.error(f"Invalid parameter type for n: {type(n)}")
        return CallToolResult(
            content=[TextContent(
                type="text",
                text=f"Parameter 'n' must be an integer, got {type(n).__name__}"
            )],
            isError=True,
        )
    
    # Validate range
    if n < 0 or n > 170:
        logger.error(f"Parameter n out of range: {n}")
        return CallToolResult(
            content=[TextContent(
                type="text",
                text="Parameter 'n' must be between 0 and 170"
            )],
            isError=True,
        )
    
    # Calculate factorial
    logger.info(f"Calculating factorial of {n}")
    result = factorial(n)
    
    result_text = f"Factorial of {n}:\n{n}! = {result}"
    
    logger.info(f"Factorial({n}) = {result}")
    
    return CallToolResult(
        content=[TextContent(type="text", text=result_text)],
        isError=False,
    )


async def handle_combinations(arguments: Any) -> CallToolResult:
    """Handle combinations tool calls."""
    # Extract and validate parameters
    n = arguments.get("n")
    r = arguments.get("r")
    
    # Validate required parameters
    if n is None:
        logger.error("Missing required parameter: n")
        return CallToolResult(
            content=[TextContent(
                type="text",
                text="Missing required parameter 'n'"
            )],
            isError=True,
        )
    
    if r is None:
        logger.error("Missing required parameter: r")
        return CallToolResult(
            content=[TextContent(
                type="text",
                text="Missing required parameter 'r'"
            )],
            isError=True,
        )
    
    # Validate parameter types
    if not isinstance(n, int):
        logger.error(f"Invalid parameter type for n: {type(n)}")
        return CallToolResult(
            content=[TextContent(
                type="text",
                text=f"Parameter 'n' must be an integer, got {type(n).__name__}"
            )],
            isError=True,
        )
    
    if not isinstance(r, int):
        logger.error(f"Invalid parameter type for r: {type(r)}")
        return CallToolResult(
            content=[TextContent(
                type="text",
                text=f"Parameter 'r' must be an integer, got {type(r).__name__}"
            )],
            isError=True,
        )
    
    # Validate ranges
    if n < 0 or n > 1000:
        logger.error(f"Parameter n out of range: {n}")
        return CallToolResult(
            content=[TextContent(
                type="text",
                text="Parameter 'n' must be between 0 and 1000"
            )],
            isError=True,
        )
    
    if r < 0:
        logger.error(f"Parameter r is negative: {r}")
        return CallToolResult(
            content=[TextContent(
                type="text",
                text="Parameter 'r' must be non-negative"
            )],
            isError=True,
        )
    
    if r > n:
        logger.error(f"Parameter r > n: {r} > {n}")
        return CallToolResult(
            content=[TextContent(
                type="text",
                text=f"Parameter 'r' ({r}) cannot be greater than 'n' ({n})"
            )],
            isError=True,
        )
    
    # Calculate combinations
    logger.info(f"Calculating C({n},{r})")
    result = combinations(n, r)
    
    result_text = (
        f"Combinations C({n},{r}):\n"
        f"Number of ways to choose {r} items from {n} items (order doesn't matter):\n"
        f"C({n},{r}) = {result}"
    )
    
    logger.info(f"C({n},{r}) = {result}")
    
    return CallToolResult(
        content=[TextContent(type="text", text=result_text)],
        isError=False,
    )


async def handle_permutations(arguments: Any) -> CallToolResult:
    """Handle permutations tool calls."""
    # Extract and validate parameters
    n = arguments.get("n")
    r = arguments.get("r")
    
    # Validate required parameters
    if n is None:
        logger.error("Missing required parameter: n")
        return CallToolResult(
            content=[TextContent(
                type="text",
                text="Missing required parameter 'n'"
            )],
            isError=True,
        )
    
    if r is None:
        logger.error("Missing required parameter: r")
        return CallToolResult(
            content=[TextContent(
                type="text",
                text="Missing required parameter 'r'"
            )],
            isError=True,
        )
    
    # Validate parameter types
    if not isinstance(n, int):
        logger.error(f"Invalid parameter type for n: {type(n)}")
        return CallToolResult(
            content=[TextContent(
                type="text",
                text=f"Parameter 'n' must be an integer, got {type(n).__name__}"
            )],
            isError=True,
        )
    
    if not isinstance(r, int):
        logger.error(f"Invalid parameter type for r: {type(r)}")
        return CallToolResult(
            content=[TextContent(
                type="text",
                text=f"Parameter 'r' must be an integer, got {type(r).__name__}"
            )],
            isError=True,
        )
    
    # Validate ranges
    if n < 0 or n > 1000:
        logger.error(f"Parameter n out of range: {n}")
        return CallToolResult(
            content=[TextContent(
                type="text",
                text="Parameter 'n' must be between 0 and 1000"
            )],
            isError=True,
        )
    
    if r < 0:
        logger.error(f"Parameter r is negative: {r}")
        return CallToolResult(
            content=[TextContent(
                type="text",
                text="Parameter 'r' must be non-negative"
            )],
            isError=True,
        )
    
    if r > n:
        logger.error(f"Parameter r > n: {r} > {n}")
        return CallToolResult(
            content=[TextContent(
                type="text",
                text=f"Parameter 'r' ({r}) cannot be greater than 'n' ({n})"
            )],
            isError=True,
        )
    
    # Calculate permutations
    logger.info(f"Calculating P({n},{r})")
    result = permutations(n, r)
    
    result_text = (
        f"Permutations P({n},{r}):\n"
        f"Number of ordered ways to choose {r} items from {n} items (order matters):\n"
        f"P({n},{r}) = {result}"
    )
    
    logger.info(f"P({n},{r}) = {result}")
    
    return CallToolResult(
        content=[TextContent(type="text", text=result_text)],
        isError=False,
    )


async def handle_pascal_triangle(arguments: Any) -> CallToolResult:
    """Handle pascal_triangle tool calls."""
    # Extract and validate parameters
    rows = arguments.get("rows")
    
    # Validate required parameter
    if rows is None:
        logger.error("Missing required parameter: rows")
        return CallToolResult(
            content=[TextContent(
                type="text",
                text="Missing required parameter 'rows'"
            )],
            isError=True,
        )
    
    # Validate parameter type
    if not isinstance(rows, int):
        logger.error(f"Invalid parameter type for rows: {type(rows)}")
        return CallToolResult(
            content=[TextContent(
                type="text",
                text=f"Parameter 'rows' must be an integer, got {type(rows).__name__}"
            )],
            isError=True,
        )
    
    # Validate range
    if rows < 1 or rows > 30:
        logger.error(f"Parameter rows out of range: {rows}")
        return CallToolResult(
            content=[TextContent(
                type="text",
                text="Parameter 'rows' must be between 1 and 30"
            )],
            isError=True,
        )
    
    # Generate Pascal's triangle
    logger.info(f"Generating Pascal's triangle with {rows} rows")
    triangle = pascal_triangle(rows)
    
    # Format the result
    result_text = f"Pascal's triangle ({rows} rows):\n"
    for i, row in enumerate(triangle):
        result_text += f"Row {i}: {row}\n"
    
    result_text += f"\nVisualization tip: Each number is the sum of the two numbers above it."
    result_text += f"\nLast row: {triangle[-1]}"
    
    logger.info(f"Generated Pascal's triangle with {rows} rows")
    
    return CallToolResult(
        content=[TextContent(type="text", text=result_text)],
        isError=False,
    )


async def handle_triangular_numbers(arguments: Any) -> CallToolResult:
    """Handle triangular_numbers tool calls."""
    # Extract parameters
    n = arguments.get("n")
    limit = arguments.get("limit")
    
    # Validate that exactly one parameter is provided
    if n is None and limit is None:
        logger.error("Missing required parameter: either n or limit")
        return CallToolResult(
            content=[TextContent(
                type="text",
                text="Either 'n' or 'limit' must be provided"
            )],
            isError=True,
        )
    
    if n is not None and limit is not None:
        logger.error("Both n and limit provided")
        return CallToolResult(
            content=[TextContent(
                type="text",
                text="Cannot provide both 'n' and 'limit', use only one"
            )],
            isError=True,
        )
    
    # Handle single triangular number calculation
    if n is not None:
        # Validate parameter type
        if not isinstance(n, int):
            logger.error(f"Invalid parameter type for n: {type(n)}")
            return CallToolResult(
                content=[TextContent(
                    type="text",
                    text=f"Parameter 'n' must be an integer, got {type(n).__name__}"
                )],
                isError=True,
            )
        
        # Validate range
        if n < 1 or n > 1000:
            logger.error(f"Parameter n out of range: {n}")
            return CallToolResult(
                content=[TextContent(
                    type="text",
                    text="Parameter 'n' must be between 1 and 1000"
                )],
                isError=True,
            )
        
        # Calculate triangular number
        logger.info(f"Calculating triangular number T({n})")
        result = triangular_numbers(n=n)
        
        result_text = (
            f"Triangular number T({n}):\n"
            f"T({n}) = {result}\n\n"
            f"Formula: T(n) = n × (n + 1) / 2 = {n} × {n + 1} / 2 = {result}"
        )
        
        logger.info(f"T({n}) = {result}")
    
    # Handle sequence generation
    else:
        # Validate parameter type
        if not isinstance(limit, int):
            logger.error(f"Invalid parameter type for limit: {type(limit)}")
            return CallToolResult(
                content=[TextContent(
                    type="text",
                    text=f"Parameter 'limit' must be an integer, got {type(limit).__name__}"
                )],
                isError=True,
            )
        
        # Validate range
        if limit < 1 or limit > 1000:
            logger.error(f"Parameter limit out of range: {limit}")
            return CallToolResult(
                content=[TextContent(
                    type="text",
                    text="Parameter 'limit' must be between 1 and 1000"
                )],
                isError=True,
            )
        
        # Generate sequence
        logger.info(f"Generating triangular number sequence up to {limit}")
        sequence = triangular_numbers(limit=limit)
        
        result_text = (
            f"Triangular number sequence (first {limit} numbers):\n"
            f"{sequence}\n\n"
            f"Last number: T({limit}) = {sequence[-1]}"
        )
        
        logger.info(f"Generated {limit} triangular numbers")
    
    return CallToolResult(
        content=[TextContent(type="text", text=result_text)],
        isError=False,
    )


async def handle_perfect_numbers(arguments: Any) -> CallToolResult:
    """Handle perfect_numbers tool calls."""
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
    
    # Validate parameter type
    if not isinstance(limit, int):
        logger.error(f"Invalid parameter type for limit: {type(limit)}")
        return CallToolResult(
            content=[TextContent(
                type="text",
                text=f"Parameter 'limit' must be an integer, got {type(limit).__name__}"
            )],
            isError=True,
        )
    
    # Validate range
    if limit < 1 or limit > 10000:
        logger.error(f"Parameter limit out of range: {limit}")
        return CallToolResult(
            content=[TextContent(
                type="text",
                text="Parameter 'limit' must be between 1 and 10000"
            )],
            isError=True,
        )
    
    # Find perfect numbers
    logger.info(f"Finding perfect numbers up to {limit}")
    perfect = perfect_numbers(limit)
    
    # Format the result
    if not perfect:
        result_text = f"No perfect numbers found up to {limit}"
    else:
        result_text = f"Perfect numbers up to {limit}:\n{perfect}\n\n"
        result_text += f"Count: {len(perfect)} perfect number(s) found\n\n"
        result_text += "Note: Perfect numbers are extremely rare. "
        result_text += "A perfect number equals the sum of its proper divisors.\n"
        result_text += "Examples: 6 (1+2+3=6), 28 (1+2+4+7+14=28)"
    
    logger.info(f"Found {len(perfect)} perfect numbers up to {limit}")
    
    return CallToolResult(
        content=[TextContent(type="text", text=result_text)],
        isError=False,
    )


async def handle_collatz_sequence(arguments: Any) -> CallToolResult:
    """Handle collatz_sequence tool calls."""
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
    
    # Validate parameter type
    if not isinstance(n, int):
        logger.error(f"Invalid parameter type for n: {type(n)}")
        return CallToolResult(
            content=[TextContent(
                type="text",
                text=f"Parameter 'n' must be an integer, got {type(n).__name__}"
            )],
            isError=True,
        )
    
    # Validate range
    if n < 1 or n > 100000:
        logger.error(f"Parameter n out of range: {n}")
        return CallToolResult(
            content=[TextContent(
                type="text",
                text="Parameter 'n' must be between 1 and 100000"
            )],
            isError=True,
        )
    
    # Generate Collatz sequence
    logger.info(f"Generating Collatz sequence for {n}")
    result = collatz_sequence(n)
    
    # Format the result
    sequence = result['sequence']
    steps = result['steps']
    max_value = result['max_value']
    
    # Show full sequence if short, otherwise show abbreviated
    if len(sequence) <= 20:
        sequence_str = str(sequence)
    else:
        # Show first 10 and last 5 elements
        sequence_str = (
            f"[{', '.join(map(str, sequence[:10]))}, ..., "
            f"{', '.join(map(str, sequence[-5:]))}]"
        )
    
    result_text = (
        f"Collatz sequence starting from {n}:\n"
        f"Sequence ({steps} steps): {sequence_str}\n\n"
        f"Steps to reach 1: {steps}\n"
        f"Maximum value reached: {max_value}\n\n"
        f"Rules: If even, divide by 2; if odd, multiply by 3 and add 1"
    )
    
    logger.info(f"Collatz sequence for {n}: {steps} steps, max value {max_value}")
    
    return CallToolResult(
        content=[TextContent(type="text", text=result_text)],
        isError=False,
    )


async def handle_generate_hash(arguments: Any) -> CallToolResult:
    """Handle generate_hash tool calls."""
    # Extract parameters
    data = arguments.get("data")
    algorithm = arguments.get("algorithm")
    output_format = arguments.get("output_format", "hex")
    
    # Validate required parameters
    if data is None:
        logger.error("Missing required parameter: data")
        return CallToolResult(
            content=[TextContent(
                type="text",
                text="Missing required parameter 'data'"
            )],
            isError=True,
        )
    
    if algorithm is None:
        logger.error("Missing required parameter: algorithm")
        return CallToolResult(
            content=[TextContent(
                type="text",
                text="Missing required parameter 'algorithm'"
            )],
            isError=True,
        )
    
    # Validate parameter types
    if not isinstance(data, str):
        logger.error(f"Invalid parameter type for data: {type(data)}")
        return CallToolResult(
            content=[TextContent(
                type="text",
                text=f"Parameter 'data' must be a string, got {type(data).__name__}"
            )],
            isError=True,
        )
    
    if not isinstance(algorithm, str):
        logger.error(f"Invalid parameter type for algorithm: {type(algorithm)}")
        return CallToolResult(
            content=[TextContent(
                type="text",
                text=f"Parameter 'algorithm' must be a string, got {type(algorithm).__name__}"
            )],
            isError=True,
        )
    
    if not isinstance(output_format, str):
        logger.error(f"Invalid parameter type for output_format: {type(output_format)}")
        return CallToolResult(
            content=[TextContent(
                type="text",
                text=f"Parameter 'output_format' must be a string, got {type(output_format).__name__}"
            )],
            isError=True,
        )
    
    # Generate hash
    try:
        logger.info(f"Generating {algorithm} hash for {len(data)} bytes of data, output format: {output_format}")
        result = generate_hash(data, algorithm, output_format)
        
        # Format the result
        result_text = (
            f"Hash generated successfully:\n\n"
            f"Algorithm: {result['algorithm'].upper()}\n"
            f"Hash ({result['output_format']}): {result['hash']}\n\n"
            f"Input size: {result['input_size']:,} bytes\n"
            f"Hash size: {result['hash_size']} bits\n\n"
            f"Security Note:\n{result['security_note']}"
        )
        
        # Add password warning for weak hashes
        if result['algorithm'] in ['md5', 'sha1']:
            result_text += (
                "\n\n⚠️ IMPORTANT: Do NOT use this hash for passwords or security-critical applications. "
                "Use SHA-256 or SHA-512 with proper salt and key derivation functions (PBKDF2, bcrypt, Argon2)."
            )
        
        logger.info(f"Successfully generated {algorithm} hash: {result['hash'][:16]}...")
        
        return CallToolResult(
            content=[TextContent(type="text", text=result_text)],
            isError=False,
        )
        
    except ValueError as e:
        # Handle validation errors from generate_hash
        logger.error(f"Hash generation error: {e}")
        return CallToolResult(
            content=[TextContent(type="text", text=f"Hash generation error: {str(e)}")],
            isError=True,
        )


async def handle_descriptive_stats(arguments: Any) -> CallToolResult:
    """Handle descriptive_stats tool calls."""
    # Extract and validate parameters
    data = arguments.get("data")
    
    # Validate required parameter
    if data is None:
        logger.error("Missing required parameter: data")
        return CallToolResult(
            content=[TextContent(
                type="text",
                text="Missing required parameter 'data'"
            )],
            isError=True,
        )
    
    # Validate parameter type
    if not isinstance(data, list):
        logger.error(f"Invalid parameter type for data: {type(data)}")
        return CallToolResult(
            content=[TextContent(
                type="text",
                text=f"Parameter 'data' must be an array, got {type(data).__name__}"
            )],
            isError=True,
        )
    
    # Calculate descriptive statistics
    try:
        logger.info(f"Calculating descriptive statistics for {len(data)} values")
        result = descriptive_stats(data)
        
        # Format mode nicely
        if len(result['mode']) == 1:
            mode_str = str(result['mode'][0])
        else:
            mode_str = f"{result['mode']} (multiple modes)"
        
        # Format the result
        result_text = (
            f"Descriptive Statistics:\n\n"
            f"Central Tendency:\n"
            f"  Mean (average):     {result['mean']:.4f}\n"
            f"  Median (middle):    {result['median']:.4f}\n"
            f"  Mode (most frequent): {mode_str}\n\n"
            f"Dispersion:\n"
            f"  Range (max - min):  {result['range']:.4f}\n"
            f"  Variance (σ²):      {result['variance']:.4f}\n"
            f"  Std Dev (σ):        {result['std_dev']:.4f}\n\n"
            f"Data Summary:\n"
            f"  Count:              {result['count']}\n"
            f"  Minimum:            {result['min']:.4f}\n"
            f"  Maximum:            {result['max']:.4f}"
        )
        
        logger.info(f"Calculated statistics: mean={result['mean']:.2f}, median={result['median']:.2f}")
        
        return CallToolResult(
            content=[TextContent(type="text", text=result_text)],
            isError=False,
        )
        
    except ValueError as e:
        logger.error(f"Validation error: {e}")
        return CallToolResult(
            content=[TextContent(type="text", text=f"Validation error: {str(e)}")],
            isError=True,
        )


async def handle_correlation(arguments: Any) -> CallToolResult:
    """Handle correlation tool calls."""
    # Extract and validate parameters
    x = arguments.get("x")
    y = arguments.get("y")
    
    # Validate required parameters
    if x is None:
        logger.error("Missing required parameter: x")
        return CallToolResult(
            content=[TextContent(
                type="text",
                text="Missing required parameter 'x'"
            )],
            isError=True,
        )
    
    if y is None:
        logger.error("Missing required parameter: y")
        return CallToolResult(
            content=[TextContent(
                type="text",
                text="Missing required parameter 'y'"
            )],
            isError=True,
        )
    
    # Validate parameter types
    if not isinstance(x, list):
        logger.error(f"Invalid parameter type for x: {type(x)}")
        return CallToolResult(
            content=[TextContent(
                type="text",
                text=f"Parameter 'x' must be an array, got {type(x).__name__}"
            )],
            isError=True,
        )
    
    if not isinstance(y, list):
        logger.error(f"Invalid parameter type for y: {type(y)}")
        return CallToolResult(
            content=[TextContent(
                type="text",
                text=f"Parameter 'y' must be an array, got {type(y).__name__}"
            )],
            isError=True,
        )
    
    # Calculate correlation
    try:
        logger.info(f"Calculating correlation between datasets of length {len(x)} and {len(y)}")
        result = correlation(x, y)
        
        # Format the result
        r = result['coefficient']
        interpretation = result['interpretation']
        
        result_text = (
            f"Pearson Correlation Analysis:\n\n"
            f"Correlation Coefficient (r): {r:.4f}\n"
            f"Interpretation: {interpretation}\n\n"
            f"What this means:\n"
        )
        
        if abs(r) >= 0.7:
            strength = "strong"
        elif abs(r) >= 0.4:
            strength = "moderate"
        elif abs(r) >= 0.1:
            strength = "weak"
        else:
            strength = "negligible"
        
        if r > 0:
            result_text += f"  • The variables show a {strength} positive linear relationship\n"
            result_text += f"  • As one variable increases, the other tends to increase\n"
        elif r < 0:
            result_text += f"  • The variables show a {strength} negative linear relationship\n"
            result_text += f"  • As one variable increases, the other tends to decrease\n"
        else:
            result_text += f"  • The variables show no linear relationship\n"
        
        result_text += f"\nDataset sizes: x has {len(x)} values, y has {len(y)} values"
        
        logger.info(f"Correlation coefficient: {r:.4f} ({interpretation})")
        
        return CallToolResult(
            content=[TextContent(type="text", text=result_text)],
            isError=False,
        )
        
    except ValueError as e:
        logger.error(f"Validation error: {e}")
        return CallToolResult(
            content=[TextContent(type="text", text=f"Validation error: {str(e)}")],
            isError=True,
        )


async def handle_percentile(arguments: Any) -> CallToolResult:
    """Handle percentile tool calls."""
    # Extract and validate parameters
    data = arguments.get("data")
    p = arguments.get("percentile")
    
    # Validate required parameters
    if data is None:
        logger.error("Missing required parameter: data")
        return CallToolResult(
            content=[TextContent(
                type="text",
                text="Missing required parameter 'data'"
            )],
            isError=True,
        )
    
    if p is None:
        logger.error("Missing required parameter: percentile")
        return CallToolResult(
            content=[TextContent(
                type="text",
                text="Missing required parameter 'percentile'"
            )],
            isError=True,
        )
    
    # Validate parameter types
    if not isinstance(data, list):
        logger.error(f"Invalid parameter type for data: {type(data)}")
        return CallToolResult(
            content=[TextContent(
                type="text",
                text=f"Parameter 'data' must be an array, got {type(data).__name__}"
            )],
            isError=True,
        )
    
    if not isinstance(p, (int, float)):
        logger.error(f"Invalid parameter type for percentile: {type(p)}")
        return CallToolResult(
            content=[TextContent(
                type="text",
                text=f"Parameter 'percentile' must be a number, got {type(p).__name__}"
            )],
            isError=True,
        )
    
    # Calculate percentile
    try:
        logger.info(f"Calculating {p}th percentile for {len(data)} values")
        result = percentile(data, p)
        
        # Format the result with contextual information
        result_text = f"Percentile Calculation:\n\n"
        result_text += f"The {p}th percentile: {result:.4f}\n\n"
        result_text += f"What this means:\n"
        result_text += f"  • {p}% of the data values are below {result:.4f}\n"
        result_text += f"  • {100 - p}% of the data values are above {result:.4f}\n"
        
        # Add special names for common percentiles
        if p == 0:
            result_text += f"\nNote: The 0th percentile is the minimum value"
        elif p == 25:
            result_text += f"\nNote: The 25th percentile is also called Q1 (first quartile)"
        elif p == 50:
            result_text += f"\nNote: The 50th percentile is the median (middle value)"
        elif p == 75:
            result_text += f"\nNote: The 75th percentile is also called Q3 (third quartile)"
        elif p == 100:
            result_text += f"\nNote: The 100th percentile is the maximum value"
        
        result_text += f"\n\nDataset size: {len(data)} values"
        
        logger.info(f"{p}th percentile = {result:.4f}")
        
        return CallToolResult(
            content=[TextContent(type="text", text=result_text)],
            isError=False,
        )
        
    except ValueError as e:
        logger.error(f"Validation error: {e}")
        return CallToolResult(
            content=[TextContent(type="text", text=f"Validation error: {str(e)}")],
            isError=True,
        )


async def handle_outliers(arguments: Any) -> CallToolResult:
    """Handle outliers tool calls."""
    # Extract and validate parameters
    data = arguments.get("data")
    
    # Validate required parameter
    if data is None:
        logger.error("Missing required parameter: data")
        return CallToolResult(
            content=[TextContent(
                type="text",
                text="Missing required parameter 'data'"
            )],
            isError=True,
        )
    
    # Validate parameter type
    if not isinstance(data, list):
        logger.error(f"Invalid parameter type for data: {type(data)}")
        return CallToolResult(
            content=[TextContent(
                type="text",
                text=f"Parameter 'data' must be an array, got {type(data).__name__}"
            )],
            isError=True,
        )
    
    # Detect outliers
    try:
        logger.info(f"Detecting outliers in {len(data)} values using IQR method")
        result = outliers(data)
        
        # Format the result
        result_text = f"Outlier Detection (IQR Method):\n\n"
        
        if result['count'] == 0:
            result_text += f"✓ No outliers detected\n\n"
        else:
            result_text += f"⚠ Found {result['count']} outlier(s):\n"
            for i, (value, index) in enumerate(zip(result['outliers'], result['indices']), 1):
                result_text += f"  {i}. Value {value:.4f} at index {index}\n"
            result_text += "\n"
        
        result_text += f"Statistical Boundaries:\n"
        result_text += f"  Q1 (25th percentile):  {result['q1']:.4f}\n"
        result_text += f"  Q3 (75th percentile):  {result['q3']:.4f}\n"
        result_text += f"  IQR (Q3 - Q1):         {result['iqr']:.4f}\n"
        result_text += f"  Lower Bound:           {result['lower_bound']:.4f}\n"
        result_text += f"  Upper Bound:           {result['upper_bound']:.4f}\n\n"
        
        result_text += f"Method: Values outside [{result['lower_bound']:.4f}, {result['upper_bound']:.4f}] are outliers\n"
        result_text += f"Formula: [Q1 - 1.5×IQR, Q3 + 1.5×IQR]\n\n"
        result_text += f"Dataset size: {len(data)} values"
        
        logger.info(f"Found {result['count']} outliers")
        
        return CallToolResult(
            content=[TextContent(type="text", text=result_text)],
            isError=False,
        )
        
    except ValueError as e:
        logger.error(f"Validation error: {e}")
        return CallToolResult(
            content=[TextContent(type="text", text=f"Validation error: {str(e)}")],
            isError=True,
        )


async def handle_unit_convert(arguments: Any) -> CallToolResult:
    """Handle unit_convert tool calls."""
    # Extract and validate parameters
    value = arguments.get("value")
    from_unit = arguments.get("from_unit")
    to_unit = arguments.get("to_unit")
    
    # Validate required parameters
    if value is None:
        logger.error("Missing required parameter: value")
        return CallToolResult(
            content=[TextContent(
                type="text",
                text="Missing required parameter 'value'"
            )],
            isError=True,
        )
    
    if from_unit is None:
        logger.error("Missing required parameter: from_unit")
        return CallToolResult(
            content=[TextContent(
                type="text",
                text="Missing required parameter 'from_unit'"
            )],
            isError=True,
        )
    
    if to_unit is None:
        logger.error("Missing required parameter: to_unit")
        return CallToolResult(
            content=[TextContent(
                type="text",
                text="Missing required parameter 'to_unit'"
            )],
            isError=True,
        )
    
    # Validate parameter types
    if not isinstance(value, (int, float)):
        logger.error(f"Invalid parameter type for value: {type(value)}")
        return CallToolResult(
            content=[TextContent(
                type="text",
                text=f"Parameter 'value' must be a number, got {type(value).__name__}"
            )],
            isError=True,
        )
    
    if not isinstance(from_unit, str):
        logger.error(f"Invalid parameter type for from_unit: {type(from_unit)}")
        return CallToolResult(
            content=[TextContent(
                type="text",
                text=f"Parameter 'from_unit' must be a string, got {type(from_unit).__name__}"
            )],
            isError=True,
        )
    
    if not isinstance(to_unit, str):
        logger.error(f"Invalid parameter type for to_unit: {type(to_unit)}")
        return CallToolResult(
            content=[TextContent(
                type="text",
                text=f"Parameter 'to_unit' must be a string, got {type(to_unit).__name__}"
            )],
            isError=True,
        )
    
    # Perform unit conversion
    try:
        logger.info(f"Converting {value} {from_unit} to {to_unit}")
        result = unit_convert(value, from_unit, to_unit)
        
        # Format the result
        result_text = (
            f"Unit Conversion:\n\n"
            f"{result['formatted']}\n\n"
            f"Details:\n"
            f"  Category: {result['category'].title()}\n"
            f"  Original: {result['value']} {result['from_unit']}\n"
            f"  Result: {result['result']} {result['to_unit']}\n"
        )
        
        # Add formula/factor information
        if result['formula']:
            if result['category'] == 'temperature':
                result_text += f"  Formula: {result['formula']}\n"
            else:
                result_text += f"  Conversion Factor: {result['formula']}\n"
        
        logger.info(f"Conversion successful: {value} {from_unit} = {result['result']} {to_unit}")
        
        return CallToolResult(
            content=[TextContent(type="text", text=result_text)],
            isError=False,
        )
        
    except ValueError as e:
        # Handle validation errors from unit_convert
        logger.error(f"Unit conversion error: {e}")
        return CallToolResult(
            content=[TextContent(type="text", text=f"Unit conversion error: {str(e)}")],
            isError=True,
        )


async def handle_date_diff(arguments: Any) -> CallToolResult:
    """Handle date_diff tool calls."""
    # Extract and validate parameters
    date1 = arguments.get("date1")
    date2 = arguments.get("date2")
    unit = arguments.get("unit", "all")
    
    # Validate required parameters
    if date1 is None:
        logger.error("Missing required parameter: date1")
        return CallToolResult(
            content=[TextContent(
                type="text",
                text="Missing required parameter 'date1'"
            )],
            isError=True,
        )
    
    if date2 is None:
        logger.error("Missing required parameter: date2")
        return CallToolResult(
            content=[TextContent(
                type="text",
                text="Missing required parameter 'date2'"
            )],
            isError=True,
        )
    
    # Validate parameter types
    if not isinstance(date1, str):
        logger.error(f"Invalid parameter type for date1: {type(date1)}")
        return CallToolResult(
            content=[TextContent(
                type="text",
                text=f"Parameter 'date1' must be a string, got {type(date1).__name__}"
            )],
            isError=True,
        )
    
    if not isinstance(date2, str):
        logger.error(f"Invalid parameter type for date2: {type(date2)}")
        return CallToolResult(
            content=[TextContent(
                type="text",
                text=f"Parameter 'date2' must be a string, got {type(date2).__name__}"
            )],
            isError=True,
        )
    
    if not isinstance(unit, str):
        logger.error(f"Invalid parameter type for unit: {type(unit)}")
        return CallToolResult(
            content=[TextContent(
                type="text",
                text=f"Parameter 'unit' must be a string, got {type(unit).__name__}"
            )],
            isError=True,
        )
    
    # Calculate date difference
    try:
        logger.info(f"Calculating date difference: {date1} to {date2}, unit: {unit}")
        result = date_diff(date1, date2, unit)
        
        # Format the result
        result_text = (
            f"Date Difference:\n\n"
            f"From: {date1}\n"
            f"To: {date2}\n\n"
            f"Result: {result['formatted']}\n"
        )
        
        # Add detailed breakdown
        if unit == "all" or unit == "days":
            result_text += f"\nTotal Days: {result.get('total_days', result.get('days'))}"
        
        if unit == "all":
            result_text += (
                f"\n\nDetailed Breakdown:"
                f"\n  Years: {result['years']}"
                f"\n  Months: {result['months']}"
                f"\n  Days: {result['days']}"
                f"\n  Total Weeks: {result['total_weeks']}"
            )
        elif unit == "weeks":
            result_text += (
                f"\n\nBreakdown:"
                f"\n  Weeks: {result['weeks']}"
                f"\n  Remaining Days: {result['days']}"
                f"\n  Total Days: {result['total_days']}"
            )
        
        logger.info(f"Date difference calculated: {result['formatted']}")
        
        return CallToolResult(
            content=[TextContent(type="text", text=result_text)],
            isError=False,
        )
        
    except ValueError as e:
        logger.error(f"Date calculation error: {e}")
        return CallToolResult(
            content=[TextContent(type="text", text=f"Date calculation error: {str(e)}")],
            isError=True,
        )


async def handle_date_add(arguments: Any) -> CallToolResult:
    """Handle date_add tool calls."""
    # Extract and validate parameters
    date = arguments.get("date")
    amount = arguments.get("amount")
    unit = arguments.get("unit")
    
    # Validate required parameters
    if date is None:
        logger.error("Missing required parameter: date")
        return CallToolResult(
            content=[TextContent(
                type="text",
                text="Missing required parameter 'date'"
            )],
            isError=True,
        )
    
    if amount is None:
        logger.error("Missing required parameter: amount")
        return CallToolResult(
            content=[TextContent(
                type="text",
                text="Missing required parameter 'amount'"
            )],
            isError=True,
        )
    
    if unit is None:
        logger.error("Missing required parameter: unit")
        return CallToolResult(
            content=[TextContent(
                type="text",
                text="Missing required parameter 'unit'"
            )],
            isError=True,
        )
    
    # Validate parameter types
    if not isinstance(date, str):
        logger.error(f"Invalid parameter type for date: {type(date)}")
        return CallToolResult(
            content=[TextContent(
                type="text",
                text=f"Parameter 'date' must be a string, got {type(date).__name__}"
            )],
            isError=True,
        )
    
    if not isinstance(amount, int):
        logger.error(f"Invalid parameter type for amount: {type(amount)}")
        return CallToolResult(
            content=[TextContent(
                type="text",
                text=f"Parameter 'amount' must be an integer, got {type(amount).__name__}"
            )],
            isError=True,
        )
    
    if not isinstance(unit, str):
        logger.error(f"Invalid parameter type for unit: {type(unit)}")
        return CallToolResult(
            content=[TextContent(
                type="text",
                text=f"Parameter 'unit' must be a string, got {type(unit).__name__}"
            )],
            isError=True,
        )
    
    # Add time to date
    try:
        logger.info(f"Adding {amount} {unit} to {date}")
        result = date_add(date, amount, unit)
        
        # Format the result
        operation = "Adding" if amount >= 0 else "Subtracting"
        result_text = (
            f"Date Calculation:\n\n"
            f"{operation} {abs(amount)} {unit} {'to' if amount >= 0 else 'from'} {date}\n\n"
            f"Original Date: {result['original_date']}\n"
            f"New Date: {result['new_date']}\n\n"
            f"Calculation: {result['formatted']}"
        )
        
        logger.info(f"Date calculation: {date} + {amount} {unit} = {result['new_date']}")
        
        return CallToolResult(
            content=[TextContent(type="text", text=result_text)],
            isError=False,
        )
        
    except ValueError as e:
        logger.error(f"Date calculation error: {e}")
        return CallToolResult(
            content=[TextContent(type="text", text=f"Date calculation error: {str(e)}")],
            isError=True,
        )


async def handle_business_days(arguments: Any) -> CallToolResult:
    """Handle business_days tool calls."""
    # Extract and validate parameters
    start_date = arguments.get("start_date")
    end_date = arguments.get("end_date")
    exclude_holidays = arguments.get("exclude_holidays", [])
    
    # Validate required parameters
    if start_date is None:
        logger.error("Missing required parameter: start_date")
        return CallToolResult(
            content=[TextContent(
                type="text",
                text="Missing required parameter 'start_date'"
            )],
            isError=True,
        )
    
    if end_date is None:
        logger.error("Missing required parameter: end_date")
        return CallToolResult(
            content=[TextContent(
                type="text",
                text="Missing required parameter 'end_date'"
            )],
            isError=True,
        )
    
    # Validate parameter types
    if not isinstance(start_date, str):
        logger.error(f"Invalid parameter type for start_date: {type(start_date)}")
        return CallToolResult(
            content=[TextContent(
                type="text",
                text=f"Parameter 'start_date' must be a string, got {type(start_date).__name__}"
            )],
            isError=True,
        )
    
    if not isinstance(end_date, str):
        logger.error(f"Invalid parameter type for end_date: {type(end_date)}")
        return CallToolResult(
            content=[TextContent(
                type="text",
                text=f"Parameter 'end_date' must be a string, got {type(end_date).__name__}"
            )],
            isError=True,
        )
    
    if not isinstance(exclude_holidays, list):
        logger.error(f"Invalid parameter type for exclude_holidays: {type(exclude_holidays)}")
        return CallToolResult(
            content=[TextContent(
                type="text",
                text=f"Parameter 'exclude_holidays' must be an array, got {type(exclude_holidays).__name__}"
            )],
            isError=True,
        )
    
    # Calculate business days
    try:
        logger.info(f"Calculating business days from {start_date} to {end_date}")
        result = business_days(start_date, end_date, exclude_holidays if exclude_holidays else None)
        
        # Format the result
        result_text = (
            f"Business Days Calculation:\n\n"
            f"Period: {result['start_date']} to {result['end_date']}\n\n"
            f"Business Days: {result['business_days']}\n"
            f"Total Calendar Days: {result['total_days']}\n"
            f"Weekend Days: {result['weekend_days']}\n"
        )
        
        if result['holidays_excluded'] > 0:
            result_text += f"Holidays Excluded: {result['holidays_excluded']}\n"
        
        result_text += f"\n{result['formatted']}"
        
        if exclude_holidays and len(exclude_holidays) > 0:
            result_text += f"\n\nNote: Excluded {len(exclude_holidays)} custom holiday(s)"
        
        logger.info(f"Business days: {result['business_days']} between {start_date} and {end_date}")
        
        return CallToolResult(
            content=[TextContent(type="text", text=result_text)],
            isError=False,
        )
        
    except ValueError as e:
        logger.error(f"Business days calculation error: {e}")
        return CallToolResult(
            content=[TextContent(type="text", text=f"Business days calculation error: {str(e)}")],
            isError=True,
        )


async def handle_age_calculator(arguments: Any) -> CallToolResult:
    """Handle age_calculator tool calls."""
    # Extract and validate parameters
    birthdate = arguments.get("birthdate")
    reference_date = arguments.get("reference_date")
    
    # Validate required parameter
    if birthdate is None:
        logger.error("Missing required parameter: birthdate")
        return CallToolResult(
            content=[TextContent(
                type="text",
                text="Missing required parameter 'birthdate'"
            )],
            isError=True,
        )
    
    # Validate parameter types
    if not isinstance(birthdate, str):
        logger.error(f"Invalid parameter type for birthdate: {type(birthdate)}")
        return CallToolResult(
            content=[TextContent(
                type="text",
                text=f"Parameter 'birthdate' must be a string, got {type(birthdate).__name__}"
            )],
            isError=True,
        )
    
    if reference_date is not None and not isinstance(reference_date, str):
        logger.error(f"Invalid parameter type for reference_date: {type(reference_date)}")
        return CallToolResult(
            content=[TextContent(
                type="text",
                text=f"Parameter 'reference_date' must be a string, got {type(reference_date).__name__}"
            )],
            isError=True,
        )
    
    # Calculate age
    try:
        logger.info(f"Calculating age from birthdate: {birthdate}")
        result = age_calculator(birthdate, reference_date)
        
        # Format the result
        result_text = (
            f"Age Calculation:\n\n"
            f"Birthdate: {result['birthdate']}\n"
            f"Reference Date: {result['reference_date']}\n\n"
            f"Age: {result['formatted']}\n\n"
            f"Detailed Breakdown:"
            f"\n  Years: {result['years']}"
            f"\n  Months: {result['months']}"
            f"\n  Days: {result['days']}"
            f"\n  Total Days Lived: {result['total_days']:,}"
        )
        
        logger.info(f"Age calculated: {result['formatted']}")
        
        return CallToolResult(
            content=[TextContent(type="text", text=result_text)],
            isError=False,
        )
        
    except ValueError as e:
        logger.error(f"Age calculation error: {e}")
        return CallToolResult(
            content=[TextContent(type="text", text=f"Age calculation error: {str(e)}")],
            isError=True,
        )


async def handle_day_of_week(arguments: Any) -> CallToolResult:
    """Handle day_of_week tool calls."""
    # Extract and validate parameters
    date = arguments.get("date")
    
    # Validate required parameter
    if date is None:
        logger.error("Missing required parameter: date")
        return CallToolResult(
            content=[TextContent(
                type="text",
                text="Missing required parameter 'date'"
            )],
            isError=True,
        )
    
    # Validate parameter type
    if not isinstance(date, str):
        logger.error(f"Invalid parameter type for date: {type(date)}")
        return CallToolResult(
            content=[TextContent(
                type="text",
                text=f"Parameter 'date' must be a string, got {type(date).__name__}"
            )],
            isError=True,
        )
    
    # Get day of week information
    try:
        logger.info(f"Getting day of week for: {date}")
        result = day_of_week(date)
        
        # Format the result
        weekend_note = " (Weekend)" if result['is_weekend'] else " (Weekday)"
        
        result_text = (
            f"Day of Week Information:\n\n"
            f"Date: {result['date']}\n"
            f"Day: {result['day_name']}{weekend_note}\n\n"
            f"Calendar Information:"
            f"\n  ISO Week Number: {result['week_number']}"
            f"\n  Day of Year: {result['day_of_year']}"
            f"\n  ISO Year: {result['iso_year']}"
            f"\n  Day Number (Mon=0): {result['day_number']}"
            f"\n  ISO Weekday (Mon=1): {result['iso_weekday']}"
            f"\n\nSummary: {result['formatted']}"
        )
        
        logger.info(f"Day of week: {result['day_name']}")
        
        return CallToolResult(
            content=[TextContent(type="text", text=result_text)],
            isError=False,
        )
        
    except ValueError as e:
        logger.error(f"Day of week calculation error: {e}")
        return CallToolResult(
            content=[TextContent(type="text", text=f"Day of week calculation error: {str(e)}")],
            isError=True,
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
