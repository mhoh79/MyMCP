"""
Main MCP server implementation for Fibonacci calculations.
This server exposes a tool that calculates Fibonacci numbers.
"""

import asyncio
import base64
import hashlib
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
