# Add Prime Number Tools to MCP Server

## Overview
Add prime number calculation tools to the Fibonacci MCP server.

## Tools to Implement

### 1. is_prime
Check if a number is prime.
- **Input**: `n` (integer, 2-1000000)
- **Output**: Boolean with explanation
- **Algorithm**: Trial division with optimization (check up to sqrt(n))

### 2. generate_primes
Generate all prime numbers up to n.
- **Input**: `limit` (integer, 2-10000)
- **Output**: List of prime numbers
- **Algorithm**: Sieve of Eratosthenes

### 3. nth_prime
Find the nth prime number.
- **Input**: `n` (integer, 1-10000)
- **Output**: The nth prime number
- **Algorithm**: Generate primes until we reach the nth one

### 4. prime_factorization
Find prime factors of a number.
- **Input**: `n` (integer, 2-1000000)
- **Output**: List of prime factors with exponents
- **Example**: 24 → [[2, 3], [3, 1]] (2³ × 3¹)

## Implementation Requirements
- Add functions to server.py after existing Fibonacci functions
- Include comprehensive docstrings with examples
- Add input validation and error handling
- Update list_tools() to register new tools
- Add cases in call_tool() handler
- Include logging for all operations
- Add unit tests (optional but recommended)

## Acceptance Criteria
- [ ] All 4 prime number tools implemented
- [ ] Full code annotations explaining algorithms
- [ ] Input validation with clear error messages
- [ ] Proper MCP response formatting
- [ ] Tools appear and work in Claude Desktop
- [ ] README updated with new tool documentation

## Example Usage
```
User: "Is 97 a prime number?"
Tool: is_prime with n=97
Result: "Yes, 97 is a prime number."

User: "Show me all prime numbers up to 50"
Tool: generate_primes with limit=50
Result: [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47]
```

## References
- [Sieve of Eratosthenes](https://en.wikipedia.org/wiki/Sieve_of_Eratosthenes)
- [Primality Test](https://en.wikipedia.org/wiki/Primality_test)

## Labels
enhancement, good first issue, math-tools
