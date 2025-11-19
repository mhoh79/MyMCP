# Add Number Theory Tools to MCP Server

## Overview
Implement fundamental number theory operations as MCP tools.

## Tools to Implement

### 1. gcd (Greatest Common Divisor)
Find the greatest common divisor of two or more numbers.
- **Input**: `numbers` (array of integers, 2-10 numbers)
- **Output**: GCD value
- **Algorithm**: Euclidean algorithm
- **Example**: gcd(48, 18) = 6

### 2. lcm (Least Common Multiple)
Find the least common multiple of two or more numbers.
- **Input**: `numbers` (array of integers, 2-10 numbers)
- **Output**: LCM value
- **Algorithm**: LCM(a,b) = (a × b) / GCD(a,b)
- **Example**: lcm(12, 18) = 36

### 3. factorial
Calculate factorial of a number.
- **Input**: `n` (integer, 0-170)
- **Output**: n! value
- **Note**: Limit to 170 to avoid overflow
- **Example**: factorial(5) = 120

### 4. combinations
Calculate combinations (nCr) - ways to choose r items from n.
- **Input**: `n` (integer, 0-1000), `r` (integer, 0-n)
- **Output**: nCr value
- **Formula**: n! / (r! × (n-r)!)
- **Example**: C(5,2) = 10

### 5. permutations
Calculate permutations (nPr) - ordered ways to choose r items from n.
- **Input**: `n` (integer, 0-1000), `r` (integer, 0-n)
- **Output**: nPr value
- **Formula**: n! / (n-r)!
- **Example**: P(5,2) = 20

## Implementation Requirements
- Implement efficient algorithms (use recursion with memoization where appropriate)
- Handle edge cases (n=0, r=0, r>n)
- Add overflow protection for large numbers
- Include comprehensive docstrings
- Update list_tools() and call_tool() handlers
- Add detailed logging

## Acceptance Criteria
- [ ] All 5 number theory tools implemented
- [ ] Edge cases handled properly
- [ ] Full code annotations
- [ ] Input validation with clear error messages
- [ ] Works correctly in Claude Desktop
- [ ] README documentation updated

## Example Usage
```
User: "What's the GCD of 48 and 18?"
Tool: gcd with numbers=[48, 18]
Result: "The greatest common divisor is 6"

User: "How many ways can I choose 3 items from 10?"
Tool: combinations with n=10, r=3
Result: "C(10,3) = 120 combinations"
```

## References
- [Euclidean Algorithm](https://en.wikipedia.org/wiki/Euclidean_algorithm)
- [Factorial](https://en.wikipedia.org/wiki/Factorial)
- [Combinations and Permutations](https://en.wikipedia.org/wiki/Combination)

## Labels
enhancement, math-tools
