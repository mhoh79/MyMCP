# Add Sequence Generator Tools to MCP Server

## Overview
Implement mathematical sequence generators beyond Fibonacci.

## Tools to Implement

### 1. pascal_triangle
Generate Pascal's triangle up to n rows.
- **Input**: `rows` (integer, 1-30)
- **Output**: 2D array representing Pascal's triangle
- **Pattern**: Each number is sum of two numbers above it
- **Example**: Row 5: [1, 4, 6, 4, 1]

### 2. triangular_numbers
Generate or calculate triangular numbers.
- **Input**: `n` (integer, 1-1000) OR `limit` for sequence
- **Output**: nth triangular number or sequence
- **Formula**: T(n) = n × (n + 1) / 2
- **Example**: T(5) = 15, sequence: [1, 3, 6, 10, 15]

### 3. perfect_numbers
Find perfect numbers up to a limit.
- **Input**: `limit` (integer, up to 10000)
- **Output**: List of perfect numbers
- **Definition**: Number equals sum of its proper divisors
- **Example**: 6 (1+2+3=6), 28 (1+2+4+7+14=28)
- **Note**: Very rare, only a few below 10000

### 4. collatz_sequence
Generate the Collatz sequence for a number.
- **Input**: `n` (integer, 1-100000)
- **Output**: Complete sequence until reaching 1
- **Rules**: If even, divide by 2; if odd, multiply by 3 and add 1
- **Example**: 13 → [13, 40, 20, 10, 5, 16, 8, 4, 2, 1]

## Implementation Requirements
- Optimize for performance (memoization for Pascal's triangle)
- Set reasonable limits to prevent long computations
- Include iteration count/steps information where relevant
- Add comprehensive docstrings with mathematical background
- Update list_tools() and call_tool() handlers
- Include visualization suggestions in output

## Acceptance Criteria
- [ ] All 4 sequence generator tools implemented
- [ ] Efficient algorithms with appropriate limits
- [ ] Full code annotations explaining the mathematics
- [ ] Input validation with clear error messages
- [ ] Works correctly in Claude Desktop
- [ ] README documentation updated with examples

## Example Usage
```
User: "Generate the first 6 rows of Pascal's triangle"
Tool: pascal_triangle with rows=6
Result: [[1], [1,1], [1,2,1], [1,3,3,1], [1,4,6,4,1], [1,5,10,10,5,1]]

User: "What's the Collatz sequence for 27?"
Tool: collatz_sequence with n=27
Result: "Sequence (111 steps): [27, 82, 41, 124, ... , 2, 1]"
```

## References
- [Pascal's Triangle](https://en.wikipedia.org/wiki/Pascal%27s_triangle)
- [Triangular Number](https://en.wikipedia.org/wiki/Triangular_number)
- [Perfect Number](https://en.wikipedia.org/wiki/Perfect_number)
- [Collatz Conjecture](https://en.wikipedia.org/wiki/Collatz_conjecture)

## Labels
enhancement, math-tools, sequences
