# Mathematical Tools MCP Server

A complete Model Context Protocol (MCP) server implementation in Python that provides mathematical calculation tools for AI assistants like Claude Desktop and other MCP-compatible clients.

## 🌟 Features

- **Fibonacci Calculations**: Calculate individual Fibonacci numbers or generate complete sequences
- **Prime Number Tools**: Check primality, generate primes, find nth prime, and factorize numbers
- **Number Theory Tools**: GCD, LCM, factorial, combinations, and permutations
- **Sequence Generators**: Pascal's triangle, triangular numbers, perfect numbers, and Collatz sequences
- **Cryptographic Hash Generator**: Generate MD5, SHA-1, SHA-256, SHA-512, and BLAKE2b hashes with security notes
- **High Performance**: Optimized algorithms (Euclidean algorithm, Sieve of Eratosthenes, efficient combinatorics, memoization)
- **Robust Validation**: Comprehensive error handling and input validation with appropriate ranges
- **Production Ready**: Full logging, proper error messages, and graceful shutdown
- **Well Documented**: Extensive code annotations and inline documentation for learning
- **Type Safe**: Complete type hints throughout the codebase
- **MCP Compliant**: Implements the Model Context Protocol specification using stdio transport

## 📋 Prerequisites

- Python 3.10 or higher
- pip (Python package manager)
- Virtual environment (recommended)

## 🚀 Quick Start

### 1. Installation

Clone the repository and set up the environment:

```bash
# Navigate to the project directory
cd MyMCP

# Create a virtual environment
python -m venv venv

# Activate the virtual environment
# Windows:
.\venv\Scripts\Activate.ps1
# macOS/Linux:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Test the Server

Run the server directly to verify it works:

```bash
python src/fibonacci_server/server.py
```

The server will start and wait for MCP protocol messages on stdin. Press `Ctrl+C` to stop.

### 3. Configure with Claude Desktop

Add the server to Claude Desktop's configuration file:

**Windows:** `%APPDATA%\Claude\claude_desktop_config.json`
**macOS:** `~/Library/Application Support/Claude/claude_desktop_config.json`
**Linux:** `~/.config/Claude/claude_desktop_config.json`

```json
{
  "mcpServers": {
    "fibonacci": {
      "command": "c:/Users/YOUR_USERNAME/path/to/MyMCP/venv/Scripts/python.exe",
      "args": [
        "c:/Users/YOUR_USERNAME/path/to/MyMCP/src/fibonacci_server/server.py"
      ]
    }
  }
}
```

**Important:** 
- Replace `YOUR_USERNAME` and the path with your actual paths
- Use forward slashes (`/`) or escaped backslashes (`\\`) in paths
- Use absolute paths for both the Python executable and server script

### 4. Restart Claude Desktop

Completely quit Claude Desktop and restart it. The Fibonacci tool should now be available.

## 💡 Usage Examples

Once configured, you can interact with the server through Claude:

### Fibonacci Calculations
```
"What is the 50th Fibonacci number?"
"Calculate the Fibonacci number at position 100"
"Show me the first 20 Fibonacci numbers"
"Generate a Fibonacci sequence with 15 numbers"
```

### Prime Number Tools
```
"Is 97 a prime number?"
"Show me all prime numbers up to 50"
"What is the 100th prime number?"
"What are the prime factors of 24?"
"Find the prime factorization of 360"
```

### Number Theory Tools
```
"What's the GCD of 48 and 18?"
"Find the LCM of 12, 18, and 24"
"Calculate 10 factorial"
"How many ways can I choose 3 items from 10?"
"How many ways can I arrange 3 items from 5?"
```

### Sequence Generators
```
"Generate the first 6 rows of Pascal's triangle"
"What's the 10th triangular number?"
"Show me the first 10 triangular numbers"
"Find all perfect numbers up to 10000"
"What's the Collatz sequence for 27?"
```

### Cryptographic Hash Generator
```
"Generate an SHA-256 hash of 'Hello World'"
"What's the MD5 hash of 'password123'?"
"Create a SHA-512 hash of my data"
"Generate a BLAKE2b hash in base64 format"
"Hash this text using SHA-256"
```

**⚠️ Security Warning**: MD5 and SHA-1 are cryptographically broken. Use SHA-256 or higher for security purposes. Never use plain hashes for passwords - always use salt and key derivation functions (PBKDF2, bcrypt, Argon2).

## 🔧 Tool Specifications

### Tool: `calculate_fibonacci`

**Parameters:**
- `n` (integer, required): Position in the Fibonacci sequence (0-1000)
  - `0` returns `0`
  - `1` returns `1`
  - Higher values return the nth Fibonacci number
- `return_sequence` (boolean, optional, default: false): 
  - `false`: Returns only the nth Fibonacci number
  - `true`: Returns the complete sequence from position 0 to n

**Response Format:**
- Single number: `"Fibonacci number at position {n}: {result}"`
- Sequence: `"Fibonacci sequence (first {n} numbers): [0, 1, 1, 2, 3, ...]"`

**Error Handling:**
- Invalid input types
- Out of range values (< 0 or > 1000)
- Negative numbers
- Missing required parameters

### Tool: `is_prime`

**Parameters:**
- `n` (integer, required): Number to check for primality (2-1000000)

**Response Format:**
- `"Yes, {n} is a prime number."` or `"No, {n} is not a prime number."`

**Algorithm:** Trial division with optimization (checks up to √n)

**Error Handling:**
- Invalid input types
- Out of range values (< 2 or > 1000000)
- Missing required parameters

### Tool: `generate_primes`

**Parameters:**
- `limit` (integer, required): Upper bound for prime generation (2-10000)

**Response Format:**
- List of all prime numbers from 2 to limit
- Includes count of primes found

**Algorithm:** Sieve of Eratosthenes (O(n log log n) time complexity)

**Error Handling:**
- Invalid input types
- Out of range values (< 2 or > 10000)
- Missing required parameters

### Tool: `nth_prime`

**Parameters:**
- `n` (integer, required): Position of the desired prime (1-10000, 1-indexed)

**Response Format:**
- `"The {n}th prime number is: {result}"`

**Examples:**
- 1st prime = 2
- 5th prime = 11
- 10th prime = 29
- 100th prime = 541

**Algorithm:** Generates primes using Sieve of Eratosthenes with estimated upper bound

**Error Handling:**
- Invalid input types
- Out of range values (< 1 or > 10000)
- Missing required parameters

### Tool: `prime_factorization`

**Parameters:**
- `n` (integer, required): Number to factorize (2-1000000)

**Response Format:**
- List of [prime, exponent] pairs
- Example: `24 = 2³ × 3¹` returns `[[2, 3], [3, 1]]`

**Algorithm:** Trial division checking divisors up to √n

**Error Handling:**
- Invalid input types
- Out of range values (< 2 or > 1000000)
- Missing required parameters

### Tool: `gcd`

**Parameters:**
- `numbers` (array of integers, required): 2-10 numbers to find GCD of

**Response Format:**
- Single integer representing the GCD
- Example: `GCD(48, 18) = 6`

**Algorithm:** Euclidean algorithm applied pairwise

**Error Handling:**
- Invalid input types
- Array length not between 2 and 10
- Non-integer elements in array
- Missing required parameters

### Tool: `lcm`

**Parameters:**
- `numbers` (array of integers, required): 2-10 numbers to find LCM of

**Response Format:**
- Single integer representing the LCM
- Example: `LCM(12, 18) = 36`

**Algorithm:** Uses GCD to compute LCM(a,b) = (a × b) / GCD(a,b)

**Error Handling:**
- Invalid input types
- Array length not between 2 and 10
- Non-integer elements in array
- Zero values in array (LCM undefined)
- Missing required parameters

### Tool: `factorial`

**Parameters:**
- `n` (integer, required): Number to calculate factorial of (0-170)

**Response Format:**
- Single integer representing n!
- Example: `5! = 120`

**Algorithm:** Iterative multiplication for efficiency

**Error Handling:**
- Invalid input types
- Out of range values (< 0 or > 170)
- Missing required parameters

**Note:** Limited to 170 to avoid overflow issues

### Tool: `combinations`

**Parameters:**
- `n` (integer, required): Total number of items (0-1000)
- `r` (integer, required): Number of items to choose (0 to n)

**Response Format:**
- Single integer representing C(n,r)
- Example: `C(5,2) = 10`

**Formula:** C(n,r) = n! / (r! × (n-r)!)

**Algorithm:** Uses multiplicative formula to avoid large factorials

**Error Handling:**
- Invalid input types
- Out of range values (n < 0, n > 1000, r < 0)
- r > n
- Missing required parameters

### Tool: `permutations`

**Parameters:**
- `n` (integer, required): Total number of items (0-1000)
- `r` (integer, required): Number of items to arrange (0 to n)

**Response Format:**
- Single integer representing P(n,r)
- Example: `P(5,2) = 20`

**Formula:** P(n,r) = n! / (n-r)!

**Algorithm:** Direct multiplication of consecutive integers

**Error Handling:**
- Invalid input types
- Out of range values (n < 0, n > 1000, r < 0)
- r > n
- Missing required parameters

### Tool: `pascal_triangle`

**Parameters:**
- `rows` (integer, required): Number of rows to generate (1-30)

**Response Format:**
- 2D array representing Pascal's triangle
- Each inner list is a row in the triangle
- Example: Row 5: `[1, 4, 6, 4, 1]`

**Mathematical Background:**
- Each number is the sum of the two numbers above it
- Entry in row n, position k is the binomial coefficient C(n,k)
- Used in probability, combinatorics, and algebra

**Algorithm:** Iterative generation with memoization for performance

**Error Handling:**
- Invalid input types
- Out of range values (< 1 or > 30)
- Missing required parameters

### Tool: `triangular_numbers`

**Parameters:**
- `n` (integer, optional): Position of triangular number to calculate (1-1000)
- `limit` (integer, optional): Generate sequence of first 'limit' numbers (1-1000)
- Note: Exactly one of `n` or `limit` must be provided

**Response Format:**
- If `n` provided: Single integer (the nth triangular number)
- If `limit` provided: List of triangular numbers
- Example: T(5) = 15, sequence: [1, 3, 6, 10, 15]

**Formula:** T(n) = n × (n + 1) / 2

**Mathematical Background:**
- Represents dots that can form an equilateral triangle
- Sum of first n natural numbers: 1 + 2 + 3 + ... + n

**Algorithm:** Direct formula for single value, iterative for sequence

**Error Handling:**
- Invalid input types
- Out of range values (< 1 or > 1000)
- Neither or both parameters provided
- Missing required parameters

### Tool: `perfect_numbers`

**Parameters:**
- `limit` (integer, required): Upper bound for searching (1-10000)

**Response Format:**
- List of all perfect numbers up to limit
- Example: [6, 28, 496, 8128]

**Mathematical Background:**
- A perfect number equals the sum of its proper divisors
- Examples: 6 (1+2+3=6), 28 (1+2+4+7+14=28)
- Extremely rare - only 4 exist below 10000
- Related to Mersenne primes via Euclid-Euler theorem

**Algorithm:** Check each number by finding divisors up to sqrt(n)

**Error Handling:**
- Invalid input types
- Out of range values (< 1 or > 10000)
- Missing required parameters

### Tool: `collatz_sequence`

**Parameters:**
- `n` (integer, required): Starting number for the sequence (1-100000)

**Response Format:**
- Dictionary containing:
  - `sequence`: List of numbers in the sequence
  - `steps`: Number of steps to reach 1
  - `max_value`: Maximum value reached
- Example: 13 → [13, 40, 20, 10, 5, 16, 8, 4, 2, 1] (9 steps)

**Rules:**
- If even: divide by 2
- If odd: multiply by 3 and add 1
- Continue until reaching 1

**Mathematical Background:**
- Also known as the 3n+1 problem or Ulam conjecture
- One of the most famous unsolved problems in mathematics
- Tested for all numbers up to 2^68, but unproven in general

**Algorithm:** Iterative application of rules, tracking sequence and statistics

**Error Handling:**
- Invalid input types
- Out of range values (< 1 or > 100000)
- Missing required parameters

### Tool: `generate_hash`

**Parameters:**
- `data` (string, required): Input text or data to hash (max 1MB)
- `algorithm` (string, required): Hash algorithm to use
  - `md5`: 128-bit, fast but INSECURE (use only for checksums)
  - `sha1`: 160-bit, DEPRECATED (avoid for new applications)
  - `sha256`: 256-bit, SECURE and recommended
  - `sha512`: 512-bit, HIGH security
  - `blake2b`: 512-bit, MODERN and fast
- `output_format` (string, optional, default: "hex"):
  - `hex`: Hexadecimal output
  - `base64`: Base64-encoded output

**Response Format:**
- Hash value in requested format
- Algorithm and format used
- Input size in bytes
- Hash size in bits
- Security note with recommendations

**Examples:**
```
"Hello World" + SHA-256 → "a591a6d40bf420404a011733cfb7b190d62c65bf0bcda32b57b277d9ad9f146e"
"password123" + MD5 → "482c811da5d5b4bc6d497ffa98491e38"
"test" + SHA-256 + base64 → "n4bQgYhMfWWaL+qgxVrQFaO/TxsrC4Is0V1sFbDwCgg="
```

**Security Notes:**
- **MD5**: ⚠️ BROKEN - collision attacks are practical. Use ONLY for non-security purposes.
- **SHA-1**: ⚠️ DEPRECATED - collision attacks demonstrated. Avoid for new applications.
- **SHA-256**: ✓ SECURE - recommended for most security applications.
- **SHA-512**: ✓ HIGH SECURITY - enhanced security with larger output.
- **BLAKE2b**: ✓ MODERN - fast and secure, comparable to SHA-3.

**Password Hashing:**
- ⚠️ NEVER use plain hashes for passwords
- Always use salt (unique random value per password)
- Use key derivation functions: PBKDF2, bcrypt, or Argon2
- Minimum: SHA-256 with salt, better: use bcrypt/Argon2

**Algorithm Characteristics:**
- MD5: 32 hex chars (128 bits), very fast, INSECURE
- SHA-1: 40 hex chars (160 bits), fast, DEPRECATED
- SHA-256: 64 hex chars (256 bits), medium speed, SECURE
- SHA-512: 128 hex chars (512 bits), slower, HIGH security
- BLAKE2b: 128 hex chars (512 bits), very fast, MODERN

**Use Cases:**
- MD5: File checksums, cache keys (non-security only)
- SHA-1: Git commits (legacy), HMAC in legacy systems
- SHA-256: Digital signatures, SSL/TLS, Bitcoin, file integrity
- SHA-512: High-security applications, hash truncation needs
- BLAKE2b: Modern file integrity, cryptocurrencies (Zcash)

**Error Handling:**
- Invalid input types
- Unsupported algorithm
- Unsupported output format
- Data size exceeds 1MB limit
- Missing required parameters

## 📁 Project Structure

```
MyMCP/
├── src/
│   └── fibonacci_server/
│       ├── __init__.py          # Package initialization
│       └── server.py            # Main MCP server implementation
├── venv/                        # Virtual environment (created during setup)
├── .env.example                 # Environment variable template
├── .gitignore                   # Git ignore patterns
├── requirements.txt             # Python dependencies
├── mcp-config.json             # MCP Inspector config (for testing)
└── README.md                    # This file
```

## 🧪 Testing

### Using MCP Inspector

The MCP Inspector is a web-based tool for testing MCP servers:

```bash
npx @modelcontextprotocol/inspector c:/path/to/venv/Scripts/python.exe c:/path/to/src/fibonacci_server/server.py
```

This will:
1. Start the MCP Inspector web interface
2. Launch your Fibonacci server
3. Open a browser at `http://localhost:6274`
4. Allow you to test tool calls interactively

### Manual Testing

You can also test the calculations directly in Python:

```python
from src.fibonacci_server.server import (
    calculate_fibonacci, 
    calculate_fibonacci_sequence,
    is_prime,
    generate_primes,
    nth_prime,
    prime_factorization,
    gcd,
    lcm,
    factorial,
    combinations,
    permutations,
    pascal_triangle,
    triangular_numbers,
    perfect_numbers,
    collatz_sequence,
    generate_hash
)

# Fibonacci calculations
print(calculate_fibonacci(10))  # Output: 55
print(calculate_fibonacci_sequence(10))  # Output: [0, 1, 1, 2, 3, 5, 8, 13, 21, 34]

# Prime number tools
print(is_prime(97))  # Output: True
print(generate_primes(50))  # Output: [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47]
print(nth_prime(10))  # Output: 29
print(prime_factorization(24))  # Output: [[2, 3], [3, 1]]

# Number theory tools
print(gcd([48, 18]))  # Output: 6
print(lcm([12, 18]))  # Output: 36
print(factorial(5))  # Output: 120
print(combinations(5, 2))  # Output: 10
print(permutations(5, 2))  # Output: 20

# Sequence generators
print(pascal_triangle(5))  # Output: [[1], [1, 1], [1, 2, 1], [1, 3, 3, 1], [1, 4, 6, 4, 1]]
print(triangular_numbers(n=5))  # Output: 15
print(triangular_numbers(limit=5))  # Output: [1, 3, 6, 10, 15]
print(perfect_numbers(100))  # Output: [6, 28]
print(collatz_sequence(13))  # Output: {'sequence': [13, 40, 20, 10, 5, 16, 8, 4, 2, 1], 'steps': 9, 'max_value': 40}

# Cryptographic hash generator
print(generate_hash("Hello World", "sha256", "hex"))  # Output: {'hash': 'a591a6d40bf420404a011733cfb7b190d62c65bf0bcda32b57b277d9ad9f146e', ...}
print(generate_hash("password123", "md5", "hex"))  # Output: {'hash': '482c811da5d5b4bc6d497ffa98491e38', ...}
print(generate_hash("test", "sha512", "base64"))  # Output: Base64-encoded SHA-512 hash
```

## 🏗️ Architecture

### MCP Protocol Flow

1. **Initialization**: Claude Desktop launches the server via the configured command
2. **Handshake**: Server and client negotiate capabilities via JSON-RPC 2.0
3. **Discovery**: Client calls `list_tools()` to discover available tools
4. **Execution**: Client invokes `call_tool()` with parameters
5. **Response**: Server returns results in MCP format
6. **Shutdown**: Server gracefully closes when client disconnects

### Key Components

- **Transport Layer**: stdio (standard input/output)
- **Protocol**: JSON-RPC 2.0 over MCP
- **Async Runtime**: Python asyncio for non-blocking I/O
- **Logging**: Structured logging to stderr (stdout reserved for protocol)
- **Error Handling**: Try-catch blocks with detailed error messages

## 🔍 Debugging

### Check Server Logs

Server logs are written to stderr. When running via Claude Desktop, check:

**Windows:** `%APPDATA%\Claude\logs\main.log`

Look for messages containing "fibonacci", "prime", or "mcp" to see connection attempts and errors.

### Common Issues

1. **Server not appearing in Claude:**
   - Verify the config file path and JSON syntax
   - Check that Python executable path is correct
   - Ensure virtual environment is activated when testing manually
   - Completely restart Claude Desktop (not just close the window)

2. **Import errors:**
   - Verify all dependencies are installed: `pip install -r requirements.txt`
   - Check that you're using the virtual environment's Python

3. **Path issues on Windows:**
   - Use forward slashes or escaped backslashes in JSON
   - Use absolute paths, not relative paths
   - Avoid spaces in paths if possible

## 📚 Dependencies

- `mcp>=1.0.0` - Model Context Protocol SDK
- `pydantic>=2.0.0` - Data validation and settings management
- `python-dotenv>=1.0.0` - Environment variable management

All dependencies are automatically installed via `pip install -r requirements.txt`.

## 🤝 Contributing

This is a learning project demonstrating MCP server implementation. Feel free to:

- Add new mathematical tools (number theory, combinatorics, etc.)
- Improve error handling
- Add unit tests
- Enhance documentation
- Create additional example servers
- Optimize algorithms

## 📖 Learning Resources

- [Model Context Protocol Documentation](https://modelcontextprotocol.io/)
- [MCP Python SDK](https://github.com/modelcontextprotocol/python-sdk)
- [MCP Specification](https://spec.modelcontextprotocol.io/)

## ✅ Verification

Your server is working correctly if you can:

1. ✅ Start the server manually without errors
2. ✅ See the server in Claude Desktop's tool list
3. ✅ Ask Claude to calculate Fibonacci numbers and receive correct results
4. ✅ Use prime number tools and verify the mathematical correctness
5. ✅ Use number theory tools and verify the mathematical correctness
6. ✅ Use sequence generators and verify the mathematical correctness

Example verifications:
- Request: "Calculate the first 10 Fibonacci numbers"
- Expected: `[0, 1, 1, 2, 3, 5, 8, 13, 21, 34]`

- Request: "Show me all prime numbers up to 50"
- Expected: `[2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47]`

- Request: "Is 97 a prime number?"
- Expected: "Yes, 97 is a prime number."

- Request: "What are the prime factors of 24?"
- Expected: `[[2, 3], [3, 1]]` (2³ × 3¹)

- Request: "What's the GCD of 48 and 18?"
- Expected: `6`

- Request: "Find the LCM of 12 and 18"
- Expected: `36`

- Request: "Calculate 5 factorial"
- Expected: `120`

- Request: "How many ways can I choose 3 items from 10?"
- Expected: `120` (C(10,3))

- Request: "How many ways can I arrange 2 items from 5?"
- Expected: `20` (P(5,2))

- Request: "Generate the first 6 rows of Pascal's triangle"
- Expected: `[[1], [1, 1], [1, 2, 1], [1, 3, 3, 1], [1, 4, 6, 4, 1], [1, 5, 10, 10, 5, 1]]`

- Request: "What's the 10th triangular number?"
- Expected: `55`

- Request: "Find all perfect numbers up to 10000"
- Expected: `[6, 28, 496, 8128]`

- Request: "What's the Collatz sequence for 13?"
- Expected: `[13, 40, 20, 10, 5, 16, 8, 4, 2, 1]` (9 steps)

- Request: "Generate an SHA-256 hash of 'Hello World'"
- Expected: `a591a6d40bf420404a011733cfb7b190d62c65bf0bcda32b57b277d9ad9f146e`

- Request: "What's the MD5 hash of 'password123'?"
- Expected: `482c811da5d5b4bc6d497ffa98491e38` (with security warning about MD5)

## 📄 License

MIT

## 🎓 Educational Value

This project demonstrates:
- MCP server implementation from scratch
- Python async/await patterns
- JSON-RPC protocol handling
- Proper error handling and validation
- Production-ready logging practices
- Type-safe Python development
- Virtual environment management
- Integration with AI assistants
- Algorithm implementation:
  - Fibonacci sequence (iterative approach)
  - Prime number algorithms (Sieve of Eratosthenes, trial division)
  - Number theory algorithms (Euclidean algorithm for GCD)
  - Combinatorics (efficient computation of combinations and permutations)
  - Sequence generators (Pascal's triangle with memoization, triangular numbers, perfect numbers, Collatz conjecture)
- Mathematical computation and number theory

---

**Built with Python 3.12 • MCP SDK 1.21.2 • Tested on Windows ARM64**
