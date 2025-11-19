# Mathematical Tools MCP Server

A complete Model Context Protocol (MCP) server implementation in Python that provides mathematical calculation tools for AI assistants like Claude Desktop and other MCP-compatible clients.

## 🌟 Features

- **Fibonacci Calculations**: Calculate individual Fibonacci numbers or generate complete sequences
- **Prime Number Tools**: Check primality, generate primes, find nth prime, and factorize numbers
- **Number Theory Tools**: GCD, LCM, factorial, combinations, and permutations
- **High Performance**: Optimized algorithms (Euclidean algorithm, Sieve of Eratosthenes, efficient combinatorics)
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
    permutations
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
- Mathematical computation and number theory

---

**Built with Python 3.12 • MCP SDK 1.21.2 • Tested on Windows ARM64**
