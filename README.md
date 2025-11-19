# Fibonacci MCP Server

A complete Model Context Protocol (MCP) server implementation in Python that provides Fibonacci calculation tools for AI assistants like Claude Desktop and other MCP-compatible clients.

## 🌟 Features

- **Dual Calculation Modes**: Calculate individual Fibonacci numbers or generate complete sequences
- **High Performance**: Fast iterative algorithm with O(n) time complexity
- **Robust Validation**: Comprehensive error handling and input validation (range: 0-1000)
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

### Calculate a Single Fibonacci Number
```
"What is the 50th Fibonacci number?"
"Calculate the Fibonacci number at position 100"
```

### Generate Fibonacci Sequences
```
"Show me the first 20 Fibonacci numbers"
"Generate a Fibonacci sequence with 15 numbers"
```

## 🔧 Tool Specification

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

You can also test the Fibonacci calculations directly in Python:

```python
from src.fibonacci_server.server import calculate_fibonacci, calculate_fibonacci_sequence

# Single number
print(calculate_fibonacci(10))  # Output: 55

# Sequence
print(calculate_fibonacci_sequence(10))  # Output: [0, 1, 1, 2, 3, 5, 8, 13, 21, 34]
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

Look for messages containing "fibonacci" or "mcp" to see connection attempts and errors.

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

- Add new mathematical tools (primes, factorials, etc.)
- Improve error handling
- Add unit tests
- Enhance documentation
- Create additional example servers

## 📖 Learning Resources

- [Model Context Protocol Documentation](https://modelcontextprotocol.io/)
- [MCP Python SDK](https://github.com/modelcontextprotocol/python-sdk)
- [MCP Specification](https://spec.modelcontextprotocol.io/)

## ✅ Verification

Your server is working correctly if you can:

1. ✅ Start the server manually without errors
2. ✅ See the server in Claude Desktop's tool list
3. ✅ Ask Claude to calculate Fibonacci numbers and receive correct results
4. ✅ Generate sequences and verify the mathematical correctness

Example verification:
- Request: "Calculate the first 10 Fibonacci numbers"
- Expected: `[0, 1, 1, 2, 3, 5, 8, 13, 21, 34]`

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

---

**Built with Python 3.12 • MCP SDK 1.21.2 • Tested on Windows ARM64**
