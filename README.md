# Mathematical & Statistical Tools MCP Servers

A complete Model Context Protocol (MCP) server implementation in Python that provides mathematical calculation and statistical analysis tools for AI assistants like Claude Desktop and other MCP-compatible clients.

## 🏗️ Architecture

This repository contains **two specialized MCP servers**:

### 1. **Math Calculator Server** (`src/math_server/`)
Mathematical calculations, sequences, and utility tools:
- **Fibonacci Calculations**: Calculate individual Fibonacci numbers or generate complete sequences
- **Prime Number Tools**: Check primality, generate primes, find nth prime, and factorize numbers
- **Number Theory Tools**: GCD, LCM, factorial, combinations, and permutations
- **Sequence Generators**: Pascal's triangle, triangular numbers, perfect numbers, and Collatz sequences
- **Cryptographic Hash Generator**: Generate MD5, SHA-1, SHA-256, SHA-512, and BLAKE2b hashes with security notes
- **Unit Converter**: Convert between units across 7 categories (length, weight, temperature, volume, time, digital storage, speed)
- **Date Calculator**: Calculate date differences, add/subtract time, count business days, calculate age, and day of week
- **Text Processing Tools**: Text statistics, word frequency analysis, text transformations, and encoding/decoding

### 2. **Statistical Analysis Server** (`src/stats_server/`)
Dedicated statistical analysis tools:
- **Descriptive Statistics**: Calculate mean, median, mode, standard deviation, variance, min, max, quartiles, and range
- **Correlation Analysis**: Pearson correlation coefficient and covariance between two datasets
- **Percentile Calculations**: Calculate specific percentiles from datasets with linear interpolation
- **Outlier Detection**: Identify outliers using IQR method with configurable threshold
- **Time Series Analysis**: 6 comprehensive tools for process monitoring and predictive maintenance
  - **Moving Average**: Simple, Exponential, and Weighted MA for smoothing sensor data
  - **Trend Detection**: Linear regression for equipment degradation and efficiency analysis
  - **Autocorrelation**: Identify cyclic patterns and batch process cycles
  - **Change Point Detection**: Detect process upsets, modifications, and regime changes
  - **Rate of Change**: Monitor acceleration/deceleration for safety and startup monitoring
  - **Rolling Statistics**: Continuous windowed statistics for SCADA displays
- **Regression Analysis**: 5 comprehensive tools for predictive modeling and equipment performance
  - **Linear Regression**: Simple and multiple regression with diagnostics (R², RMSE, confidence intervals, Durbin-Watson)
  - **Polynomial Regression**: Fit non-linear curves (degree 2-6) with turning points and optimization
  - **Residual Analysis**: Validate model assumptions (normality, autocorrelation, heteroscedasticity)
  - **Prediction with Intervals**: Generate forecasts with confidence and prediction intervals
  - **Multivariate Regression**: Multiple independent variables with VIF for multicollinearity detection
- **Signal Processing**: 6 comprehensive tools for vibration monitoring and electrical analysis
  - **FFT Analysis**: Frequency domain analysis for bearing defects, motor faults, and gear mesh
  - **Power Spectral Density**: Energy distribution across frequencies for vibration and noise
  - **RMS Value**: Overall signal energy with ISO 10816 compliance and rolling statistics
  - **Peak Detection**: Identify dominant frequencies, harmonics, and resonances
  - **Signal-to-Noise Ratio**: Assess sensor health and data acquisition quality
  - **Harmonic Analysis**: THD calculation, power quality, and IEEE 519 compliance
- **Statistical Process Control (SPC)**: 5 essential tools for manufacturing quality control and Six Sigma
  - **Control Limits**: Calculate UCL, LCL, centerline for X-bar, Individuals, Range, S, p, np, c, u charts
  - **Process Capability**: Cp, Cpk, Pp, Ppk indices with sigma level and estimated PPM defects
  - **Western Electric Rules**: 8 run rules for early detection of non-random patterns
  - **CUSUM Chart**: Cumulative sum for detecting small persistent shifts (<1.5σ)
  - **EWMA Chart**: Exponentially weighted moving average for balanced shift detection

## 🌟 Benefits of Split Architecture

- **Separation of Concerns**: Mathematical and statistical tools are logically separated
- **Focused Development**: Easier to extend specific capabilities independently
- **Performance**: Lighter weight servers with specific purposes
- **Deployment Flexibility**: Run only the servers you need
- **Easier Testing**: Isolated testing of specific functionality
- **Clear Organization**: Each server has a well-defined scope

## 🚀 Key Features

- **High Performance**: Optimized algorithms (Euclidean algorithm, Sieve of Eratosthenes, efficient combinatorics, memoization)
- **Robust Validation**: Comprehensive error handling and input validation with appropriate ranges
- **Production Ready**: Full logging, proper error messages, and graceful shutdown
- **Security Features**: API key authentication, rate limiting, CORS support (see [SECURITY.md](SECURITY.md))
- **Configurable**: YAML-based configuration with environment variable overrides
- **Well Documented**: Extensive code annotations and inline documentation for learning
- **Type Safe**: Complete type hints throughout the codebase
- **MCP Compliant**: Implements the Model Context Protocol specification using stdio and HTTP transports

## 🌐 HTTP/HTTPS Transport

The MCP servers support **dual transport modes**, allowing flexibility in how clients connect:

### Transport Modes

#### 📟 stdio Mode (Default)
- **Use Case**: Claude Desktop, local MCP clients
- **Transport**: Standard input/output (stdin/stdout)
- **Protocol**: JSON-RPC 2.0 over MCP
- **Best For**: Desktop applications, CLI tools, direct process communication

#### 🌍 HTTP Mode
- **Use Case**: Web clients, remote connections, GitHub Codespaces, browser-based applications
- **Transport**: HTTP/HTTPS with Server-Sent Events (SSE)
- **Protocol**: JSON-RPC 2.0 over HTTP + SSE for streaming
- **Best For**: Remote access, cloud deployments, web applications, Codespaces

### Architecture Diagram

```
┌─────────────────────────────────────────────────────────────────┐
│                    MCP Server (Dual Mode)                       │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  ┌────────────────────────┐      ┌─────────────────────────┐   │
│  │   stdio Transport      │      │   HTTP Transport        │   │
│  │                        │      │                         │   │
│  │  • stdin/stdout        │      │  • FastAPI Framework    │   │
│  │  • JSON-RPC 2.0        │      │  • REST Endpoints       │   │
│  │  • Direct process      │      │  • Server-Sent Events   │   │
│  │    communication       │      │  • JSON-RPC over HTTP   │   │
│  │                        │      │                         │   │
│  │  Clients:              │      │  Endpoints:             │   │
│  │  • Claude Desktop      │      │  • POST /messages       │   │
│  │  • MCP Inspector       │      │  • GET  /sse            │   │
│  │  • CLI tools           │      │  • GET  /health         │   │
│  │                        │      │  • GET  /ready          │   │
│  │                        │      │  • GET  /metrics        │   │
│  │                        │      │                         │   │
│  │                        │      │  Clients:               │   │
│  │                        │      │  • Web browsers         │   │
│  │                        │      │  • HTTP clients         │   │
│  │                        │      │  • GitHub Codespaces    │   │
│  │                        │      │  • Remote applications  │   │
│  └────────────────────────┘      └─────────────────────────┘   │
│                                                                 │
│              ┌────────────────────────────────┐                 │
│              │   MCP Tools Layer (24 tools)   │                 │
│              │   • Math calculations          │                 │
│              │   • Statistical analysis       │                 │
│              │   • Text processing            │                 │
│              │   • Unit conversions           │                 │
│              └────────────────────────────────┘                 │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### HTTP Endpoints Reference

#### POST `/messages`
**Purpose**: Main JSON-RPC 2.0 endpoint for tool invocation

**Methods Supported**:
- `initialize` - Initialize MCP connection
- `tools/list` - List available tools
- `tools/call` - Execute a specific tool

**Request Format**:
```json
{
  "jsonrpc": "2.0",
  "method": "tools/call",
  "params": {
    "name": "fibonacci",
    "arguments": {"n": 10}
  },
  "id": 1
}
```

**Response Format**:
```json
{
  "jsonrpc": "2.0",
  "result": {
    "content": [
      {
        "type": "text",
        "text": "Fibonacci number at position 10: 55"
      }
    ],
    "isError": false
  },
  "id": 1
}
```

#### GET `/sse`
**Purpose**: Server-Sent Events stream for server-to-client notifications

**Features**:
- Real-time event streaming
- Automatic keepalive (every 30 seconds)
- Connection status notifications

**Events**:
- `connected` - Initial connection established
- `ping` - Keepalive heartbeat

#### GET `/health`
**Purpose**: Basic liveness check

**Response**:
```json
{
  "status": "ok",
  "server": "math-server",
  "uptime_seconds": 123.45,
  "timestamp": "2025-11-22T10:30:00Z"
}
```

#### GET `/ready`
**Purpose**: Readiness check for load balancers

**Response (Ready)**:
```json
{
  "status": "ready",
  "mcp_initialized": true,
  "tools_count": 24
}
```

**Response (Not Ready)**: HTTP 503

#### GET `/metrics`
**Purpose**: Operational metrics for monitoring

**Response**:
```json
{
  "total_requests": 1234,
  "active_connections": 5,
  "tools_available": 24,
  "uptime_seconds": 3600.00,
  "requests_per_minute": 20.50
}
```

### Starting Servers in HTTP Mode

**Quick Start** (both servers):
```bash
# Linux/Mac
./start-http-servers.sh start

# Windows
.\start-http-servers.ps1
```

**Individual Servers**:
```bash
# Math server on port 8000
python src/math_server/server.py --transport http --port 8000

# Stats server on port 8001
python src/stats_server/server.py --transport http --port 8001
```

**With Configuration**:
```bash
python src/math_server/server.py --transport http --config config.yaml
```

## 📋 Prerequisites

### For Local Development
- Python 3.10 or higher
- pip (Python package manager)
- Virtual environment (recommended)

### For Docker Deployment
- Docker 20.10 or higher
- Docker Compose 2.0 or higher
- (Optional) SSL certificates for production HTTPS

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

### 2. Quick Start - HTTP Mode

Get started quickly with HTTP transport for web clients and remote access:

```bash
# 1. Copy example config
# Option A: Manual copy
cp config.example.yaml config.yaml
# Option B: Use launcher (will auto-create if missing)

# 2. Start both servers in HTTP mode
./start-http-servers.sh start

# 3. Test connection
curl http://localhost:8000/health

# Expected response:
# {"status":"ok","server":"math-server","uptime_seconds":1.23,"timestamp":"..."}
```

**What you get**:
- Math server running on `http://localhost:8000`
- Stats server running on `http://localhost:8001`
- REST API with JSON-RPC 2.0 protocol
- Health checks, metrics, and SSE streaming
- Ready for HTTP clients, web apps, and Codespaces

**Next steps**:
- Test the API: See [Client Connection Examples](#-client-connection-examples)
- Configure security: See [Configuration Reference](#4-configuration-reference)
- Deploy to Codespaces: See [GitHub Codespaces Setup](#-github-codespaces-setup)

### 3. Configuration (Optional)

The servers support YAML-based configuration with environment variable overrides. By default, servers run with sensible defaults, but you can customize settings using a config file.

#### Configuration File

Create or modify `config.yaml` in the project root (see `config.example.yaml` for all options):

```yaml
server:
  math:
    host: "0.0.0.0"
    port: 8000
  stats:
    host: "0.0.0.0"
    port: 8001
  cors_origins:
    - "http://localhost:*"
    - "https://*.app.github.dev"

authentication:
  enabled: false
  api_key: "your-secret-api-key-here"

rate_limiting:
  enabled: false
  requests_per_minute: 60

logging:
  level: "INFO"  # DEBUG, INFO, WARNING, ERROR, CRITICAL
```

#### Environment Variables

Override any config value using environment variables with the `MCP_` prefix:

```bash
# Server settings
export MCP_MATH_PORT=9000
export MCP_STATS_PORT=9001

# Authentication
export MCP_AUTH_ENABLED=true
export MCP_API_KEY="my-secure-api-key-1234567890"

# Rate limiting
export MCP_RATE_LIMIT_ENABLED=true
export MCP_RATE_LIMIT_RPM=120

# Logging
export MCP_LOG_LEVEL=DEBUG
```

#### Using Configuration

Run servers with configuration:

```bash
# With config file
python src/math_server/server.py --config config.yaml

# With environment overrides
MCP_LOG_LEVEL=DEBUG python src/stats_server/server.py --config config.yaml

# Without config (uses defaults)
python src/math_server/server.py
```

### 4. Security Configuration (Optional)

The servers include comprehensive security features that can be enabled via configuration. See [SECURITY.md](SECURITY.md) for detailed documentation.

#### Quick Security Setup

For production deployments with authentication and rate limiting:

```yaml
authentication:
  enabled: true
  api_key: "your-secure-api-key-with-at-least-16-characters"

rate_limiting:
  enabled: true
  requests_per_minute: 60  # Adjust based on your needs
```

Or via environment variables:

```bash
export MCP_AUTH_ENABLED=true
export MCP_API_KEY="your-secure-api-key-with-at-least-16-characters"
export MCP_RATE_LIMIT_ENABLED=true
export MCP_RATE_LIMIT_RPM=60
```

**Security Features Include**:
- 🔐 **API Key Authentication**: Bearer token validation
- 🚦 **Rate Limiting**: Token bucket algorithm per client IP
- 🌐 **CORS Support**: Configurable cross-origin access
- 📝 **Request Logging**: Comprehensive request tracking

**Important**: Always use strong API keys and HTTPS in production. See [SECURITY.md](SECURITY.md) for best practices.

### 5. Configuration Reference

Complete reference for all configuration options in `config.yaml`:

#### Server Configuration

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `server.math.host` | string | `"0.0.0.0"` | Math server bind address. Use `"0.0.0.0"` for all interfaces, `"127.0.0.1"` for localhost only |
| `server.math.port` | integer | `8000` | Math server port number (1-65535) |
| `server.stats.host` | string | `"0.0.0.0"` | Stats server bind address |
| `server.stats.port` | integer | `8001` | Stats server port number (1-65535) |
| `server.cors_origins` | array | `["http://localhost:*", "https://*.app.github.dev"]` | Allowed CORS origins. Supports wildcards (`*`) |

**CORS Origins Examples**:
```yaml
cors_origins:
  - "http://localhost:*"              # All localhost ports
  - "https://*.app.github.dev"        # All GitHub Codespaces
  - "https://example.com"             # Specific domain
  - "http://192.168.1.100:3000"      # Specific IP and port
  - "*"                               # All origins (not recommended for production)
```

#### Authentication Configuration

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `authentication.enabled` | boolean | `false` | Enable/disable API key authentication |
| `authentication.api_key` | string | `"your-secret-api-key-here"` | API key for Bearer token authentication (min 16 chars) |

**Authentication Behavior**:
- **When enabled**: All endpoints except `/health`, `/ready`, and `/metrics` require `Authorization: Bearer <api_key>` header
- **When disabled**: No authentication required (suitable for development or trusted networks)
- **Invalid/missing key**: Returns HTTP 401 Unauthorized

**Environment Variable**: `MCP_AUTH_ENABLED`, `MCP_API_KEY`

#### Rate Limiting Configuration

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `rate_limiting.enabled` | boolean | `false` | Enable/disable rate limiting per client IP |
| `rate_limiting.requests_per_minute` | integer | `60` | Maximum requests per minute per client (1-10000) |

**Rate Limiting Algorithm**: Token bucket
- Bucket capacity equals `requests_per_minute` (allows bursts)
- Tokens refill at rate of `requests_per_minute / 60` per second
- Each request consumes one token
- Returns HTTP 429 when rate limit exceeded with `Retry-After` header

**Recommended Values**:
- **Development**: 120 (2 requests/second)
- **Production**: 60 (1 request/second)
- **High-traffic**: 300 (5 requests/second)

**Environment Variable**: `MCP_RATE_LIMIT_ENABLED`, `MCP_RATE_LIMIT_RPM`

#### Logging Configuration

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `logging.level` | string | `"INFO"` | Log level: DEBUG, INFO, WARNING, ERROR, CRITICAL |

**Log Level Details**:
- **DEBUG**: Detailed information for debugging (verbose, includes request/response details)
- **INFO**: General informational messages (recommended for production)
- **WARNING**: Warning messages for potentially problematic situations
- **ERROR**: Error messages for serious problems
- **CRITICAL**: Critical messages for severe errors

**Environment Variable**: `MCP_LOG_LEVEL`

#### Environment Variable Override Reference

All configuration options can be overridden using environment variables with the `MCP_` prefix:

```bash
# Server Configuration
export MCP_MATH_HOST=127.0.0.1
export MCP_MATH_PORT=9000
export MCP_STATS_HOST=127.0.0.1
export MCP_STATS_PORT=9001

# Authentication
export MCP_AUTH_ENABLED=true
export MCP_API_KEY=my-secret-key

# Rate Limiting
export MCP_RATE_LIMIT_ENABLED=true
export MCP_RATE_LIMIT_RPM=120

# Logging
export MCP_LOG_LEVEL=DEBUG
```

**Priority Order**:
1. Environment variables (highest priority)
2. Configuration file (`config.yaml`)
3. Default values (lowest priority)

### 6. Test the Servers

#### Option A: Quick Start with Launcher Scripts (HTTP Mode)

Use the provided launcher scripts to start both servers in HTTP mode with one command:

**Linux/Mac:**
```bash
# Start both servers
./start-http-servers.sh start

# Start with hot-reload (development mode)
./start-http-servers.sh --dev

# Check status
./start-http-servers.sh status

# Stop servers
./start-http-servers.sh stop

# Restart servers
./start-http-servers.sh restart
```

**Windows (PowerShell):**
```powershell
# Start both servers
.\start-http-servers.ps1

# Start with hot-reload (development mode)
.\start-http-servers.ps1 -Dev

# Check status
.\start-http-servers.ps1 -Status

# Stop servers
.\start-http-servers.ps1 -Stop

# Restart servers
.\start-http-servers.ps1 -Restart
```

The launcher scripts will:
- Automatically create `config.yaml` from the example if it doesn't exist
- Start math_server on port 8000 and stats_server on port 8001
- Display connection URLs for localhost and GitHub Codespaces
- Save process IDs for easy management
- Provide colored output showing server status

#### Option B: Manual Testing (stdio Mode)

Run each server directly to verify they work:

```bash
# Test Math Calculator Server
python src/math_server/server.py

# Test Statistical Analysis Server (in a new terminal)
python src/stats_server/server.py

# Test with configuration
python src/math_server/server.py --config config.yaml
```

Each server will start and wait for MCP protocol messages on stdin. Press `Ctrl+C` to stop.

To validate your configuration:

```bash
# Run the configuration test suite
python test_config.py
```

## 🧪 HTTP Client Testing Suite

The repository includes a comprehensive testing suite for validating HTTP transport, authentication, rate limiting, and all MCP tools over HTTP.

### Features

The test suite validates:
- ✅ **SSE Connections**: Server-Sent Events endpoint connectivity
- ✅ **MCP Protocol**: Initialization, tool listing, tool execution
- ✅ **All Tools**: 16+ tool categories (Fibonacci, primes, number theory, hashing, dates, text processing, etc.)
- ✅ **Authentication**: Valid/invalid/missing API keys, enabled/disabled states
- ✅ **Rate Limiting**: Within/exceeding limits, disabled state
- ✅ **CORS**: Header verification and preflight requests
- ✅ **Health Endpoints**: `/health`, `/ready`, `/metrics`
- ✅ **Error Handling**: Invalid JSON-RPC, unknown methods, malformed requests
- ✅ **Concurrent Requests**: Multiple simultaneous requests
- ✅ **Performance**: Response time validation

### Quick Start

```bash
# Install test dependencies
pip install -r requirements-test.txt

# Run all tests (starts server automatically)
./run_http_tests.sh

# Or manually
python src/math_server/server.py --transport http --port 8000 &
pytest tests/test_http_client.py -v
```

### Running Tests

**Against local server:**
```bash
./run_http_tests.sh
```

**Against existing server:**
```bash
./run_http_tests.sh --no-server
```

**Against GitHub Codespaces:**
```bash
export MATH_SERVER_URL="https://your-codespace-8000.app.github.dev"
pytest tests/test_http_client.py -v
```

**Run specific test categories:**
```bash
# Tool execution tests only
pytest tests/test_http_client.py -k "tool" -v

# Authentication tests only
pytest tests/test_http_client.py -k "auth" -v

# Health endpoint tests only
pytest tests/test_http_client.py -k "health" -v

# Concurrent tests only
pytest tests/test_http_client.py -k "concurrent" -v
```

### Test Results

**Summary**: 40 passed, 4 skipped in ~1.7s

- 40 tests validate core functionality
- 4 tests skipped (require authentication or rate limiting enabled in config)
- Full coverage of HTTP transport, SSE, JSON-RPC, and all MCP tools

See [tests/README.md](tests/README.md) for comprehensive documentation including:
- Detailed test categories
- Configuration options
- Troubleshooting guide
- CI/CD integration examples

### CI/CD Integration

The repository includes GitHub Actions workflow for automated testing:

```yaml
# .github/workflows/test-http-client.yml
- Runs tests against Python 3.11 and 3.12
- Tests with authentication enabled
- Tests with rate limiting enabled
- Generates coverage reports
```

## 🔥 Development Mode

For faster iteration when developing or modifying server code, use the built-in hot-reload feature.

### Starting Servers in Development Mode

**Using Launcher Scripts:**

```bash
# Linux/Mac - start both servers with hot-reload
./start-http-servers.sh --dev

# Windows PowerShell - start both servers with hot-reload
.\start-http-servers.ps1 -Dev
```

**Manual Start:**

```bash
# Start math server with hot-reload
python src/math_server/server.py --transport http --port 8000 --dev

# Start stats server with hot-reload
python src/stats_server/server.py --transport http --port 8001 --dev
```

### VS Code Debugging

The repository includes pre-configured VS Code launch configurations in `.vscode/launch.json`:

**Available Debug Configurations:**
1. **Math Server (HTTP Dev)** - Math server with hot-reload via uvicorn
2. **Stats Server (HTTP Dev)** - Stats server with hot-reload via uvicorn
3. **Math Server (stdio)** - Math server in stdio mode for Claude Desktop testing
4. **Stats Server (stdio)** - Stats server in stdio mode for Claude Desktop testing
5. **Math Server (HTTP Dev with CLI)** - Math server using --dev flag
6. **Stats Server (HTTP Dev with CLI)** - Stats server using --dev flag

**Compound Configurations:**
- **Both Servers (HTTP Dev)** - Run both servers simultaneously with hot-reload
- **Both Servers (HTTP Dev with CLI)** - Run both servers using CLI --dev flag

**To Use:**
1. Open the project in VS Code
2. Press `F5` or go to Run and Debug (Ctrl+Shift+D)
3. Select a configuration from the dropdown
4. Click the green play button or press `F5`

### Development Workflow

1. **Start servers in dev mode:**
   ```bash
   ./start-http-servers.sh --dev
   ```

2. **Make code changes** in any Python file under `src/`

3. **Automatic reload** - The server detects changes and restarts automatically (typically takes 1-2 seconds)

4. **Test changes** - The server is immediately available with your updates

5. **Debug with breakpoints** - Use VS Code debug configurations to set breakpoints and step through code

### Hot-Reload Behavior

- **Watched directories**: `src/` and all subdirectories
- **Monitored file types**: `.py` files
- **Reload delay**: ~1-2 seconds after file save
- **Log level**: DEBUG (in dev mode) for more verbose output
- **Preserved state**: Server state is reset on reload

### Important Notes

⚠️ **Dev mode is for development only**
- Do NOT use `--dev` flag in production
- Hot-reload adds overhead and is not suitable for production use
- Use standard mode for production deployments

🔍 **Debugging Tips**
- Check `logs/math_server.log` and `logs/stats_server.log` for detailed output
- Use VS Code's integrated terminal to see reload messages
- Set breakpoints in tool handlers for step-through debugging
- Use `--log-level debug` for maximum verbosity

### Troubleshooting Hot-Reload

**Server doesn't restart on changes:**
- Ensure you're watching the correct directory (`src/`)
- Check that the file is a `.py` file
- Verify the server started with `--dev` flag
- Look for syntax errors in logs

**Port already in use:**
- Stop other instances: `./start-http-servers.sh stop`
- Check for orphaned processes: `lsof -i :8000` (Mac/Linux) or `netstat -ano | findstr :8000` (Windows)

**Changes not reflected:**
- Wait 2-3 seconds after saving
- Check logs for reload confirmation
- Verify the file you're editing is in the `src/` directory
- Try manually restarting the server

### 7. Configure with Claude Desktop

Add **both servers** to Claude Desktop's configuration file:

**Windows:** `%APPDATA%\Claude\claude_desktop_config.json`
**macOS:** `~/Library/Application Support/Claude/claude_desktop_config.json`
**Linux:** `~/.config/Claude/claude_desktop_config.json`

```json
{
  "mcpServers": {
    "math-tools": {
      "command": "c:/Users/YOUR_USERNAME/path/to/MyMCP/venv/Scripts/python.exe",
      "args": [
        "c:/Users/YOUR_USERNAME/path/to/MyMCP/src/math_server/server.py"
      ]
    },
    "stats-tools": {
      "command": "c:/Users/YOUR_USERNAME/path/to/MyMCP/venv/Scripts/python.exe",
      "args": [
        "c:/Users/YOUR_USERNAME/path/to/MyMCP/src/stats_server/server.py"
      ]
    }
  }
}
```

**Important:** 
- Replace `YOUR_USERNAME` and the path with your actual paths
- Use forward slashes (`/`) or escaped backslashes (`\\`) in paths
- Use absolute paths for both the Python executable and server scripts
- You can configure either or both servers based on your needs

**With Configuration File:**

To use a configuration file with Claude Desktop, add the `--config` argument:

```json
{
  "mcpServers": {
    "math-tools": {
      "command": "c:/Users/YOUR_USERNAME/path/to/MyMCP/venv/Scripts/python.exe",
      "args": [
        "c:/Users/YOUR_USERNAME/path/to/MyMCP/src/math_server/server.py",
        "--config",
        "c:/Users/YOUR_USERNAME/path/to/MyMCP/config.yaml"
      ]
    },
    "stats-tools": {
      "command": "c:/Users/YOUR_USERNAME/path/to/MyMCP/venv/Scripts/python.exe",
      "args": [
        "c:/Users/YOUR_USERNAME/path/to/MyMCP/src/stats_server/server.py",
        "--config",
        "c:/Users/YOUR_USERNAME/path/to/MyMCP/config.yaml"
      ]
    }
  }
}
```

### 8. Restart Claude Desktop

Completely quit Claude Desktop and restart it. Both servers should now be available with their respective tools.

## 🐳 Docker Deployment

For production deployments, Docker provides a containerized, scalable solution with nginx reverse proxy, SSL termination, and health checks.

### Prerequisites

- Docker (20.10 or higher)
- Docker Compose (2.0 or higher)

### Quick Start with Docker

1. **Create Environment File**

Create a `.env` file in the project root:

```bash
MCP_API_KEY=your-secure-api-key-here
MCP_AUTH_ENABLED=true
MCP_RATE_LIMIT_ENABLED=true
```

2. **Prepare Configuration**

Copy and customize the configuration:

```bash
cp config.example.yaml config.yaml
# Edit config.yaml with your settings
```

3. **Build and Start Services**

```bash
# Build Docker images
docker-compose build

# Start all services (math-server, stats-server, nginx)
docker-compose up -d

# View logs
docker-compose logs -f

# Check service status
docker-compose ps
```

4. **Access the Servers**

- Math Server: `http://localhost/math/` or direct `http://localhost:8000/`
- Stats Server: `http://localhost/stats/` or direct `http://localhost:8001/`
- Health checks: 
  - `http://localhost:8000/health`
  - `http://localhost:8001/health`
  - `http://localhost:8000/metrics`

### Docker Services

The `docker-compose.yml` defines three services:

1. **math-server**: Math Calculator MCP server on port 8000
2. **stats-server**: Statistical Analysis MCP server on port 8001
3. **nginx**: Reverse proxy with SSL support on ports 80/443

### Architecture Benefits

- **Isolation**: Each service runs in its own container
- **Scalability**: Easy to scale individual services
- **Health Checks**: Automatic health monitoring and restart
- **Security**: Non-root user, isolated network
- **SSL Termination**: HTTPS support via nginx
- **Rate Limiting**: Built-in API rate limiting

### SSL/HTTPS Configuration

To enable HTTPS with nginx:

1. **Generate or Obtain SSL Certificates**

For development (self-signed):
```bash
mkdir ssl
openssl req -x509 -nodes -days 365 -newkey rsa:2048 \
  -keyout ssl/key.pem -out ssl/cert.pem \
  -subj "/CN=mcp.example.com"
```

For production, use Let's Encrypt:
```bash
# Install certbot
sudo apt-get install certbot

# Obtain certificate
sudo certbot certonly --standalone -d mcp.example.com

# Copy to ssl directory
cp /etc/letsencrypt/live/mcp.example.com/fullchain.pem ssl/cert.pem
cp /etc/letsencrypt/live/mcp.example.com/privkey.pem ssl/key.pem
```

2. **Update nginx.conf**

Edit `nginx.conf` and replace `mcp.example.com` with your domain.

3. **Restart nginx**

```bash
docker-compose restart nginx
```

### Docker Commands

```bash
# Build images
docker-compose build

# Start services in background
docker-compose up -d

# Start with rebuild
docker-compose up -d --build

# View logs (all services)
docker-compose logs -f

# View logs (specific service)
docker-compose logs -f math-server

# Stop services
docker-compose down

# Stop and remove volumes
docker-compose down -v

# Restart a service
docker-compose restart math-server

# Scale a service (if needed)
docker-compose up -d --scale math-server=3
```

### Configuration in Docker

Environment variables can be set in three ways:

1. **`.env` file** (recommended for secrets):
```env
MCP_API_KEY=your-secure-api-key
MCP_AUTH_ENABLED=true
```

2. **docker-compose.yml** environment section:
```yaml
environment:
  - MCP_LOG_LEVEL=DEBUG
  - MCP_RATE_LIMIT_RPM=120
```

3. **Command line**:
```bash
MCP_API_KEY=test docker-compose up -d
```

### Production Deployment Checklist

- [ ] Set strong API key in `.env`
- [ ] Enable authentication (`MCP_AUTH_ENABLED=true`)
- [ ] Configure CORS origins restrictively in `config.yaml`
- [ ] Set up SSL certificates for nginx
- [ ] Configure log rotation
- [ ] Set up monitoring (Prometheus, Grafana)
- [ ] Configure backup strategy for configuration
- [ ] Set resource limits (CPU, memory) in docker-compose.yml
- [ ] Enable container security scanning
- [ ] Document incident response procedures
- [ ] Configure firewall rules
- [ ] Set up automated backups
- [ ] Test disaster recovery procedures

### Resource Limits

Add resource constraints to docker-compose.yml:

```yaml
services:
  math-server:
    # ... other config ...
    deploy:
      resources:
        limits:
          cpus: '0.5'
          memory: 512M
        reservations:
          cpus: '0.25'
          memory: 256M
```

### Monitoring

Access metrics endpoints:

```bash
# Math server metrics
curl http://localhost:8000/metrics

# Stats server metrics
curl http://localhost:8001/metrics

# Health checks
curl http://localhost:8000/health
curl http://localhost:8001/health

# Readiness checks
curl http://localhost:8000/ready
curl http://localhost:8001/ready
```

### Troubleshooting

**Containers not starting:**
```bash
# Check logs
docker-compose logs

# Check container status
docker-compose ps

# Inspect specific container
docker inspect mcp-math-server
```

**Health check failures:**
```bash
# Test health endpoint directly
docker exec mcp-math-server curl http://localhost:8000/health

# Check container logs
docker logs mcp-math-server
```

**Network issues:**
```bash
# Verify network
docker network ls
docker network inspect mymcp_mcp-network

# Test connectivity between containers
docker exec mcp-nginx ping math-server
```

### GitHub Codespaces Deployment

The Docker setup works seamlessly in GitHub Codespaces:

1. Codespaces automatically forwards ports 8000, 8001, 80, 443
2. Access via Codespaces URL: `https://<codespace-name>-8000.app.github.dev`
3. No SSL configuration needed (Codespaces handles HTTPS)

## 🚀 GitHub Codespaces Setup

GitHub Codespaces provides a cloud-based development environment with automatic port forwarding, making it easy to run and access the MCP servers remotely.

### Step-by-Step Setup

#### 1. Create a Codespace

1. Navigate to the repository on GitHub
2. Click the **Code** button (green button)
3. Select the **Codespaces** tab
4. Click **Create codespace on main** (or your preferred branch)
5. Wait for the Codespace to initialize (1-2 minutes)

#### 2. Automatic Setup

The repository includes a `.devcontainer/devcontainer.json` configuration that automatically:
- Installs Python 3.10+
- Installs all dependencies from `requirements.txt`
- Sets up the development environment
- Configures port forwarding for ports 8000 and 8001

#### 3. Start the Servers

Once the Codespace is ready, open the terminal and start the servers:

```bash
# Copy example config (if not already done)
cp config.example.yaml config.yaml

# Start both servers in HTTP mode
./start-http-servers.sh start
```

The startup script will display the server URLs.

#### 4. Access Public URLs

GitHub Codespaces automatically creates public URLs for forwarded ports:

**URL Format**: `https://<codespace-name>-<port>.app.github.dev`

Example URLs:
- Math server: `https://jubilant-space-waddle-5p4v997xqw5h555-8000.app.github.dev`
- Stats server: `https://jubilant-space-waddle-5p4v997xqw5h555-8001.app.github.dev`

**Finding Your URLs**:
1. In VS Code, click the **PORTS** tab in the terminal panel
2. Your forwarded ports (8000, 8001) will be listed with their URLs
3. Click the globe icon to open in browser or copy the URL

#### 5. Test Connection

Test the health endpoint in your browser or with curl:

```bash
# Test math server (replace with your actual URL)
curl https://your-codespace-8000.app.github.dev/health

# Expected response:
# {"status":"ok","server":"math-server","uptime_seconds":123.45,"timestamp":"..."}
```

#### 6. Configure MCP Client

Use the Codespaces URLs in your MCP client configuration:

```javascript
// Example for HTTP-based MCP client
const mathServerURL = "https://your-codespace-8000.app.github.dev";
const statsServerURL = "https://your-codespace-8001.app.github.dev";
```

### Codespaces Features

✅ **Automatic Port Forwarding**: Ports 8000 and 8001 are automatically forwarded  
✅ **HTTPS by Default**: All Codespaces URLs use HTTPS (no SSL setup needed)  
✅ **Public or Private**: Set port visibility to public (accessible by anyone) or private (requires GitHub authentication)  
✅ **Pre-configured Environment**: `.devcontainer` handles all setup automatically  
✅ **Persistent Storage**: Your work is saved between Codespace sessions  
✅ **VS Code Integration**: Full IDE experience in the browser or desktop VS Code  

### Port Visibility Settings

By default, forwarded ports in Codespaces are **private** (require GitHub authentication). To make them **public**:

1. Right-click the port in the **PORTS** tab
2. Select **Port Visibility** → **Public**
3. The URL will now be accessible without authentication

⚠️ **Security Note**: Only make ports public if needed. Use authentication (`MCP_AUTH_ENABLED=true`) for public ports.

### Codespaces URL Examples

Your Codespace will have a unique name. Here are examples of what the URLs look like:

```
Math Server:
https://jubilant-space-waddle-5p4v997xqw5h555-8000.app.github.dev

Stats Server:
https://jubilant-space-waddle-5p4v997xqw5h555-8001.app.github.dev

Health Check:
https://jubilant-space-waddle-5p4v997xqw5h555-8000.app.github.dev/health

Metrics:
https://jubilant-space-waddle-5p4v997xqw5h555-8000.app.github.dev/metrics
```

### Dual-Endpoint Architecture in Codespaces

The repository includes a **dual-endpoint architecture** that runs both development and production-mode servers simultaneously in the same Codespace. This allows you to test both configurations side-by-side.

#### Architecture Overview

```
┌─────────────────────────────────────────────────────────────────────┐
│              GitHub Codespaces - Dual Endpoint Architecture         │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  ┌────────────────────────────┐  ┌───────────────────────────────┐ │
│  │   DEV Endpoints (8000-8001) │  │  PROD Endpoints (9000-9001)   │ │
│  │   ❌ No Authentication       │  │  ✅ Authentication Required   │ │
│  │   🔒 Private Visibility      │  │  🌐 Public Visibility         │ │
│  │   📝 Debug Logging           │  │  📊 Info Logging              │ │
│  ├────────────────────────────┤  ├───────────────────────────────┤ │
│  │                            │  │                               │ │
│  │  Port 8000: Math Server    │  │  Port 9000: Math Server       │ │
│  │  Port 8001: Stats Server   │  │  Port 9001: Stats Server      │ │
│  │                            │  │                               │ │
│  │  Config: config.dev.yaml   │  │  Config: config.prod.yaml     │ │
│  │  Auth: MCP_AUTH_ENABLED=   │  │  Auth: MCP_AUTH_ENABLED=true  │ │
│  │        false               │  │  Key:  $MCP_API_KEY           │ │
│  │                            │  │                               │ │
│  │  Use Case:                 │  │  Use Case:                    │ │
│  │  • Quick testing           │  │  • Demo authentication        │ │
│  │  • Development             │  │  • Production simulation      │ │
│  │  • Debugging               │  │  • Security testing           │ │
│  │  • Private access via      │  │  • Public API demo            │ │
│  │    'gh codespace ports     │  │  • Client integration         │ │
│  │     forward'               │  │                               │ │
│  └────────────────────────────┘  └───────────────────────────────┘ │
│                                                                     │
│  All 4 servers start automatically via .devcontainer/startup.sh    │
│  Health checks verify all instances before startup completes       │
└─────────────────────────────────────────────────────────────────────┘
```

#### Server Endpoints

**Development Servers (No Auth)**:
- Math Server: `https://<codespace-name>-8000.app.github.dev`
- Stats Server: `https://<codespace-name>-8001.app.github.dev`
- Visibility: **Private** (requires `gh codespace ports forward` for access)
- Authentication: **Disabled**
- Config: `config.dev.yaml`

**Production Servers (Auth Required)**:
- Math Server: `https://<codespace-name>-9000.app.github.dev`
- Stats Server: `https://<codespace-name>-9001.app.github.dev`
- Visibility: **Public** (accessible via URL)
- Authentication: **Enabled** (Bearer token required)
- Config: `config.prod.yaml`

#### Testing Development Endpoints

```bash
# Direct access (no authentication needed)
curl https://<codespace-name>-8000.app.github.dev/health

# Test a tool
curl -X POST https://<codespace-name>-8000.app.github.dev/messages \
  -H "Content-Type: application/json" \
  -d '{
    "jsonrpc": "2.0",
    "method": "tools/call",
    "params": {"name": "fibonacci", "arguments": {"n": 10}},
    "id": 1
  }'
```

#### Testing Production Endpoints

```bash
# Without authentication (should fail with 401)
curl https://<codespace-name>-9000.app.github.dev/health

# With authentication (should succeed)
curl -H "Authorization: Bearer <your-api-key>" \
  https://<codespace-name>-9000.app.github.dev/health

# Test a tool with authentication
curl -X POST https://<codespace-name>-9000.app.github.dev/messages \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer <your-api-key>" \
  -d '{
    "jsonrpc": "2.0",
    "method": "tools/call",
    "params": {"name": "fibonacci", "arguments": {"n": 10}},
    "id": 1
  }'
```

#### Setting Up Production API Key

For production use, set a secure API key in Codespaces Secrets:

1. Go to your repository on GitHub
2. Click **Settings** → **Secrets and variables** → **Codespaces**
3. Click **New repository secret**
4. Name: `MCP_API_KEY`
5. Value: Your secure API key (minimum 16 characters)
6. Click **Add secret**

The production servers will automatically use this key when the Codespace starts.

#### Benefits of Dual-Endpoint Architecture

✅ **Side-by-side Testing**: Compare development and production behavior in the same environment  
✅ **Authentication Demo**: Show real authentication flow without breaking dev workflow  
✅ **Security Best Practices**: Private dev endpoints, public prod endpoints with auth  
✅ **Flexible Access**: Use private endpoints for debugging, public for demos  
✅ **Configuration Validation**: Test both `config.dev.yaml` and `config.prod.yaml`  
✅ **Production Simulation**: Match production security posture in development  

#### Accessing Private Endpoints

Development endpoints (8000-8001) are private by default. To access them:

**From Codespace Terminal**:
```bash
# Direct access from within the Codespace
curl http://localhost:8000/health
```

**From Your Local Machine**:
```bash
# Forward port from Codespace to local machine
gh codespace ports forward 8000:8000

# Then access locally
curl http://localhost:8000/health
```

### Troubleshooting Codespaces

**Port not forwarding**:
- Check the **PORTS** tab to see if ports 8000/8001 are listed
- Manually add port forwarding: Click **Forward a Port** in PORTS tab
- Verify servers are running: `./start-http-servers.sh status`

**404 Not Found**:
- Ensure servers are running
- Check the correct port number in the URL
- Verify the path (e.g., `/health`, `/messages`)

**Connection refused**:
- Servers may not be started yet
- Check logs: `cat logs/math_server.log`
- Restart servers: `./start-http-servers.sh restart`

**CORS errors**:
- Add your Codespaces origin to `config.yaml`:
  ```yaml
  cors_origins:
    - "https://*.app.github.dev"
  ```
- Restart servers after config change

## 🔌 Client Connection Examples

### Python Client Example

```python
import httpx
import asyncio
import json

async def call_mcp_tool():
    """Example: Call a tool via HTTP transport"""
    
    # Server URL (replace with your actual URL)
    server_url = "http://localhost:8000"
    
    # Optional: Add authentication header
    headers = {
        "Content-Type": "application/json",
        # "Authorization": "Bearer your-api-key"  # If auth is enabled
    }
    
    # JSON-RPC 2.0 request
    request = {
        "jsonrpc": "2.0",
        "method": "tools/call",
        "params": {
            "name": "fibonacci",
            "arguments": {"n": 10}
        },
        "id": 1
    }
    
    async with httpx.AsyncClient() as client:
        # Call the tool
        response = await client.post(
            f"{server_url}/messages",
            json=request,
            headers=headers
        )
        
        result = response.json()
        print(f"Result: {result}")

# Run the example
asyncio.run(call_mcp_tool())
```

**Expected Output**:
```json
{
  "jsonrpc": "2.0",
  "result": {
    "content": [
      {
        "type": "text",
        "text": "Fibonacci number at position 10: 55"
      }
    ],
    "isError": false
  },
  "id": 1
}
```

### JavaScript/Node.js Client Example

**Installation**: `npm install axios`

```javascript
// Node.js with CommonJS (require)
const axios = require('axios');

// Or with ES modules (import) - add "type": "module" to package.json
// import axios from 'axios';

async function callMCPTool() {
    // Server URL
    const serverUrl = 'http://localhost:8000';
    
    // Optional: Authentication header
    const headers = {
        'Content-Type': 'application/json',
        // 'Authorization': 'Bearer your-api-key'  // If auth is enabled
    };
    
    // JSON-RPC 2.0 request
    const request = {
        jsonrpc: '2.0',
        method: 'tools/call',
        params: {
            name: 'is_prime',
            arguments: { number: 17 }
        },
        id: 1
    };
    
    try {
        const response = await axios.post(
            `${serverUrl}/messages`,
            request,
            { headers }
        );
        
        console.log('Result:', response.data);
    } catch (error) {
        console.error('Error:', error.response?.data || error.message);
    }
}

callMCPTool();
```

### cURL Examples

**Health Check**:
```bash
curl http://localhost:8000/health
```

**List Available Tools**:
```bash
curl -X POST http://localhost:8000/messages \
  -H "Content-Type: application/json" \
  -d '{
    "jsonrpc": "2.0",
    "method": "tools/list",
    "params": {},
    "id": 1
  }'
```

**Call a Tool (without authentication)**:
```bash
curl -X POST http://localhost:8000/messages \
  -H "Content-Type: application/json" \
  -d '{
    "jsonrpc": "2.0",
    "method": "tools/call",
    "params": {
      "name": "fibonacci",
      "arguments": {"n": 10}
    },
    "id": 1
  }'
```

**Call a Tool (with authentication)**:
```bash
curl -X POST http://localhost:8000/messages \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer your-api-key" \
  -d '{
    "jsonrpc": "2.0",
    "method": "tools/call",
    "params": {
      "name": "is_prime",
      "arguments": {"number": 17}
    },
    "id": 1
  }'
```

**Check Metrics**:
```bash
curl http://localhost:8000/metrics
```

### Server-Sent Events (SSE) Example

**Installation**: `pip install httpx httpx-sse`

```python
import httpx
from httpx_sse import connect_sse

async def listen_to_sse():
    """Example: Listen to SSE stream"""
    
    server_url = "http://localhost:8000"
    headers = {
        # "Authorization": "Bearer your-api-key"  # If auth is enabled
    }
    
    async with httpx.AsyncClient() as client:
        async with connect_sse(
            client, "GET", f"{server_url}/sse", headers=headers
        ) as event_source:
            async for event in event_source.aiter_sse():
                print(f"Event: {event.event}")
                print(f"Data: {event.data}")
                print("---")

# Run the example
import asyncio
asyncio.run(listen_to_sse())
```

### Using with Different Languages

**Go**:
```go
package main

import (
    "bytes"
    "encoding/json"
    "fmt"
    "io"
    "net/http"
)

func main() {
    serverURL := "http://localhost:8000"
    
    request := map[string]interface{}{
        "jsonrpc": "2.0",
        "method":  "tools/call",
        "params": map[string]interface{}{
            "name": "fibonacci",
            "arguments": map[string]interface{}{
                "n": 10,
            },
        },
        "id": 1,
    }
    
    jsonData, _ := json.Marshal(request)
    
    resp, err := http.Post(
        serverURL+"/messages",
        "application/json",
        bytes.NewBuffer(jsonData),
    )
    if err != nil {
        panic(err)
    }
    defer resp.Body.Close()
    
    body, _ := io.ReadAll(resp.Body)
    fmt.Println(string(body))
}
```

**Rust**:
```rust
use reqwest;
use serde_json::json;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let server_url = "http://localhost:8000";
    
    let request = json!({
        "jsonrpc": "2.0",
        "method": "tools/call",
        "params": {
            "name": "fibonacci",
            "arguments": {"n": 10}
        },
        "id": 1
    });
    
    let client = reqwest::Client::new();
    let response = client
        .post(format!("{}/messages", server_url))
        .json(&request)
        .send()
        .await?;
    
    let result = response.json::<serde_json::Value>().await?;
    println!("{}", serde_json::to_string_pretty(&result)?);
    
    Ok(())
}
```

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

### Statistical Analysis (Stats Server)
```
"Analyze this data: [23, 45, 12, 67, 34, 89, 23, 56]"
"What's the correlation between [1, 2, 3, 4, 5] and [2, 4, 6, 8, 10]?"
"Find the 75th percentile of [10, 20, 30, 40, 50]"
"Identify outliers in [10, 12, 14, 13, 15, 100, 11, 13, 14] using threshold 1.5"
"Calculate mean and standard deviation for my dataset"
```

### Time Series Analysis (Stats Server)
```
"Calculate a 5-period moving average to smooth this temperature data"
"Analyze the trend in bearing temperature - is it increasing?"
"Find cyclic patterns in these batch cycle times"
"Detect when the process improvement took effect in production data"
"Calculate the rate of temperature change during reactor startup"
"Show rolling statistics with a 10-point window for process monitoring"
```

**For detailed time series documentation and industrial use cases, see [TIME_SERIES_TOOLS.md](TIME_SERIES_TOOLS.md)**

### Regression Analysis (Stats Server)
```
"Perform linear regression on flow rates [100, 150, 200, 250] vs power [45, 62, 82, 105]"
"Fit a polynomial curve to this compressor performance data"
"Analyze residuals to check if my regression model is valid"
"Predict pump power consumption at 225 m³/h with confidence intervals"
"Model chiller efficiency with load, ambient temperature, and condenser flow as variables"
"Check for multicollinearity in my multivariate model"
```

### Signal Processing Analysis (Stats Server)
```
"Perform FFT analysis on bearing vibration data at 10 kHz sampling rate"
"Calculate power spectral density of acoustic signal from pump"
"Compute RMS vibration level for ISO 10816 compliance check"
"Detect dominant frequency peaks in motor current spectrum"
"Calculate signal-to-noise ratio to validate accelerometer health"
"Analyze harmonics in 60 Hz voltage waveform and calculate THD"
```

**For detailed signal processing documentation and industrial workflows, see [SIGNAL_PROCESSING_TOOLS.md](SIGNAL_PROCESSING_TOOLS.md)**
### Statistical Process Control (SPC) Tools (Stats Server)
```
"Calculate control limits for these process measurements using an individuals chart"
"Determine if this process is capable of meeting specifications from 490 to 510"
"Apply Western Electric rules to detect non-random patterns in my control chart data"
"Use CUSUM to detect small shifts in compressor efficiency over 90 days"
"Apply EWMA chart with lambda=0.2 to smooth noisy pH measurements"
"Calculate Cp and Cpk indices for this manufacturing process"
"Check if there are any out-of-control points in my X-bar chart data"
```

**SPC Use Cases:**
- Manufacturing quality control and Six Sigma programs
- Process monitoring and stability assessment
- Supplier qualification and process validation
- Regulatory compliance (ISO 9001, FDA)
- Early detection of process shifts and trends
- Capability analysis and continuous improvement

### Unit Converter
```
"Convert 100 kilometers to miles"
"How many celsius is 75 fahrenheit?"
"Convert 2 gigabytes to megabytes"
"What is 5 feet in meters?"
"Convert 1 gallon to liters"
"How many seconds in 2 hours?"
"Convert 60 miles per hour to kilometers per hour"
```

### Date Calculator
```
"How many days until Christmas 2025?"
"What date is 90 days from today?"
"How many business days between January 1 and January 31, 2025?"
"How old is someone born on January 1, 2000?"
"What day of the week was January 1, 2000?"
"Calculate the difference between 2025-01-01 and 2025-12-31 in weeks"
"Add 6 months to 2025-01-31"
```

### Text Processing Tools
```
"Analyze this text: 'Hello world! This is a test.'"
"What are the most common words in this document?"
"Convert 'Hello World' to base64"
"Transform this text to camelCase"
"Calculate statistics for this paragraph"
"Encode this URL with percent encoding"
"Reverse the words in this sentence"
"Get word frequency for top 20 words, skip common words"
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

### Tool: `descriptive_stats`

**Parameters:**
- `data` (array of numbers, required): Dataset to analyze (1-10000 items)

**Response Format:**
- `mean`: Arithmetic average (μ = Σx / n)
- `median`: Middle value when sorted
- `mode`: Most frequent value(s)
- `range`: Difference between max and min
- `variance`: Average of squared deviations (σ² = Σ(x - μ)² / n)
- `std_dev`: Square root of variance (σ)
- `count`: Number of data points
- `min`: Minimum value
- `max`: Maximum value

**Examples:**
```
[23, 45, 12, 67, 34, 89, 23, 56] → mean: 43.625, median: 39.5, mode: 23, std_dev: 24.25
[1, 2, 3, 4, 5] → mean: 3.0, median: 3, variance: 2.0
```

**Algorithm:** Uses pure Python calculations with O(n log n) complexity for sorting (median)

**Error Handling:**
- Empty or invalid data
- Non-numeric values
- Data size out of range

### Tool: `correlation`

**Parameters:**
- `x` (array of numbers, required): First dataset (2-1000 items)
- `y` (array of numbers, required): Second dataset (must be same length as x)

**Response Format:**
- `coefficient`: Pearson correlation coefficient (r, range: -1 to 1)
- `interpretation`: Human-readable interpretation

**Formula:** r = Σ[(x - x̄)(y - ȳ)] / √[Σ(x - x̄)² × Σ(y - ȳ)²]

**Interpretation:**
- r = 1.0: Perfect positive correlation
- 0.7 ≤ r < 1.0: Strong positive correlation
- 0.4 ≤ r < 0.7: Moderate positive correlation
- -0.1 < r < 0.1: No correlation
- -1.0 < r ≤ -0.7: Strong negative correlation
- r = -1.0: Perfect negative correlation

**Examples:**
```
[1, 2, 3, 4, 5] and [2, 4, 6, 8, 10] → r = 1.0 (perfect positive)
[1, 2, 3, 4, 5] and [5, 4, 3, 2, 1] → r = -1.0 (perfect negative)
```

**Algorithm:** Calculates covariance and standard deviations in O(n) time

**Error Handling:**
- Arrays of different lengths
- Non-numeric values
- Zero variance (all values identical)
- Array size out of range

### Tool: `percentile`

**Parameters:**
- `data` (array of numbers, required): Dataset (1-10000 items)
- `percentile` (number, required): Percentile to calculate (0-100)

**Response Format:**
- Numeric value at the specified percentile

**Formula:** Uses linear interpolation method:
- k = (p / 100) × (n - 1)
- If k is integer, return data[k]
- Otherwise, interpolate between floor(k) and ceil(k)

**Special Percentiles:**
- 0th: Minimum value
- 25th: First quartile (Q1)
- 50th: Median (Q2)
- 75th: Third quartile (Q3)
- 100th: Maximum value

**Examples:**
```
[15, 20, 35, 40, 50], 50th → 35 (median)
[1, 2, 3, 4, 5, 6, 7, 8, 9, 10], 90th → 9.1
```

**Algorithm:** Sorts data and uses linear interpolation, O(n log n) complexity

**Error Handling:**
- Invalid percentile value (outside 0-100)
- Non-numeric data
- Empty dataset

### Tool: `detect_outliers`

**Parameters:**
- `data` (array of numbers, required): Dataset to analyze (4-10000 items)
- `threshold` (number, optional, default: 1.5): IQR multiplier (0.1-10, use 1.5 for standard outliers, 3.0 for extreme outliers)

**Response Format:**
- `outliers`: List of outlier values
- `indices`: Indices of outliers in original data
- `count`: Number of outliers
- `lower_bound`: Lower threshold (Q1 - threshold×IQR)
- `upper_bound`: Upper threshold (Q3 + threshold×IQR)
- `q1`: First quartile (25th percentile)
- `q3`: Third quartile (75th percentile)
- `iqr`: Interquartile range (Q3 - Q1)
- `threshold`: The threshold multiplier used

**Method:** IQR (Interquartile Range) method with configurable threshold
- Calculate Q1 and Q3
- IQR = Q3 - Q1
- Lower bound = Q1 - threshold × IQR
- Upper bound = Q3 + threshold × IQR
- Values outside [lower_bound, upper_bound] are outliers

**Examples:**
```
detect_outliers([10, 12, 14, 13, 15, 100, 11, 13, 14]) → outliers: [100], count: 1
detect_outliers([10, 11, 12, 13, 14, 15]) → outliers: [], count: 0 (no outliers)
detect_outliers([10, 12, 14, 13, 15, 100, 11, 13, 14], threshold=3.0) → fewer/different outliers detected
```

**Algorithm:** Uses percentile calculations with O(n log n) complexity

**Error Handling:**
- Non-numeric values
- Empty dataset
- Data size out of range

### Tool: `unit_convert`

**Parameters:**
- `value` (number, required): The numeric value to convert
- `from_unit` (string, required): Source unit (case-insensitive, accepts abbreviations)
- `to_unit` (string, required): Target unit (case-insensitive, accepts abbreviations)

**Response Format:**
- Converted value with formatted result
- Conversion formula (for temperature) or factor
- Category information (length, weight, temperature, volume, time, storage, speed)

**Supported Unit Categories:**

**Length:**
- Metric: millimeter (mm), centimeter (cm), meter (m), kilometer (km)
- Imperial: inch (in), foot (ft), yard (yd), mile (mi)
- Uses exact conversion factors (1 inch = 2.54 cm exactly)

**Weight/Mass:**
- Metric: milligram (mg), gram (g), kilogram (kg), tonne (t)
- Imperial: ounce (oz), pound (lb), ton (US short ton)
- Precise factors from international standards

**Temperature:**
- Celsius (C), Fahrenheit (F), Kelvin (K)
- Uses conversion formulas, not factors
- Displays the formula used (e.g., "C × 9/5 + 32")

**Volume:**
- Metric: milliliter (ml), liter (l), cubic meter (m3)
- Imperial: fluid ounce (fl oz), cup, pint (pt), quart (qt), gallon (gal)
- US liquid measures

**Time:**
- millisecond (ms), second (s), minute (min), hour (h)
- day (d), week, year (365 days)

**Digital Storage:**
- bit, byte (B), kilobyte (KB), megabyte (MB), gigabyte (GB), terabyte (TB)
- Uses binary (1024-based) not decimal (1000-based)
- Standard computer storage convention

**Speed:**
- meters per second (m/s), kilometers per hour (km/h), miles per hour (mph)

**Examples:**
```
100 km → 62.137 mi
75°F → 23.89°C (Formula: (F - 32) × 5/9)
2 GB → 2048 MB
5 ft → 1.524 m
1 gallon → 3.785 liters
2 hours → 7200 seconds
60 mph → 96.561 km/h
```

**Features:**
- Case-insensitive unit names
- Accepts both full names and abbreviations
- Validates units are in the same category
- Shows conversion formula for temperature
- Shows conversion factor for other categories
- Appropriate precision for results
- Exact conversion factors (1 inch = 2.54 cm, 1 lb = 453.59237 g)

**Algorithm:** O(1) - direct calculation using lookup tables and formulas

**Error Handling:**
- Unknown units with helpful message
- Attempt to convert between different categories
- Invalid parameter types
- Missing required parameters

### Tool: `date_diff`

**Parameters:**
- `date1` (string, required): First date in ISO format (YYYY-MM-DD)
- `date2` (string, required): Second date in ISO format (YYYY-MM-DD)
- `unit` (string, optional, default: "all"): Unit of difference
  - `days`: Total days between dates
  - `weeks`: Weeks and remaining days
  - `months`: Total months
  - `years`: Years, months, and days
  - `all`: Complete breakdown in all units

**Response Format:**
- Formatted difference string with detailed breakdown
- Total days count
- Human-readable summary

**Examples:**
```
date_diff("2025-01-01", "2025-12-31", "days") → 364 days
date_diff("2025-01-01", "2025-12-31", "all") → "0 years, 11 months, 30 days"
date_diff("2020-01-01", "2025-01-01", "years") → "5 years"
```

**Algorithm:** Uses relativedelta for accurate month/year calculations with proper leap year handling

**Error Handling:**
- Invalid ISO date format
- Unrecognized unit parameter
- Missing required parameters

### Tool: `date_add`

**Parameters:**
- `date` (string, required): Starting date in ISO format (YYYY-MM-DD)
- `amount` (integer, required): Amount to add (negative to subtract)
- `unit` (string, required): Time unit
  - `days`: Add/subtract days
  - `weeks`: Add/subtract weeks
  - `months`: Add/subtract months (handles month-end dates)
  - `years`: Add/subtract years (handles leap years)

**Response Format:**
- Original date
- New date after calculation
- Formatted calculation string

**Examples:**
```
date_add("2025-01-15", 30, "days") → "2025-02-14"
date_add("2025-01-31", 1, "months") → "2025-02-28" (month-end handling)
date_add("2024-02-29", 1, "years") → "2025-02-28" (leap year handling)
date_add("2025-06-15", -90, "days") → "2025-03-17" (subtraction)
```

**Algorithm:** Uses relativedelta for months/years and timedelta for days/weeks

**Error Handling:**
- Invalid ISO date format
- Unrecognized unit parameter
- Invalid amount type
- Missing required parameters

### Tool: `business_days`

**Parameters:**
- `start_date` (string, required): Start date in ISO format (YYYY-MM-DD)
- `end_date` (string, required): End date in ISO format (YYYY-MM-DD)
- `exclude_holidays` (array, optional): List of holiday dates in ISO format to exclude

**Response Format:**
- Business day count (Monday-Friday, excluding holidays)
- Total calendar days
- Weekend days count
- Holidays excluded count
- Formatted summary

**Examples:**
```
business_days("2025-01-06", "2025-01-10") → 5 business days (Mon-Fri)
business_days("2025-01-04", "2025-01-11") → 6 business days (Sat-Sat, excludes weekends)
business_days("2025-12-22", "2025-12-26", ["2025-12-25"]) → 3 business days (excluding Christmas)
```

**Algorithm:** Iterates through date range, counting weekdays and excluding specified holidays

**Error Handling:**
- Invalid ISO date format
- Invalid holiday date format
- Missing required parameters

### Tool: `age_calculator`

**Parameters:**
- `birthdate` (string, required): Birth date in ISO format (YYYY-MM-DD)
- `reference_date` (string, optional): Reference date for calculation (default: today)

**Response Format:**
- Age in years, months, and days
- Total days lived
- Formatted age string

**Examples:**
```
age_calculator("1990-05-15") → "35 years, 6 months, 4 days old"
age_calculator("2000-01-01", "2025-01-01") → "25 years, 0 months, 0 days old"
age_calculator("2020-02-29", "2021-03-01") → "1 year, 0 months, 1 day old"
```

**Algorithm:** Uses relativedelta for accurate age calculation with leap year support

**Error Handling:**
- Invalid ISO date format
- Birthdate in the future
- Missing required parameters

### Tool: `day_of_week`

**Parameters:**
- `date` (string, required): Date in ISO format (YYYY-MM-DD)

**Response Format:**
- Day name (e.g., "Wednesday")
- ISO week number
- Day of year (1-366)
- ISO year
- Weekend indicator
- Formatted summary

**Examples:**
```
day_of_week("2025-11-19") → "Wednesday, Week 47, Day 323"
day_of_week("2000-01-01") → "Saturday, Week 52, Day 1"
day_of_week("2024-12-25") → "Wednesday, Week 52, Day 360"
```

**Algorithm:** Uses Python's datetime methods to extract calendar information

**Error Handling:**
- Invalid ISO date format
- Missing required parameters

### Tool: `text_stats`

**Parameters:**
- `text` (string, required): Text to analyze (up to 100KB)

**Response Format:**
- Dictionary containing:
  - `characters`: Total character count
  - `characters_no_spaces`: Character count excluding spaces
  - `words`: Word count
  - `sentences`: Sentence count
  - `paragraphs`: Paragraph count
  - `avg_word_length`: Average word length in characters
  - `avg_sentence_length`: Average sentence length in words
  - `reading_time`: Estimated reading time (based on 200 words/min)

**Examples:**
```
text_stats("Hello world! This is a test.") → 
{
  "characters": 30,
  "characters_no_spaces": 25,
  "words": 6,
  "sentences": 2,
  "paragraphs": 1,
  "avg_word_length": 3.33,
  "avg_sentence_length": 3.0,
  "reading_time": "2 seconds"
}
```

**Algorithm:** 
- Uses regex for word and sentence detection
- Counts paragraphs by blank line separation
- Handles Unicode text properly

**Error Handling:**
- Text exceeds 100KB size limit
- Missing required parameters

### Tool: `word_frequency`

**Parameters:**
- `text` (string, required): Text to analyze
- `top_n` (integer, optional, default: 10): Number of most frequent words to return
- `skip_common` (boolean, optional, default: false): Skip common English words

**Response Format:**
- List of [word, count] pairs sorted by frequency (descending)

**Examples:**
```
word_frequency("The cat sat on the mat. The cat was happy.", top_n=3)
→ [["the", 3], ["cat", 2], ["sat", 1]]

word_frequency("The cat sat on the mat", top_n=5, skip_common=true)
→ [["cat", 1], ["mat", 1], ["sat", 1]]
```

**Features:**
- Case-insensitive analysis
- Removes punctuation automatically
- Optional filtering of 50+ common English words (the, a, is, etc.)

**Algorithm:**
- Converts to lowercase
- Extracts words using regex
- Counts frequencies with dictionary
- Sorts by count (descending) and alphabetically for ties

**Error Handling:**
- Invalid top_n value (< 1)
- Missing required parameters

### Tool: `text_transform`

**Parameters:**
- `text` (string, required): Text to transform
- `operation` (string, required): Transformation operation

**Operations:**
- `uppercase`: Convert to UPPERCASE
- `lowercase`: Convert to lowercase
- `titlecase`: Convert To Title Case
- `camelcase`: convertToCamelCase
- `snakecase`: convert_to_snake_case
- `reverse`: esreveR txet (reverse characters)
- `words_reverse`: Reverse word order
- `remove_spaces`: Removeallspaces
- `remove_punctuation`: Remove all punctuation

**Examples:**
```
text_transform("Hello World", "uppercase") → "HELLO WORLD"
text_transform("Hello World", "camelcase") → "helloWorld"
text_transform("Hello World", "snakecase") → "hello_world"
text_transform("Hello World", "reverse") → "dlroW olleH"
text_transform("Hello World!", "words_reverse") → "World! Hello"
```

**Algorithm:**
- Direct string manipulation for simple cases
- Regex for complex transformations (camelCase, snake_case)
- Word-based operations split and rejoin text

**Error Handling:**
- Invalid operation name
- Missing required parameters

### Tool: `encode_decode`

**Parameters:**
- `text` (string, required): Text to encode or decode
- `operation` (string, required): 'encode' or 'decode'
- `format` (string, required): Encoding format - 'base64', 'hex', or 'url'

**Response Format:**
- Encoded or decoded text string

**Formats:**
- **base64**: Standard Base64 encoding (RFC 4648)
  - Uses UTF-8 encoding for text
  - Safe for binary data representation
- **hex**: Hexadecimal encoding (lowercase)
  - Each byte becomes 2 hex characters
  - Commonly used in hashes and binary data
- **url**: URL percent-encoding (RFC 3986)
  - Encodes special characters as %XX
  - Safe for URLs and query strings

**Examples:**
```
encode_decode("Hello World", "encode", "base64") → "SGVsbG8gV29ybGQ="
encode_decode("SGVsbG8gV29ybGQ=", "decode", "base64") → "Hello World"
encode_decode("Hello World", "encode", "hex") → "48656c6c6f20576f726c64"
encode_decode("Hello World!", "encode", "url") → "Hello%20World%21"
```

**Use Cases:**
- Base64: Email attachments, embedding data in JSON/XML
- Hex: Hash representations, color codes, binary debugging
- URL: Query parameters, path encoding in URLs

**Algorithm:**
- Uses Python standard library (base64, urllib.parse)
- UTF-8 encoding/decoding for text
- Proper error handling for invalid input

**Error Handling:**
- Invalid operation ('encode' or 'decode')
- Invalid format ('base64', 'hex', or 'url')
- Malformed encoded input during decode
- Invalid UTF-8 sequences

## 📁 Project Structure

```
MyMCP/
├── src/
│   ├── math_server/             # Mathematical tools MCP server
│   │   ├── __init__.py          # Package initialization
│   │   └── server.py            # Math calculator server implementation
│   ├── stats_server/            # Statistical analysis MCP server
│   │   ├── __init__.py          # Package initialization
│   │   └── server.py            # Statistical tools server implementation
│   └── config.py                # Configuration management module
├── venv/                        # Virtual environment (created during setup)
├── ssl/                         # SSL certificates directory (for Docker)
├── .dockerignore                # Docker build exclusions
├── .env.example                 # Environment variable template
├── .gitignore                   # Git ignore patterns
├── config.example.yaml          # Example configuration file
├── config.yaml                  # User configuration (not in git)
├── docker-compose.yml           # Docker Compose orchestration
├── Dockerfile                   # Docker image definition
├── nginx.conf                   # Nginx reverse proxy configuration
├── requirements.txt             # Python dependencies
├── mcp-config.json             # MCP Inspector config (for testing)
└── README.md                    # This file
```

## 🧪 Testing

### Using MCP Inspector

The MCP Inspector is a web-based tool for testing MCP servers:

```bash
# Test Math Calculator Server
npx @modelcontextprotocol/inspector c:/path/to/venv/Scripts/python.exe c:/path/to/src/math_server/server.py

# Test Statistical Analysis Server
npx @modelcontextprotocol/inspector c:/path/to/venv/Scripts/python.exe c:/path/to/src/stats_server/server.py
```

This will:
1. Start the MCP Inspector web interface
2. Launch the selected server
3. Open a browser at `http://localhost:6274`
4. Allow you to test tool calls interactively

### Manual Testing

You can also test the functions directly in Python:

**Math Server Functions:**
```python
from src.math_server.server import (
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
    generate_hash,
    unit_convert,
    date_diff,
    date_add,
    business_days,
    age_calculator,
    day_of_week,
    text_stats,
    word_frequency,
    text_transform,
    encode_decode
)

# Test examples from math server...
```

**Stats Server Functions:**
```python
from src.stats_server.server import (
    descriptive_stats,
    correlation,
    percentile,
    detect_outliers
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

# Statistical analysis tools (from stats_server)
print(descriptive_stats([23, 45, 12, 67, 34, 89, 23, 56]))  
# Output: {'mean': 43.625, 'median': 39.5, 'mode': [23], 'range': 77, 'variance': 587.98, 'std_dev': 24.25, ...}
print(correlation([1, 2, 3, 4, 5], [2, 4, 6, 8, 10]))  
# Output: {'coefficient': 1.0, 'interpretation': 'Perfect positive correlation'}
print(percentile([23, 45, 12, 67, 34, 89, 23, 56], 50))  # Output: 39.5 (median)
print(detect_outliers([10, 12, 14, 13, 15, 100, 11, 13, 14], threshold=1.5))  
# Output: {'outliers': [100], 'indices': [5], 'count': 1, 'q1': 12.0, 'q3': 14.0, 'iqr': 2.0, ...}

# Unit converter
print(unit_convert(100, "km", "mi"))  # Output: {'result': 62.137..., 'formatted': '100 km = 62.14 mi', ...}
print(unit_convert(75, "F", "C"))  # Output: {'result': 23.89, 'formatted': '75°F = 23.89°C (Formula: (F - 32) × 5/9)', ...}
print(unit_convert(2, "GB", "MB"))  # Output: {'result': 2048, 'formatted': '2 GB = 2048 MB', ...}
print(unit_convert(1, "gallon", "liter"))  # Output: {'result': 3.785..., 'formatted': '1 gallon = 3.785 liter', ...}

# Date calculator
print(date_diff("2025-01-01", "2025-12-31", "days"))  # Output: {'days': 364, 'formatted': '364 days'}
print(date_diff("2025-01-01", "2025-12-31", "all"))  # Output: {'years': 0, 'months': 11, 'days': 30, ...}
print(date_add("2025-01-15", 30, "days"))  # Output: {'new_date': '2025-02-14', ...}
print(date_add("2025-01-31", 1, "months"))  # Output: {'new_date': '2025-02-28', ...}
print(business_days("2025-01-06", "2025-01-10"))  # Output: {'business_days': 5, ...}
print(age_calculator("2000-01-01", "2025-01-01"))  # Output: {'years': 25, 'months': 0, 'days': 0, ...}
print(day_of_week("2000-01-01"))  # Output: {'day_name': 'Saturday', 'week_number': 52, ...}

# Text processing tools
print(text_stats("Hello world! This is a test."))
# Output: {'characters': 30, 'words': 6, 'sentences': 2, 'paragraphs': 1, ...}

print(word_frequency("The cat sat on the mat. The cat was happy.", top_n=3))
# Output: [['the', 3], ['cat', 2], ['sat', 1]]

print(text_transform("Hello World", "camelcase"))  # Output: 'helloWorld'
print(text_transform("Hello World", "snakecase"))  # Output: 'hello_world'
print(text_transform("Hello World", "reverse"))    # Output: 'dlroW olleH'

print(encode_decode("Hello World", "encode", "base64"))  # Output: 'SGVsbG8gV29ybGQ='
print(encode_decode("SGVsbG8gV29ybGQ=", "decode", "base64"))  # Output: 'Hello World'
print(encode_decode("Hello World", "encode", "hex"))     # Output: '48656c6c6f20576f726c64'
print(encode_decode("Hello World!", "encode", "url"))    # Output: 'Hello%20World%21'
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

## 🔒 Security Best Practices

When deploying MCP servers in production or exposing them over HTTP, follow these security best practices:

### API Key Management

✅ **DO**:
- Generate strong, random API keys (minimum 32 characters)
- Use a password manager or key generation tool (e.g., `openssl rand -hex 32`)
- Store API keys in environment variables, never in code
- Use different API keys for different environments (dev, staging, production)
- Rotate API keys regularly (e.g., every 90 days)
- Use `.gitignore` to prevent committing `config.yaml` with real keys

❌ **DON'T**:
- Never commit API keys to version control
- Don't use simple or guessable keys like "password123" or "my-api-key"
- Don't share API keys in chat, email, or documentation
- Don't use the same key across multiple services
- Don't disable authentication in production

**Example - Generate Strong API Key**:
```bash
# Linux/Mac
openssl rand -hex 32

# Python
python -c "import secrets; print(secrets.token_urlsafe(32))"

# PowerShell (using cryptographically secure random)
-join ((1..32) | ForEach-Object { "{0:X2}" -f (Get-Random -Minimum 0 -Maximum 256) })

# Alternative: Use Python on Windows for consistency
python -c "import secrets; print(secrets.token_urlsafe(32))"
```

### HTTPS in Production

⚠️ **Never use HTTP in production** - always use HTTPS:

- **Development**: HTTP is acceptable for `localhost` or private networks
- **Production**: Always use HTTPS with valid SSL certificates
- **Codespaces**: HTTPS is automatic (GitHub handles SSL)
- **Docker**: Configure nginx with SSL certificates (see Docker section)

**Why HTTPS**:
- Encrypts API keys in transit (prevents interception)
- Prevents man-in-the-middle attacks
- Required for modern web security standards
- Enables secure CORS from web browsers

### CORS Configuration

Configure CORS origins restrictively to prevent unauthorized cross-origin requests:

```yaml
# Bad - allows all origins (security risk)
cors_origins:
  - "*"

# Good - specific origins only
cors_origins:
  - "http://localhost:3000"           # Local development
  - "https://app.example.com"         # Production app
  - "https://*.app.github.dev"        # Codespaces only
```

**CORS Security Rules**:
1. Never use `"*"` in production
2. Use specific domains and protocols
3. Use wildcards sparingly (e.g., `*.app.github.dev` for Codespaces)
4. Review CORS origins regularly
5. Remove unused origins

### Rate Limiting

Enable rate limiting to prevent abuse and DoS attacks:

```yaml
rate_limiting:
  enabled: true
  requests_per_minute: 60  # Adjust based on your needs
```

**Recommended Limits**:
- **Development**: 120-300 requests/minute (liberal for testing)
- **Production API**: 60-120 requests/minute (1-2 requests/second)
- **Public endpoints**: 30-60 requests/minute (more restrictive)

**Monitor `/metrics` endpoint** for unusual activity:
- Spike in requests from single IP
- High rate of 401 (authentication failures)
- Pattern of 429 (rate limit exceeded)

### Network Security

**Firewall Rules**:
```bash
# Allow only necessary ports
sudo ufw allow 22/tcp      # SSH
sudo ufw allow 443/tcp     # HTTPS
sudo ufw deny 8000/tcp     # Deny direct access to app ports
sudo ufw enable
```

**Bind to Localhost** when not needed remotely:
```yaml
server:
  math:
    host: "127.0.0.1"  # Localhost only, not accessible remotely
    port: 8000
```

**Use Reverse Proxy** (nginx, Caddy) for production:
- Terminates SSL/TLS
- Handles rate limiting at network level
- Provides additional security headers
- Hides internal server details

### Environment Variables for Production

**Never store secrets in `config.yaml` for production**. Use environment variables:

```bash
# .env file (never commit to git)
MCP_AUTH_ENABLED=true
MCP_API_KEY=your-strong-randomly-generated-api-key-here
MCP_RATE_LIMIT_ENABLED=true
MCP_RATE_LIMIT_RPM=60
MCP_LOG_LEVEL=INFO
```

**Load environment variables**:
```bash
# Linux/Mac
source .env

# Or use docker-compose with .env file
docker-compose up -d
```

### Monitoring and Alerting

Monitor these endpoints for security issues:

1. **`/metrics`** - Track request rates and patterns
2. **`/health`** - Ensure server is operational
3. **Server logs** - Watch for authentication failures

**Set up alerts for**:
- High rate of 401 responses (unauthorized attempts)
- Unusual spike in requests
- Server downtime or health check failures
- High error rates (5xx responses)

### Security Checklist

Before deploying to production:

- [ ] Authentication enabled (`MCP_AUTH_ENABLED=true`)
- [ ] Strong API key generated and stored securely
- [ ] HTTPS configured (not HTTP)
- [ ] CORS origins configured restrictively (no `*`)
- [ ] Rate limiting enabled with appropriate limits
- [ ] Firewall rules configured
- [ ] Environment variables used for secrets (not config file)
- [ ] Server logs monitored regularly
- [ ] `.gitignore` includes `config.yaml` and `.env`
- [ ] API keys documented in password manager
- [ ] Incident response plan documented
- [ ] Regular security audits scheduled

### Additional Resources

- [SECURITY.md](SECURITY.md) - Detailed security features documentation
- [OWASP API Security Top 10](https://owasp.org/www-project-api-security/)
- [GitHub Security Best Practices](https://docs.github.com/en/code-security)

## 🔧 Troubleshooting

Comprehensive troubleshooting guide for common issues with MCP servers.

### HTTP Transport Issues

#### Connection Refused

**Symptoms**: `curl: (7) Failed to connect to localhost port 8000: Connection refused`

**Solutions**:
1. **Check if server is running**:
   ```bash
   # Check process
   ./start-http-servers.sh status
   
   # Or check ports
   lsof -i :8000    # Mac/Linux
   netstat -ano | findstr :8000  # Windows
   ```

2. **Start the server**:
   ```bash
   ./start-http-servers.sh start
   ```

3. **Check server logs**:
   ```bash
   cat logs/math_server.log
   tail -f logs/math_server.log  # Follow logs in real-time
   ```

4. **Verify port not in use by another process**:
   ```bash
   # Kill process on port 8000 (if needed)
   # Try graceful shutdown first (SIGTERM)
   lsof -ti:8000 | xargs kill
   
   # Wait 5 seconds for graceful shutdown
   sleep 5
   
   # If process still running, force kill as last resort (SIGKILL)
   lsof -ti:8000 | xargs kill -9
   ```

#### CORS Errors in Browser

**Symptoms**: 
- Browser console shows: `Access to fetch at 'http://localhost:8000' from origin 'http://localhost:3000' has been blocked by CORS policy`
- HTTP error: `No 'Access-Control-Allow-Origin' header`

**Solutions**:

1. **Add your origin to config.yaml**:
   ```yaml
   server:
     cors_origins:
       - "http://localhost:3000"
       - "http://localhost:*"  # All localhost ports
   ```

2. **For development, allow all localhost ports**:
   ```yaml
   cors_origins:
     - "http://localhost:*"
     - "http://127.0.0.1:*"
   ```

3. **Restart server after config change**:
   ```bash
   ./start-http-servers.sh restart
   ```

4. **Verify CORS headers in response**:
   ```bash
   curl -i -H "Origin: http://localhost:3000" http://localhost:8000/health
   # Look for: Access-Control-Allow-Origin: http://localhost:3000
   ```

#### 401 Unauthorized

**Symptoms**: HTTP 401 response with `{"error": "Invalid or missing API key"}`

**Solutions**:

1. **Check if authentication is enabled**:
   ```bash
   # Look in config.yaml
   grep -A 2 "authentication:" config.yaml
   ```

2. **Include Authorization header**:
   ```bash
   curl -H "Authorization: Bearer your-api-key" \
        http://localhost:8000/messages \
        -d '{"jsonrpc":"2.0","method":"tools/list","params":{},"id":1}'
   ```

3. **Verify API key matches config**:
   ```bash
   # Check config file
   grep "api_key:" config.yaml
   
   # Or check environment variable
   echo $MCP_API_KEY
   ```

4. **Disable authentication for testing** (development only):
   ```yaml
   authentication:
     enabled: false
   ```

#### 429 Rate Limit Exceeded

**Symptoms**: HTTP 429 response with `{"error": "Rate limit exceeded"}`

**Solutions**:

1. **Check rate limit settings**:
   ```yaml
   rate_limiting:
     enabled: true
     requests_per_minute: 60
   ```

2. **Increase rate limit** (if appropriate):
   ```yaml
   rate_limiting:
     requests_per_minute: 120  # 2 requests/second
   ```

3. **Wait before retrying**:
   - Check `Retry-After` header in 429 response
   - Implement exponential backoff in client

4. **Disable rate limiting for testing** (development only):
   ```yaml
   rate_limiting:
     enabled: false
   ```

5. **Check metrics for request pattern**:
   ```bash
   curl http://localhost:8000/metrics
   ```

#### SSE Connection Drops

**Symptoms**: Server-Sent Events connection closes unexpectedly

**Solutions**:

1. **Check network stability**:
   - Wi-Fi connection stable
   - No firewall blocking persistent connections
   - Proxy/VPN not interfering

2. **Verify keepalive is working**:
   - Server sends ping every 30 seconds
   - Check client receives ping events

3. **Increase client timeout**:
   ```javascript
   // JavaScript EventSource
   const evtSource = new EventSource("http://localhost:8000/sse");
   evtSource.onerror = (err) => {
       console.error("EventSource error:", err);
       // Implement reconnection logic
   };
   ```

4. **Check server logs for connection errors**:
   ```bash
   grep "SSE" logs/math_server.log
   ```

### Codespaces Issues

#### Port Not Forwarding

**Solutions**:

1. **Manually forward port**:
   - Open **PORTS** tab in VS Code terminal
   - Click **Forward a Port**
   - Enter port number (8000 or 8001)

2. **Check `.devcontainer/devcontainer.json`**:
   ```json
   "forwardPorts": [8000, 8001]
   ```

3. **Verify servers are running**:
   ```bash
   ./start-http-servers.sh status
   ps aux | grep python
   ```

#### 404 Not Found on Codespaces URL

**Solutions**:

1. **Verify correct URL format**:
   ```
   https://<codespace-name>-8000.app.github.dev/health
   ```

2. **Check port number in URL matches server port**

3. **Ensure path is correct** (e.g., `/health`, not `/health/`)

4. **Check server is listening on 0.0.0.0**:
   ```yaml
   server:
     math:
       host: "0.0.0.0"  # Not 127.0.0.1
   ```

#### Codespaces URL Returns 502 Bad Gateway

**Solutions**:

1. **Server not started yet**:
   ```bash
   ./start-http-servers.sh start
   ```

2. **Server crashed - check logs**:
   ```bash
   cat logs/math_server.log
   ```

3. **Port mismatch - verify configuration**:
   ```bash
   grep "port:" config.yaml
   ```

### stdio Mode Issues (Claude Desktop)

#### Server Not Appearing in Claude

**Solutions**:

1. **Verify config file location**:
   - **Windows**: `%APPDATA%\Claude\claude_desktop_config.json`
   - **macOS**: `~/Library/Application Support/Claude/claude_desktop_config.json`
   - **Linux**: `~/.config/Claude/claude_desktop_config.json`

2. **Check JSON syntax**:
   ```bash
   # Validate JSON
   python -m json.tool claude_desktop_config.json
   ```

3. **Use absolute paths**:
   ```json
   {
     "mcpServers": {
       "math-server": {
         "command": "/absolute/path/to/venv/bin/python",
         "args": ["/absolute/path/to/src/math_server/server.py"]
       }
     }
   }
   ```

4. **Restart Claude Desktop completely**:
   - Quit Claude Desktop (not just close window)
   - On Windows: Check Task Manager and end process if needed
   - Relaunch Claude Desktop

5. **Check Claude logs**:
   - **Windows**: `%APPDATA%\Claude\logs\main.log`
   - Look for MCP-related errors

#### Import Errors

**Symptoms**: `ModuleNotFoundError: No module named 'mcp'`

**Solutions**:

1. **Verify virtual environment activated**:
   ```bash
   which python  # Should show venv path
   ```

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Use correct Python path in config**:
   ```bash
   # Find venv Python
   which python  # Mac/Linux
   where python  # Windows
   ```

### Configuration Issues

#### Config File Not Loaded

**Symptoms**: Server uses default values instead of config.yaml

**Solutions**:

1. **Specify config file explicitly**:
   ```bash
   python src/math_server/server.py --config config.yaml
   ```

2. **Check file exists**:
   ```bash
   ls -la config.yaml
   ```

3. **Verify YAML syntax**:
   ```bash
   python -c "import yaml; yaml.safe_load(open('config.yaml'))"
   ```

4. **Check file permissions**:
   ```bash
   chmod 644 config.yaml
   ```

#### Environment Variables Not Working

**Solutions**:

1. **Verify export syntax**:
   ```bash
   export MCP_API_KEY="your-key"
   echo $MCP_API_KEY  # Should print your key
   ```

2. **Use correct variable names** (with `MCP_` prefix):
   ```bash
   MCP_AUTH_ENABLED=true  # Correct
   AUTH_ENABLED=true      # Wrong - missing MCP_ prefix
   ```

3. **Load .env file**:
   ```bash
   source .env  # Or use python-dotenv
   ```

### Performance Issues

#### High Memory Usage

**Solutions**:

1. **Check for large datasets in tool calls**
2. **Reduce logging level**:
   ```yaml
   logging:
     level: "WARNING"  # Instead of DEBUG
   ```
3. **Monitor with metrics**:
   ```bash
   curl http://localhost:8000/metrics
   ```

#### Slow Response Times

**Solutions**:

1. **Check CPU usage**:
   ```bash
   top -p $(pgrep -f math_server)
   ```

2. **Enable hot-reload only in development**:
   ```bash
   # Don't use --dev in production
   python src/math_server/server.py --transport http
   ```

3. **Consider scaling** (run multiple instances behind load balancer)

### Docker Issues

#### Container Won't Start

**Solutions**:

1. **Check Docker logs**:
   ```bash
   docker-compose logs math-server
   docker-compose logs stats-server
   ```

2. **Verify ports not in use**:
   ```bash
   docker-compose ps
   lsof -i :8000
   ```

3. **Rebuild containers**:
   ```bash
   docker-compose down
   docker-compose up --build
   ```

#### Can't Access Docker Services

**Solutions**:

1. **Check container is running**:
   ```bash
   docker-compose ps
   ```

2. **Test from inside container**:
   ```bash
   docker exec mcp-math-server curl http://localhost:8000/health
   ```

3. **Check network configuration**:
   ```bash
   docker network inspect mymcp_mcp-network
   ```

### Getting Help

If you're still experiencing issues:

1. **Check logs** (most important):
   ```bash
   cat logs/math_server.log
   cat logs/stats_server.log
   ```

2. **Enable DEBUG logging**:
   ```yaml
   logging:
     level: "DEBUG"
   ```

3. **Test with curl** (isolate client issues):
   ```bash
   curl -v http://localhost:8000/health
   ```

4. **Create GitHub issue** with:
   - Exact error message
   - Log file contents (remove API keys!)
   - Steps to reproduce
   - Environment details (OS, Python version)

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

- `mcp>=1.10.0` - Model Context Protocol SDK
- `pydantic>=2.0.0` - Data validation and settings management
- `pydantic-settings>=2.0.0` - Settings management with environment variable support
- `pyyaml>=6.0` - YAML configuration file parsing
- `python-dotenv>=1.0.0` - Environment variable management
- `python-dateutil>=2.8.0` - Date and time calculations with timezone support
- `scipy>=1.11.0` - Scientific computing library for statistical tests (Shapiro-Wilk, Durbin-Watson, VIF)

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

- Request: "Analyze this data: [23, 45, 12, 67, 34, 89, 23, 56]"
- Expected: Mean: 43.625, Median: 39.5, Mode: 23, Std Dev: ~24.25

- Request: "What's the correlation between [1, 2, 3, 4, 5] and [2, 4, 6, 8, 10]?"
- Expected: r = 1.0 (Perfect positive correlation)

- Request: "Find the 75th percentile of [23, 45, 12, 67, 34, 89, 23, 56]"
- Expected: 58.75

- Request: "Detect outliers in [10, 12, 14, 13, 15, 100, 11, 13, 14]"
- Expected: Outliers: [100], Count: 1

- Request: "Convert 100 kilometers to miles"
- Expected: `100 km = 62.14 mi`

- Request: "How many celsius is 75 fahrenheit?"
- Expected: `75°F = 23.89°C (Formula: (F - 32) × 5/9)`

- Request: "Convert 2 gigabytes to megabytes"
- Expected: `2 GB = 2048 MB`

- Request: "Analyze this text: 'Hello world! This is a test.'"
- Expected: 6 words, 2 sentences, character counts, reading time

- Request: "What are the most common words in 'The cat sat on the mat. The cat was happy.'?"
- Expected: [["the", 3], ["cat", 2], ...]

- Request: "Convert 'Hello World' to base64"
- Expected: `SGVsbG8gV29ybGQ=`

- Request: "Transform 'Hello World' to camelCase"
- Expected: `helloWorld`

## 📄 License

MIT

## 🎓 Educational Value

This project demonstrates:
- **Multi-server MCP architecture**: Two specialized servers working together
- **Separation of concerns**: Mathematical vs. statistical domains
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
  - Statistical analysis (descriptive statistics, Pearson correlation, percentile calculation, outlier detection using IQR method)
- Mathematical computation and number theory
- Pure Python implementations without external numerical libraries

---

**Built with Python 3.12 • MCP SDK 1.21.2 • Tested on Windows ARM64**
