# MyMCP Architecture Documentation

This document provides a comprehensive overview of the MyMCP project architecture, design patterns, and implementation details.

## Table of Contents

- [System Overview](#system-overview)
- [Repository Structure](#repository-structure)
- [Core Architecture](#core-architecture)
- [Design Patterns](#design-patterns)
- [Transport Layer](#transport-layer)
- [Server Lifecycle](#server-lifecycle)
- [Tool Registry](#tool-registry)
- [Configuration System](#configuration-system)
- [Middleware Architecture](#middleware-architecture)
- [Security Model](#security-model)
- [Testing Strategy](#testing-strategy)
- [Deployment Architecture](#deployment-architecture)

## System Overview

MyMCP is a Python-based framework for building Model Context Protocol (MCP) servers. It provides:

- **Dual Transport Support**: stdio (for desktop apps) and HTTP/SSE (for web clients)
- **Extensible Framework**: Core base classes for rapid server development
- **Built-in Servers**: Production-ready math and statistics servers
- **Template System**: Skeleton templates for quick starts
- **Type Safety**: Full type hints using MCP SDK types

### High-Level Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                         MyMCP System                            │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐          │
│  │   Custom     │  │   Builtin    │  │  Templates   │          │
│  │   Servers    │  │   Servers    │  │   Skeleton   │          │
│  │              │  │              │  │              │          │
│  │  • my_server │  │  • math      │  │  • skeleton  │          │
│  │  • ...       │  │  • stats     │  │              │          │
│  └──────┬───────┘  └──────┬───────┘  └──────┬───────┘          │
│         │                 │                 │                  │
│         └─────────────────┴─────────────────┘                  │
│                           │                                    │
│         ┌─────────────────▼─────────────────┐                  │
│         │   Core Framework (BaseMCPServer)  │                  │
│         │                                   │                  │
│         │  • ToolRegistry                   │                  │
│         │  • ServerState                    │                  │
│         │  • Transport abstraction          │                  │
│         └─────────────────┬─────────────────┘                  │
│                           │                                    │
│         ┌─────────────────▼─────────────────┐                  │
│         │        MCP SDK (Official)         │                  │
│         │                                   │                  │
│         │  • Protocol types                 │                  │
│         │  • Server implementation          │                  │
│         │  • Transport layers               │                  │
│         └───────────────────────────────────┘                  │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

## Repository Structure

```
MyMCP/
├── src/
│   ├── core/                    # Core framework
│   │   ├── __init__.py
│   │   ├── mcp_server.py        # BaseMCPServer class
│   │   ├── tool_registry.py     # Tool management
│   │   ├── server_state.py      # State tracking
│   │   └── README.md
│   │
│   ├── builtin/                 # Built-in servers
│   │   ├── __init__.py
│   │   ├── math_server/         # Math calculation server
│   │   │   ├── __init__.py
│   │   │   ├── __main__.py
│   │   │   ├── server.py
│   │   │   └── tools.py
│   │   └── stats_server/        # Statistical analysis server
│   │       ├── __init__.py
│   │       ├── __main__.py
│   │       ├── server.py
│   │       └── tools.py
│   │
│   ├── custom/                  # User custom servers
│   │   ├── __init__.py
│   │   └── README.md            # Instructions for users
│   │
│   ├── templates/               # Server templates
│   │   ├── __init__.py
│   │   └── skeleton_server/     # Minimal working template
│   │       ├── __init__.py
│   │       ├── server.py
│   │       └── README.md
│   │
│   ├── config.py                # Configuration loader
│   └── middleware.py            # HTTP middleware
│
├── tests/                       # Test suite
│   ├── core/                    # Core framework tests
│   ├── builtin/                 # Built-in server tests
│   ├── custom/                  # Custom server tests
│   └── templates/               # Template tests
│
├── docs/                        # Documentation
│   ├── ARCHITECTURE.md          # This file
│   └── ADDING_CUSTOM_SERVERS.md # Developer guide
│
├── config.yaml                  # Default configuration
├── requirements.txt             # Python dependencies
├── requirements-test.txt        # Test dependencies
└── README.md                    # Project overview
```

### Directory Purpose

- **`src/core/`**: Reusable framework code that all servers inherit from
- **`src/builtin/`**: Production-ready servers maintained by the project
- **`src/custom/`**: User-created servers (gitignored for privacy)
- **`src/templates/`**: Starting points for new servers
- **`tests/`**: Comprehensive test suite mirroring src structure
- **`docs/`**: Detailed documentation

## Core Architecture

### BaseMCPServer

The foundation of all MCP servers in this project.

```python
class BaseMCPServer(ABC):
    """
    Abstract base class for MCP servers with dual transport support.
    
    Provides:
    - Automatic transport handling (stdio/HTTP)
    - Built-in monitoring endpoints
    - Tool registry management
    - Configuration loading
    - Graceful shutdown
    """
    
    # Abstract methods (must be implemented by subclasses)
    @abstractmethod
    def register_tools(self) -> None:
        """Register server-specific tools."""
        pass
    
    @abstractmethod
    def get_server_name(self) -> str:
        """Return server name."""
        pass
    
    @abstractmethod
    def get_server_version(self) -> str:
        """Return server version."""
        pass
```

#### Key Responsibilities

1. **Initialization**
   - Load configuration from YAML files
   - Set up logging
   - Initialize tool registry and server state

2. **Transport Management**
   - Abstract transport selection (stdio vs HTTP)
   - Handle client connections
   - Manage message routing

3. **Tool Management**
   - Provide tool registry
   - Handle tool invocations
   - Route requests to handlers

4. **Monitoring**
   - Expose health check endpoints
   - Track metrics (requests, uptime, etc.)
   - Provide readiness status

### Component Diagram

```
┌─────────────────────────────────────────────────────────────┐
│                    BaseMCPServer                            │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  ┌──────────────────────┐  ┌──────────────────────┐        │
│  │   Configuration      │  │   Logging            │        │
│  │   • Load YAML        │  │   • Setup logger     │        │
│  │   • Environment vars │  │   • Level control    │        │
│  └──────────────────────┘  └──────────────────────┘        │
│                                                             │
│  ┌──────────────────────┐  ┌──────────────────────┐        │
│  │   ToolRegistry       │  │   ServerState        │        │
│  │   • Register tools   │  │   • Track requests   │        │
│  │   • Get handlers     │  │   • Monitor uptime   │        │
│  │   • List tools       │  │   • Connection count │        │
│  └──────────────────────┘  └──────────────────────┘        │
│                                                             │
│  ┌──────────────────────┐  ┌──────────────────────┐        │
│  │   stdio Transport    │  │   HTTP Transport     │        │
│  │   • MCP Server       │  │   • FastAPI app      │        │
│  │   • stdin/stdout     │  │   • SSE events       │        │
│  │   • JSON-RPC         │  │   • REST endpoints   │        │
│  └──────────────────────┘  └──────────────────────┘        │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

## Design Patterns

### 1. Template Method Pattern

`BaseMCPServer` uses the Template Method pattern:

```python
class BaseMCPServer:
    def run(self, transport, host, port, dev_mode):
        """Template method - defines algorithm structure."""
        # Step 1: Initialize (common)
        self._setup()
        
        # Step 2: Register tools (subclass-specific)
        self.register_tools()  # Abstract - must override
        
        # Step 3: Start transport (conditional)
        if transport == "stdio":
            self.run_stdio_server()
        else:
            self.run_http_server(host, port, dev_mode)
```

Benefits:
- Enforces consistent server initialization
- Allows customization through abstract methods
- Reduces code duplication

### 2. Registry Pattern

`ToolRegistry` implements the Registry pattern:

```python
class ToolRegistry:
    def __init__(self):
        self._tools = {}      # Tool definitions
        self._handlers = {}   # Tool handlers
    
    def register_tool(self, tool, handler):
        """Register a tool and its handler."""
        self._tools[tool.name] = tool
        self._handlers[tool.name] = handler
    
    def get_handler(self, tool_name):
        """Retrieve handler by tool name."""
        return self._handlers.get(tool_name)
```

Benefits:
- Central tool management
- Dynamic tool registration
- Easy tool lookup

### 3. Strategy Pattern

Transport selection uses the Strategy pattern:

```python
class BaseMCPServer:
    def run(self, transport, ...):
        # Select strategy based on transport type
        if transport == "stdio":
            strategy = self._stdio_strategy
        else:
            strategy = self._http_strategy
        
        strategy.execute()
```

Benefits:
- Flexible transport switching
- Easy to add new transports
- Clean separation of concerns

### 4. Singleton Pattern (State)

`ServerState` is designed as a singleton per server:

```python
class BaseMCPServer:
    def __init__(self):
        # Single instance per server
        self.server_state = ServerState()
```

Benefits:
- Consistent state across components
- Thread-safe metrics tracking
- Easy monitoring

## Transport Layer

### stdio Transport

Used for Claude Desktop and local CLI tools.

**Flow:**
```
User Input (Claude Desktop)
       ↓
stdin (JSON-RPC message)
       ↓
MCP Server (mcp.server.stdio)
       ↓
Tool Handler
       ↓
stdout (JSON-RPC response)
       ↓
User Output (Claude Desktop)
```

**Implementation:**
```python
def run_stdio_server(self):
    """Run server using stdio transport."""
    async def _run():
        async with mcp.server.stdio.stdio_server() as (read_stream, write_stream):
            await self.app.run(
                read_stream,
                write_stream,
                InitializationOptions(
                    server_name=self.get_server_name(),
                    server_version=self.get_server_version()
                )
            )
    
    asyncio.run(_run())
```

### HTTP Transport

Used for web clients, remote access, and cloud deployments.

**Flow:**
```
HTTP Client
       ↓
POST /messages (JSON-RPC)
       ↓
FastAPI Application
       ↓
MCP Server (mcp.server.sse)
       ↓
Tool Handler
       ↓
HTTP Response (JSON-RPC)
       ↓
HTTP Client
```

**Endpoints:**

| Endpoint | Method | Purpose |
|----------|--------|---------|
| `/messages` | POST | JSON-RPC tool invocation |
| `/sse` | GET | Server-sent events stream |
| `/health` | GET | Liveness check |
| `/ready` | GET | Readiness check |
| `/metrics` | GET | Operational metrics |

**Implementation:**
```python
def run_http_server(self, host, port, dev_mode):
    """Run server using HTTP/SSE transport."""
    # Create FastAPI app
    http_app = FastAPI(title=self.get_server_name())
    
    # Add endpoints
    http_app.post("/messages")(self._handle_message)
    http_app.get("/sse")(self._handle_sse)
    http_app.get("/health")(self._health_check)
    http_app.get("/ready")(self._readiness_check)
    http_app.get("/metrics")(self._get_metrics)
    
    # Apply middleware
    http_app.add_middleware(CORSMiddleware, ...)
    
    # Run with uvicorn
    uvicorn.run(http_app, host=host, port=port)
```

## Server Lifecycle

### Startup Sequence

```
1. Configuration Loading
   ├─ Load config.yaml
   ├─ Apply environment overrides
   └─ Validate settings

2. Initialization
   ├─ Create logger
   ├─ Initialize ServerState
   ├─ Create ToolRegistry
   └─ Create MCP Server instance

3. Tool Registration
   ├─ Call register_tools()
   ├─ Define Tool objects
   ├─ Create handler functions
   └─ Register with ToolRegistry

4. Transport Setup
   ├─ stdio: Connect stdin/stdout
   └─ HTTP: Start FastAPI + uvicorn

5. Ready State
   ├─ Accept connections
   ├─ Process tool calls
   └─ Monitor health
```

### Request Processing

**stdio mode:**
```
Client sends JSON-RPC via stdin
       ↓
MCP SDK parses message
       ↓
Route to appropriate handler
       ↓
Extract tool name and arguments
       ↓
Look up handler in ToolRegistry
       ↓
Execute handler function
       ↓
Format response as CallToolResult
       ↓
MCP SDK serializes response
       ↓
Write to stdout
       ↓
Client receives response
```

**HTTP mode:**
```
Client POSTs to /messages
       ↓
FastAPI receives request
       ↓
Apply middleware (auth, rate limit, CORS)
       ↓
MCP SSE handler processes
       ↓
Extract tool name and arguments
       ↓
Look up handler in ToolRegistry
       ↓
Execute handler function
       ↓
Format response as CallToolResult
       ↓
Return HTTP 200 with JSON body
       ↓
Client receives response
```

### Shutdown Sequence

```
1. Receive SIGTERM/SIGINT
       ↓
2. Stop accepting new requests
       ↓
3. Wait for active requests (graceful timeout)
       ↓
4. Close active SSE connections
       ↓
5. Cleanup resources
       ↓
6. Exit
```

## Tool Registry

### Purpose

Manages the mapping between tool definitions and their handlers.

### Implementation

```python
class ToolRegistry:
    def __init__(self):
        self._tools: Dict[str, Tool] = {}
        self._handlers: Dict[str, Callable] = {}
    
    def register_tool(self, tool: Tool, handler: Callable):
        """Register a tool with its handler."""
        if tool.name in self._tools:
            raise ValueError(f"Tool {tool.name} already registered")
        
        self._tools[tool.name] = tool
        self._handlers[tool.name] = handler
    
    def get_handler(self, tool_name: str):
        """Get handler for a tool."""
        return self._handlers.get(tool_name)
    
    def list_tools(self) -> List[Tool]:
        """List all registered tools."""
        return list(self._tools.values())
```

### Tool Definition Structure

```python
Tool(
    name="tool_name",           # Unique identifier
    description="What it does", # Human-readable description
    inputSchema={                # JSON Schema
        "type": "object",
        "properties": {
            "param1": {
                "type": "string",
                "description": "Parameter description"
            }
        },
        "required": ["param1"]
    }
)
```

### Handler Contract

```python
async def handle_tool(self, arguments: dict) -> CallToolResult:
    """
    All handlers must:
    1. Be async functions
    2. Accept arguments dict
    3. Return CallToolResult
    4. Handle errors gracefully
    """
    try:
        # Process
        result = process(arguments)
        
        # Success
        return CallToolResult(
            content=[TextContent(type="text", text=result)],
            isError=False
        )
    except Exception as e:
        # Error
        return CallToolResult(
            content=[TextContent(type="text", text=str(e))],
            isError=True
        )
```

## Configuration System

### Configuration File Structure

```yaml
# config.yaml
server:
  name: "my-server"
  version: "1.0.0"
  log_level: "INFO"

http:
  host: "0.0.0.0"
  port: 8000
  enable_cors: true
  allowed_origins:
    - "http://localhost:3000"
    - "https://example.com"

security:
  enable_auth: true
  api_keys:
    - "key1"
    - "key2"
  rate_limit:
    enabled: true
    requests_per_minute: 60

features:
  enable_caching: true
  cache_ttl: 300
```

### Configuration Loading

```python
class Config:
    def __init__(self, config_path=None):
        # 1. Load from file
        if config_path and os.path.exists(config_path):
            with open(config_path) as f:
                self.config = yaml.safe_load(f)
        else:
            self.config = {}
        
        # 2. Override with environment variables
        for key, value in os.environ.items():
            if key.startswith("MCP_"):
                config_key = key[4:].lower()
                self.config[config_key] = value
    
    def get(self, key, default=None):
        """Get configuration value."""
        return self.config.get(key, default)
```

### Environment Variable Overrides

```bash
# Override config values with environment variables
export MCP_LOG_LEVEL=DEBUG
export MCP_PORT=8080
export MCP_API_KEY=secret123

python -m src.custom.my_server.server
```

## Middleware Architecture

### HTTP Middleware Stack

```
Client Request
       ↓
┌──────────────────┐
│  CORS Middleware │  Add CORS headers
└────────┬─────────┘
         ↓
┌──────────────────┐
│  Auth Middleware │  Validate API key
└────────┬─────────┘
         ↓
┌──────────────────┐
│ Rate Limit       │  Check request rate
│ Middleware       │
└────────┬─────────┘
         ↓
┌──────────────────┐
│ Logging          │  Log request details
│ Middleware       │
└────────┬─────────┘
         ↓
    Route Handler
         ↓
    Response
```

### CORS Configuration

```python
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure based on needs
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
```

### Authentication Middleware

```python
async def auth_middleware(request: Request, call_next):
    """Validate API key from header."""
    if not config.get("enable_auth"):
        return await call_next(request)
    
    api_key = request.headers.get("X-API-Key")
    if not api_key or api_key not in valid_keys:
        return JSONResponse(
            status_code=401,
            content={"error": "Invalid API key"}
        )
    
    return await call_next(request)
```

## Security Model

### Authentication

- **API Key based**: Simple key-based authentication for HTTP mode
- **No auth in stdio**: stdio mode runs locally, no network exposure

### Rate Limiting

```python
# Per-client rate limiting
rate_limiter = {
    "window": 60,  # seconds
    "max_requests": 100
}
```

### Input Validation

- JSON Schema validation for all tool inputs
- Type checking at runtime
- Range and format validation

### Best Practices

1. **Never log sensitive data**
2. **Validate all inputs**
3. **Use HTTPS in production**
4. **Rotate API keys regularly**
5. **Monitor for abuse**

See [SECURITY.md](../SECURITY.md) for detailed security guidelines.

## Testing Strategy

### Test Structure

```
tests/
├── core/                    # Framework tests
│   ├── test_tool_registry.py
│   ├── test_server_state.py
│   └── test_base_server.py
│
├── builtin/                 # Built-in server tests
│   ├── math_server/
│   │   └── test_math_tools.py
│   └── stats_server/
│       └── test_stats_tools.py
│
├── templates/               # Template tests
│   └── test_skeleton_server.py
│
└── conftest.py             # Shared fixtures
```

### Test Categories

1. **Unit Tests**: Individual functions and methods
2. **Integration Tests**: Server initialization and tool execution
3. **HTTP Tests**: API endpoints and middleware
4. **End-to-End Tests**: Full request/response cycles

### Running Tests

```bash
# All tests
pytest

# Specific test file
pytest tests/core/test_tool_registry.py

# With coverage
pytest --cov=src --cov-report=html

# Verbose output
pytest -v

# Stop on first failure
pytest -x
```

## Deployment Architecture

### Local Development

```
Developer Machine
├── Clone repository
├── Install dependencies
├── Run in stdio mode
└── Test with Claude Desktop
```

### Docker Deployment

```
Docker Container
├── Python runtime
├── Application code
├── Configuration
└── Exposed ports (8000-8010)
```

**docker-compose.yml:**
```yaml
services:
  math-server:
    build: .
    ports:
      - "8000:8000"
    environment:
      - MCP_LOG_LEVEL=INFO
    command: python -m src.builtin.math_server --transport http
  
  stats-server:
    build: .
    ports:
      - "8001:8001"
    command: python -m src.builtin.stats_server --transport http
```

### Cloud Deployment

```
Load Balancer
       ↓
┌──────────────────┐
│  nginx (reverse  │
│     proxy)       │
└────────┬─────────┘
         ↓
┌─────────────────────────┐
│  Multiple Server         │
│  Instances               │
│                          │
│  ┌────────┐  ┌────────┐ │
│  │Server 1│  │Server 2│ │
│  └────────┘  └────────┘ │
└─────────────────────────┘
```

### Scaling Considerations

1. **Horizontal Scaling**: Run multiple instances behind load balancer
2. **Vertical Scaling**: Increase resources per instance
3. **Caching**: Use Redis for shared state
4. **Monitoring**: Prometheus + Grafana for metrics

## Design Principles

### 1. Minimal Changes

Framework extracts common patterns without changing existing behavior.

### 2. Separation of Concerns

- **Core**: Framework code
- **Builtin**: Production servers
- **Custom**: User servers
- **Templates**: Starting points

### 3. Extensibility

Easy to add:
- New tools
- New servers
- New transports
- New middleware

### 4. Type Safety

- Full type hints
- MCP SDK types
- Runtime validation

### 5. Production Ready

- Logging
- Monitoring
- Error handling
- Graceful shutdown

## Future Enhancements

### Planned Features

- [ ] WebSocket transport
- [ ] GraphQL API
- [ ] Built-in metrics (Prometheus)
- [ ] Plugin system
- [ ] Server clustering
- [ ] Advanced caching
- [ ] Request batching
- [ ] Tool versioning

### Extension Points

1. **New Transports**: Implement transport interface
2. **New Middleware**: Add to middleware stack
3. **New Monitoring**: Integrate with metrics system
4. **New Auth Methods**: Extend auth middleware

## Conclusion

The MyMCP architecture provides:
- **Solid foundation** for MCP server development
- **Clear separation** between framework and applications
- **Flexible deployment** options (local, Docker, cloud)
- **Extensible design** for future enhancements

For practical implementation, see:
- [Adding Custom Servers Guide](./ADDING_CUSTOM_SERVERS.md)
- [Core Framework README](../src/core/README.md)
- [Built-in Server Examples](../src/builtin/)

---

**Questions or feedback?** Please open an issue on GitHub!
