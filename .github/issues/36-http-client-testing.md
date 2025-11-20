# Implement HTTP Client Testing Suite

## Overview
Create comprehensive test suite for validating HTTP transport, authentication, rate limiting, and all MCP tools over HTTP.

## Dependencies
- Issue #30 (HTTP/HTTPS Transport Infrastructure)
- Issue #32 (Security Features)
- Issue #33 (Health Check Endpoints)

## Implementation Tasks
- [ ] Create `tests/test_http_client.py` test module
- [ ] Add function to test SSE connection establishment at `/sse`
- [ ] Add function to test `/messages` POST with sample MCP requests
- [ ] Test authentication when enabled (valid key, invalid key, missing key)
- [ ] Test authentication when disabled (no key required)
- [ ] Test rate limiting behavior (within limit, exceeding limit)
- [ ] Test rate limiting when disabled
- [ ] Verify CORS headers in responses
- [ ] Test all health endpoints (`/health`, `/ready`, `/metrics`)
- [ ] Test calling each MCP tool via HTTP POST
- [ ] Add test for graceful error handling
- [ ] Add test for concurrent requests

## Acceptance Criteria
- Tests cover all HTTP endpoints
- Authentication scenarios fully tested
- Rate limiting validation included
- CORS headers verified
- All MCP tools callable via HTTP
- Tests can run against local and Codespaces servers

## Test Structure

```python
# tests/test_http_client.py

import asyncio
import httpx
from sse_client import SSEClient

class MCPHTTPClient:
    """Client for testing MCP over HTTP"""
    
    async def test_sse_connection(self):
        """Test SSE stream connection"""
        pass
    
    async def test_list_tools(self):
        """Test listing available tools"""
        pass
    
    async def test_call_tool(self, tool_name, arguments):
        """Test calling a specific tool"""
        pass
    
    async def test_authentication(self):
        """Test auth with valid/invalid/missing keys"""
        pass
    
    async def test_rate_limiting(self):
        """Test rate limit enforcement"""
        pass
    
    async def test_health_endpoints(self):
        """Test /health, /ready, /metrics"""
        pass
```

## Test Scenarios

### Authentication Tests
1. **Auth Disabled**: All requests succeed without key
2. **Auth Enabled + Valid Key**: Requests succeed with correct key
3. **Auth Enabled + Invalid Key**: Requests return 401
4. **Auth Enabled + Missing Key**: Requests return 401

### Rate Limiting Tests
1. **Within Limit**: All requests succeed
2. **Exceeding Limit**: Requests return 429 after threshold
3. **Rate Limit Disabled**: No limit enforced

### Tool Execution Tests
Test each tool category:
- Fibonacci tools (fibonacci, is_fibonacci)
- Prime number tools (is_prime, prime_factors, nth_prime)
- Number theory tools (gcd, lcm, factorial, combinations, permutations)
- Hash tools (hash_text)
- Date calculator tools (date_diff, add_time)
- Text processing tools (text_stats, word_frequency)
- Unit converter (convert_units)

### Concurrent Request Tests
- Multiple simultaneous tool calls
- SSE stream handling with concurrent messages
- Race condition testing

## Usage

```bash
# Run all tests
pytest tests/test_http_client.py -v

# Run specific test
pytest tests/test_http_client.py::test_authentication -v

# Run against Codespaces
MATH_SERVER_URL=https://xyz-8000.app.github.dev pytest tests/test_http_client.py
```

## Labels
testing, http-transport
