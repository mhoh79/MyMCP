# MCP HTTP Client Testing Suite

Comprehensive test suite for validating HTTP transport, authentication, rate limiting, and all MCP tools over HTTP.

## Overview

This test suite validates:
- **HTTP Transport**: SSE connections and JSON-RPC over HTTP
- **Authentication**: Valid/invalid/missing API keys, enabled/disabled states
- **Rate Limiting**: Within/exceeding limits, disabled state
- **CORS**: Header verification
- **Health Endpoints**: `/health`, `/ready`, `/metrics`
- **MCP Tools**: All available tools across multiple categories
- **Concurrent Requests**: Multiple simultaneous requests
- **Error Handling**: Invalid inputs, malformed JSON, unknown methods

## Installation

Install test dependencies:

```bash
pip install -r requirements-test.txt
```

Or install individually:

```bash
pip install pytest pytest-asyncio httpx
```

## Running Tests

### Run all tests

```bash
pytest tests/test_http_client.py -v
```

### Run specific test categories

```bash
# Run only tool execution tests
pytest tests/test_http_client.py -k "tool" -v

# Run only health endpoint tests
pytest tests/test_http_client.py -k "health" -v

# Run only authentication tests
pytest tests/test_http_client.py -k "auth" -v

# Run only concurrent tests
pytest tests/test_http_client.py -k "concurrent" -v
```

### Run with coverage

```bash
pytest tests/test_http_client.py --cov=tests --cov-report=html
```

### Run specific test

```bash
pytest tests/test_http_client.py::test_fibonacci_tool -v
```

## Configuration

### Environment Variables

Configure the test suite using environment variables:

```bash
# Server URL (default: http://localhost:8000)
export MATH_SERVER_URL="http://localhost:8000"

# API key for authentication tests (default: test-api-key-12345678)
export MCP_API_KEY="your-api-key-here"
```

### Testing Against Different Servers

#### Local Server

```bash
# Start the server
python src/math_server/server.py --transport http --host 0.0.0.0 --port 8000

# Run tests in another terminal
pytest tests/test_http_client.py -v
```

#### GitHub Codespaces

```bash
# Set the Codespaces URL
export MATH_SERVER_URL="https://your-codespace-8000.app.github.dev"

# Run tests
pytest tests/test_http_client.py -v
```

#### Custom Server

```bash
export MATH_SERVER_URL="https://your-server.example.com:8000"
pytest tests/test_http_client.py -v
```

## Test Categories

### SSE Connection Tests
- `test_sse_connection_success`: Verify SSE endpoint connectivity
- `test_sse_connection_headers`: Validate SSE response headers

### MCP Protocol Tests
- `test_initialize_method`: Test MCP initialization
- `test_list_tools_success`: Verify tool listing
- `test_list_tools_structure`: Validate tool metadata structure

### Tool Execution Tests

#### Fibonacci Tools
- `test_fibonacci_tool`: Calculate Fibonacci numbers
- `test_is_fibonacci_tool`: Check if number is Fibonacci

#### Prime Number Tools
- `test_is_prime_tool`: Check if number is prime
- `test_prime_factors_tool`: Get prime factorization
- `test_nth_prime_tool`: Find nth prime number

#### Number Theory Tools
- `test_gcd_tool`: Greatest common divisor
- `test_lcm_tool`: Least common multiple
- `test_factorial_tool`: Calculate factorial
- `test_combinations_tool`: Combinations (nCr)
- `test_permutations_tool`: Permutations (nPr)

#### Hash Tools
- `test_hash_text_tool`: Generate cryptographic hashes

#### Unit Conversion Tools
- `test_convert_units_tool`: Convert between units

#### Date Calculator Tools
- `test_date_diff_tool`: Calculate date differences
- `test_date_add_tool`: Add time to dates

#### Text Processing Tools
- `test_text_stats_tool`: Text statistics
- `test_word_frequency_tool`: Word frequency analysis

### Authentication Tests
- `test_auth_disabled_no_key`: No key required when disabled
- `test_auth_disabled_with_key`: Key accepted when disabled
- `test_auth_enabled_valid_key`: Valid key accepted (requires auth enabled)
- `test_auth_enabled_invalid_key`: Invalid key rejected (requires auth enabled)
- `test_auth_enabled_missing_key`: Missing key rejected (requires auth enabled)

### Rate Limiting Tests
- `test_rate_limit_within_limit`: Requests within limit succeed
- `test_rate_limit_exceeding_limit`: Exceeding limit returns 429 (requires rate limiting)
- `test_rate_limit_headers`: Verify rate limit headers

### CORS Tests
- `test_cors_headers_present`: CORS headers in responses
- `test_cors_preflight`: CORS preflight requests

### Health Endpoint Tests
- `test_health_endpoint`: `/health` endpoint
- `test_ready_endpoint`: `/ready` endpoint  
- `test_metrics_endpoint`: `/metrics` endpoint

### Error Handling Tests
- `test_invalid_json_rpc`: Invalid JSON-RPC format
- `test_unknown_method`: Unknown MCP method
- `test_invalid_tool_name`: Non-existent tool
- `test_invalid_tool_arguments`: Invalid tool arguments
- `test_malformed_json`: Malformed JSON requests

### Concurrent Request Tests
- `test_concurrent_tool_calls`: Multiple tool calls simultaneously
- `test_concurrent_different_tools`: Different tools simultaneously
- `test_concurrent_list_tools`: Multiple list_tools calls

### Performance Tests
- `test_response_time`: Response time validation
- `test_large_response_handling`: Large response handling

## Skipped Tests

Some tests are skipped by default because they require specific server configuration:

### Authentication Tests (Skipped)
Require `authentication.enabled: true` in config.yaml:

```bash
# Enable authentication in config.yaml
# Then run:
pytest tests/test_http_client.py -k "auth_enabled" -v
```

### Rate Limiting Tests (Skipped)
Require `rate_limiting.enabled: true` in config.yaml:

```bash
# Enable rate limiting in config.yaml
# Then run:
pytest tests/test_http_client.py::test_rate_limit_exceeding_limit -v
```

## CI/CD Integration

### GitHub Actions Example

```yaml
name: Test MCP HTTP Client

on: [push, pull_request]

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - uses: actions/setup-python@v4
        with:
          python-version: '3.11'
      - name: Install dependencies
        run: |
          pip install -r requirements.txt
          pip install -r requirements-test.txt
      - name: Start MCP server
        run: |
          python src/math_server/server.py --transport http --port 8000 &
          sleep 5
      - name: Run tests
        run: pytest tests/test_http_client.py -v --cov=tests
```

## Troubleshooting

### Connection Refused

```
httpx.ConnectError: [Errno 111] Connection refused
```

**Solution**: Make sure the MCP server is running:

```bash
python src/math_server/server.py --transport http --host 0.0.0.0 --port 8000
```

### Timeout Errors

```
httpx.TimeoutException: timed out
```

**Solution**: Server may be slow to start. Increase timeout or wait longer before running tests.

### Import Errors

```
ModuleNotFoundError: No module named 'httpx'
```

**Solution**: Install test dependencies:

```bash
pip install -r requirements-test.txt
```

## Development

### Adding New Tests

1. Add test function to `test_http_client.py`
2. Use `@pytest.mark.asyncio` for async tests
3. Use appropriate fixtures (`client` or `auth_client`)
4. Follow naming convention: `test_<feature>_<scenario>`

### Test Markers

Use markers to categorize tests:

```python
@pytest.mark.slow
async def test_long_running_operation(client):
    """Test that takes a long time."""
    pass

@pytest.mark.integration
async def test_full_workflow(client):
    """Test complete workflow."""
    pass
```

Run specific markers:

```bash
pytest -m "not slow"  # Skip slow tests
pytest -m "integration"  # Run only integration tests
```

## Test Coverage

Generate coverage report:

```bash
pytest tests/test_http_client.py --cov=tests --cov-report=html
open htmlcov/index.html
```

## Support

For issues or questions:
1. Check this README
2. Review server logs
3. Verify server configuration
4. Check GitHub issues

## License

Same as parent project.
