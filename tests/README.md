# MCP HTTP Client Testing Suite

Comprehensive test suite for validating HTTP transport, authentication, rate limiting, and all MCP tools over HTTP for **both Math and Stats servers**.

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

## Test Files

- **`test_http_client.py`**: Math Server tests (25 tools, 44 test functions)
- **`test_http_stats_server.py`**: Stats Server tests (32 tools, 47 test functions)

Each server can be tested individually or together.

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

### Run all tests for both servers

```bash
./run_all_tests.sh
```

### Run Math Server tests only

```bash
./run_http_tests.sh
# or
pytest tests/test_http_client.py -v
```

### Run Stats Server tests only

```bash
./run_stats_tests.sh
# or
pytest tests/test_http_stats_server.py -v
```

### Run specific test categories

**Math Server:**
```bash
# Run only tool execution tests
pytest tests/test_http_client.py -k "tool" -v

# Run only health endpoint tests
pytest tests/test_http_client.py -k "health" -v

# Run only authentication tests
pytest tests/test_http_client.py -k "auth" -v
```

**Stats Server:**
```bash
# Run only descriptive stats tests
pytest tests/test_http_stats_server.py -k "descriptive" -v

# Run only time series tests
pytest tests/test_http_stats_server.py -k "moving_average or detect_trend" -v

# Run only signal processing tests
pytest tests/test_http_stats_server.py -k "fft or rms" -v

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
# Math Server URL (default: http://localhost:8000)
export MATH_SERVER_URL="http://localhost:8000"

# Stats Server URL (default: http://localhost:8001)
export STATS_SERVER_URL="http://localhost:8001"

# API key for authentication tests (default: test-api-key-12345678)
export MCP_API_KEY="your-api-key-here"
```

### Testing Individual Servers

#### Math Server (Port 8000)

```bash
# Start the server
python -m src.builtin.math_server --transport http --host 0.0.0.0 --port 8000

# Run tests in another terminal
pytest tests/test_http_client.py -v

# Or use the test runner script
./run_http_tests.sh
```

#### Stats Server (Port 8001)

```bash
# Start the server
python -m src.builtin.stats_server --transport http --host 0.0.0.0 --port 8001

# Run tests in another terminal
pytest tests/test_http_stats_server.py -v

# Or use the test runner script
./run_stats_tests.sh
```

#### Both Servers Together

```bash
# Use the combined test runner
./run_all_tests.sh

# Test only Math server
./run_all_tests.sh --math-only

# Test only Stats server
./run_all_tests.sh --stats-only

# Test against existing servers
./run_all_tests.sh --no-servers
```

### Testing Against Different Servers

#### Local Servers

```bash
# Start both servers
python -m src.builtin.math_server --transport http --host 0.0.0.0 --port 8000 &
python -m src.builtin.stats_server --transport http --host 0.0.0.0 --port 8001 &

# Run all tests
./run_all_tests.sh --no-servers
```

#### GitHub Codespaces

```bash
# Set the Codespaces URLs
export MATH_SERVER_URL="https://your-codespace-8000.app.github.dev"
export STATS_SERVER_URL="https://your-codespace-8001.app.github.dev"

# Run tests for both servers
./run_all_tests.sh --no-servers

# Or test individually
pytest tests/test_http_client.py -v
pytest tests/test_http_stats_server.py -v
```

#### Custom Servers

```bash
export MATH_SERVER_URL="https://your-math-server.example.com:8000"
export STATS_SERVER_URL="https://your-stats-server.example.com:8001"
./run_all_tests.sh --no-servers
```

## Test Categories

### Math Server Tests (`test_http_client.py`)

**Total: 44 tests covering 25 tools**

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

---

### Stats Server Tests (`test_http_stats_server.py`)

**Total: 47 tests covering 32 tools**

#### Descriptive Statistics (4 tests)
- `test_descriptive_stats_tool`: Calculate mean, median, mode, std dev, variance
- `test_correlation_tool`: Pearson correlation coefficient
- `test_percentile_tool`: Calculate percentiles
- `test_detect_outliers_tool`: IQR-based outlier detection

#### Time Series Analysis (6 tests)
- `test_moving_average_tool`: Simple/Exponential/Weighted moving averages
- `test_detect_trend_tool`: Linear regression trend detection
- `test_autocorrelation_tool`: Identify cyclic patterns
- `test_change_point_detection_tool`: Detect regime changes
- `test_rate_of_change_tool`: Monitor acceleration/deceleration
- `test_rolling_statistics_tool`: Windowed statistics

#### Statistical Process Control (5 tests)
- `test_control_limits_tool`: UCL, LCL, centerline for control charts
- `test_process_capability_tool`: Cp, Cpk, Pp, Ppk indices
- `test_western_electric_rules_tool`: 8 run rules for pattern detection
- `test_cusum_chart_tool`: Cumulative sum for shift detection
- `test_ewma_chart_tool`: Exponentially weighted moving average

#### Signal Processing (6 tests)
- `test_fft_analysis_tool`: Frequency domain analysis
- `test_power_spectral_density_tool`: Energy distribution across frequencies
- `test_rms_value_tool`: Overall signal energy
- `test_peak_detection_tool`: Identify dominant frequencies
- `test_signal_to_noise_ratio_tool`: SNR calculation
- `test_harmonic_analysis_tool`: THD and power quality

#### Regression Analysis (5 tests)
- `test_linear_regression_tool`: Simple/multiple regression with diagnostics
- `test_polynomial_regression_tool`: Non-linear curve fitting
- `test_residual_analysis_tool`: Model assumption validation
- `test_prediction_with_intervals_tool`: Forecasts with confidence intervals
- `test_multivariate_regression_tool`: Multiple independent variables

#### Advanced Outlier Detection (6 tests)
- `test_z_score_detection_tool`: Z-score based outlier detection
- `test_grubbs_test_tool`: Grubbs' test for outliers
- `test_dixon_q_test_tool`: Dixon's Q test
- `test_isolation_forest_tool`: ML-based outlier detection
- `test_mahalanobis_distance_tool`: Multivariate outlier detection
- `test_streaming_outlier_detection_tool`: Real-time outlier detection

#### Protocol & Infrastructure (11 tests)
- SSE connection tests (2)
- MCP protocol tests (3)
- Authentication tests (2)
- Health endpoint tests (3)
- Error handling tests (3)
- Concurrent request tests (3)
- Performance tests (1)

---

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
          python -m src.builtin.math_server --transport http --port 8000 &
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
python -m src.builtin.math_server --transport http --host 0.0.0.0 --port 8000
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
