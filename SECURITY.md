# Security Features

This document describes the security features implemented in the MCP servers.

## Overview

The Math and Stats MCP servers include comprehensive security middleware that can be configured via the `config.yaml` file. All security features can be independently enabled or disabled based on your deployment requirements.

## Features

### 1. API Key Authentication

**Status**: ✅ Implemented  
**Configuration**: `config.authentication.enabled`

#### Description
Protects server endpoints by requiring a valid API key in the `Authorization` header. Uses Bearer token authentication scheme.

#### Configuration

```yaml
authentication:
  enabled: true  # Enable authentication
  api_key: "your-secure-api-key-here"  # Minimum 16 characters
```

#### Usage

Include the API key in the `Authorization` header:

```bash
curl -X POST http://localhost:8000/messages \
  -H "Authorization: Bearer your-secure-api-key-here" \
  -H "Content-Type: application/json" \
  -d '{"jsonrpc": "2.0", "method": "tools/list", "params": {}, "id": 1}'
```

#### Behavior

- **Enabled**: All endpoints except `/health`, `/ready`, and `/metrics` require authentication
- **Disabled**: No authentication required (suitable for development or trusted networks)
- **Invalid/Missing Key**: Returns 401 Unauthorized with descriptive error message

#### Security Notes

- ⚠️ **Change the default API key** before enabling authentication
- Use a strong, randomly generated key (minimum 16 characters)
- Store API keys securely (e.g., environment variables, secret manager)
- Never commit API keys to version control

#### Environment Variable Override

```bash
export MCP_AUTH_ENABLED=true
export MCP_API_KEY="your-secure-api-key-here"
python src/math_server/server.py --config config.yaml
```

---

### 2. Rate Limiting

**Status**: ✅ Implemented  
**Configuration**: `config.rate_limiting.enabled`

#### Description
Prevents abuse by limiting the number of requests per client IP address. Uses the token bucket algorithm for smooth rate limiting with burst capacity.

#### Configuration

```yaml
rate_limiting:
  enabled: true  # Enable rate limiting
  requests_per_minute: 60  # Maximum requests per minute (1-10000)
```

#### How It Works

**Token Bucket Algorithm**:
- Each client IP gets its own "bucket" of tokens
- Bucket capacity = `requests_per_minute` (allows bursts)
- Tokens refill at rate of `requests_per_minute / 60` per second
- Each request consumes one token
- If no tokens available, request is rate limited

**Example**: With `requests_per_minute: 60`
- Client can make 60 requests instantly (burst)
- Then limited to 1 request per second (sustained)
- Bucket refills gradually, allowing occasional bursts

#### Behavior

- **Enabled**: Enforces rate limits per client IP
- **Disabled**: No rate limiting (unlimited requests)
- **Rate Limited**: Returns 429 Too Many Requests with `Retry-After` header

#### Response Headers

When rate limiting is enabled, all responses include:
- `X-RateLimit-Limit`: Maximum requests per minute
- `X-RateLimit-Remaining`: Remaining tokens in bucket

When rate limited:
- `Retry-After`: Seconds to wait before retrying
- `X-RateLimit-Remaining`: 0

#### Example Response (Rate Limited)

```json
{
  "error": "Too Many Requests",
  "message": "Rate limit exceeded. Maximum 60 requests per minute.",
  "retry_after": 1
}
```

#### Client IP Detection

Rate limiting tracks clients by IP address:
1. Checks `X-Forwarded-For` header (for proxies/load balancers)
2. Falls back to direct client IP
3. Uses first IP in X-Forwarded-For chain (original client)

#### Memory Management

To prevent memory leaks in long-running servers:
- Maximum 10,000 client buckets tracked
- Old buckets cleaned up every hour
- Most recently used buckets are kept

#### Environment Variable Override

```bash
export MCP_RATE_LIMIT_ENABLED=true
export MCP_RATE_LIMIT_RPM=120
python src/math_server/server.py --config config.yaml
```

---

### 3. CORS (Cross-Origin Resource Sharing)

**Status**: ✅ Implemented  
**Configuration**: `config.server.cors_origins`

#### Description
Allows web browsers to make requests from specific origins. Essential for web-based clients.

#### Configuration

```yaml
server:
  cors_origins:
    - "http://localhost:*"           # All localhost ports
    - "https://*.app.github.dev"     # GitHub Codespaces
    - "https://example.com"          # Specific domain
```

#### Features

- Supports wildcard patterns (`*`)
- Allows credentials (cookies, authorization headers)
- Permits all HTTP methods and headers
- Required for web-based MCP clients

#### Example Usage

```javascript
// Web browser can make requests from allowed origins
fetch('http://localhost:8000/messages', {
  method: 'POST',
  headers: {
    'Content-Type': 'application/json',
    'Authorization': 'Bearer your-api-key'
  },
  body: JSON.stringify({
    jsonrpc: '2.0',
    method: 'tools/list',
    params: {},
    id: 1
  })
})
```

---

### 4. Request Logging

**Status**: ✅ Implemented  
**Always Enabled**

#### Description
Logs all HTTP requests for debugging and monitoring. Includes timing information.

#### Log Format

```
2025-11-22 09:26:43,175 - middleware - INFO - POST /messages from 127.0.0.1 -> 200 (0.001s)
```

Includes:
- Timestamp
- HTTP method and path
- Client IP address
- Response status code
- Request duration in seconds

#### Log Level Configuration

Configure via `config.logging.level`:
- `DEBUG`: Detailed middleware debug logs
- `INFO`: Request logs and important events
- `WARNING`: Authentication failures, rate limits
- `ERROR`: Server errors

```yaml
logging:
  level: "INFO"  # DEBUG, INFO, WARNING, ERROR, CRITICAL
```

---

## Deployment Recommendations

### Development Environment

```yaml
authentication:
  enabled: false  # Easy testing
rate_limiting:
  enabled: false  # No restrictions
logging:
  level: "DEBUG"  # Detailed logs
```

### Production Environment

```yaml
authentication:
  enabled: true
  api_key: "${MCP_API_KEY}"  # From environment variable
rate_limiting:
  enabled: true
  requests_per_minute: 60  # Adjust based on expected load
logging:
  level: "INFO"  # Production-appropriate
```

### High-Traffic Environment

```yaml
authentication:
  enabled: true
  api_key: "${MCP_API_KEY}"
rate_limiting:
  enabled: true
  requests_per_minute: 300  # 5 requests/second sustained
logging:
  level: "WARNING"  # Reduce log volume
```

---

## Testing Security Features

### Test Authentication

```bash
# Without auth (should fail with 401)
curl http://localhost:8000/messages

# With invalid key (should fail with 401)
curl -H "Authorization: Bearer wrong-key" http://localhost:8000/messages

# With valid key (should succeed)
curl -H "Authorization: Bearer your-api-key" http://localhost:8000/messages
```

### Test Rate Limiting

```bash
# Make rapid requests to trigger rate limit
for i in {1..100}; do
  curl -H "Authorization: Bearer your-api-key" \
    http://localhost:8000/messages &
done
wait

# Check for 429 responses
```

### Test CORS

```bash
# Preflight request
curl -X OPTIONS http://localhost:8000/messages \
  -H "Origin: http://localhost:3000" \
  -H "Access-Control-Request-Method: POST"

# Check for CORS headers in response
```

---

## Monitoring

### Health Check Endpoint

Always accessible without authentication:

```bash
curl http://localhost:8000/health
# Returns: {"status": "ok", "server": "math-server", ...}
```

### Metrics Endpoint

Always accessible without authentication:

```bash
curl http://localhost:8000/metrics
# Returns: {"total_requests": 1234, "active_connections": 5, ...}
```

### Readiness Check

Always accessible without authentication:

```bash
curl http://localhost:8000/ready
# Returns 200 if ready, 503 if not ready
```

---

## Security Best Practices

### API Keys

1. **Use Strong Keys**: Minimum 16 characters, preferably 32+
2. **Rotate Regularly**: Change API keys periodically
3. **Environment Variables**: Never hardcode keys in config files
4. **Secret Management**: Use tools like AWS Secrets Manager, HashiCorp Vault
5. **Least Privilege**: Use different keys for different clients if possible

### Rate Limiting

1. **Monitor Patterns**: Watch for unusual traffic patterns
2. **Adjust Limits**: Tune based on legitimate usage patterns
3. **Client Communication**: Document rate limits for API consumers
4. **Retry Logic**: Implement exponential backoff in clients

### CORS

1. **Specific Origins**: Avoid `*` wildcards in production
2. **Review Regularly**: Audit allowed origins periodically
3. **Minimal Exposure**: Only allow necessary origins

### Logging

1. **Sensitive Data**: Never log API keys or request bodies containing secrets
2. **Log Rotation**: Configure log rotation to manage disk space
3. **Monitoring**: Set up alerts for authentication failures and rate limit events
4. **Compliance**: Ensure logging complies with privacy regulations (GDPR, etc.)

---

## Troubleshooting

### Authentication Issues

**Problem**: Getting 401 even with correct API key

**Solutions**:
1. Verify `authentication.enabled: true` in config
2. Check API key has minimum 16 characters
3. Ensure header format: `Authorization: Bearer <key>` (note the space)
4. Check for whitespace/encoding issues in API key

### Rate Limiting Issues

**Problem**: Getting rate limited unexpectedly

**Solutions**:
1. Check `requests_per_minute` setting
2. Verify client IP isn't shared (NAT, proxy)
3. Review X-Forwarded-For header handling
4. Increase limit for legitimate high-traffic use cases

### CORS Issues

**Problem**: Browser requests failing with CORS errors

**Solutions**:
1. Verify origin is in `cors_origins` list
2. Check wildcard patterns match correctly
3. Ensure preflight requests (OPTIONS) are succeeding
4. Verify browser is sending Origin header

---

## Security Considerations

### Known Limitations

1. **IP-Based Rate Limiting**: Can be circumvented by rotating IPs
2. **Shared IPs**: Clients behind NAT may share rate limits
3. **No Request Signing**: API keys can be intercepted without HTTPS
4. **Memory-Based State**: Rate limits reset on server restart

### Recommended Additional Security

1. **HTTPS**: Always use HTTPS in production (TLS encryption)
2. **Reverse Proxy**: Use nginx/Apache for additional security layers
3. **WAF**: Consider Web Application Firewall for advanced protection
4. **Network Isolation**: Deploy in private network when possible
5. **Monitoring**: Set up intrusion detection and monitoring

---

## Support

For security concerns or questions:
- Review this documentation
- Check configuration examples in `config.example.yaml`
- Review code in `src/middleware.py`
- Report security issues privately to repository maintainers

**Note**: Do not include actual API keys or sensitive information in public issues or discussions.
