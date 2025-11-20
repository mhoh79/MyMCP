# Implement Security Features (Auth & Rate Limiting)

## Overview
Add configurable API key authentication and rate limiting middleware for HTTP transport, controllable via configuration file.

## Dependencies
- Issue #30 (HTTP/HTTPS Transport Infrastructure)
- Issue #31 (Configuration System)

## Implementation Tasks
- [ ] Create `src/middleware.py` module
- [ ] Implement authentication middleware checking `Authorization: Bearer <key>` header when `config.authentication.enabled` is true
- [ ] Return 401 Unauthorized with proper error message when auth fails
- [ ] Skip authentication when `config.authentication.enabled` is false
- [ ] Implement rate limiting middleware using token bucket algorithm per client IP
- [ ] Return 429 Too Many Requests when rate limit exceeded
- [ ] Skip rate limiting when `config.rate_limiting.enabled` is false
- [ ] Add request logging middleware for debugging
- [ ] Apply middleware conditionally to FastAPI app in both servers
- [ ] Add CORS middleware with origins from `config.server.cors_origins`

## Acceptance Criteria
- Authentication can be enabled/disabled in config
- Valid API key grants access, invalid returns 401
- Rate limiting enforces requests per minute limit
- Rate limiting can be disabled in config
- CORS headers configured from config file
- Middleware logs requests appropriately

## Security Implementation Details

### Authentication Middleware
```python
# Header format: Authorization: Bearer <api-key>
# Returns 401 if:
#   - Authentication enabled and header missing
#   - Authentication enabled and key invalid
# Allows through if:
#   - Authentication disabled
#   - Authentication enabled and key valid
```

### Rate Limiting Algorithm
- Token bucket algorithm per client IP
- Tokens refill at configured rate (requests per minute)
- Burst capacity = requests_per_minute
- Track state in memory (consider Redis for production)

### CORS Configuration
- Support wildcard patterns in origins
- Allow credentials for authenticated requests
- Configure allowed methods: GET, POST, OPTIONS
- Configure allowed headers: Content-Type, Authorization

## Example Responses

### 401 Unauthorized
```json
{
  "error": "Unauthorized",
  "message": "Invalid or missing API key"
}
```

### 429 Too Many Requests
```json
{
  "error": "Rate Limit Exceeded",
  "message": "Too many requests. Please try again later.",
  "retry_after": 30
}
```

## Labels
enhancement, security, authentication, rate-limiting
