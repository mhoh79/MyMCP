# Add Configuration System with Validation

## Overview
Implement YAML-based configuration system with Pydantic validation for server settings, authentication, rate limiting, and CORS configuration.

## Dependencies
- Issue #30 (HTTP/HTTPS Transport Infrastructure)

## Implementation Tasks
- [ ] Add dependencies to `requirements.txt`: `pyyaml>=6.0`, `pydantic-settings>=2.0.0`
- [ ] Create `config.yaml` with sections: server (host, ports, cors_origins), authentication (enabled, api_key), rate_limiting (enabled, requests_per_minute), logging (level)
- [ ] Create `config.example.yaml` with documented examples and defaults
- [ ] Create `src/config.py` module with Pydantic models for each config section
- [ ] Implement config loading with file path argument and environment variable overrides
- [ ] Add validation with detailed error messages on schema violations
- [ ] Add `--config` CLI argument to both servers
- [ ] Load and apply configuration in `run_http_server()` functions

## Acceptance Criteria
- Config file uses YAML format with clear structure
- Pydantic validates all settings on startup
- Invalid config shows helpful error messages
- Environment variables can override config file values
- Example config file documents all available options

## Configuration Schema

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
  level: "INFO"
```

## Environment Variable Overrides
- `MCP_HOST` - Override server host
- `MCP_PORT` - Override server port
- `MCP_AUTH_ENABLED` - Enable/disable authentication
- `MCP_API_KEY` - Set API key
- `MCP_RATE_LIMIT_ENABLED` - Enable/disable rate limiting

## Labels
enhancement, configuration
