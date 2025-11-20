# Add HTTP/HTTPS Transport Infrastructure

## Overview
Add HTTP/HTTPS transport layer to MCP servers using FastAPI and SSE, enabling remote access from GitHub Codespaces while maintaining backward compatibility with stdio transport.

## Dependencies
None

## Implementation Tasks
- [ ] Add dependencies to `requirements.txt`: `fastapi>=0.104.0`, `uvicorn[standard]>=0.24.0`, `sse-starlette>=1.8.0`, `python-multipart>=0.0.6`
- [ ] Add CLI argument parsing to `src/math_server/server.py`: `--transport stdio|http`, `--host`, `--port`
- [ ] Add CLI argument parsing to `src/stats_server/server.py`: `--transport stdio|http`, `--host`, `--port`
- [ ] Implement `run_http_server()` in `math_server` with FastAPI app, `/sse` endpoint for SSE stream, `/messages` POST endpoint for JSON-RPC
- [ ] Implement `run_http_server()` in `stats_server` with same FastAPI structure
- [ ] Preserve existing `run_stdio_server()` functionality unchanged
- [ ] Test both transports work independently

## Acceptance Criteria
- Servers support `--transport` flag to choose stdio or http mode
- HTTP mode exposes MCP protocol over SSE at `/sse` endpoint
- `/messages` endpoint handles JSON-RPC 2.0 requests
- Stdio mode continues working for Claude Desktop
- Both servers can run simultaneously on different ports

## Technical Details
### MCP over SSE Protocol
- SSE endpoint (`/sse`) streams server-to-client messages
- Messages POST endpoint (`/messages`) receives client-to-server requests
- JSON-RPC 2.0 format for all messages
- Proper Content-Type headers (text/event-stream for SSE)

### Example Usage
```bash
# Start in HTTP mode
python src/math_server/server.py --transport http --host 0.0.0.0 --port 8000

# Start in stdio mode (default)
python src/math_server/server.py --transport stdio
```

## References
- [MCP Specification](https://spec.modelcontextprotocol.io/)
- [FastAPI Documentation](https://fastapi.tiangolo.com/)
- [SSE Starlette](https://github.com/sysid/sse-starlette)

## Labels
enhancement, infrastructure, http-transport
