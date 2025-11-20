# Create Server Launcher Scripts

## Overview
Develop cross-platform launcher scripts for starting both MCP servers in HTTP mode with proper configuration and process management.

## Dependencies
- Issue #30 (HTTP/HTTPS Transport Infrastructure)
- Issue #31 (Configuration System)

## Implementation Tasks
- [ ] Create `start-http-servers.ps1` PowerShell script
- [ ] Create `start-http-servers.sh` Bash script
- [ ] Check if `config.yaml` exists, copy from example if missing
- [ ] Start math_server on port 8000 in background
- [ ] Start stats_server on port 8001 in background
- [ ] Capture and store process IDs for management
- [ ] Display connection URLs (localhost and Codespaces format)
- [ ] Add option to stop servers by PID
- [ ] Handle existing server processes gracefully
- [ ] Add logging output showing startup progress

## Acceptance Criteria
- Scripts work on Windows (PowerShell) and Linux/Mac (Bash)
- Both servers start with correct ports and config
- Process IDs saved for later management
- Clear output shows how to connect to servers
- Can stop servers cleanly using saved PIDs

## Script Specifications

### `start-http-servers.ps1` (Windows)
```powershell
# Features:
# - Check for config.yaml, create from example if needed
# - Start both servers with Start-Process
# - Save PIDs to .pids file
# - Display URLs (localhost and Codespaces)
# - Color-coded output
# - Error handling

# Usage:
# .\start-http-servers.ps1           # Start servers
# .\start-http-servers.ps1 -Stop     # Stop servers
```

### `start-http-servers.sh` (Linux/Mac)
```bash
# Features:
# - Check for config.yaml, create from example if needed
# - Start both servers with nohup
# - Save PIDs to .pids file
# - Display URLs (localhost and Codespaces)
# - Color-coded output
# - Error handling

# Usage:
# ./start-http-servers.sh start      # Start servers
# ./start-http-servers.sh stop       # Stop servers
# ./start-http-servers.sh status     # Check status
```

## Process Management

### PID File Format (`.pids`)
```
math_server=12345
stats_server=12346
```

### Stop Functionality
- Read PIDs from `.pids` file
- Send SIGTERM (graceful shutdown)
- Wait up to 10 seconds
- Send SIGKILL if still running
- Remove `.pids` file

## Output Format
```
🚀 Starting MCP Servers in HTTP mode...

✓ Config file found: config.yaml
✓ Math Server started on port 8000 (PID: 12345)
✓ Stats Server started on port 8001 (PID: 12346)

📡 Connection URLs:
  Math Server:  http://localhost:8000
  Stats Server: http://localhost:8001

🌐 Codespaces URLs (if applicable):
  Math Server:  https://{codespace}-8000.app.github.dev
  Stats Server: https://{codespace}-8001.app.github.dev

ℹ️  To stop servers: ./start-http-servers.sh stop
```

## Error Handling
- Check Python installation
- Verify virtual environment
- Validate config file
- Check port availability
- Handle missing dependencies

## Labels
enhancement, tooling, scripts
