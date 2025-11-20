# Configure GitHub Codespaces Environment

## Overview
Set up devcontainer configuration for automatic server deployment in GitHub Codespaces with public port forwarding.

## Dependencies
- Issue #30 (HTTP/HTTPS Transport Infrastructure)
- Issue #31 (Configuration System)

## Implementation Tasks
- [ ] Create `.devcontainer/devcontainer.json` configuration file
- [ ] Add `forwardPorts` array with [8000, 8001] for math and stats servers
- [ ] Set `portsAttributes` with visibility "public" for both ports
- [ ] Add `postCreateCommand` to run `pip install -r requirements.txt`
- [ ] Add `postStartCommand` to execute startup script
- [ ] Define `containerEnv` with default environment variables
- [ ] Create `.devcontainer/startup.sh` bash script
- [ ] Script copies `config.example.yaml` to `config.yaml` if missing
- [ ] Script launches both servers in HTTP mode on background
- [ ] Script logs public URLs for accessing servers
- [ ] Test in actual Codespaces environment

## Acceptance Criteria
- Codespace automatically installs dependencies on creation
- Servers start automatically when Codespace starts
- Ports 8000 and 8001 are publicly accessible
- Public URLs display in terminal on startup
- Configuration file created if missing

## Devcontainer Configuration

### `.devcontainer/devcontainer.json`
```json
{
  "name": "MCP Servers",
  "image": "mcr.microsoft.com/devcontainers/python:3.11",
  "forwardPorts": [8000, 8001],
  "portsAttributes": {
    "8000": {
      "label": "Math Server",
      "onAutoForward": "notify",
      "visibility": "public"
    },
    "8001": {
      "label": "Stats Server",
      "onAutoForward": "notify",
      "visibility": "public"
    }
  },
  "postCreateCommand": "pip install -r requirements.txt",
  "postStartCommand": "bash .devcontainer/startup.sh",
  "containerEnv": {
    "MCP_AUTH_ENABLED": "false",
    "MCP_RATE_LIMIT_ENABLED": "false"
  },
  "customizations": {
    "vscode": {
      "extensions": [
        "ms-python.python",
        "ms-python.vscode-pylance"
      ]
    }
  }
}
```

### Public URL Format
```
Math Server:  https://{codespace-name}-8000.app.github.dev
Stats Server: https://{codespace-name}-8001.app.github.dev
```

## Testing Checklist
- [ ] Create new Codespace from branch
- [ ] Verify dependencies install automatically
- [ ] Verify servers start without errors
- [ ] Verify ports are forwarded and public
- [ ] Test accessing public URLs
- [ ] Verify MCP tools work over HTTP

## Labels
enhancement, devcontainer, codespaces
