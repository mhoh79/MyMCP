# Add Hot-Reload Development Mode

## Overview
Implement development mode with hot-reload for faster iteration when modifying server code.

## Dependencies
- Issue #30 (HTTP/HTTPS Transport Infrastructure)

## Implementation Tasks
- [ ] Add `--dev` CLI flag to both servers
- [ ] Configure uvicorn with `reload=True` when in dev mode
- [ ] Add uvicorn `reload_dirs` to watch `src/` directory
- [ ] Update launcher scripts with `--dev` option
- [ ] Create VS Code launch configuration in `.vscode/launch.json`
- [ ] Add debug configurations for both servers in HTTP mode
- [ ] Document development workflow in README
- [ ] Test hot-reload triggers on file changes

## Acceptance Criteria
- `--dev` flag enables hot-reload
- Server restarts automatically on code changes
- VS Code launch configs work for debugging
- Development workflow documented
- Works on local and Codespaces environments

## Implementation Details

### CLI Flag
```python
# Add to argument parser
parser.add_argument(
    '--dev',
    action='store_true',
    help='Enable development mode with hot-reload'
)

# In run_http_server()
if args.dev:
    uvicorn.run(
        app,
        host=host,
        port=port,
        reload=True,
        reload_dirs=["src"],
        log_level="debug"
    )
else:
    uvicorn.run(app, host=host, port=port)
```

### VS Code Launch Configuration

`.vscode/launch.json`:
```json
{
  "version": "0.2.0",
  "configurations": [
    {
      "name": "Math Server (HTTP Dev)",
      "type": "python",
      "request": "launch",
      "module": "uvicorn",
      "args": [
        "src.math_server.server:app",
        "--host", "0.0.0.0",
        "--port", "8000",
        "--reload",
        "--reload-dir", "src"
      ],
      "console": "integratedTerminal",
      "justMyCode": false
    },
    {
      "name": "Stats Server (HTTP Dev)",
      "type": "python",
      "request": "launch",
      "module": "uvicorn",
      "args": [
        "src.stats_server.server:app",
        "--host", "0.0.0.0",
        "--port", "8001",
        "--reload",
        "--reload-dir", "src"
      ],
      "console": "integratedTerminal",
      "justMyCode": false
    },
    {
      "name": "Math Server (stdio)",
      "type": "python",
      "request": "launch",
      "program": "${workspaceFolder}/src/math_server/server.py",
      "args": ["--transport", "stdio"],
      "console": "integratedTerminal"
    }
  ],
  "compounds": [
    {
      "name": "Both Servers (HTTP Dev)",
      "configurations": [
        "Math Server (HTTP Dev)",
        "Stats Server (HTTP Dev)"
      ]
    }
  ]
}
```

### Launcher Script Updates

Add `--dev` flag support:
```bash
# start-http-servers.sh
./start-http-servers.sh --dev    # Start with hot-reload
```

### Development Workflow

1. **Start in Dev Mode**:
   ```bash
   python src/math_server/server.py --transport http --port 8000 --dev
   ```

2. **Make Code Changes**: Edit tool implementations or add new features

3. **Automatic Reload**: Server detects changes and restarts

4. **Test Changes**: Run client tests against reloaded server

5. **Debug**: Set breakpoints in VS Code, use debug configurations

### Watch Patterns
Monitor changes in:
- `src/**/*.py` - All Python source files
- `config.yaml` - Configuration file
- Exclude: `__pycache__/`, `*.pyc`, `.venv/`

### Performance Considerations
- Dev mode only for development (not production)
- Reload adds ~1-2 second delay on changes
- Disable in production for performance
- Keep reload_dirs focused to minimize watch overhead

## Usage Examples

```bash
# Development with hot-reload
python src/math_server/server.py --transport http --dev

# VS Code: Press F5, select "Math Server (HTTP Dev)"

# Both servers with hot-reload
./start-http-servers.sh --dev
```

## Documentation Updates
- Add "Development Mode" section to README
- Document VS Code setup
- Include hot-reload behavior explanation
- Add troubleshooting for reload issues

## Labels
enhancement, developer-experience
