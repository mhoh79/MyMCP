# Scripts Directory

This directory contains utility scripts for the MyMCP project.

## Available Scripts

### create-server.ps1

PowerShell script to generate new MCP servers from the skeleton template.

#### Features

- **Interactive Mode**: Prompts for server details with helpful guidance
- **Non-Interactive Mode**: Accepts parameters for automation
- **Input Validation**: 
  - Server names (alphanumeric and hyphens only)
  - Port numbers (checks for conflicts with existing servers)
  - Directory name conflicts
- **Automatic Setup**:
  - Copies skeleton template to `src/custom/{server_name}/`
  - Replaces placeholders in all files
  - Updates `config.yaml` with new server configuration
  - Creates test stub in `tests/custom/`
- **Error Handling**: Comprehensive error checking and user-friendly messages
- **Success Guidance**: Displays next steps after server creation

#### Usage

**Interactive Mode** (recommended):
```powershell
.\scripts\create-server.ps1
```

**Non-Interactive Mode**:
```powershell
.\scripts\create-server.ps1 -Name my-weather-server -Description "Weather data integration" -Port 8002
```

**With Author**:
```powershell
.\scripts\create-server.ps1 -Name my-api-server -Description "API integration" -Author "John Doe" -NonInteractive
```

#### Parameters

- **Name** (required): Server name using alphanumeric characters and hyphens (e.g., `my-weather-server`)
- **Description** (optional): Brief description of the server's purpose
- **Port** (optional): Port number (default: auto-detects next available starting from 8002)
- **Author** (optional): Author name for documentation
- **NonInteractive** (switch): Run without prompts (all required parameters must be provided)

#### Examples

Create a weather server interactively:
```powershell
cd /path/to/MyMCP
.\scripts\create-server.ps1
# Follow the prompts...
```

Create a server for CI/CD:
```powershell
.\scripts\create-server.ps1 `
  -Name my-automation-server `
  -Description "Automation and workflow integration" `
  -Port 8005 `
  -Author "DevOps Team" `
  -NonInteractive
```

#### What Gets Created

When you create a server named `my-weather-server`:

1. **Server Directory**: `src/custom/my_weather_server/`
   - `server.py` - Main server implementation with placeholder tools
   - `__init__.py` - Package initialization
   - `README.md` - Documentation template

2. **Test File**: `tests/custom/test_my_weather_server.py`
   - Test stubs for your server
   - Basic initialization tests
   - Ready to extend with your custom tests

3. **Config Update**: Entry added to `config.yaml`:
   ```yaml
   custom_servers:
     - name: my-weather-server
       module: src.custom.my_weather_server.server
       host: 0.0.0.0
       port: 8002
       enabled: true
   ```

#### Server Naming Conventions

- Use **lowercase letters** only
- Use **hyphens** to separate words (e.g., `my-weather-server`)
- **Alphanumeric characters and hyphens** only
- Must be **at least 3 characters** long
- Cannot **start or end with a hyphen**

The directory will be created with underscores (e.g., `my_weather_server`) for Python compatibility.

#### Port Selection

Reserved ports:
- **8000**: Math server
- **8001**: Stats server
- **9000-9001**: Reserved for builtin servers

Custom servers typically use ports **8002+**. The script auto-detects the next available port if not specified.

#### Validation

The script validates:
- ✓ Server name format and uniqueness
- ✓ Port availability and conflicts
- ✓ Template file existence
- ✓ Directory creation permissions
- ✓ Config file format

#### Troubleshooting

**Error: "Server directory already exists"**
- Solution: Choose a different server name or remove the existing directory

**Error: "Port already in use"**
- Solution: Specify a different port number or omit to auto-select

**Error: "Template not found"**
- Solution: Run the script from the repository root directory

**Error: "PowerShell not found"**
- Solution: Install PowerShell Core or use Windows PowerShell

#### Next Steps After Creation

1. **Customize your server**:
   ```bash
   # Edit the main server file
   code src/custom/my_weather_server/server.py
   ```

2. **Test your server**:
   ```bash
   # Run in stdio mode
   python -m src.custom.my_weather_server.server
   
   # Run in HTTP mode
   python -m src.custom.my_weather_server.server --transport http --port 8002
   ```

3. **Write tests**:
   ```bash
   # Edit test file
   code tests/custom/test_my_weather_server.py
   
   # Run tests
   pytest tests/custom/test_my_weather_server.py -v
   ```

4. **Update documentation**:
   - Edit `src/custom/my_weather_server/README.md`
   - Document your custom tools
   - Add usage examples

#### Testing the Script

Run automated tests:
```bash
pytest tests/test_create_server_script.py -v
```

## Contributing

When adding new scripts:
1. Add comprehensive help documentation
2. Include parameter validation
3. Provide clear error messages
4. Add automated tests in `tests/`
5. Update this README

## Requirements

- PowerShell Core 7.0+ or Windows PowerShell 5.1+
- Python 3.8+
- Git (for config file management)

## Support

For issues or questions:
- Check the main project README
- Review example servers in `src/builtin/`
- See the skeleton template in `src/templates/skeleton_server/`
