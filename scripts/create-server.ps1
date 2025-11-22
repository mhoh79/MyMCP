#!/usr/bin/env pwsh

<#
.SYNOPSIS
    Interactive Server Generator for MCP Custom Servers
    
.DESCRIPTION
    This script automates the creation of new MCP servers from the skeleton template.
    It provides interactive prompts, validates inputs, copies template files, replaces
    placeholders, updates config.yaml, and creates test stubs.
    
.PARAMETER Name
    Server name (alphanumeric and hyphens only, e.g., 'my-weather-server')
    
.PARAMETER Description
    Brief description of what the server does
    
.PARAMETER Port
    Port number (default: auto-detect next available port starting from 8002)
    
.PARAMETER Author
    Author name (optional)
    
.PARAMETER NonInteractive
    Run in non-interactive mode with provided parameters (no prompts)
    
.EXAMPLE
    .\scripts\create-server.ps1
    Run interactively with prompts
    
.EXAMPLE
    .\scripts\create-server.ps1 -Name my-weather-server -Description "Weather data server" -Port 8002
    Create server with specified parameters
    
.EXAMPLE
    .\scripts\create-server.ps1 -Name my-api-server -Description "API integration server" -Author "John Doe" -NonInteractive
    Create server non-interactively
#>

param(
    [string]$Name,
    [string]$Description,
    [int]$Port = 0,
    [string]$Author,
    [switch]$NonInteractive
)

# Color output functions
function Write-Success {
    param([string]$Message)
    Write-Host "✓ " -ForegroundColor Green -NoNewline
    Write-Host $Message
}

function Write-Error-Custom {
    param([string]$Message)
    Write-Host "✗ " -ForegroundColor Red -NoNewline
    Write-Host $Message
}

function Write-Info {
    param([string]$Message)
    Write-Host "ℹ  " -ForegroundColor Blue -NoNewline
    Write-Host $Message
}

function Write-Warning-Custom {
    param([string]$Message)
    Write-Host "⚠  " -ForegroundColor Yellow -NoNewline
    Write-Host $Message
}

function Write-Section {
    param([string]$Message)
    Write-Host ""
    Write-Host "═══════════════════════════════════════════════════════════════" -ForegroundColor Cyan
    Write-Host "  $Message" -ForegroundColor Cyan
    Write-Host "═══════════════════════════════════════════════════════════════" -ForegroundColor Cyan
    Write-Host ""
}

# Validate server name
function Test-ServerName {
    param([string]$ServerName)
    
    if ([string]::IsNullOrWhiteSpace($ServerName)) {
        return @{ Valid = $false; Error = "Server name cannot be empty" }
    }
    
    # Must contain only alphanumeric characters and hyphens
    if ($ServerName -notmatch '^[a-zA-Z0-9\-]+$') {
        return @{ Valid = $false; Error = "Server name must contain only alphanumeric characters and hyphens" }
    }
    
    # Cannot start or end with hyphen
    if ($ServerName -match '^-|-$') {
        return @{ Valid = $false; Error = "Server name cannot start or end with a hyphen" }
    }
    
    # Must be at least 3 characters
    if ($ServerName.Length -lt 3) {
        return @{ Valid = $false; Error = "Server name must be at least 3 characters long" }
    }
    
    # Convert to lowercase (MCP convention)
    $ServerName = $ServerName.ToLower()
    
    # Check if directory already exists (directory name uses underscores)
    $serverNameUnderscore = $ServerName -replace '-', '_'
    $customPath = "src/custom/$serverNameUnderscore"
    if (Test-Path $customPath) {
        return @{ Valid = $false; Error = "Server directory already exists at $customPath" }
    }
    
    return @{ Valid = $true; Name = $ServerName }
}

# Parse config.yaml and extract used ports
function Get-UsedPorts {
    param([string]$ConfigPath)
    
    try {
        if (-not (Test-Path $ConfigPath)) {
            Write-Warning-Custom "Config file not found at $ConfigPath"
            return @(8000, 8001, 9000, 9001)  # Default reserved ports
        }
        
        $usedPorts = @()
        $content = Get-Content $ConfigPath -Raw
        
        # Extract port numbers from config (simple regex approach)
        # Look for 'port: NNNN' patterns
        $portMatches = [regex]::Matches($content, 'port:\s*(\d+)')
        foreach ($match in $portMatches) {
            $port = [int]$match.Groups[1].Value
            if ($port -notin $usedPorts) {
                $usedPorts += $port
            }
        }
        
        # Always include reserved builtin ports
        $builtinPorts = @(8000, 8001, 9000, 9001)
        foreach ($port in $builtinPorts) {
            if ($port -notin $usedPorts) {
                $usedPorts += $port
            }
        }
        
        return $usedPorts | Sort-Object
    }
    catch {
        Write-Warning-Custom "Error parsing config.yaml: $_"
        return @(8000, 8001, 9000, 9001)
    }
}

# Find next available port
function Get-NextAvailablePort {
    param([string]$ConfigPath, [int]$StartPort = 8002)
    
    $usedPorts = Get-UsedPorts -ConfigPath $ConfigPath
    
    $port = $StartPort
    while ($port -in $usedPorts -and $port -le 65535) {
        $port++
    }
    
    if ($port -gt 65535) {
        return $null
    }
    
    return $port
}

# Validate port number
function Test-Port {
    param([int]$PortNumber, [string]$ConfigPath)
    
    if ($PortNumber -lt 1 -or $PortNumber -gt 65535) {
        return @{ Valid = $false; Error = "Port must be between 1 and 65535" }
    }
    
    $usedPorts = Get-UsedPorts -ConfigPath $ConfigPath
    
    if ($PortNumber -in $usedPorts) {
        return @{ Valid = $false; Error = "Port $PortNumber is already in use. Used ports: $($usedPorts -join ', ')" }
    }
    
    return @{ Valid = $true; Port = $PortNumber }
}

# Copy template files to custom directory
function Copy-TemplateFiles {
    param(
        [string]$ServerName,
        [string]$TemplatePath,
        [string]$DestinationPath
    )
    
    try {
        Write-Info "Copying template files from $TemplatePath to $DestinationPath..."
        
        # Create destination directory
        New-Item -ItemType Directory -Path $DestinationPath -Force | Out-Null
        
        # Copy all files except __pycache__
        Get-ChildItem -Path $TemplatePath -Exclude '__pycache__' | ForEach-Object {
            if ($_.PSIsContainer -and $_.Name -ne '__pycache__') {
                # Skip __pycache__ directories
                return
            }
            elseif (-not $_.PSIsContainer) {
                Copy-Item -Path $_.FullName -Destination $DestinationPath -Force
                Write-Success "  Copied $($_.Name)"
            }
        }
        
        Write-Success "Template files copied successfully"
        return $true
    }
    catch {
        Write-Error-Custom "Failed to copy template files: $_"
        return $false
    }
}

# Replace placeholders in files
function Update-PlaceholdersInFile {
    param(
        [string]$FilePath,
        [hashtable]$Replacements
    )
    
    try {
        $content = Get-Content $FilePath -Raw -Encoding UTF8
        
        foreach ($key in $Replacements.Keys) {
            $content = $content -replace [regex]::Escape($key), $Replacements[$key]
        }
        
        Set-Content -Path $FilePath -Value $content -Encoding UTF8 -NoNewline
        return $true
    }
    catch {
        Write-Error-Custom "Failed to update placeholders in ${FilePath}: $_"
        return $false
    }
}

# Update config.yaml with new server entry
function Update-ConfigYaml {
    param(
        [string]$ConfigPath,
        [string]$ServerName,
        [int]$Port,
        [string]$ModulePath
    )
    
    try {
        Write-Info "Updating config.yaml..."
        
        $content = Get-Content $ConfigPath -Raw -Encoding UTF8
        
        # Find the custom_servers section
        if ($content -match 'custom_servers:') {
            # Check if there are existing custom servers
            $customServersSection = $content -split 'custom_servers:'
            $afterCustomServers = $customServersSection[1]
            
            # Prepare the new entry
            $newEntry = @"

  - name: $ServerName
    module: $ModulePath
    host: 0.0.0.0
    port: $Port
    enabled: true
"@
            
            # Find where to insert (before the next major section or end of file)
            # Look for the next section starting with a non-indented line
            $lines = $afterCustomServers -split "`n"
            $insertIndex = 0
            $foundNextSection = $false
            
            for ($i = 0; $i -lt $lines.Count; $i++) {
                $line = $lines[$i]
                # Check if this is a new section (starts at column 0 and ends with :)
                if ($line -match '^[a-zA-Z_].*:' -and $line -notmatch '^\s') {
                    $insertIndex = $i
                    $foundNextSection = $true
                    break
                }
            }
            
            if ($foundNextSection) {
                # Insert before the next section
                $beforeNextSection = ($lines[0..($insertIndex - 1)] -join "`n")
                $afterNextSection = ($lines[$insertIndex..($lines.Count - 1)] -join "`n")
                $afterCustomServers = $beforeNextSection + $newEntry + "`n" + $afterNextSection
            }
            else {
                # Append at the end of the custom_servers section
                $afterCustomServers = $afterCustomServers.TrimEnd() + $newEntry + "`n"
            }
            
            $content = $customServersSection[0] + 'custom_servers:' + $afterCustomServers
        }
        else {
            Write-Warning-Custom "custom_servers section not found in config.yaml"
            # Add the section at the end
            $content = $content.TrimEnd() + @"

# Custom Servers Configuration
custom_servers:
  - name: $ServerName
    module: $ModulePath
    host: 0.0.0.0
    port: $Port
    enabled: true
"@
        }
        
        Set-Content -Path $ConfigPath -Value $content -Encoding UTF8 -NoNewline
        Write-Success "config.yaml updated successfully"
        return $true
    }
    catch {
        Write-Error-Custom "Failed to update config.yaml: $_"
        return $false
    }
}

# Create test file stub
function New-TestFile {
    param(
        [string]$ServerName,
        [string]$ClassName,
        [string]$ModulePath,
        [string]$TestsPath
    )
    
    try {
        # Create tests/custom directory if it doesn't exist
        New-Item -ItemType Directory -Path $TestsPath -Force | Out-Null
        
        # Create __init__.py if it doesn't exist
        $initFile = Join-Path $TestsPath "__init__.py"
        if (-not (Test-Path $initFile)) {
            Set-Content -Path $initFile -Value '"""Custom server tests."""' -Encoding UTF8
        }
        
        # Create test file
        $testFileName = "test_$($ServerName -replace '-', '_').py"
        $testFilePath = Join-Path $TestsPath $testFileName
        
        $testContent = @"
"""
Tests for the $ClassName MCP Server.

This is a test stub created by the server generator.
Add comprehensive tests for your custom server here.
"""

import pytest
from $ModulePath import $ClassName


class Test${ClassName}Initialization:
    """Test $ClassName initialization."""
    
    def test_server_can_be_instantiated(self):
        """Test that $ClassName can be instantiated."""
        server = $ClassName()
        assert server is not None
    
    def test_server_inherits_from_base(self):
        """Test that $ClassName properly inherits from BaseMCPServer."""
        server = $ClassName()
        assert hasattr(server, 'tool_registry')
        assert hasattr(server, 'server_state')
        assert hasattr(server, 'logger')
        assert hasattr(server, 'config')
    
    def test_server_name(self):
        """Test that server name is correct."""
        server = $ClassName()
        assert server.get_server_name() == "$ServerName"
    
    def test_server_version(self):
        """Test that server version is correct."""
        server = $ClassName()
        version = server.get_server_version()
        # Version should follow semantic versioning
        parts = version.split(".")
        assert len(parts) == 3
        assert all(part.isdigit() for part in parts)


class Test${ClassName}ToolRegistration:
    """Test $ClassName tool registration."""
    
    def test_tools_are_registered(self):
        """Test that tools are registered during initialization."""
        server = $ClassName()
        # Update this test based on your actual number of tools
        assert server.tool_registry.count() >= 0
    
    # Add more tests for your specific tools here
    # Example:
    # def test_my_tool_exists(self):
    #     server = $ClassName()
    #     assert server.tool_registry.tool_exists("my_tool")


# Add more test classes for your specific tools
# Example:
# class TestMyTool:
#     """Test the my_tool functionality."""
#     
#     @pytest.mark.asyncio
#     async def test_my_tool_with_valid_input(self):
#         server = $ClassName()
#         handler = server.tool_registry.get_handler("my_tool")
#         result = await handler({"input": "test"})
#         assert result.isError is False
"@
        
        Set-Content -Path $testFilePath -Value $testContent -Encoding UTF8
        Write-Success "Test file created at $testFilePath"
        return $true
    }
    catch {
        Write-Error-Custom "Failed to create test file: $_"
        return $false
    }
}

# Display next steps
function Show-NextSteps {
    param(
        [string]$ServerName,
        [string]$ServerPath,
        [string]$TestPath,
        [int]$Port
    )
    
    Write-Section "SUCCESS! Server Created"
    
    Write-Host "Your new MCP server has been created successfully!" -ForegroundColor Green
    Write-Host ""
    
    Write-Host "📁 Server Location:" -ForegroundColor Cyan
    Write-Host "   $ServerPath" -ForegroundColor White
    Write-Host ""
    
    Write-Host "🧪 Test Location:" -ForegroundColor Cyan
    Write-Host "   $TestPath" -ForegroundColor White
    Write-Host ""
    
    Write-Host "⚙️  Configuration:" -ForegroundColor Cyan
    Write-Host "   Server added to config.yaml on port $Port" -ForegroundColor White
    Write-Host ""
    
    Write-Section "NEXT STEPS"
    
    Write-Host "1. Customize your server:" -ForegroundColor Yellow
    Write-Host "   • Edit $ServerPath/server.py" -ForegroundColor Gray
    Write-Host "   • Update the class name and server info" -ForegroundColor Gray
    Write-Host "   • Add your custom tools in register_tools()" -ForegroundColor Gray
    Write-Host ""
    
    Write-Host "2. Test your server:" -ForegroundColor Yellow
    Write-Host "   # Run in stdio mode (for Claude Desktop)" -ForegroundColor Gray
    Write-Host "   python -m src.custom.$($ServerName -replace '-', '_').server" -ForegroundColor Gray
    Write-Host ""
    Write-Host "   # Run in HTTP mode" -ForegroundColor Gray
    Write-Host "   python -m src.custom.$($ServerName -replace '-', '_').server --transport http --port $Port" -ForegroundColor Gray
    Write-Host ""
    
    Write-Host "3. Write tests:" -ForegroundColor Yellow
    Write-Host "   • Edit $TestPath" -ForegroundColor Gray
    Write-Host "   • Add comprehensive tests for your tools" -ForegroundColor Gray
    Write-Host "   • Run tests:" -ForegroundColor Gray
    Write-Host "     pytest $TestPath -v" -ForegroundColor Gray
    Write-Host ""
    
    Write-Host "4. Update documentation:" -ForegroundColor Yellow
    Write-Host "   • Edit $ServerPath/README.md" -ForegroundColor Gray
    Write-Host "   • Document your tools and usage examples" -ForegroundColor Gray
    Write-Host ""
    
    Write-Host "5. Start developing:" -ForegroundColor Yellow
    Write-Host "   • See $ServerPath/server.py for the skeleton template" -ForegroundColor Gray
    Write-Host "   • Check src/builtin/math_server/ for examples" -ForegroundColor Gray
    Write-Host "   • Read src/custom/README.md for best practices" -ForegroundColor Gray
    Write-Host ""
    
    Write-Section "HELPFUL RESOURCES"
    
    Write-Host "📖 Documentation:" -ForegroundColor Cyan
    Write-Host "   • Skeleton Template: src/templates/skeleton_server/README.md" -ForegroundColor Gray
    Write-Host "   • Custom Servers Guide: src/custom/README.md" -ForegroundColor Gray
    Write-Host "   • Core Framework: src/core/README.md" -ForegroundColor Gray
    Write-Host ""
    
    Write-Host "💡 Examples:" -ForegroundColor Cyan
    Write-Host "   • Math Server: src/builtin/math_server/" -ForegroundColor Gray
    Write-Host "   • Stats Server: src/builtin/stats_server/" -ForegroundColor Gray
    Write-Host ""
    
    Write-Success "Happy coding! 🚀"
    Write-Host ""
}

# Main script execution
function Main {
    Write-Host ""
    Write-Host "╔════════════════════════════════════════════════════════════════╗" -ForegroundColor Cyan
    Write-Host "║          MCP Server Generator - Interactive Setup             ║" -ForegroundColor Cyan
    Write-Host "╚════════════════════════════════════════════════════════════════╝" -ForegroundColor Cyan
    Write-Host ""
    
    # Check if we're in the correct directory
    if (-not (Test-Path "src/templates/skeleton_server")) {
        Write-Error-Custom "Template not found. Please run this script from the repository root."
        Write-Info "Expected: src/templates/skeleton_server/"
        exit 1
    }
    
    $configPath = "config.yaml"
    if (-not (Test-Path $configPath)) {
        Write-Warning-Custom "config.yaml not found, using config.example.yaml"
        $configPath = "config.example.yaml"
        if (-not (Test-Path $configPath)) {
            Write-Error-Custom "No config file found. Please create config.yaml first."
            exit 1
        }
    }
    
    # Get server name
    if (-not $Name -and -not $NonInteractive) {
        Write-Info "Enter server name (alphanumeric and hyphens, e.g., 'my-weather-server'):"
        $Name = Read-Host "Server name"
    }
    
    # Validate server name
    $nameValidation = Test-ServerName -ServerName $Name
    if (-not $nameValidation.Valid) {
        Write-Error-Custom $nameValidation.Error
        exit 1
    }
    $serverName = $nameValidation.Name
    Write-Success "Server name: $serverName"
    
    # Get description
    if (-not $Description -and -not $NonInteractive) {
        Write-Info "Enter a brief description of your server:"
        $Description = Read-Host "Description"
    }
    
    if ([string]::IsNullOrWhiteSpace($Description)) {
        $Description = "Custom MCP server created from skeleton template"
    }
    Write-Success "Description: $Description"
    
    # Get port
    if ($Port -eq 0) {
        $defaultPort = Get-NextAvailablePort -ConfigPath $configPath
        if (-not $NonInteractive) {
            Write-Info "Enter port number (default: $defaultPort):"
            $portInput = Read-Host "Port"
            if ([string]::IsNullOrWhiteSpace($portInput)) {
                $Port = $defaultPort
            }
            else {
                $Port = [int]$portInput
            }
        }
        else {
            $Port = $defaultPort
        }
    }
    
    # Validate port
    $portValidation = Test-Port -PortNumber $Port -ConfigPath $configPath
    if (-not $portValidation.Valid) {
        Write-Error-Custom $portValidation.Error
        exit 1
    }
    Write-Success "Port: $Port"
    
    # Get author
    if (-not $Author -and -not $NonInteractive) {
        Write-Info "Enter author name (optional, press Enter to skip):"
        $Author = Read-Host "Author"
    }
    
    if (-not [string]::IsNullOrWhiteSpace($Author)) {
        Write-Success "Author: $Author"
    }
    
    Write-Host ""
    Write-Section "CREATING SERVER"
    
    # Define paths
    # Note: Python package names must use underscores, but server name can use hyphens
    $serverNameUnderscore = $serverName -replace '-', '_'
    $templatePath = "src/templates/skeleton_server"
    $serverPath = "src/custom/$serverNameUnderscore"
    $testPath = "tests/custom/test_$serverNameUnderscore.py"
    $serverNameWithSpaces = $serverName -replace '-', ' '
    $className = ((Get-Culture).TextInfo.ToTitleCase($serverNameWithSpaces)) -replace ' ', ''
    $modulePath = "src.custom.$serverNameUnderscore.server"
    
    Write-Info "Template: $templatePath"
    Write-Info "Destination: $serverPath"
    Write-Info "Module: $modulePath"
    Write-Info "Class: $className"
    Write-Host ""
    
    # Step 1: Copy template files
    $success = Copy-TemplateFiles -ServerName $serverName -TemplatePath $templatePath -DestinationPath $serverPath
    if (-not $success) {
        Write-Error-Custom "Failed to copy template files"
        exit 1
    }
    
    # Step 2: Update placeholders in server.py
    Write-Info "Updating placeholders in server.py..."
    $serverFile = Join-Path $serverPath "server.py"
    
    $replacements = @{
        'SkeletonServer' = $className
        'skeleton-server' = $serverName
        'Skeleton MCP Server - Template for Creating New MCP Servers' = $Description
        'Skeleton MCP Server - A minimal template for creating new servers' = $Description
    }
    
    $success = Update-PlaceholdersInFile -FilePath $serverFile -Replacements $replacements
    if (-not $success) {
        Write-Error-Custom "Failed to update server.py"
        exit 1
    }
    Write-Success "server.py updated successfully"
    
    # Step 3: Update __init__.py
    Write-Info "Updating placeholders in __init__.py..."
    $initFile = Join-Path $serverPath "__init__.py"
    
    $initReplacements = @{
        'SkeletonServer' = $className
        'Skeleton MCP Server Template' = "$className Template"
    }
    
    $success = Update-PlaceholdersInFile -FilePath $initFile -Replacements $initReplacements
    if (-not $success) {
        Write-Error-Custom "Failed to update __init__.py"
        exit 1
    }
    Write-Success "__init__.py updated successfully"
    
    # Step 4: Update config.yaml
    $success = Update-ConfigYaml -ConfigPath $configPath -ServerName $serverName -Port $Port -ModulePath $modulePath
    if (-not $success) {
        Write-Error-Custom "Failed to update config.yaml"
        exit 1
    }
    
    # Step 5: Create test file
    Write-Info "Creating test file stub..."
    $success = New-TestFile -ServerName $serverName -ClassName $className -ModulePath $modulePath -TestsPath "tests/custom"
    if (-not $success) {
        Write-Error-Custom "Failed to create test file"
        exit 1
    }
    
    # Display next steps
    Show-NextSteps -ServerName $serverName -ServerPath $serverPath -TestPath $testPath -Port $Port
}

# Run main function
try {
    Main
}
catch {
    Write-Error-Custom "An error occurred: $_"
    Write-Host $_.ScriptStackTrace -ForegroundColor Red
    exit 1
}
