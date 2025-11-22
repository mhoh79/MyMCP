#!/usr/bin/env pwsh

<#
.SYNOPSIS
    MCP Servers Launcher Script (Windows/PowerShell)
    
.DESCRIPTION
    This script starts both math_server and stats_server in HTTP mode
    with proper configuration and process management.
    
.PARAMETER Stop
    Stop running servers instead of starting them
    
.PARAMETER Status
    Check status of running servers
    
.PARAMETER Restart
    Restart servers (stop if running, then start)
    
.EXAMPLE
    .\start-http-servers.ps1
    Start both MCP servers
    
.EXAMPLE
    .\start-http-servers.ps1 -Stop
    Stop running servers
    
.EXAMPLE
    .\start-http-servers.ps1 -Status
    Check server status
    
.EXAMPLE
    .\start-http-servers.ps1 -Restart
    Restart servers
#>

param(
    [switch]$Stop,
    [switch]$Status,
    [switch]$Restart
)

# Configuration
$MathPort = 8000
$StatsPort = 8001
$ConfigFile = "config.yaml"
$ConfigExample = "config.example.yaml"
$PidsFile = ".pids"
$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path

# Function to print colored output
function Write-Status {
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

# Function to check if a command exists
function Test-Command {
    param([string]$Command)
    $null -ne (Get-Command $Command -ErrorAction SilentlyContinue)
}

# Function to check if a port is in use
function Test-PortInUse {
    param([int]$Port)
    
    # Try Windows-specific command first
    if (Test-Command "Get-NetTCPConnection") {
        $connections = Get-NetTCPConnection -LocalPort $Port -State Listen -ErrorAction SilentlyContinue
        return $null -ne $connections
    }
    
    # Fallback for Linux/Mac using netstat or lsof
    try {
        if (Test-Command "lsof") {
            $result = & lsof -Pi :$Port -sTCP:LISTEN -t 2>$null
            return $LASTEXITCODE -eq 0 -and $result
        }
        elseif (Test-Command "netstat") {
            $result = & netstat -tuln 2>$null | Select-String ":$Port "
            return $null -ne $result
        }
        else {
            # Last resort: try to bind to the port using Python
            $pythonCmd = if (Test-Command python) { "python" } else { "python3" }
            $testScript = "import socket; s = socket.socket(socket.AF_INET, socket.SOCK_STREAM); s.bind(('', $Port)); s.close()"
            & $pythonCmd -c $testScript 2>$null
            return $LASTEXITCODE -ne 0
        }
    }
    catch {
        # If all else fails, assume port is available
        return $false
    }
}

# Function to check if a process is running
function Test-ProcessRunning {
    param([int]$ProcessId)
    try {
        $process = Get-Process -Id $ProcessId -ErrorAction SilentlyContinue
        return $null -ne $process
    }
    catch {
        return $false
    }
}

# Function to get the Codespace name
function Get-CodespaceName {
    if ($env:CODESPACE_NAME) {
        return $env:CODESPACE_NAME
    }
    elseif ($env:GITHUB_CODESPACES_PORT_FORWARDING_DOMAIN) {
        return "YOUR-CODESPACE"
    }
    else {
        return ""
    }
}

# Function to read PIDs from file
function Read-PidsFile {
    if (Test-Path $PidsFile) {
        $pids = @{}
        Get-Content $PidsFile | ForEach-Object {
            if ($_ -match '^(.+?)=(.+)$') {
                $pids[$matches[1]] = [int]$matches[2]
            }
        }
        return $pids
    }
    return @{}
}

# Function to write PIDs to file
function Write-PidsFile {
    param([hashtable]$Pids)
    $content = $Pids.GetEnumerator() | ForEach-Object { "$($_.Key)=$($_.Value)" }
    $content | Set-Content $PidsFile
}

# Function to start servers
function Start-Servers {
    Write-Host "🚀 Starting MCP Servers in HTTP mode..." -ForegroundColor Blue
    Write-Host ""

    # Change to script directory
    Set-Location $ScriptDir

    # Check if Python is installed
    if (-not (Test-Command python) -and -not (Test-Command python3)) {
        Write-Error-Custom "Python is not installed. Please install Python 3 to continue."
        exit 1
    }

    $pythonCmd = if (Test-Command python) { "python" } else { "python3" }
    $pythonVersion = & $pythonCmd --version 2>&1
    Write-Status "Python found: $pythonVersion"

    # Check if config file exists, create from example if missing
    if (-not (Test-Path $ConfigFile)) {
        if (Test-Path $ConfigExample) {
            Copy-Item $ConfigExample $ConfigFile
            Write-Warning-Custom "Config file not found. Created $ConfigFile from $ConfigExample"
        }
        else {
            Write-Error-Custom "Config file $ConfigFile and example $ConfigExample not found!"
            exit 1
        }
    }
    else {
        Write-Status "Config file found: $ConfigFile"
    }

    # Check if dependencies are installed
    $mcpInstalled = & $pythonCmd -c "import mcp" 2>&1
    if ($LASTEXITCODE -ne 0) {
        Write-Warning-Custom "MCP dependencies not found. Installing from requirements.txt..."
        if (Test-Path "requirements.txt") {
            & $pythonCmd -m pip install -r requirements.txt
            if ($LASTEXITCODE -ne 0) {
                Write-Error-Custom "Failed to install dependencies. Please run: pip install -r requirements.txt"
                exit 1
            }
        }
        else {
            Write-Error-Custom "requirements.txt not found!"
            exit 1
        }
    }

    # Check if servers are already running
    $existingPids = Read-PidsFile
    if ($existingPids.Count -gt 0) {
        Write-Warning-Custom "Server PIDs file found. Checking if servers are still running..."
        
        foreach ($entry in $existingPids.GetEnumerator()) {
            if (Test-ProcessRunning $entry.Value) {
                Write-Error-Custom "Server $($entry.Key) is already running with PID $($entry.Value)"
                Write-Info "To stop servers, run: .\start-http-servers.ps1 -Stop"
                exit 1
            }
        }
        
        # Remove stale PID file
        Remove-Item $PidsFile -ErrorAction SilentlyContinue
    }

    # Check if ports are available
    if (Test-PortInUse $MathPort) {
        Write-Error-Custom "Port $MathPort is already in use. Cannot start math_server."
        exit 1
    }

    if (Test-PortInUse $StatsPort) {
        Write-Error-Custom "Port $StatsPort is already in use. Cannot start stats_server."
        exit 1
    }

    # Create logs directory if it doesn't exist
    if (-not (Test-Path "logs")) {
        New-Item -ItemType Directory -Path "logs" | Out-Null
    }

    # Start math_server
    Write-Host ""
    Write-Info "Starting Math Server on port $MathPort..."
    
    $startProcessParams = @{
        FilePath = $pythonCmd
        ArgumentList = "src/math_server/server.py", "--transport", "http", "--host", "0.0.0.0", "--port", $MathPort, "--config", $ConfigFile
        RedirectStandardOutput = "logs/math_server.log"
        RedirectStandardError = "logs/math_server_error.log"
        PassThru = $true
    }
    
    # Add WindowStyle only on Windows
    if ($IsWindows -or $null -eq $IsWindows) {
        $startProcessParams['WindowStyle'] = 'Hidden'
    }
    
    $mathProcess = Start-Process @startProcessParams

    Start-Sleep -Seconds 2

    if (Test-ProcessRunning $mathProcess.Id) {
        Write-Status "Math Server started on port $MathPort (PID: $($mathProcess.Id))"
        $pids = @{ "math_server" = $mathProcess.Id }
    }
    else {
        Write-Error-Custom "Failed to start Math Server. Check logs/math_server_error.log for details."
        exit 1
    }

    # Start stats_server
    Write-Host ""
    Write-Info "Starting Stats Server on port $StatsPort..."
    
    $startProcessParams = @{
        FilePath = $pythonCmd
        ArgumentList = "src/stats_server/server.py", "--transport", "http", "--host", "0.0.0.0", "--port", $StatsPort, "--config", $ConfigFile
        RedirectStandardOutput = "logs/stats_server.log"
        RedirectStandardError = "logs/stats_server_error.log"
        PassThru = $true
    }
    
    # Add WindowStyle only on Windows
    if ($IsWindows -or $null -eq $IsWindows) {
        $startProcessParams['WindowStyle'] = 'Hidden'
    }
    
    $statsProcess = Start-Process @startProcessParams

    Start-Sleep -Seconds 2

    if (Test-ProcessRunning $statsProcess.Id) {
        Write-Status "Stats Server started on port $StatsPort (PID: $($statsProcess.Id))"
        $pids["stats_server"] = $statsProcess.Id
    }
    else {
        Write-Error-Custom "Failed to start Stats Server. Check logs/stats_server_error.log for details."
        # Kill math_server if stats_server failed
        Stop-Process -Id $mathProcess.Id -Force -ErrorAction SilentlyContinue
        exit 1
    }

    # Save PIDs to file
    Write-PidsFile $pids

    # Display connection URLs
    Write-Host ""
    Write-Host "📡 Connection URLs:" -ForegroundColor Blue
    Write-Host "  Math Server:  http://localhost:$MathPort"
    Write-Host "  Stats Server: http://localhost:$StatsPort"
    Write-Host ""

    # Display Codespaces URLs if applicable
    $codespace = Get-CodespaceName
    if ($codespace) {
        Write-Host "🌐 Codespaces URLs:" -ForegroundColor Blue
        if ($codespace -eq "YOUR-CODESPACE") {
            Write-Host "  Math Server:  https://YOUR-CODESPACE-$MathPort.app.github.dev"
            Write-Host "  Stats Server: https://YOUR-CODESPACE-$StatsPort.app.github.dev"
            Write-Host ""
            Write-Info "Replace YOUR-CODESPACE with your actual Codespace name"
        }
        else {
            Write-Host "  Math Server:  https://$codespace-$MathPort.app.github.dev"
            Write-Host "  Stats Server: https://$codespace-$StatsPort.app.github.dev"
        }
        Write-Host ""
    }

    Write-Info "To stop servers: .\start-http-servers.ps1 -Stop"
    Write-Info "To check status: .\start-http-servers.ps1 -Status"
    Write-Info "Server logs: logs/math_server.log, logs/stats_server.log"
    Write-Host ""
}

# Function to stop servers
function Stop-Servers {
    Write-Host "🛑 Stopping MCP Servers..." -ForegroundColor Blue
    Write-Host ""

    $pids = Read-PidsFile
    if ($pids.Count -eq 0) {
        Write-Error-Custom "No PID file found. Servers may not be running."
        exit 1
    }

    $stopped = 0
    foreach ($entry in $pids.GetEnumerator()) {
        $name = $entry.Key
        $procId = $entry.Value

        if (Test-ProcessRunning $procId) {
            Write-Info "Stopping $name (PID: $procId)..."
            
            try {
                # Try graceful shutdown first
                Stop-Process -Id $procId -ErrorAction Stop
                
                # Wait up to 10 seconds for graceful shutdown
                $count = 0
                while ((Test-ProcessRunning $procId) -and ($count -lt 10)) {
                    Start-Sleep -Seconds 1
                    $count++
                }
                
                # Force kill if still running
                if (Test-ProcessRunning $procId) {
                    Write-Warning-Custom "Forcing $name to stop..."
                    Stop-Process -Id $procId -Force -ErrorAction SilentlyContinue
                    Start-Sleep -Seconds 1
                }
                
                if (-not (Test-ProcessRunning $procId)) {
                    Write-Status "$name stopped successfully"
                    $stopped++
                }
                else {
                    Write-Error-Custom "Failed to stop $name (PID: $procId)"
                }
            }
            catch {
                Write-Error-Custom "Error stopping $name : $_"
            }
        }
        else {
            Write-Warning-Custom "$name (PID: $procId) is not running"
        }
    }

    # Remove PID file
    Remove-Item $PidsFile -ErrorAction SilentlyContinue

    if ($stopped -gt 0) {
        Write-Host ""
        Write-Status "All servers stopped"
    }
}

# Function to check server status
function Get-ServerStatus {
    Write-Host "📊 Server Status:" -ForegroundColor Blue
    Write-Host ""

    $pids = Read-PidsFile
    if ($pids.Count -eq 0) {
        Write-Info "No servers are currently running (no PID file found)"
        return
    }

    $running = 0
    foreach ($entry in $pids.GetEnumerator()) {
        $name = $entry.Key
        $procId = $entry.Value

        if (Test-ProcessRunning $procId) {
            Write-Status "$name is running (PID: $procId)"
            $running++
        }
        else {
            Write-Error-Custom "$name is not running (PID: $procId - stale)"
        }
    }

    if ($running -eq 0) {
        Write-Host ""
        Write-Warning-Custom "No servers are running. Removing stale PID file."
        Remove-Item $PidsFile -ErrorAction SilentlyContinue
    }
    
    Write-Host ""
}

# Main script logic
try {
    if ($Stop) {
        Stop-Servers
    }
    elseif ($Status) {
        Get-ServerStatus
    }
    elseif ($Restart) {
        $pids = Read-PidsFile
        if ($pids.Count -gt 0) {
            Stop-Servers
            Write-Host ""
        }
        Start-Servers
    }
    else {
        Start-Servers
    }
}
catch {
    Write-Error-Custom "An error occurred: $_"
    exit 1
}
