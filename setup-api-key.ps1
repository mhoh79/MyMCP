#!/usr/bin/env pwsh

<#
.SYNOPSIS
    Automated API Key Generation and Setup Script for MCP Servers
    
.DESCRIPTION
    This script automates the generation and setup of secure API keys for the MCP servers.
    It supports multiple deployment methods: local .env files, Windows environment variables,
    and GitHub Codespaces configuration with step-by-step instructions.
    
.PARAMETER Method
    Local setup method: 'env' (default) for .env file, 'envvar' for Windows environment variable
    
.PARAMETER Environment
    Target environment: 'local' (default), 'codespace', or 'both'
    
.EXAMPLE
    .\setup-api-key.ps1
    Creates .env file with generated API key (default local setup)
    
.EXAMPLE
    .\setup-api-key.ps1 -Method envvar
    Sets Windows user-level environment variable with generated API key
    
.EXAMPLE
    .\setup-api-key.ps1 -Environment codespace
    Generates API key and provides GitHub Codespaces setup instructions
    
.EXAMPLE
    .\setup-api-key.ps1 -Environment both
    Sets up local .env file and provides Codespaces instructions
    
.EXAMPLE
    .\setup-api-key.ps1 -Method envvar -Environment both
    Sets Windows environment variable and provides Codespaces instructions
#>

param(
    [ValidateSet('env', 'envvar')]
    [string]$Method = 'env',
    
    [ValidateSet('local', 'codespace', 'both')]
    [string]$Environment = 'local'
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

# Function to generate cryptographically secure API key
function New-ApiKey {
    Write-Info "Generating cryptographically secure API key..."
    
    try {
        # Use .NET's RNGCryptoServiceProvider for secure random bytes
        $rng = [System.Security.Cryptography.RandomNumberGenerator]::Create()
        $bytes = New-Object byte[] 24  # 24 bytes = 32 base64 characters
        $rng.GetBytes($bytes)
        $rng.Dispose()
        
        # Convert to base64 and add prefix
        $base64 = [Convert]::ToBase64String($bytes)
        # Remove padding and make URL-safe
        $base64 = $base64.Replace('+', '-').Replace('/', '_').Replace('=', '')
        
        # Add MCP-specific prefix
        $apiKey = "sk_mcp_$base64"
        
        Write-Success "API key generated successfully"
        return $apiKey
    }
    catch {
        Write-Error-Custom "Failed to generate API key: $_"
        exit 1
    }
}

# Function to copy to clipboard
function Set-ClipboardText {
    param([string]$Text)
    
    try {
        # Try Windows clipboard first
        if (Get-Command Set-Clipboard -ErrorAction SilentlyContinue) {
            Set-Clipboard -Value $Text
            return $true
        }
        # Try xclip on Linux
        elseif (Get-Command xclip -ErrorAction SilentlyContinue) {
            $Text | xclip -selection clipboard
            return $true
        }
        # Try pbcopy on macOS
        elseif (Get-Command pbcopy -ErrorAction SilentlyContinue) {
            $Text | pbcopy
            return $true
        }
        else {
            return $false
        }
    }
    catch {
        return $false
    }
}

# Function to check if python-dotenv is installed
function Test-PythonDotenv {
    $pythonCmd = if (Get-Command python -ErrorAction SilentlyContinue) { "python" } else { "python3" }
    
    try {
        $result = & $pythonCmd -c "import dotenv" 2>&1
        return $LASTEXITCODE -eq 0
    }
    catch {
        return $false
    }
}

# Function to install python-dotenv
function Install-PythonDotenv {
    Write-Info "Installing python-dotenv..."
    
    $pythonCmd = if (Get-Command python -ErrorAction SilentlyContinue) { "python" } else { "python3" }
    
    try {
        & $pythonCmd -m pip install python-dotenv --quiet
        if ($LASTEXITCODE -eq 0) {
            Write-Success "python-dotenv installed successfully"
            return $true
        }
        else {
            Write-Error-Custom "Failed to install python-dotenv"
            return $false
        }
    }
    catch {
        Write-Error-Custom "Failed to install python-dotenv: $_"
        return $false
    }
}

# Function to verify .env is in .gitignore
function Test-GitIgnoreEntry {
    param([string]$Entry)
    
    if (Test-Path ".gitignore") {
        $content = Get-Content ".gitignore" -Raw
        return $content -match [regex]::Escape($Entry)
    }
    return $false
}

# Function to add entry to .gitignore
function Add-GitIgnoreEntry {
    param([string]$Entry)
    
    if (-not (Test-Path ".gitignore")) {
        Write-Warning-Custom ".gitignore file not found, creating one..."
        New-Item -ItemType File -Path ".gitignore" | Out-Null
    }
    
    # Add entry with a comment
    Add-Content -Path ".gitignore" -Value "`n# Environment variables (contains secrets)`n$Entry"
    Write-Success "Added '$Entry' to .gitignore"
}

# Function to create/update .env file
function Set-EnvFile {
    param([string]$ApiKey)
    
    Write-Section "LOCAL SETUP: .env FILE"
    
    # Check if .env exists
    $envExists = Test-Path ".env"
    
    if ($envExists) {
        Write-Warning-Custom ".env file already exists"
        $overwrite = Read-Host "Do you want to update the API key? (y/N)"
        if ($overwrite -ne 'y' -and $overwrite -ne 'Y') {
            Write-Info "Skipping .env file update"
            return
        }
    }
    
    # Verify .gitignore entry
    if (-not (Test-GitIgnoreEntry ".env")) {
        Write-Warning-Custom ".env is not in .gitignore"
        $addToGitignore = Read-Host "Add .env to .gitignore? (Y/n)"
        if ($addToGitignore -ne 'n' -and $addToGitignore -ne 'N') {
            Add-GitIgnoreEntry ".env"
        }
        else {
            Write-Warning-Custom "⚠️  WARNING: .env file may be committed to version control!"
        }
    }
    else {
        Write-Success ".env is already in .gitignore"
    }
    
    # Read existing .env or use .env.example as template
    $envContent = @()
    if ($envExists) {
        $envContent = Get-Content ".env"
    }
    elseif (Test-Path ".env.example") {
        Write-Info "Using .env.example as template"
        $envContent = Get-Content ".env.example"
    }
    
    # Update or add MCP_API_KEY
    $keyFound = $false
    $updatedContent = @()
    
    foreach ($line in $envContent) {
        if ($line -match '^MCP_API_KEY=') {
            $updatedContent += "MCP_API_KEY=$ApiKey"
            $keyFound = $true
        }
        else {
            $updatedContent += $line
        }
    }
    
    # Add MCP_API_KEY if not found
    if (-not $keyFound) {
        if ($updatedContent.Count -gt 0 -and $updatedContent[-1] -ne "") {
            $updatedContent += ""
        }
        $updatedContent += "# API Key for authentication (generated by setup-api-key.ps1)"
        $updatedContent += "MCP_API_KEY=$ApiKey"
    }
    
    # Write to .env file
    try {
        $updatedContent | Set-Content ".env" -Encoding UTF8
        Write-Success ".env file created/updated with new API key"
        
        # Verify python-dotenv is installed
        Write-Host ""
        Write-Info "Checking python-dotenv installation..."
        if (-not (Test-PythonDotenv)) {
            Write-Warning-Custom "python-dotenv is not installed"
            $installDotenv = Read-Host "Install python-dotenv? (Y/n)"
            if ($installDotenv -ne 'n' -and $installDotenv -ne 'N') {
                Install-PythonDotenv | Out-Null
            }
            else {
                Write-Warning-Custom "You'll need to install python-dotenv manually: pip install python-dotenv"
            }
        }
        else {
            Write-Success "python-dotenv is already installed"
        }
        
        Write-Host ""
        Write-Success "Local .env setup complete!"
        Write-Info "The .env file is located at: $(Get-Location)\.env"
        
    }
    catch {
        Write-Error-Custom "Failed to write .env file: $_"
        exit 1
    }
}

# Function to set Windows environment variable
function Set-WindowsEnvVar {
    param([string]$ApiKey)
    
    Write-Section "LOCAL SETUP: WINDOWS ENVIRONMENT VARIABLE"
    
    # Check if MCP_API_KEY already exists
    $existingKey = [System.Environment]::GetEnvironmentVariable("MCP_API_KEY", "User")
    if ($existingKey) {
        Write-Warning-Custom "MCP_API_KEY environment variable already exists"
        $overwrite = Read-Host "Do you want to update it? (y/N)"
        if ($overwrite -ne 'y' -and $overwrite -ne 'Y') {
            Write-Info "Skipping environment variable update"
            return
        }
    }
    
    try {
        # Set user-level environment variable (permanent)
        [System.Environment]::SetEnvironmentVariable("MCP_API_KEY", $ApiKey, "User")
        Write-Success "MCP_API_KEY environment variable set successfully"
        
        Write-Host ""
        Write-Info "The environment variable has been set at the USER level (permanent)"
        Write-Info "Restart your terminal/PowerShell for changes to take effect"
        Write-Info "To verify: `$env:MCP_API_KEY (may require restart)"
        
    }
    catch {
        Write-Error-Custom "Failed to set environment variable: $_"
        exit 1
    }
}

# Function to display Codespaces instructions
function Show-CodespacesInstructions {
    param([string]$ApiKey)
    
    Write-Section "GITHUB CODESPACES SETUP"
    
    Write-Host "Follow these steps to configure your API key in GitHub Codespaces:" -ForegroundColor Cyan
    Write-Host ""
    
    Write-Host "1. Go to GitHub Codespaces Secrets:" -ForegroundColor White
    Write-Host "   https://github.com/settings/codespaces" -ForegroundColor Blue
    Write-Host ""
    
    Write-Host "2. Click 'New secret' button" -ForegroundColor White
    Write-Host ""
    
    Write-Host "3. Configure the secret:" -ForegroundColor White
    Write-Host "   Name:  " -NoNewline -ForegroundColor White
    Write-Host "MCP_API_KEY" -ForegroundColor Yellow
    Write-Host "   Value: " -NoNewline -ForegroundColor White
    Write-Host $ApiKey -ForegroundColor Yellow
    Write-Host ""
    
    Write-Host "4. Set repository access:" -ForegroundColor White
    Write-Host "   • Select repositories that need access to this secret" -ForegroundColor Gray
    Write-Host "   • Or choose 'All repositories' (less secure)" -ForegroundColor Gray
    Write-Host ""
    
    Write-Host "5. Click 'Add secret'" -ForegroundColor White
    Write-Host ""
    
    Write-Host "6. Rebuild your Codespace:" -ForegroundColor White
    Write-Host "   • Open Command Palette (Ctrl+Shift+P or Cmd+Shift+P)" -ForegroundColor Gray
    Write-Host "   • Search for 'Codespaces: Rebuild Container'" -ForegroundColor Gray
    Write-Host "   • Or restart Codespace to apply the new secret" -ForegroundColor Gray
    Write-Host ""
    
    Write-Warning-Custom "IMPORTANT: Codespaces secrets are only available after container rebuild!"
    Write-Host ""
}

# Function to display verification commands
function Show-VerificationCommands {
    param([string]$Method, [string]$Environment)
    
    Write-Section "VERIFICATION & TESTING"
    
    Write-Host "Test your API key configuration:" -ForegroundColor Cyan
    Write-Host ""
    
    if ($Environment -eq 'local' -or $Environment -eq 'both') {
        if ($Method -eq 'env') {
            Write-Host "1. Verify .env file:" -ForegroundColor White
            Write-Host "   Get-Content .env | Select-String MCP_API_KEY" -ForegroundColor Gray
            Write-Host ""
            
            Write-Host "2. Test with Python:" -ForegroundColor White
            Write-Host "   python -c `"from dotenv import load_dotenv; import os; load_dotenv(); print('API Key loaded:', 'MCP_API_KEY' in os.environ)`"" -ForegroundColor Gray
            Write-Host ""
        }
        else {
            Write-Host "1. Verify environment variable (may require terminal restart):" -ForegroundColor White
            Write-Host "   `$env:MCP_API_KEY" -ForegroundColor Gray
            Write-Host ""
            
            Write-Host "2. Or check in System Properties:" -ForegroundColor White
            Write-Host "   [System.Environment]::GetEnvironmentVariable('MCP_API_KEY', 'User')" -ForegroundColor Gray
            Write-Host ""
        }
        
        Write-Host "3. Start the server with authentication enabled:" -ForegroundColor White
        Write-Host "   `$env:MCP_AUTH_ENABLED='true'; python src/math_server/server.py --transport http" -ForegroundColor Gray
        Write-Host ""
        
        Write-Host "4. Test API request with authentication:" -ForegroundColor White
        Write-Host "   curl -H `"Authorization: Bearer `$env:MCP_API_KEY`" http://localhost:8000/health" -ForegroundColor Gray
        Write-Host ""
    }
    
    if ($Environment -eq 'codespace' -or $Environment -eq 'both') {
        Write-Host "Codespaces Testing:" -ForegroundColor White
        Write-Host "1. After rebuilding, verify secret is loaded:" -ForegroundColor Gray
        Write-Host "   echo `$MCP_API_KEY" -ForegroundColor Gray
        Write-Host ""
        
        Write-Host "2. Start servers with authentication:" -ForegroundColor Gray
        Write-Host "   export MCP_AUTH_ENABLED=true" -ForegroundColor Gray
        Write-Host "   ./start-http-servers.sh start" -ForegroundColor Gray
        Write-Host ""
        
        Write-Host "3. Test with your Codespace URL:" -ForegroundColor Gray
        Write-Host "   curl -H `"Authorization: Bearer `$MCP_API_KEY`" https://YOUR-CODESPACE-8000.app.github.dev/health" -ForegroundColor Gray
        Write-Host ""
    }
}

# Function to display security reminders
function Show-SecurityReminders {
    Write-Section "SECURITY BEST PRACTICES"
    
    Write-Host "🔒 Keep Your API Key Secure:" -ForegroundColor Yellow
    Write-Host ""
    Write-Host "✓ Never commit .env files to version control" -ForegroundColor Green
    Write-Host "✓ Never share your API key in chat, email, or documentation" -ForegroundColor Green
    Write-Host "✓ Use different keys for dev/staging/production environments" -ForegroundColor Green
    Write-Host "✓ Rotate API keys regularly (every 90 days recommended)" -ForegroundColor Green
    Write-Host "✓ Store keys in secure password managers" -ForegroundColor Green
    Write-Host ""
    Write-Host "⚠  Revoke immediately if key is exposed" -ForegroundColor Yellow
    Write-Host "⚠  Always use HTTPS in production (never HTTP)" -ForegroundColor Yellow
    Write-Host "⚠  Enable rate limiting in production deployments" -ForegroundColor Yellow
    Write-Host ""
    
    Write-Info "For more security guidance, see SECURITY.md in the repository"
    Write-Host ""
}

# Main script execution
function Main {
    Write-Host ""
    Write-Host "╔════════════════════════════════════════════════════════════════╗" -ForegroundColor Cyan
    Write-Host "║       MCP Servers - API Key Generation & Setup Script         ║" -ForegroundColor Cyan
    Write-Host "╚════════════════════════════════════════════════════════════════╝" -ForegroundColor Cyan
    Write-Host ""
    
    # Display configuration
    Write-Info "Configuration:"
    Write-Host "  Method:      $Method" -ForegroundColor Gray
    Write-Host "  Environment: $Environment" -ForegroundColor Gray
    Write-Host ""
    
    # Generate API key
    $apiKey = New-ApiKey
    Write-Host ""
    
    # Display generated key
    Write-Host "Generated API Key:" -ForegroundColor Green
    Write-Host $apiKey -ForegroundColor Yellow
    Write-Host ""
    
    # Try to copy to clipboard
    if (Set-ClipboardText $apiKey) {
        Write-Success "API key copied to clipboard!"
    }
    else {
        Write-Warning-Custom "Could not copy to clipboard automatically"
        Write-Info "Please copy the key manually if needed"
    }
    Write-Host ""
    
    # Execute based on environment and method
    if ($Environment -eq 'local' -or $Environment -eq 'both') {
        if ($Method -eq 'env') {
            Set-EnvFile -ApiKey $apiKey
        }
        else {
            Set-WindowsEnvVar -ApiKey $apiKey
        }
    }
    
    if ($Environment -eq 'codespace' -or $Environment -eq 'both') {
        Show-CodespacesInstructions -ApiKey $apiKey
    }
    
    # Show verification commands
    Show-VerificationCommands -Method $Method -Environment $Environment
    
    # Show security reminders
    Show-SecurityReminders
    
    Write-Success "Setup complete!"
    Write-Host ""
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
