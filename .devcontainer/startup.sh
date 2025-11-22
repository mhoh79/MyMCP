#!/bin/bash

# Startup script for MCP servers in GitHub Codespaces
# This script:
# 1. Creates config.yaml from config.example.yaml if it doesn't exist
# 2. Starts both math and stats servers in HTTP mode in the background
# 3. Displays public URLs for accessing the servers

set -e  # Exit on error

echo "============================================"
echo "  MCP Servers - GitHub Codespaces Startup  "
echo "============================================"
echo ""

# Step 1: Create config.yaml if it doesn't exist
if [ ! -f "config.yaml" ]; then
    echo "📝 Creating config.yaml from config.example.yaml..."
    cp config.example.yaml config.yaml
    echo "✅ Configuration file created"
else
    echo "✅ Configuration file already exists"
fi

echo ""

# Step 2: Start Math Server (port 8000)
echo "🚀 Starting Math Server on port 8000..."
nohup python src/math_server/server.py --transport http --host 0.0.0.0 --port 8000 --config config.yaml > /tmp/math_server.log 2>&1 &
MATH_PID=$!
echo "✅ Math Server started (PID: $MATH_PID)"

# Step 3: Start Stats Server (port 8001)
echo "🚀 Starting Stats Server on port 8001..."
nohup python src/stats_server/server.py --transport http --host 0.0.0.0 --port 8001 --config config.yaml > /tmp/stats_server.log 2>&1 &
STATS_PID=$!
echo "✅ Stats Server started (PID: $STATS_PID)"

echo ""

# Step 4: Wait a moment for servers to start
echo "⏳ Waiting for servers to initialize..."
sleep 3

# Step 5: Display public URLs
echo ""
echo "============================================"
echo "  📡 Server URLs                           "
echo "============================================"
echo ""

# Get the Codespace name from environment variable
if [ -n "$CODESPACE_NAME" ]; then
    echo "Math Server:  https://${CODESPACE_NAME}-8000.app.github.dev"
    echo "Stats Server: https://${CODESPACE_NAME}-8001.app.github.dev"
    echo ""
    echo "🌐 Servers are publicly accessible via these URLs"
else
    echo "Math Server:  http://localhost:8000"
    echo "Stats Server: http://localhost:8001"
    echo ""
    echo "ℹ️  Running in local environment (not Codespaces)"
fi

echo ""
echo "============================================"
echo "  📋 Additional Information                "
echo "============================================"
echo ""
echo "Server logs:"
echo "  - Math Server:  /tmp/math_server.log"
echo "  - Stats Server: /tmp/stats_server.log"
echo ""
echo "View logs with:"
echo "  tail -f /tmp/math_server.log"
echo "  tail -f /tmp/stats_server.log"
echo ""
echo "✅ Startup complete!"
echo "============================================"
