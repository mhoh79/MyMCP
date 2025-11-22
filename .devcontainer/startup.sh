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

# Step 2: Validate server files exist
if [ ! -f "src/math_server/server.py" ]; then
    echo "❌ Error: src/math_server/server.py not found"
    exit 1
fi

if [ ! -f "src/stats_server/server.py" ]; then
    echo "❌ Error: src/stats_server/server.py not found"
    exit 1
fi

# Step 3: Start Math Server (port 8000)
echo "🚀 Starting Math Server on port 8000..."
nohup python src/math_server/server.py --transport http --host 0.0.0.0 --port 8000 --config config.yaml > /tmp/math_server.log 2>&1 &
MATH_PID=$!
echo "   Process started (PID: $MATH_PID)"

# Step 4: Start Stats Server (port 8001)
echo "🚀 Starting Stats Server on port 8001..."
nohup python src/stats_server/server.py --transport http --host 0.0.0.0 --port 8001 --config config.yaml > /tmp/stats_server.log 2>&1 &
STATS_PID=$!
echo "   Process started (PID: $STATS_PID)"

echo ""

# Step 5: Wait for servers to start and verify they're healthy
echo "⏳ Waiting for servers to initialize..."
MATH_READY=false
STATS_READY=false
MAX_ATTEMPTS=30  # 30 seconds max wait time

for _ in $(seq 1 $MAX_ATTEMPTS); do
    # Check if Math Server is ready
    if [ "$MATH_READY" = false ] && curl -s http://localhost:8000/health > /dev/null 2>&1; then
        MATH_READY=true
        echo "✅ Math Server is healthy"
    fi
    
    # Check if Stats Server is ready
    if [ "$STATS_READY" = false ] && curl -s http://localhost:8001/health > /dev/null 2>&1; then
        STATS_READY=true
        echo "✅ Stats Server is healthy"
    fi
    
    # Exit loop if both servers are ready
    if [ "$MATH_READY" = true ] && [ "$STATS_READY" = true ]; then
        break
    fi
    
    sleep 1
done

# Check if servers started successfully
if [ "$MATH_READY" = false ]; then
    echo "⚠️  Math Server failed to start or is not responding"
    echo "   Check logs: tail -f /tmp/math_server.log"
fi

if [ "$STATS_READY" = false ]; then
    echo "⚠️  Stats Server failed to start or is not responding"
    echo "   Check logs: tail -f /tmp/stats_server.log"
fi

# Exit if both servers failed
if [ "$MATH_READY" = false ] && [ "$STATS_READY" = false ]; then
    echo ""
    echo "❌ Both servers failed to start. Please check the logs above."
    exit 1
fi

# Step 5: Display public URLs
echo ""
echo "============================================"
echo "  📡 Server URLs                           "
echo "============================================"
echo ""

# Get the Codespace name from environment variable
# Note: GitHub Codespaces URL format is subject to change by GitHub
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
