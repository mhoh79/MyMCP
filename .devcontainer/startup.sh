#!/bin/bash

# Startup script for MCP servers in GitHub Codespaces
# This script:
# 1. Creates config.yaml from config.example.yaml if it doesn't exist
# 2. Starts dev servers (8000-8001) with no authentication
# 3. Starts prod servers (9000-9001) with authentication
# 4. Verifies health of all server instances
# 5. Displays public URLs for accessing the servers

set -e  # Exit on error

echo "============================================"
echo "  MCP Servers - GitHub Codespaces Startup  "
echo "============================================"
echo ""

# Detect environment mode (default: "codespace")
ENVIRONMENT="${ENVIRONMENT:-codespace}"
echo "🔧 Environment Mode: $ENVIRONMENT"
echo ""

# Step 1: Create config.yaml if it doesn't exist (for backward compatibility)
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

# Step 3: Start DEV servers (ports 8000-8001, no authentication)
echo "🚀 Starting DEV servers (no authentication)..."
echo ""

# Start Math Server (Dev - port 8000)
echo "  📊 Math Server (Dev) on port 8000..."
MCP_AUTH_ENABLED=false nohup python src/math_server/server.py --transport http --host 0.0.0.0 --port 8000 --config config.dev.yaml > /tmp/math_server_dev.log 2>&1 &
MATH_DEV_PID=$!
echo "     Process started (PID: $MATH_DEV_PID)"

# Start Stats Server (Dev - port 8001)
echo "  📈 Stats Server (Dev) on port 8001..."
MCP_AUTH_ENABLED=false nohup python src/stats_server/server.py --transport http --host 0.0.0.0 --port 8001 --config config.dev.yaml > /tmp/stats_server_dev.log 2>&1 &
STATS_DEV_PID=$!
echo "     Process started (PID: $STATS_DEV_PID)"

echo ""

# Step 4: Start PROD servers (ports 9000-9001, with authentication)
echo "🔐 Starting PROD servers (authentication required)..."
echo ""

# Get API key from environment or generate a random one for demo
if [ -z "$MCP_API_KEY" ]; then
    # Generate a random API key for this session using openssl
    if command -v openssl >/dev/null 2>&1; then
        API_KEY="demo-$(openssl rand -hex 16)"
    else
        # Fallback: generate using /dev/urandom with validation
        API_KEY="demo-$(head -c 16 /dev/urandom | xxd -p -c 32)"
    fi
    echo "  ⚠️  Generated temporary API key for this session."
    echo "     For production, set MCP_API_KEY in Codespaces Secrets."
else
    API_KEY="$MCP_API_KEY"
    echo "  ✅ Using API key from MCP_API_KEY environment variable."
fi

# Start Math Server (Prod - port 9000)
echo "  📊 Math Server (Prod) on port 9000..."
MCP_AUTH_ENABLED=true MCP_API_KEY="$API_KEY" nohup python src/math_server/server.py --transport http --host 0.0.0.0 --port 9000 --config config.prod.yaml > /tmp/math_server_prod.log 2>&1 &
MATH_PROD_PID=$!
echo "     Process started (PID: $MATH_PROD_PID)"

# Start Stats Server (Prod - port 9001)
echo "  📈 Stats Server (Prod) on port 9001..."
MCP_AUTH_ENABLED=true MCP_API_KEY="$API_KEY" nohup python src/stats_server/server.py --transport http --host 0.0.0.0 --port 9001 --config config.prod.yaml > /tmp/stats_server_prod.log 2>&1 &
STATS_PROD_PID=$!
echo "     Process started (PID: $STATS_PROD_PID)"

echo ""

# Step 4.5: Start CUSTOM servers from config.yaml (if any)
if [ "$ENVIRONMENT" = "codespace" ]; then
    echo "🔧 Starting CUSTOM servers (from config.yaml)..."
    echo ""
    
    # Use Python helper to extract custom server config
    CUSTOM_SERVERS=$(python3 get_custom_servers.py config.yaml 2>/dev/null)
    
    if [ -n "$CUSTOM_SERVERS" ]; then
        # Array to track custom server PIDs and info
        declare -a CUSTOM_SERVER_PIDS=()
        declare -a CUSTOM_SERVER_NAMES=()
        declare -a CUSTOM_SERVER_PORTS=()
        
        # Read custom servers and start each one
        while IFS='|' read -r name module host port; do
            echo "  🚀 Custom Server '$name' on port $port..."
            
            # Start custom server
            MCP_AUTH_ENABLED=false nohup python3 -m "$module" --transport http --host "$host" --port "$port" --config config.yaml > "/tmp/custom_${name}.log" 2>&1 &
            CUSTOM_PID=$!
            
            # Track PID, name, and port
            CUSTOM_SERVER_PIDS+=($CUSTOM_PID)
            CUSTOM_SERVER_NAMES+=("$name")
            CUSTOM_SERVER_PORTS+=($port)
            
            echo "     Process started (PID: $CUSTOM_PID)"
        done <<< "$CUSTOM_SERVERS"
        
        echo ""
        echo "  ✅ Started ${#CUSTOM_SERVER_PIDS[@]} custom server(s)"
    else
        echo "  ℹ️  No custom servers configured"
    fi
    echo ""
fi

# Step 5: Wait for servers to start and verify they're healthy
echo ""
echo "⏳ Waiting for servers to initialize..."
MATH_DEV_READY=false
STATS_DEV_READY=false
MATH_PROD_READY=false
STATS_PROD_READY=false
MAX_ATTEMPTS=30  # 30 seconds max wait time

# Initialize custom server health status array
if [ "$ENVIRONMENT" = "codespace" ] && [ -n "$CUSTOM_SERVERS" ]; then
    declare -a CUSTOM_SERVER_READY=()
    for _ in "${CUSTOM_SERVER_NAMES[@]}"; do
        CUSTOM_SERVER_READY+=(false)
    done
fi

for _ in $(seq 1 $MAX_ATTEMPTS); do
    # Check DEV servers
    if [ "$MATH_DEV_READY" = false ] && curl -s http://localhost:8000/health > /dev/null 2>&1; then
        MATH_DEV_READY=true
        echo "✅ Math Server (Dev) is healthy"
    fi
    
    if [ "$STATS_DEV_READY" = false ] && curl -s http://localhost:8001/health > /dev/null 2>&1; then
        STATS_DEV_READY=true
        echo "✅ Stats Server (Dev) is healthy"
    fi
    
    # Check PROD servers (with authentication)
    if [ "$MATH_PROD_READY" = false ] && curl -s -H "Authorization: Bearer $API_KEY" http://localhost:9000/health > /dev/null 2>&1; then
        MATH_PROD_READY=true
        echo "✅ Math Server (Prod) is healthy"
    fi
    
    if [ "$STATS_PROD_READY" = false ] && curl -s -H "Authorization: Bearer $API_KEY" http://localhost:9001/health > /dev/null 2>&1; then
        STATS_PROD_READY=true
        echo "✅ Stats Server (Prod) is healthy"
    fi
    
    # Check CUSTOM servers (if any)
    if [ "$ENVIRONMENT" = "codespace" ] && [ -n "$CUSTOM_SERVERS" ]; then
        for i in "${!CUSTOM_SERVER_NAMES[@]}"; do
            if [ "${CUSTOM_SERVER_READY[$i]}" = false ] && curl -s "http://localhost:${CUSTOM_SERVER_PORTS[$i]}/health" > /dev/null 2>&1; then
                CUSTOM_SERVER_READY[$i]=true
                echo "✅ Custom Server '${CUSTOM_SERVER_NAMES[$i]}' is healthy"
            fi
        done
    fi
    
    # Check if all servers are ready
    ALL_READY=true
    if [ "$MATH_DEV_READY" = false ] || [ "$STATS_DEV_READY" = false ] || [ "$MATH_PROD_READY" = false ] || [ "$STATS_PROD_READY" = false ]; then
        ALL_READY=false
    fi
    
    # Check custom servers
    if [ "$ENVIRONMENT" = "codespace" ] && [ -n "$CUSTOM_SERVERS" ]; then
        for ready in "${CUSTOM_SERVER_READY[@]}"; do
            if [ "$ready" = false ]; then
                ALL_READY=false
                break
            fi
        done
    fi
    
    # Exit loop if all servers are ready
    if [ "$ALL_READY" = true ]; then
        break
    fi
    
    sleep 1
done

# Report status of servers
echo ""
if [ "$MATH_DEV_READY" = false ]; then
    echo "⚠️  Math Server (Dev) failed to start or is not responding"
    echo "   Check logs: tail -f /tmp/math_server_dev.log"
fi

if [ "$STATS_DEV_READY" = false ]; then
    echo "⚠️  Stats Server (Dev) failed to start or is not responding"
    echo "   Check logs: tail -f /tmp/stats_server_dev.log"
fi

if [ "$MATH_PROD_READY" = false ]; then
    echo "⚠️  Math Server (Prod) failed to start or is not responding"
    echo "   Check logs: tail -f /tmp/math_server_prod.log"
fi

if [ "$STATS_PROD_READY" = false ]; then
    echo "⚠️  Stats Server (Prod) failed to start or is not responding"
    echo "   Check logs: tail -f /tmp/stats_server_prod.log"
fi

# Report custom server status
if [ "$ENVIRONMENT" = "codespace" ] && [ -n "$CUSTOM_SERVERS" ]; then
    for i in "${!CUSTOM_SERVER_NAMES[@]}"; do
        if [ "${CUSTOM_SERVER_READY[$i]}" = false ]; then
            echo "⚠️  Custom Server '${CUSTOM_SERVER_NAMES[$i]}' failed to start or is not responding"
            echo "   Check logs: tail -f /tmp/custom_${CUSTOM_SERVER_NAMES[$i]}.log"
        fi
    done
fi

# Exit if all servers failed
ALL_FAILED=true
if [ "$MATH_DEV_READY" = true ] || [ "$STATS_DEV_READY" = true ] || [ "$MATH_PROD_READY" = true ] || [ "$STATS_PROD_READY" = true ]; then
    ALL_FAILED=false
fi

# Check if any custom server succeeded
if [ "$ENVIRONMENT" = "codespace" ] && [ -n "$CUSTOM_SERVERS" ]; then
    for ready in "${CUSTOM_SERVER_READY[@]}"; do
        if [ "$ready" = true ]; then
            ALL_FAILED=false
            break
        fi
    done
fi

if [ "$ALL_FAILED" = true ]; then
    echo ""
    echo "❌ All servers failed to start. Please check the logs above."
    exit 1
fi

# Step 6: Display server URLs
echo ""
echo "============================================"
echo "  📡 Server URLs                           "
echo "============================================"
echo ""

# Get the Codespace name from environment variable
if [ -n "$CODESPACE_NAME" ]; then
    echo "🔓 DEV Servers (No Authentication):"
    echo "   Math Server:  https://${CODESPACE_NAME}-8000.app.github.dev"
    echo "   Stats Server: https://${CODESPACE_NAME}-8001.app.github.dev"
    echo "   Visibility:   Private (use 'gh codespace ports forward' for access)"
    echo ""
    echo "🔐 PROD Servers (Authentication Required):"
    echo "   Math Server:  https://${CODESPACE_NAME}-9000.app.github.dev"
    echo "   Stats Server: https://${CODESPACE_NAME}-9001.app.github.dev"
    echo "   Visibility:   Public"
    echo "   Auth Header:  Authorization: Bearer <api-key>"
    echo ""
    
    # Display custom server URLs
    if [ "$ENVIRONMENT" = "codespace" ] && [ -n "$CUSTOM_SERVERS" ]; then
        echo "🔧 CUSTOM Servers (No Authentication):"
        for i in "${!CUSTOM_SERVER_NAMES[@]}"; do
            echo "   ${CUSTOM_SERVER_NAMES[$i]}:  https://${CODESPACE_NAME}-${CUSTOM_SERVER_PORTS[$i]}.app.github.dev"
        done
        echo "   Visibility:   Private"
        echo ""
    fi
    
    echo "🌐 Codespace Environment Detected"
else
    echo "🔓 DEV Servers (No Authentication):"
    echo "   Math Server:  http://localhost:8000"
    echo "   Stats Server: http://localhost:8001"
    echo ""
    echo "🔐 PROD Servers (Authentication Required):"
    echo "   Math Server:  http://localhost:9000"
    echo "   Stats Server: http://localhost:9001"
    echo "   Auth Header:  Authorization: Bearer <api-key>"
    echo ""
    
    # Display custom server URLs
    if [ "$ENVIRONMENT" = "codespace" ] && [ -n "$CUSTOM_SERVERS" ]; then
        echo "🔧 CUSTOM Servers (No Authentication):"
        for i in "${!CUSTOM_SERVER_NAMES[@]}"; do
            echo "   ${CUSTOM_SERVER_NAMES[$i]}:  http://localhost:${CUSTOM_SERVER_PORTS[$i]}"
        done
        echo ""
    fi
    
    echo "ℹ️  Running in local environment (not Codespaces)"
fi

echo ""
echo "============================================"
echo "  📋 Additional Information                "
echo "============================================"
echo ""
echo "Server logs:"
echo "  DEV:"
echo "    - Math Server:  /tmp/math_server_dev.log"
echo "    - Stats Server: /tmp/stats_server_dev.log"
echo "  PROD:"
echo "    - Math Server:  /tmp/math_server_prod.log"
echo "    - Stats Server: /tmp/stats_server_prod.log"

# Add custom server logs
if [ "$ENVIRONMENT" = "codespace" ] && [ -n "$CUSTOM_SERVERS" ]; then
    echo "  CUSTOM:"
    for i in "${!CUSTOM_SERVER_NAMES[@]}"; do
        echo "    - ${CUSTOM_SERVER_NAMES[$i]}:  /tmp/custom_${CUSTOM_SERVER_NAMES[$i]}.log"
    done
fi

echo ""
echo "View logs with:"
echo "  tail -f /tmp/math_server_dev.log"
echo "  tail -f /tmp/stats_server_dev.log"
echo "  tail -f /tmp/math_server_prod.log"
echo "  tail -f /tmp/stats_server_prod.log"

# Add custom server log commands
if [ "$ENVIRONMENT" = "codespace" ] && [ -n "$CUSTOM_SERVERS" ]; then
    for i in "${!CUSTOM_SERVER_NAMES[@]}"; do
        echo "  tail -f /tmp/custom_${CUSTOM_SERVER_NAMES[$i]}.log"
    done
fi
echo ""
echo "Test authentication:"
if [ -n "$CODESPACE_NAME" ]; then
    echo "  # Without auth (should fail on prod):"
    echo "  curl https://${CODESPACE_NAME}-9000.app.github.dev/messages"
    echo ""
    echo "  # With auth (should succeed on prod):"
    echo "  curl -H \"Authorization: Bearer \$MCP_API_KEY\" https://${CODESPACE_NAME}-9000.app.github.dev/health"
else
    echo "  # Without auth (should fail on prod):"
    echo "  curl http://localhost:9000/messages"
    echo ""
    echo "  # With auth (should succeed on prod):"
    echo "  curl -H \"Authorization: Bearer \$MCP_API_KEY\" http://localhost:9000/health"
fi
echo ""
echo "  # Note: Use the API key from your environment variable or the one generated above"
echo ""
echo "✅ Startup complete!"
echo "============================================"
