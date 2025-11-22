#!/bin/bash
# Script to run Stats Server HTTP client tests
# Usage: ./run_stats_tests.sh [options]

set -e

# Default values
SERVER_URL="${STATS_SERVER_URL:-http://localhost:8001}"
START_SERVER="${START_SERVER:-yes}"
SERVER_PORT=8001
CONFIG_FILE=""

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --no-server)
            START_SERVER="no"
            shift
            ;;
        --url)
            SERVER_URL="$2"
            START_SERVER="no"
            shift 2
            ;;
        --config)
            CONFIG_FILE="$2"
            shift 2
            ;;
        --port)
            SERVER_PORT="$2"
            shift 2
            ;;
        --help)
            echo "Usage: ./run_stats_tests.sh [options]"
            echo ""
            echo "Options:"
            echo "  --no-server         Don't start the server (assume it's already running)"
            echo "  --url URL           Use specific server URL (implies --no-server)"
            echo "  --config FILE       Use specific config file for server"
            echo "  --port PORT         Server port (default: 8001)"
            echo "  --help              Show this help message"
            echo ""
            echo "Examples:"
            echo "  ./run_stats_tests.sh                    # Start server and run tests"
            echo "  ./run_stats_tests.sh --no-server        # Run tests against existing server"
            echo "  ./run_stats_tests.sh --port 9001        # Use port 9001"
            echo "  ./run_stats_tests.sh --url http://example.com:8001  # Test remote server"
            exit 0
            ;;
        *)
            echo -e "${RED}Unknown option: $1${NC}"
            echo "Use --help for usage information"
            exit 1
            ;;
    esac
done

# Function to check if server is running
check_server() {
    local url=$1
    local max_attempts=30
    local attempt=1
    
    echo -e "${YELLOW}Waiting for server at $url...${NC}"
    
    while [ $attempt -le $max_attempts ]; do
        if curl -s "${url}/health" > /dev/null 2>&1; then
            echo -e "${GREEN}✓ Server is ready!${NC}"
            return 0
        fi
        
        if [ $attempt -eq $max_attempts ]; then
            echo -e "${RED}✗ Server failed to start within 30 seconds${NC}"
            return 1
        fi
        
        echo "Attempt $attempt/$max_attempts..."
        sleep 1
        ((attempt++))
    done
}

# Function to cleanup on exit
cleanup() {
    if [ -n "$SERVER_PID" ] && [ "$START_SERVER" = "yes" ]; then
        echo -e "\n${YELLOW}Stopping server (PID: $SERVER_PID)...${NC}"
        kill $SERVER_PID 2>/dev/null || true
        wait $SERVER_PID 2>/dev/null || true
        echo -e "${GREEN}✓ Server stopped${NC}"
    fi
}

# Register cleanup function
trap cleanup EXIT INT TERM

# Start server if needed
if [ "$START_SERVER" = "yes" ]; then
    echo -e "${GREEN}Starting Stats MCP HTTP server...${NC}"
    
    # Build command as array for proper quoting
    CMD_ARGS=(python src/stats_server/server.py --transport http --host 127.0.0.1 --port "$SERVER_PORT")
    
    if [ -n "$CONFIG_FILE" ]; then
        CMD_ARGS+=(--config "$CONFIG_FILE")
        echo "Using config file: $CONFIG_FILE"
    fi
    
    echo "Command: ${CMD_ARGS[*]}"
    
    # Start server in background
    "${CMD_ARGS[@]}" &
    SERVER_PID=$!
    echo "Server PID: $SERVER_PID"
    
    # Update server URL if using custom port
    if [ $SERVER_PORT -ne 8001 ]; then
        SERVER_URL="http://127.0.0.1:$SERVER_PORT"
    else
        SERVER_URL="http://127.0.0.1:8001"
    fi
    
    # Wait for server to be ready
    if ! check_server "$SERVER_URL"; then
        echo -e "${RED}✗ Failed to start server${NC}"
        exit 1
    fi
else
    echo -e "${YELLOW}Using existing server at $SERVER_URL${NC}"
    
    # Verify server is accessible
    if ! curl -s "${SERVER_URL}/health" > /dev/null 2>&1; then
        echo -e "${RED}✗ Server at $SERVER_URL is not accessible${NC}"
        exit 1
    fi
    echo -e "${GREEN}✓ Server is accessible${NC}"
fi

# Run tests
echo -e "\n${GREEN}Running Stats Server HTTP client tests...${NC}"
echo "Server URL: $SERVER_URL"
echo ""

export STATS_SERVER_URL="$SERVER_URL"

# Run pytest with various options
if ! pytest tests/test_http_stats_server.py -v --tb=short --strict-markers; then
    echo -e "\n${RED}✗ Tests failed${NC}"
    exit 1
fi

echo -e "\n${GREEN}✓ All tests passed!${NC}"
