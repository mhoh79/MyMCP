#!/bin/bash
# Script to run HTTP client tests for both Math and Stats servers
# Usage: ./run_all_tests.sh [options]

set -e

# Default values
MATH_PORT=8000
STATS_PORT=8001
START_SERVERS="${START_SERVERS:-yes}"
CONFIG_FILE=""
TEST_MATH=true
TEST_STATS=true

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --no-servers)
            START_SERVERS="no"
            shift
            ;;
        --math-only)
            TEST_STATS=false
            shift
            ;;
        --stats-only)
            TEST_MATH=false
            shift
            ;;
        --config)
            CONFIG_FILE="$2"
            shift 2
            ;;
        --math-port)
            MATH_PORT="$2"
            shift 2
            ;;
        --stats-port)
            STATS_PORT="$2"
            shift 2
            ;;
        --help)
            echo "Usage: ./run_all_tests.sh [options]"
            echo ""
            echo "Options:"
            echo "  --no-servers        Don't start servers (assume they're already running)"
            echo "  --math-only         Test only the Math server"
            echo "  --stats-only        Test only the Stats server"
            echo "  --config FILE       Use specific config file for servers"
            echo "  --math-port PORT    Math server port (default: 8000)"
            echo "  --stats-port PORT   Stats server port (default: 8001)"
            echo "  --help              Show this help message"
            echo ""
            echo "Examples:"
            echo "  ./run_all_tests.sh                     # Start both servers and run all tests"
            echo "  ./run_all_tests.sh --math-only         # Test only Math server"
            echo "  ./run_all_tests.sh --stats-only        # Test only Stats server"
            echo "  ./run_all_tests.sh --no-servers        # Test against existing servers"
            exit 0
            ;;
        *)
            echo -e "${RED}Unknown option: $1${NC}"
            echo "Use --help for usage information"
            exit 1
            ;;
    esac
done

# PIDs for cleanup
MATH_PID=""
STATS_PID=""

# Function to check if server is running
check_server() {
    local url=$1
    local name=$2
    local max_attempts=30
    local attempt=1
    
    echo -e "${YELLOW}Waiting for $name at $url...${NC}"
    
    while [ $attempt -le $max_attempts ]; do
        if curl -s "${url}/health" > /dev/null 2>&1; then
            echo -e "${GREEN}✓ $name is ready!${NC}"
            return 0
        fi
        
        if [ $attempt -eq $max_attempts ]; then
            echo -e "${RED}✗ $name failed to start within 30 seconds${NC}"
            return 1
        fi
        
        echo "Attempt $attempt/$max_attempts..."
        sleep 1
        ((attempt++))
    done
}

# Function to cleanup on exit
cleanup() {
    if [ "$START_SERVERS" = "yes" ]; then
        if [ -n "$MATH_PID" ]; then
            echo -e "\n${YELLOW}Stopping Math server (PID: $MATH_PID)...${NC}"
            kill $MATH_PID 2>/dev/null || true
            wait $MATH_PID 2>/dev/null || true
        fi
        if [ -n "$STATS_PID" ]; then
            echo -e "${YELLOW}Stopping Stats server (PID: $STATS_PID)...${NC}"
            kill $STATS_PID 2>/dev/null || true
            wait $STATS_PID 2>/dev/null || true
        fi
        echo -e "${GREEN}✓ Servers stopped${NC}"
    fi
}

# Register cleanup function
trap cleanup EXIT INT TERM

# Start servers if needed
if [ "$START_SERVERS" = "yes" ]; then
    echo -e "${BLUE}=====================================${NC}"
    echo -e "${BLUE}Starting MCP HTTP Servers${NC}"
    echo -e "${BLUE}=====================================${NC}"
    
    if [ "$TEST_MATH" = true ]; then
        echo -e "\n${GREEN}Starting Math MCP HTTP server on port $MATH_PORT...${NC}"
        
        CMD_ARGS=(python -m src.builtin.math_server --transport http --host 127.0.0.1 --port "$MATH_PORT")
        if [ -n "$CONFIG_FILE" ]; then
            CMD_ARGS+=(--config "$CONFIG_FILE")
        fi
        
        "${CMD_ARGS[@]}" &
        MATH_PID=$!
        echo "Math Server PID: $MATH_PID"
        
        if ! check_server "http://127.0.0.1:$MATH_PORT" "Math Server"; then
            echo -e "${RED}✗ Failed to start Math server${NC}"
            exit 1
        fi
    fi
    
    if [ "$TEST_STATS" = true ]; then
        echo -e "\n${GREEN}Starting Stats MCP HTTP server on port $STATS_PORT...${NC}"
        
        CMD_ARGS=(python -m src.builtin.stats_server --transport http --host 127.0.0.1 --port "$STATS_PORT")
        if [ -n "$CONFIG_FILE" ]; then
            CMD_ARGS+=(--config "$CONFIG_FILE")
        fi
        
        "${CMD_ARGS[@]}" &
        STATS_PID=$!
        echo "Stats Server PID: $STATS_PID"
        
        if ! check_server "http://127.0.0.1:$STATS_PORT" "Stats Server"; then
            echo -e "${RED}✗ Failed to start Stats server${NC}"
            exit 1
        fi
    fi
else
    echo -e "${YELLOW}Using existing servers${NC}"
fi

# Export environment variables
export MATH_SERVER_URL="http://127.0.0.1:$MATH_PORT"
export STATS_SERVER_URL="http://127.0.0.1:$STATS_PORT"

# Run tests
echo -e "\n${BLUE}=====================================${NC}"
echo -e "${BLUE}Running HTTP Client Tests${NC}"
echo -e "${BLUE}=====================================${NC}"

TEST_FAILED=false

if [ "$TEST_MATH" = true ]; then
    echo -e "\n${GREEN}Testing Math Server at $MATH_SERVER_URL${NC}"
    echo "-----------------------------------"
    
    if ! pytest tests/test_http_client.py -v --tb=short --strict-markers; then
        echo -e "\n${RED}✗ Math Server tests failed${NC}"
        TEST_FAILED=true
    else
        echo -e "\n${GREEN}✓ Math Server tests passed!${NC}"
    fi
fi

if [ "$TEST_STATS" = true ]; then
    echo -e "\n${GREEN}Testing Stats Server at $STATS_SERVER_URL${NC}"
    echo "-----------------------------------"
    
    if ! pytest tests/test_http_stats_server.py -v --tb=short --strict-markers; then
        echo -e "\n${RED}✗ Stats Server tests failed${NC}"
        TEST_FAILED=true
    else
        echo -e "\n${GREEN}✓ Stats Server tests passed!${NC}"
    fi
fi

# Final summary
echo -e "\n${BLUE}=====================================${NC}"
if [ "$TEST_FAILED" = true ]; then
    echo -e "${RED}✗ Some tests failed${NC}"
    exit 1
else
    echo -e "${GREEN}✓ All tests passed!${NC}"
fi
echo -e "${BLUE}=====================================${NC}"
