#!/bin/bash

# MCP Servers Launcher Script (Linux/Mac)
# This script starts both math_server and stats_server in HTTP mode
# with proper configuration and process management.

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
MATH_PORT=8000
STATS_PORT=8001
CONFIG_FILE="config.yaml"
CONFIG_EXAMPLE="config.example.yaml"
PIDS_FILE=".pids"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DEV_MODE=false

# Function to print colored output
print_status() {
    echo -e "${GREEN}✓${NC} $1"
}

print_error() {
    echo -e "${RED}✗${NC} $1"
}

print_info() {
    echo -e "${BLUE}ℹ${NC}  $1"
}

print_warning() {
    echo -e "${YELLOW}⚠${NC}  $1"
}

# Function to check if a command exists
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

# Function to check if a port is in use
port_in_use() {
    local port=$1
    if command_exists lsof; then
        lsof -Pi :$port -sTCP:LISTEN -t >/dev/null 2>&1
    elif command_exists netstat; then
        netstat -tuln | grep -q ":$port "
    else
        # Fallback: try to bind to the port
        # If bind succeeds (exit 0), port is available, so we return 1 (port NOT in use)
        # If bind fails (exit 1), port is in use, so we return 0 (port IS in use)
        python3 -c "import socket; s = socket.socket(socket.AF_INET, socket.SOCK_STREAM); s.bind(('', $port)); s.close()" 2>/dev/null
        # Invert the exit code
        return $(( $? == 0 ? 1 : 0 ))
    fi
}

# Function to check if a process is running
is_running() {
    local pid=$1
    kill -0 "$pid" 2>/dev/null
}

# Function to get the Codespace name
get_codespace_name() {
    if [ -n "$CODESPACE_NAME" ]; then
        echo "$CODESPACE_NAME"
    elif [ -n "$GITHUB_CODESPACES_PORT_FORWARDING_DOMAIN" ]; then
        echo "YOUR-CODESPACE"
    else
        echo ""
    fi
}

# Function to start servers
start_servers() {
    if [ "$DEV_MODE" = true ]; then
        echo -e "${BLUE}🚀 Starting MCP Servers in HTTP mode with hot-reload...${NC}"
        print_warning "Development mode enabled - servers will restart on code changes"
    else
        echo -e "${BLUE}🚀 Starting MCP Servers in HTTP mode...${NC}"
    fi
    echo ""

    # Change to script directory
    cd "$SCRIPT_DIR"

    # Check if Python 3 is installed
    if ! command_exists python3; then
        print_error "Python 3 is not installed. Please install Python 3 to continue."
        exit 1
    fi

    print_status "Python 3 found: $(python3 --version)"

    # Check if config file exists, create from example if missing
    if [ ! -f "$CONFIG_FILE" ]; then
        if [ -f "$CONFIG_EXAMPLE" ]; then
            cp "$CONFIG_EXAMPLE" "$CONFIG_FILE"
            print_warning "Config file not found. Created $CONFIG_FILE from $CONFIG_EXAMPLE"
        else
            print_error "Config file $CONFIG_FILE and example $CONFIG_EXAMPLE not found!"
            exit 1
        fi
    else
        print_status "Config file found: $CONFIG_FILE"
    fi

    # Check if dependencies are installed
    if ! python3 -c "import mcp" 2>/dev/null; then
        print_warning "MCP dependencies not found. Installing from requirements.txt..."
        if [ -f "requirements.txt" ]; then
            pip3 install -r requirements.txt || {
                print_error "Failed to install dependencies. Please run: pip3 install -r requirements.txt"
                exit 1
            }
        else
            print_error "requirements.txt not found!"
            exit 1
        fi
    fi

    # Check if servers are already running
    if [ -f "$PIDS_FILE" ]; then
        print_warning "Server PIDs file found. Checking if servers are still running..."
        
        while IFS='=' read -r name pid; do
            if [ -n "$pid" ] && is_running "$pid"; then
                print_error "Server $name is already running with PID $pid"
                print_info "To stop servers, run: $0 stop"
                exit 1
            fi
        done < "$PIDS_FILE"
        
        # Remove stale PID file
        rm -f "$PIDS_FILE"
    fi

    # Check if ports are available
    if port_in_use $MATH_PORT; then
        print_error "Port $MATH_PORT is already in use. Cannot start math_server."
        exit 1
    fi

    if port_in_use $STATS_PORT; then
        print_error "Port $STATS_PORT is already in use. Cannot start stats_server."
        exit 1
    fi

    # Create logs directory if it doesn't exist
    mkdir -p logs

    # Start math_server
    echo ""
    print_info "Starting Math Server on port $MATH_PORT..."
    
    # Build command with optional --dev flag
    MATH_CMD="python3 src/math_server/server.py --transport http --host 0.0.0.0 --port $MATH_PORT --config \"$CONFIG_FILE\""
    if [ "$DEV_MODE" = true ]; then
        MATH_CMD="$MATH_CMD --dev"
    fi
    
    nohup bash -c "$MATH_CMD" > logs/math_server.log 2>&1 &
    MATH_PID=$!
    sleep 2

    if is_running $MATH_PID; then
        print_status "Math Server started on port $MATH_PORT (PID: $MATH_PID)"
        echo "math_server=$MATH_PID" >> "$PIDS_FILE"
    else
        print_error "Failed to start Math Server. Check logs/math_server.log for details."
        exit 1
    fi

    # Start stats_server
    echo ""
    print_info "Starting Stats Server on port $STATS_PORT..."
    
    # Build command with optional --dev flag
    STATS_CMD="python3 src/stats_server/server.py --transport http --host 0.0.0.0 --port $STATS_PORT --config \"$CONFIG_FILE\""
    if [ "$DEV_MODE" = true ]; then
        STATS_CMD="$STATS_CMD --dev"
    fi
    
    nohup bash -c "$STATS_CMD" > logs/stats_server.log 2>&1 &
    STATS_PID=$!
    sleep 2

    if is_running $STATS_PID; then
        print_status "Stats Server started on port $STATS_PORT (PID: $STATS_PID)"
        echo "stats_server=$STATS_PID" >> "$PIDS_FILE"
    else
        print_error "Failed to start Stats Server. Check logs/stats_server.log for details."
        # Kill math_server if stats_server failed
        kill $MATH_PID 2>/dev/null
        rm -f "$PIDS_FILE"
        exit 1
    fi

    # Display connection URLs
    echo ""
    echo -e "${BLUE}📡 Connection URLs:${NC}"
    echo "  Math Server:  http://localhost:$MATH_PORT"
    echo "  Stats Server: http://localhost:$STATS_PORT"
    echo ""

    # Display Codespaces URLs if applicable
    CODESPACE=$(get_codespace_name)
    if [ -n "$CODESPACE" ]; then
        echo -e "${BLUE}🌐 Codespaces URLs:${NC}"
        if [ "$CODESPACE" = "YOUR-CODESPACE" ]; then
            echo "  Math Server:  https://YOUR-CODESPACE-$MATH_PORT.app.github.dev"
            echo "  Stats Server: https://YOUR-CODESPACE-$STATS_PORT.app.github.dev"
            echo ""
            print_info "Replace YOUR-CODESPACE with your actual Codespace name"
        else
            echo "  Math Server:  https://$CODESPACE-$MATH_PORT.app.github.dev"
            echo "  Stats Server: https://$CODESPACE-$STATS_PORT.app.github.dev"
        fi
        echo ""
    fi

    print_info "To stop servers: $0 stop"
    print_info "To check status: $0 status"
    print_info "Server logs: logs/math_server.log, logs/stats_server.log"
    echo ""
}

# Function to stop servers
stop_servers() {
    echo -e "${BLUE}🛑 Stopping MCP Servers...${NC}"
    echo ""

    if [ ! -f "$PIDS_FILE" ]; then
        print_error "No PID file found. Servers may not be running."
        exit 1
    fi

    local stopped=0
    while IFS='=' read -r name pid; do
        if [ -z "$pid" ]; then
            continue
        fi

        if is_running "$pid"; then
            print_info "Stopping $name (PID: $pid)..."
            kill "$pid" 2>/dev/null || true
            
            # Wait up to 10 seconds for graceful shutdown
            local count=0
            while is_running "$pid" && [ $count -lt 10 ]; do
                sleep 1
                count=$((count + 1))
            done
            
            # Force kill if still running
            if is_running "$pid"; then
                print_warning "Forcing $name to stop..."
                kill -9 "$pid" 2>/dev/null || true
                sleep 1
            fi
            
            if ! is_running "$pid"; then
                print_status "$name stopped successfully"
                stopped=$((stopped + 1))
            else
                print_error "Failed to stop $name (PID: $pid)"
            fi
        else
            print_warning "$name (PID: $pid) is not running"
        fi
    done < "$PIDS_FILE"

    # Remove PID file
    rm -f "$PIDS_FILE"
    
    if [ $stopped -gt 0 ]; then
        echo ""
        print_status "All servers stopped"
    fi
}

# Function to check server status
check_status() {
    echo -e "${BLUE}📊 Server Status:${NC}"
    echo ""

    if [ ! -f "$PIDS_FILE" ]; then
        print_info "No servers are currently running (no PID file found)"
        exit 0
    fi

    local running=0
    while IFS='=' read -r name pid; do
        if [ -z "$pid" ]; then
            continue
        fi

        if is_running "$pid"; then
            print_status "$name is running (PID: $pid)"
            running=$((running + 1))
        else
            print_error "$name is not running (PID: $pid - stale)"
        fi
    done < "$PIDS_FILE"

    if [ $running -eq 0 ]; then
        echo ""
        print_warning "No servers are running. Removing stale PID file."
        rm -f "$PIDS_FILE"
    fi
    
    echo ""
}

# Parse command-line arguments
while [ $# -gt 0 ]; do
    case "$1" in
        --dev)
            DEV_MODE=true
            shift
            ;;
        start|stop|status|restart)
            COMMAND="$1"
            shift
            ;;
        *)
            echo "Unknown option: $1"
            echo "Usage: $0 [start|stop|status|restart] [--dev]"
            exit 1
            ;;
    esac
done

# Default command is start if not specified
COMMAND="${COMMAND:-start}"

# Main script logic
case "$COMMAND" in
    start)
        start_servers
        ;;
    stop)
        stop_servers
        ;;
    status)
        check_status
        ;;
    restart)
        if [ -f "$PIDS_FILE" ]; then
            stop_servers
            echo ""
        fi
        start_servers
        ;;
    *)
        echo "Usage: $0 {start|stop|status|restart} [--dev]"
        echo ""
        echo "Commands:"
        echo "  start    - Start both MCP servers in HTTP mode (default)"
        echo "  stop     - Stop running servers"
        echo "  status   - Check server status"
        echo "  restart  - Restart servers"
        echo ""
        echo "Options:"
        echo "  --dev    - Enable development mode with hot-reload"
        exit 1
        ;;
esac

exit 0
