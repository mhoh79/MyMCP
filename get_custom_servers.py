#!/usr/bin/env python3
"""
Helper script to extract custom server configuration from config.yaml.
This script outputs custom server information in a format that can be easily
parsed by shell scripts.

Usage:
    python get_custom_servers.py [config_path]

Output format (one line per enabled server):
    name|module|host|port

Exit codes:
    0 - Success
    1 - Error (config not found, validation failed, etc.)
"""

import sys
from pathlib import Path

# Add src directory to path to import config module
sys.path.insert(0, str(Path(__file__).parent / "src"))

from config import Config


def main():
    """Extract custom server configuration from YAML file."""
    # Get config path from command line or use default
    config_path = sys.argv[1] if len(sys.argv) > 1 else "config.yaml"
    
    try:
        # Load configuration
        config = Config.from_yaml(config_path)
        
        # Output enabled custom servers in shell-friendly format
        for server in config.custom_servers:
            if server.enabled:
                # Format: name|module|host|port
                print(f"{server.name}|{server.module}|{server.host}|{server.port}")
        
        # Exit successfully
        sys.exit(0)
        
    except FileNotFoundError as e:
        # Config file not found - this is OK, means no custom servers
        sys.exit(0)
        
    except Exception as e:
        # Configuration error - print to stderr and exit with error
        print(f"Error loading configuration: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
