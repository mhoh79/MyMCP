"""
Configuration module for MCP servers.

This module provides YAML-based configuration with Pydantic validation
for server settings, authentication, rate limiting, and CORS configuration.
Environment variables can override configuration file values.
"""

import os
from pathlib import Path
from typing import Any, Optional

import yaml
from pydantic import BaseModel, Field, field_validator, ValidationError
from pydantic_settings import BaseSettings, SettingsConfigDict


class ServerConfig(BaseModel):
    """Configuration for a single server instance."""
    
    host: str = Field(default="0.0.0.0", description="Server host address")
    port: int = Field(default=8000, ge=1, le=65535, description="Server port")
    
    @field_validator("host")
    @classmethod
    def validate_host(cls, v: str) -> str:
        """Validate host address format."""
        if not v:
            raise ValueError("Host cannot be empty")
        return v


class CustomServerConfig(ServerConfig):
    """Configuration for a custom MCP server."""
    
    name: str = Field(description="Unique name for the custom server")
    module: str = Field(description="Python module path (e.g., 'custom.my_server')")
    enabled: bool = Field(default=True, description="Whether the server is enabled")
    
    # Override port to remove default value (required for custom servers)
    port: int = Field(ge=1, le=65535, description="Server port")
    
    @field_validator("name")
    @classmethod
    def validate_name(cls, v: str) -> str:
        """Validate server name format."""
        if not v:
            raise ValueError("Server name cannot be empty")
        if not v.replace("-", "").replace("_", "").isalnum():
            raise ValueError(
                f"Server name '{v}' must contain only alphanumeric characters, hyphens, and underscores"
            )
        return v
    
    @field_validator("module")
    @classmethod
    def validate_module(cls, v: str) -> str:
        """Validate module path format."""
        if not v:
            raise ValueError("Module path cannot be empty")
        # Basic validation of module path format
        if not all(part.isidentifier() for part in v.split(".")):
            raise ValueError(
                f"Module path '{v}' must be a valid Python module path (e.g., 'custom.my_server')"
            )
        return v


class ServersConfig(BaseModel):
    """Configuration for both math and stats servers."""
    
    math: ServerConfig = Field(default_factory=lambda: ServerConfig(host="0.0.0.0", port=8000))
    stats: ServerConfig = Field(default_factory=lambda: ServerConfig(host="0.0.0.0", port=8001))
    cors_origins: list[str] = Field(
        default_factory=lambda: ["http://localhost:*", "https://*.app.github.dev"],
        description="List of allowed CORS origins (supports wildcards)"
    )


class AuthenticationConfig(BaseModel):
    """Authentication configuration."""
    
    enabled: bool = Field(default=False, description="Enable authentication")
    api_key: str = Field(default="your-secret-api-key-here", description="API key for authentication")
    
    @field_validator("api_key")
    @classmethod
    def validate_api_key(cls, v: str, info: Any) -> str:
        """Validate API key when authentication is enabled."""
        # Get the enabled value from the validation context
        if info.data.get("enabled", False):
            if not v or v == "your-secret-api-key-here":
                raise ValueError(
                    "API key must be set to a secure value when authentication is enabled. "
                    "Please provide a strong API key."
                )
            if len(v) < 16:
                raise ValueError(
                    "API key must be at least 16 characters long for security. "
                    f"Current length: {len(v)}"
                )
        return v


class RateLimitingConfig(BaseModel):
    """Rate limiting configuration."""
    
    enabled: bool = Field(default=False, description="Enable rate limiting")
    requests_per_minute: int = Field(
        default=60,
        ge=1,
        le=10000,
        description="Maximum requests per minute per client"
    )


class LoggingConfig(BaseModel):
    """Logging configuration."""
    
    level: str = Field(default="INFO", description="Logging level")
    
    @field_validator("level")
    @classmethod
    def validate_level(cls, v: str) -> str:
        """Validate logging level."""
        valid_levels = ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
        v_upper = v.upper()
        if v_upper not in valid_levels:
            raise ValueError(
                f"Invalid logging level '{v}'. Must be one of: {', '.join(valid_levels)}"
            )
        return v_upper


class Config(BaseSettings):
    """
    Main configuration class with environment variable support.
    
    Environment variables can override config file values:
    - MCP_MATH_HOST / MCP_STATS_HOST - Override server hosts
    - MCP_MATH_PORT / MCP_STATS_PORT - Override server ports
    - MCP_AUTH_ENABLED - Enable/disable authentication
    - MCP_API_KEY - Set API key
    - MCP_RATE_LIMIT_ENABLED - Enable/disable rate limiting
    - MCP_RATE_LIMIT_RPM - Set requests per minute
    - MCP_LOG_LEVEL - Set logging level
    """
    
    model_config = SettingsConfigDict(
        env_prefix="MCP_",
        env_nested_delimiter="__",
        case_sensitive=False,
        extra="ignore"
    )
    
    server: ServersConfig = Field(default_factory=ServersConfig)
    authentication: AuthenticationConfig = Field(default_factory=AuthenticationConfig)
    rate_limiting: RateLimitingConfig = Field(default_factory=RateLimitingConfig)
    logging: LoggingConfig = Field(default_factory=LoggingConfig)
    custom_servers: list[CustomServerConfig] = Field(
        default_factory=list,
        description="List of custom MCP servers to register"
    )
    
    @field_validator("custom_servers")
    @classmethod
    def validate_custom_servers(cls, v: list[CustomServerConfig], info: Any) -> list[CustomServerConfig]:
        """Validate custom servers configuration."""
        if not v:
            return v  # Empty list is valid
        
        # Check for duplicate server names
        names = [server.name for server in v]
        duplicate_names = [name for name in names if names.count(name) > 1]
        if duplicate_names:
            raise ValueError(
                f"Duplicate server names found: {', '.join(set(duplicate_names))}. "
                "Each custom server must have a unique name."
            )
        
        # Collect all ports to check for conflicts
        builtin_ports = {8000, 8001, 9000, 9001}  # Reserved for builtin servers
        custom_ports = [server.port for server in v if server.enabled]
        
        # Check for duplicate ports among custom servers
        duplicate_ports = [port for port in custom_ports if custom_ports.count(port) > 1]
        if duplicate_ports:
            raise ValueError(
                f"Duplicate ports found in custom servers: {', '.join(map(str, set(duplicate_ports)))}. "
                "Each custom server must use a unique port."
            )
        
        # Check for conflicts with builtin server ports
        conflicting_ports = [port for port in custom_ports if port in builtin_ports]
        if conflicting_ports:
            raise ValueError(
                f"Custom server ports conflict with builtin server ports: {', '.join(map(str, conflicting_ports))}. "
                f"Ports {', '.join(map(str, sorted(builtin_ports)))} are reserved for builtin servers. "
                "Please use different ports for custom servers."
            )
        
        return v
    
    @classmethod
    def from_yaml(cls, config_path: str | Path) -> "Config":
        """
        Load configuration from a YAML file with environment variable overrides.
        
        Args:
            config_path: Path to the YAML configuration file
            
        Returns:
            Config instance with loaded and validated configuration
            
        Raises:
            FileNotFoundError: If the config file doesn't exist
            ValueError: If the YAML is invalid or validation fails
        """
        config_path = Path(config_path)
        
        if not config_path.exists():
            raise FileNotFoundError(
                f"Configuration file not found: {config_path}\n"
                f"Please create a config file or use the default configuration."
            )
        
        try:
            with open(config_path, "r", encoding="utf-8") as f:
                yaml_data = yaml.safe_load(f)
                
            if yaml_data is None:
                yaml_data = {}
                
        except yaml.YAMLError as e:
            raise ValueError(
                f"Invalid YAML in configuration file '{config_path}':\n{e}\n"
                f"Please check the YAML syntax and try again."
            ) from e
        except Exception as e:
            raise ValueError(
                f"Error reading configuration file '{config_path}': {e}"
            ) from e
        
        # Apply environment variable overrides
        yaml_data = cls._apply_env_overrides(yaml_data)
        
        try:
            return cls(**yaml_data)
        except ValidationError as e:
            # Format validation errors in a user-friendly way
            error_messages = []
            for error in e.errors():
                loc = " -> ".join(str(x) for x in error["loc"])
                msg = error["msg"]
                error_messages.append(f"  • {loc}: {msg}")
            
            raise ValueError(
                f"Configuration validation failed for '{config_path}':\n" +
                "\n".join(error_messages) +
                "\n\nPlease fix these errors in your configuration file."
            ) from e
    
    @staticmethod
    def _apply_env_overrides(yaml_data: dict[str, Any]) -> dict[str, Any]:
        """
        Apply environment variable overrides to YAML data.
        
        This allows environment variables to override specific configuration values:
        - MCP_MATH_HOST -> server.math.host
        - MCP_MATH_PORT -> server.math.port
        - MCP_STATS_HOST -> server.stats.host
        - MCP_STATS_PORT -> server.stats.port
        - MCP_AUTH_ENABLED -> authentication.enabled
        - MCP_API_KEY -> authentication.api_key
        - MCP_RATE_LIMIT_ENABLED -> rate_limiting.enabled
        - MCP_RATE_LIMIT_RPM -> rate_limiting.requests_per_minute
        - MCP_LOG_LEVEL -> logging.level
        """
        # Ensure nested structures exist
        if "server" not in yaml_data:
            yaml_data["server"] = {}
        if "math" not in yaml_data["server"]:
            yaml_data["server"]["math"] = {}
        if "stats" not in yaml_data["server"]:
            yaml_data["server"]["stats"] = {}
        if "authentication" not in yaml_data:
            yaml_data["authentication"] = {}
        if "rate_limiting" not in yaml_data:
            yaml_data["rate_limiting"] = {}
        if "logging" not in yaml_data:
            yaml_data["logging"] = {}
        
        # Math server overrides
        if "MCP_MATH_HOST" in os.environ:
            yaml_data["server"]["math"]["host"] = os.environ["MCP_MATH_HOST"]
        if "MCP_MATH_PORT" in os.environ:
            try:
                yaml_data["server"]["math"]["port"] = int(os.environ["MCP_MATH_PORT"])
            except ValueError:
                raise ValueError(
                    f"MCP_MATH_PORT must be an integer, got: {os.environ['MCP_MATH_PORT']}"
                )
        
        # Stats server overrides
        if "MCP_STATS_HOST" in os.environ:
            yaml_data["server"]["stats"]["host"] = os.environ["MCP_STATS_HOST"]
        if "MCP_STATS_PORT" in os.environ:
            try:
                yaml_data["server"]["stats"]["port"] = int(os.environ["MCP_STATS_PORT"])
            except ValueError:
                raise ValueError(
                    f"MCP_STATS_PORT must be an integer, got: {os.environ['MCP_STATS_PORT']}"
                )
        
        # Authentication overrides
        if "MCP_AUTH_ENABLED" in os.environ:
            auth_enabled = os.environ["MCP_AUTH_ENABLED"].lower()
            if auth_enabled in ("true", "1", "yes", "on"):
                yaml_data["authentication"]["enabled"] = True
            elif auth_enabled in ("false", "0", "no", "off"):
                yaml_data["authentication"]["enabled"] = False
            else:
                raise ValueError(
                    f"MCP_AUTH_ENABLED must be a boolean value (true/false), got: {os.environ['MCP_AUTH_ENABLED']}"
                )
        
        if "MCP_API_KEY" in os.environ:
            yaml_data["authentication"]["api_key"] = os.environ["MCP_API_KEY"]
        
        # Rate limiting overrides
        if "MCP_RATE_LIMIT_ENABLED" in os.environ:
            rate_enabled = os.environ["MCP_RATE_LIMIT_ENABLED"].lower()
            if rate_enabled in ("true", "1", "yes", "on"):
                yaml_data["rate_limiting"]["enabled"] = True
            elif rate_enabled in ("false", "0", "no", "off"):
                yaml_data["rate_limiting"]["enabled"] = False
            else:
                raise ValueError(
                    f"MCP_RATE_LIMIT_ENABLED must be a boolean value (true/false), got: {os.environ['MCP_RATE_LIMIT_ENABLED']}"
                )
        
        if "MCP_RATE_LIMIT_RPM" in os.environ:
            try:
                yaml_data["rate_limiting"]["requests_per_minute"] = int(os.environ["MCP_RATE_LIMIT_RPM"])
            except ValueError:
                raise ValueError(
                    f"MCP_RATE_LIMIT_RPM must be an integer, got: {os.environ['MCP_RATE_LIMIT_RPM']}"
                )
        
        # Logging overrides
        if "MCP_LOG_LEVEL" in os.environ:
            yaml_data["logging"]["level"] = os.environ["MCP_LOG_LEVEL"]
        
        return yaml_data
    
    @classmethod
    def create_default(cls) -> "Config":
        """Create a Config instance with default values."""
        return cls()


def load_config(config_path: Optional[str | Path] = None) -> Config:
    """
    Load configuration from a file or create default configuration.
    
    Args:
        config_path: Optional path to configuration file. If None, uses default values.
        
    Returns:
        Config instance with loaded configuration
        
    Raises:
        FileNotFoundError: If config_path is provided but file doesn't exist
        ValueError: If configuration validation fails
    """
    if config_path:
        return Config.from_yaml(config_path)
    else:
        return Config.create_default()
