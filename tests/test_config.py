"""
Tests for configuration module with custom server support.

This test suite validates:
- Basic config loading
- Custom server validation
- Port conflict detection
- Name uniqueness
- Module path validation
- Backward compatibility
"""

import pytest
import tempfile
from pathlib import Path
from pydantic import ValidationError

from src.config import (
    Config,
    ServerConfig,
    ServersConfig,
    CustomServerConfig,
    AuthenticationConfig,
    RateLimitingConfig,
    LoggingConfig,
    load_config,
)


class TestServerConfig:
    """Tests for ServerConfig model."""
    
    def test_default_values(self):
        """Test ServerConfig with default values."""
        config = ServerConfig()
        assert config.host == "0.0.0.0"
        assert config.port == 8000
    
    def test_custom_values(self):
        """Test ServerConfig with custom values."""
        config = ServerConfig(host="127.0.0.1", port=9000)
        assert config.host == "127.0.0.1"
        assert config.port == 9000
    
    def test_empty_host_raises_error(self):
        """Test that empty host raises validation error."""
        with pytest.raises(ValidationError) as exc_info:
            ServerConfig(host="")
        assert "Host cannot be empty" in str(exc_info.value)
    
    def test_invalid_port_range(self):
        """Test that invalid port range raises validation error."""
        with pytest.raises(ValidationError):
            ServerConfig(port=0)
        
        with pytest.raises(ValidationError):
            ServerConfig(port=65536)


class TestCustomServerConfig:
    """Tests for CustomServerConfig model."""
    
    def test_valid_custom_server(self):
        """Test creating a valid custom server config."""
        config = CustomServerConfig(
            name="my-server",
            module="custom.my_server",
            host="0.0.0.0",
            port=8002,
            enabled=True
        )
        assert config.name == "my-server"
        assert config.module == "custom.my_server"
        assert config.host == "0.0.0.0"
        assert config.port == 8002
        assert config.enabled is True
    
    def test_default_values(self):
        """Test CustomServerConfig with default values."""
        config = CustomServerConfig(
            name="test-server",
            module="custom.test",
            port=8002
        )
        assert config.host == "0.0.0.0"
        assert config.enabled is True
    
    def test_empty_name_raises_error(self):
        """Test that empty name raises validation error."""
        with pytest.raises(ValidationError) as exc_info:
            CustomServerConfig(name="", module="custom.test", port=8002)
        assert "Server name cannot be empty" in str(exc_info.value)
    
    def test_invalid_name_characters(self):
        """Test that invalid name characters raise validation error."""
        with pytest.raises(ValidationError) as exc_info:
            CustomServerConfig(name="my@server", module="custom.test", port=8002)
        assert "must contain only alphanumeric characters" in str(exc_info.value)
    
    def test_valid_name_formats(self):
        """Test various valid name formats."""
        valid_names = ["my-server", "my_server", "MyServer", "server1", "server-1-test"]
        for name in valid_names:
            config = CustomServerConfig(name=name, module="custom.test", port=8002)
            assert config.name == name
    
    def test_empty_module_raises_error(self):
        """Test that empty module raises validation error."""
        with pytest.raises(ValidationError) as exc_info:
            CustomServerConfig(name="test", module="", port=8002)
        assert "Module path cannot be empty" in str(exc_info.value)
    
    def test_invalid_module_format(self):
        """Test that invalid module format raises validation error."""
        with pytest.raises(ValidationError) as exc_info:
            CustomServerConfig(name="test", module="custom.my-server", port=8002)
        assert "must be a valid Python module path" in str(exc_info.value)
    
    def test_valid_module_formats(self):
        """Test various valid module formats."""
        valid_modules = [
            "custom.my_server",
            "custom.my_server.server",
            "src.custom.test",
            "mymodule"
        ]
        for module in valid_modules:
            config = CustomServerConfig(name="test", module=module, port=8002)
            assert config.module == module
    
    def test_port_range_validation(self):
        """Test port range validation."""
        # Valid ports
        CustomServerConfig(name="test", module="custom.test", port=1)
        CustomServerConfig(name="test", module="custom.test", port=65535)
        CustomServerConfig(name="test", module="custom.test", port=8080)
        
        # Invalid ports
        with pytest.raises(ValidationError):
            CustomServerConfig(name="test", module="custom.test", port=0)
        
        with pytest.raises(ValidationError):
            CustomServerConfig(name="test", module="custom.test", port=65536)


class TestConfig:
    """Tests for main Config class."""
    
    def test_default_config(self):
        """Test creating config with defaults."""
        config = Config()
        assert config.server.math.port == 8000
        assert config.server.stats.port == 8001
        assert config.custom_servers == []
        assert config.logging.level == "INFO"
    
    def test_config_with_empty_custom_servers(self):
        """Test config with empty custom_servers list."""
        config = Config(custom_servers=[])
        assert config.custom_servers == []
    
    def test_config_with_single_custom_server(self):
        """Test config with a single custom server."""
        custom_server = CustomServerConfig(
            name="my-server",
            module="custom.my_server",
            port=8002
        )
        config = Config(custom_servers=[custom_server])
        assert len(config.custom_servers) == 1
        assert config.custom_servers[0].name == "my-server"
    
    def test_config_with_multiple_custom_servers(self):
        """Test config with multiple custom servers."""
        servers = [
            CustomServerConfig(name="server1", module="custom.server1", port=8002),
            CustomServerConfig(name="server2", module="custom.server2", port=8003),
            CustomServerConfig(name="server3", module="custom.server3", port=8004),
        ]
        config = Config(custom_servers=servers)
        assert len(config.custom_servers) == 3


class TestCustomServerValidation:
    """Tests for custom server validation rules."""
    
    def test_duplicate_names_raises_error(self):
        """Test that duplicate server names raise validation error."""
        servers = [
            CustomServerConfig(name="my-server", module="custom.server1", port=8002),
            CustomServerConfig(name="my-server", module="custom.server2", port=8003),
        ]
        with pytest.raises(ValidationError) as exc_info:
            Config(custom_servers=servers)
        assert "Duplicate server names found: my-server" in str(exc_info.value)
    
    def test_duplicate_ports_raises_error(self):
        """Test that duplicate ports raise validation error."""
        servers = [
            CustomServerConfig(name="server1", module="custom.server1", port=8002),
            CustomServerConfig(name="server2", module="custom.server2", port=8002),
        ]
        with pytest.raises(ValidationError) as exc_info:
            Config(custom_servers=servers)
        assert "Duplicate ports found" in str(exc_info.value)
        assert "8002" in str(exc_info.value)
    
    def test_disabled_servers_port_not_checked(self):
        """Test that disabled servers don't cause port conflicts."""
        servers = [
            CustomServerConfig(name="server1", module="custom.server1", port=8002, enabled=True),
            CustomServerConfig(name="server2", module="custom.server2", port=8002, enabled=False),
        ]
        # Should not raise error because server2 is disabled
        config = Config(custom_servers=servers)
        assert len(config.custom_servers) == 2
    
    def test_port_conflict_with_builtin_math(self):
        """Test that custom server port conflicts with builtin math server."""
        servers = [
            CustomServerConfig(name="my-server", module="custom.server", port=8000),
        ]
        with pytest.raises(ValidationError) as exc_info:
            Config(custom_servers=servers)
        assert "conflict with builtin server ports" in str(exc_info.value)
        assert "8000" in str(exc_info.value)
    
    def test_port_conflict_with_builtin_stats(self):
        """Test that custom server port conflicts with builtin stats server."""
        servers = [
            CustomServerConfig(name="my-server", module="custom.server", port=8001),
        ]
        with pytest.raises(ValidationError) as exc_info:
            Config(custom_servers=servers)
        assert "conflict with builtin server ports" in str(exc_info.value)
        assert "8001" in str(exc_info.value)
    
    def test_port_conflict_with_reserved_ports(self):
        """Test that custom server port conflicts with reserved ports."""
        reserved_ports = [8000, 8001, 9000, 9001]
        for port in reserved_ports:
            servers = [
                CustomServerConfig(name=f"server-{port}", module="custom.server", port=port),
            ]
            with pytest.raises(ValidationError) as exc_info:
                Config(custom_servers=servers)
            assert "conflict with builtin server ports" in str(exc_info.value)
    
    def test_valid_custom_ports(self):
        """Test that non-conflicting ports are accepted."""
        valid_ports = [8002, 8003, 8100, 9002, 9500, 10000]
        for port in valid_ports:
            servers = [
                CustomServerConfig(name=f"server-{port}", module="custom.server", port=port),
            ]
            config = Config(custom_servers=servers)
            assert len(config.custom_servers) == 1
    
    def test_unique_names_unique_ports(self):
        """Test multiple servers with unique names and ports."""
        servers = [
            CustomServerConfig(name="server1", module="custom.server1", port=8002),
            CustomServerConfig(name="server2", module="custom.server2", port=8003),
            CustomServerConfig(name="server3", module="custom.server3", port=8004),
        ]
        config = Config(custom_servers=servers)
        assert len(config.custom_servers) == 3


class TestConfigFromYAML:
    """Tests for loading config from YAML files."""
    
    def test_load_minimal_config(self):
        """Test loading minimal config without custom servers."""
        yaml_content = """
server:
  math:
    host: "0.0.0.0"
    port: 8000
  stats:
    host: "0.0.0.0"
    port: 8001
logging:
  level: "INFO"
"""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            f.write(yaml_content)
            config_path = f.name
        
        try:
            config = Config.from_yaml(config_path)
            assert config.server.math.port == 8000
            assert config.custom_servers == []
        finally:
            Path(config_path).unlink()
    
    def test_load_config_with_empty_custom_servers(self):
        """Test loading config with empty custom_servers list."""
        yaml_content = """
server:
  math:
    port: 8000
  stats:
    port: 8001
custom_servers: []
"""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            f.write(yaml_content)
            config_path = f.name
        
        try:
            config = Config.from_yaml(config_path)
            assert config.custom_servers == []
        finally:
            Path(config_path).unlink()
    
    def test_load_config_with_custom_servers(self):
        """Test loading config with custom servers."""
        yaml_content = """
server:
  math:
    port: 8000
  stats:
    port: 8001
custom_servers:
  - name: my-server
    module: custom.my_server
    host: 0.0.0.0
    port: 8002
    enabled: true
  - name: another-server
    module: custom.another_server
    host: 127.0.0.1
    port: 8003
    enabled: false
"""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            f.write(yaml_content)
            config_path = f.name
        
        try:
            config = Config.from_yaml(config_path)
            assert len(config.custom_servers) == 2
            assert config.custom_servers[0].name == "my-server"
            assert config.custom_servers[0].port == 8002
            assert config.custom_servers[0].enabled is True
            assert config.custom_servers[1].name == "another-server"
            assert config.custom_servers[1].port == 8003
            assert config.custom_servers[1].enabled is False
        finally:
            Path(config_path).unlink()
    
    def test_load_config_with_duplicate_names(self):
        """Test that loading config with duplicate names fails."""
        yaml_content = """
custom_servers:
  - name: my-server
    module: custom.server1
    port: 8002
  - name: my-server
    module: custom.server2
    port: 8003
"""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            f.write(yaml_content)
            config_path = f.name
        
        try:
            with pytest.raises(ValueError) as exc_info:
                Config.from_yaml(config_path)
            assert "Duplicate server names" in str(exc_info.value)
        finally:
            Path(config_path).unlink()
    
    def test_load_config_with_duplicate_ports(self):
        """Test that loading config with duplicate ports fails."""
        yaml_content = """
custom_servers:
  - name: server1
    module: custom.server1
    port: 8002
  - name: server2
    module: custom.server2
    port: 8002
"""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            f.write(yaml_content)
            config_path = f.name
        
        try:
            with pytest.raises(ValueError) as exc_info:
                Config.from_yaml(config_path)
            assert "Duplicate ports" in str(exc_info.value)
        finally:
            Path(config_path).unlink()
    
    def test_load_config_with_port_conflict(self):
        """Test that loading config with builtin port conflict fails."""
        yaml_content = """
custom_servers:
  - name: my-server
    module: custom.server
    port: 8000
"""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            f.write(yaml_content)
            config_path = f.name
        
        try:
            with pytest.raises(ValueError) as exc_info:
                Config.from_yaml(config_path)
            assert "conflict with builtin server ports" in str(exc_info.value)
        finally:
            Path(config_path).unlink()
    
    def test_load_config_with_invalid_port_range(self):
        """Test that loading config with invalid port range fails."""
        yaml_content = """
custom_servers:
  - name: my-server
    module: custom.server
    port: 70000
"""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            f.write(yaml_content)
            config_path = f.name
        
        try:
            with pytest.raises(ValueError) as exc_info:
                Config.from_yaml(config_path)
            assert "validation failed" in str(exc_info.value).lower()
        finally:
            Path(config_path).unlink()
    
    def test_load_config_with_invalid_module_path(self):
        """Test that loading config with invalid module path fails."""
        yaml_content = """
custom_servers:
  - name: my-server
    module: custom.my-server
    port: 8002
"""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            f.write(yaml_content)
            config_path = f.name
        
        try:
            with pytest.raises(ValueError) as exc_info:
                Config.from_yaml(config_path)
            assert "validation failed" in str(exc_info.value).lower()
        finally:
            Path(config_path).unlink()


class TestBackwardCompatibility:
    """Tests for backward compatibility."""
    
    def test_old_config_without_custom_servers(self):
        """Test that old config files without custom_servers work."""
        yaml_content = """
server:
  math:
    host: "0.0.0.0"
    port: 8000
  stats:
    host: "0.0.0.0"
    port: 8001
  cors_origins:
    - "http://localhost:*"

authentication:
  enabled: false
  api_key: "test-key"

rate_limiting:
  enabled: false
  requests_per_minute: 60

logging:
  level: "INFO"
"""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            f.write(yaml_content)
            config_path = f.name
        
        try:
            config = Config.from_yaml(config_path)
            assert config.server.math.port == 8000
            assert config.server.stats.port == 8001
            assert config.custom_servers == []  # Should default to empty list
            assert config.authentication.enabled is False
            assert config.logging.level == "INFO"
        finally:
            Path(config_path).unlink()
    
    def test_create_default_config(self):
        """Test creating default config."""
        config = Config.create_default()
        assert config.server.math.port == 8000
        assert config.server.stats.port == 8001
        assert config.custom_servers == []
        assert config.logging.level == "INFO"
    
    def test_load_config_without_file(self):
        """Test load_config without file path returns default."""
        config = load_config(None)
        assert config.server.math.port == 8000
        assert config.custom_servers == []


class TestEdgeCases:
    """Tests for edge cases and boundary conditions."""
    
    def test_many_custom_servers(self):
        """Test config with many custom servers."""
        servers = [
            CustomServerConfig(
                name=f"server-{i}",
                module=f"custom.server{i}",
                port=8002 + i
            )
            for i in range(10)
        ]
        config = Config(custom_servers=servers)
        assert len(config.custom_servers) == 10
    
    def test_custom_server_with_minimal_fields(self):
        """Test custom server with only required fields."""
        servers = [
            CustomServerConfig(name="test", module="custom.test", port=8002)
        ]
        config = Config(custom_servers=servers)
        assert config.custom_servers[0].host == "0.0.0.0"  # Default
        assert config.custom_servers[0].enabled is True  # Default
    
    def test_mixed_enabled_disabled_servers(self):
        """Test config with mix of enabled and disabled servers."""
        servers = [
            CustomServerConfig(name="server1", module="custom.s1", port=8002, enabled=True),
            CustomServerConfig(name="server2", module="custom.s2", port=8003, enabled=False),
            CustomServerConfig(name="server3", module="custom.s3", port=8004, enabled=True),
            CustomServerConfig(name="server4", module="custom.s4", port=8005, enabled=False),
        ]
        config = Config(custom_servers=servers)
        enabled_servers = [s for s in config.custom_servers if s.enabled]
        disabled_servers = [s for s in config.custom_servers if not s.enabled]
        assert len(enabled_servers) == 2
        assert len(disabled_servers) == 2
    
    def test_high_port_numbers(self):
        """Test custom servers with high port numbers."""
        servers = [
            CustomServerConfig(name="server1", module="custom.s1", port=60000),
            CustomServerConfig(name="server2", module="custom.s2", port=65535),
        ]
        config = Config(custom_servers=servers)
        assert len(config.custom_servers) == 2
    
    def test_low_port_numbers(self):
        """Test custom servers with low port numbers (not reserved)."""
        servers = [
            CustomServerConfig(name="server1", module="custom.s1", port=1),
            CustomServerConfig(name="server2", module="custom.s2", port=100),
        ]
        config = Config(custom_servers=servers)
        assert len(config.custom_servers) == 2
