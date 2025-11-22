"""Tests for BaseMCPServer class."""

import pytest
import asyncio
from unittest.mock import Mock, AsyncMock, patch
from mcp.types import Tool, TextContent, CallToolResult
from src.core.mcp_server import BaseMCPServer


class MockServer(BaseMCPServer):
    """Mock server for testing BaseMCPServer."""
    
    def __init__(self, config_path=None):
        """Initialize mock server."""
        super().__init__(config_path)
        self.tools_registered = False
    
    def register_tools(self):
        """Register mock tools."""
        self.tools_registered = True
        
        # Register a simple test tool
        test_tool = Tool(
            name="test_tool",
            description="A test tool",
            inputSchema={
                "type": "object",
                "properties": {
                    "value": {"type": "string"}
                },
                "required": ["value"]
            }
        )
        
        async def test_handler(arguments):
            value = arguments.get("value", "")
            return CallToolResult(
                content=[TextContent(
                    type="text",
                    text=f"Test result: {value}"
                )],
                isError=False
            )
        
        self.tool_registry.register_tool(test_tool, test_handler)
    
    def get_server_name(self):
        """Return mock server name."""
        return "mock-server"
    
    def get_server_version(self):
        """Return mock server version."""
        return "1.0.0"


class TestBaseMCPServer:
    """Test BaseMCPServer class functionality."""
    
    def test_initialization(self):
        """Test server initialization."""
        server = MockServer()
        
        assert server.get_server_name() == "mock-server"
        assert server.get_server_version() == "1.0.0"
        # Note: tools_registered is set to True after super().__init__() completes
        # But at this point register_tools() has already been called
        assert server.tool_registry.count() == 1
        assert server.tool_registry.tool_exists("test_tool")
    
    def test_server_state_initialized(self):
        """Test that server state is initialized."""
        server = MockServer()
        
        assert hasattr(server, 'server_state')
        assert server.server_state.total_requests == 0
        assert server.server_state.active_connections == 0
        assert server.server_state.mcp_initialized is True
    
    def test_tool_registry_initialized(self):
        """Test that tool registry is initialized."""
        server = MockServer()
        
        assert hasattr(server, 'tool_registry')
        assert server.tool_registry.count() > 0  # Should have registered tools
    
    def test_logger_initialized(self):
        """Test that logger is initialized."""
        server = MockServer()
        
        assert hasattr(server, 'logger')
        assert server.logger.name == "mock-server"
    
    def test_config_loaded(self):
        """Test that config is loaded."""
        server = MockServer()
        
        assert hasattr(server, 'config')
        assert hasattr(server.config, 'logging')
    
    @pytest.mark.asyncio
    async def test_tool_execution(self):
        """Test tool execution through handler."""
        server = MockServer()
        
        # Get handler for test tool
        handler = server.tool_registry.get_handler("test_tool")
        assert handler is not None
        
        # Execute tool
        result = await handler({"value": "hello"})
        
        assert isinstance(result, CallToolResult)
        assert result.isError is False
        assert len(result.content) == 1
        assert result.content[0].text == "Test result: hello"
    
    def test_argument_parser_creation(self):
        """Test argument parser creation."""
        parser = BaseMCPServer.create_argument_parser("Test Server")
        
        # Test that parser has expected arguments
        args = parser.parse_args([])
        
        assert args.config is None
        assert args.transport == "stdio"
        assert args.host == "0.0.0.0"
        assert args.port == 8000
        assert args.dev is False
    
    def test_argument_parser_with_args(self):
        """Test argument parser with custom arguments."""
        parser = BaseMCPServer.create_argument_parser("Test Server")
        
        args = parser.parse_args([
            "--transport", "http",
            "--host", "127.0.0.1",
            "--port", "9000",
            "--config", "/path/to/config.yaml",
            "--dev"
        ])
        
        assert args.transport == "http"
        assert args.host == "127.0.0.1"
        assert args.port == 9000
        assert args.config == "/path/to/config.yaml"
        assert args.dev is True
    
    def test_run_with_invalid_transport(self):
        """Test that invalid transport raises error."""
        server = MockServer()
        
        with pytest.raises(ValueError, match="Invalid transport mode"):
            server.run(transport="invalid")
    
    @pytest.mark.asyncio
    async def test_list_tools_handler(self):
        """Test that list_tools handler works."""
        server = MockServer()
        
        # The MCP app should have list_tools handler registered
        # We can verify by checking the tool registry
        tools = server.tool_registry.list_tools()
        
        assert len(tools) == 1
        assert tools[0].name == "test_tool"
        assert tools[0].description == "A test tool"
    
    @pytest.mark.asyncio
    async def test_call_tool_handler(self):
        """Test that call_tool handler works."""
        server = MockServer()
        
        # Get handler and call it
        handler = server.tool_registry.get_handler("test_tool")
        result = await handler({"value": "test_value"})
        
        assert result.isError is False
        assert "test_value" in result.content[0].text


class TestAbstractMethods:
    """Test that abstract methods must be implemented."""
    
    def test_cannot_instantiate_base_class(self):
        """Test that BaseMCPServer cannot be instantiated directly."""
        with pytest.raises(TypeError, match="Can't instantiate abstract class"):
            BaseMCPServer()
    
    def test_subclass_must_implement_register_tools(self):
        """Test that subclass must implement register_tools."""
        
        class IncompleteServer1(BaseMCPServer):
            def get_server_name(self):
                return "incomplete"
            
            def get_server_version(self):
                return "1.0.0"
        
        with pytest.raises(TypeError, match="Can't instantiate abstract class"):
            IncompleteServer1()
    
    def test_subclass_must_implement_get_server_name(self):
        """Test that subclass must implement get_server_name."""
        
        class IncompleteServer2(BaseMCPServer):
            def register_tools(self):
                pass
            
            def get_server_version(self):
                return "1.0.0"
        
        with pytest.raises(TypeError, match="Can't instantiate abstract class"):
            IncompleteServer2()
    
    def test_subclass_must_implement_get_server_version(self):
        """Test that subclass must implement get_server_version."""
        
        class IncompleteServer3(BaseMCPServer):
            def register_tools(self):
                pass
            
            def get_server_name(self):
                return "incomplete"
        
        with pytest.raises(TypeError, match="Can't instantiate abstract class"):
            IncompleteServer3()


class TestServerIntegration:
    """Integration tests for server functionality."""
    
    def test_server_lifecycle(self):
        """Test complete server lifecycle."""
        # Create server
        server = MockServer()
        
        # Verify initialization
        assert server.get_server_name() == "mock-server"
        assert server.tool_registry.count() == 1
        
        # Simulate some activity
        server.server_state.increment_requests()
        server.server_state.increment_connections()
        
        assert server.server_state.total_requests == 1
        assert server.server_state.active_connections == 1
        
        # Get uptime
        uptime = server.server_state.get_uptime_seconds()
        assert uptime >= 0
    
    @pytest.mark.asyncio
    async def test_tool_registration_and_execution(self):
        """Test tool registration and execution flow."""
        server = MockServer()
        
        # Verify tool is registered
        assert server.tool_registry.tool_exists("test_tool")
        
        # Get tool
        tool = server.tool_registry.get_tool("test_tool")
        assert tool.name == "test_tool"
        
        # Get and execute handler
        handler = server.tool_registry.get_handler("test_tool")
        result = await handler({"value": "integration_test"})
        
        assert result.isError is False
        assert "integration_test" in result.content[0].text
