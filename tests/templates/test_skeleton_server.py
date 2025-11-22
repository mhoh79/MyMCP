"""
Tests for the Skeleton MCP Server Template.

These tests verify that the skeleton server works correctly and
can serve as a reliable template for creating new MCP servers.
"""

import pytest
from src.templates.skeleton_server import SkeletonServer


class TestSkeletonServerInitialization:
    """Test skeleton server initialization."""
    
    def test_skeleton_server_can_be_instantiated(self):
        """Test that SkeletonServer can be instantiated."""
        server = SkeletonServer()
        assert server is not None
    
    def test_skeleton_server_inherits_from_base(self):
        """Test that SkeletonServer properly inherits from BaseMCPServer."""
        server = SkeletonServer()
        assert hasattr(server, 'tool_registry')
        assert hasattr(server, 'server_state')
        assert hasattr(server, 'logger')
        assert hasattr(server, 'config')
    
    def test_server_name(self):
        """Test that server name is correct."""
        server = SkeletonServer()
        assert server.get_server_name() == "skeleton-server"
    
    def test_server_version(self):
        """Test that server version is correct."""
        server = SkeletonServer()
        assert server.get_server_version() == "1.0.0"


class TestSkeletonServerToolRegistration:
    """Test skeleton server tool registration."""
    
    def test_tools_are_registered(self):
        """Test that tools are registered during initialization."""
        server = SkeletonServer()
        
        # Should have exactly 2 tools (echo and reverse)
        assert server.tool_registry.count() == 2
    
    def test_echo_tool_exists(self):
        """Test that echo tool is registered."""
        server = SkeletonServer()
        assert server.tool_registry.tool_exists("echo")
    
    def test_reverse_tool_exists(self):
        """Test that reverse tool is registered."""
        server = SkeletonServer()
        assert server.tool_registry.tool_exists("reverse")
    
    def test_echo_tool_has_handler(self):
        """Test that echo tool has a handler."""
        server = SkeletonServer()
        handler = server.tool_registry.get_handler("echo")
        assert handler is not None
        assert callable(handler)
    
    def test_reverse_tool_has_handler(self):
        """Test that reverse tool has a handler."""
        server = SkeletonServer()
        handler = server.tool_registry.get_handler("reverse")
        assert handler is not None
        assert callable(handler)
    
    def test_tool_list_structure(self):
        """Test that tools have proper structure."""
        server = SkeletonServer()
        tools = server.tool_registry.list_tools()
        
        assert len(tools) == 2
        
        for tool in tools:
            # Each tool should have required attributes
            assert hasattr(tool, 'name')
            assert hasattr(tool, 'description')
            assert hasattr(tool, 'inputSchema')
            
            # Verify tool names
            assert tool.name in ["echo", "reverse"]
            
            # Verify description is not empty
            assert len(tool.description) > 0
            
            # Verify input schema structure
            assert isinstance(tool.inputSchema, dict)
            assert "type" in tool.inputSchema
            assert "properties" in tool.inputSchema
            assert "required" in tool.inputSchema


class TestEchoTool:
    """Test the echo tool functionality."""
    
    @pytest.mark.asyncio
    async def test_echo_with_valid_input(self):
        """Test echo tool with valid input."""
        server = SkeletonServer()
        handler = server.tool_registry.get_handler("echo")
        
        result = await handler({"text": "Hello, World!"})
        
        assert result.isError is False
        assert len(result.content) == 1
        assert "Hello, World!" in result.content[0].text
        assert "Echo:" in result.content[0].text
    
    @pytest.mark.asyncio
    async def test_echo_with_empty_string(self):
        """Test echo tool with empty string."""
        server = SkeletonServer()
        handler = server.tool_registry.get_handler("echo")
        
        result = await handler({"text": ""})
        
        assert result.isError is True
        assert "required" in result.content[0].text.lower() or "empty" in result.content[0].text.lower()
    
    @pytest.mark.asyncio
    async def test_echo_with_missing_parameter(self):
        """Test echo tool with missing text parameter."""
        server = SkeletonServer()
        handler = server.tool_registry.get_handler("echo")
        
        result = await handler({})
        
        assert result.isError is True
        assert "required" in result.content[0].text.lower() or "empty" in result.content[0].text.lower()
    
    @pytest.mark.asyncio
    async def test_echo_with_special_characters(self):
        """Test echo tool with special characters."""
        server = SkeletonServer()
        handler = server.tool_registry.get_handler("echo")
        
        special_text = "Hello! @#$%^&*() 你好 🎉"
        result = await handler({"text": special_text})
        
        assert result.isError is False
        assert special_text in result.content[0].text
    
    @pytest.mark.asyncio
    async def test_echo_with_multiline_text(self):
        """Test echo tool with multiline text."""
        server = SkeletonServer()
        handler = server.tool_registry.get_handler("echo")
        
        multiline_text = "Line 1\nLine 2\nLine 3"
        result = await handler({"text": multiline_text})
        
        assert result.isError is False
        assert multiline_text in result.content[0].text


class TestReverseTool:
    """Test the reverse tool functionality."""
    
    @pytest.mark.asyncio
    async def test_reverse_with_valid_input(self):
        """Test reverse tool with valid input."""
        server = SkeletonServer()
        handler = server.tool_registry.get_handler("reverse")
        
        result = await handler({"text": "Hello"})
        
        assert result.isError is False
        assert len(result.content) == 1
        assert "olleH" in result.content[0].text
        assert "Reversed:" in result.content[0].text
    
    @pytest.mark.asyncio
    async def test_reverse_with_palindrome(self):
        """Test reverse tool with palindrome."""
        server = SkeletonServer()
        handler = server.tool_registry.get_handler("reverse")
        
        result = await handler({"text": "racecar"})
        
        assert result.isError is False
        assert "racecar" in result.content[0].text
    
    @pytest.mark.asyncio
    async def test_reverse_with_empty_string(self):
        """Test reverse tool with empty string."""
        server = SkeletonServer()
        handler = server.tool_registry.get_handler("reverse")
        
        result = await handler({"text": ""})
        
        assert result.isError is True
        assert "required" in result.content[0].text.lower() or "empty" in result.content[0].text.lower()
    
    @pytest.mark.asyncio
    async def test_reverse_with_missing_parameter(self):
        """Test reverse tool with missing text parameter."""
        server = SkeletonServer()
        handler = server.tool_registry.get_handler("reverse")
        
        result = await handler({})
        
        assert result.isError is True
        assert "required" in result.content[0].text.lower() or "empty" in result.content[0].text.lower()
    
    @pytest.mark.asyncio
    async def test_reverse_with_numbers(self):
        """Test reverse tool with numeric text."""
        server = SkeletonServer()
        handler = server.tool_registry.get_handler("reverse")
        
        result = await handler({"text": "12345"})
        
        assert result.isError is False
        assert "54321" in result.content[0].text
    
    @pytest.mark.asyncio
    async def test_reverse_with_special_characters(self):
        """Test reverse tool with special characters."""
        server = SkeletonServer()
        handler = server.tool_registry.get_handler("reverse")
        
        result = await handler({"text": "A!B@C#"})
        
        assert result.isError is False
        assert "#C@B!A" in result.content[0].text
    
    @pytest.mark.asyncio
    async def test_reverse_with_unicode(self):
        """Test reverse tool with unicode characters."""
        server = SkeletonServer()
        handler = server.tool_registry.get_handler("reverse")
        
        result = await handler({"text": "Hello 世界"})
        
        assert result.isError is False
        # Unicode should be reversed correctly
        assert "界世 olleH" in result.content[0].text


class TestErrorHandling:
    """Test error handling in skeleton server."""
    
    @pytest.mark.asyncio
    async def test_unknown_tool_handling(self):
        """Test that unknown tools are handled gracefully."""
        server = SkeletonServer()
        
        # Try to get a handler for a non-existent tool
        handler = server.tool_registry.get_handler("nonexistent_tool")
        
        assert handler is None
    
    @pytest.mark.asyncio
    async def test_echo_error_contains_useful_info(self):
        """Test that echo tool errors contain useful information."""
        server = SkeletonServer()
        handler = server.tool_registry.get_handler("echo")
        
        result = await handler({})
        
        assert result.isError is True
        # Error message should mention 'text' parameter
        assert "text" in result.content[0].text.lower()
    
    @pytest.mark.asyncio
    async def test_reverse_error_contains_useful_info(self):
        """Test that reverse tool errors contain useful information."""
        server = SkeletonServer()
        handler = server.tool_registry.get_handler("reverse")
        
        result = await handler({})
        
        assert result.isError is True
        # Error message should mention 'text' parameter
        assert "text" in result.content[0].text.lower()


class TestServerIntegration:
    """Integration tests for skeleton server."""
    
    def test_complete_server_lifecycle(self):
        """Test complete server initialization and setup."""
        # Create server
        server = SkeletonServer()
        
        # Verify initialization
        assert server.get_server_name() == "skeleton-server"
        assert server.get_server_version() == "1.0.0"
        
        # Verify tools are registered
        assert server.tool_registry.count() == 2
        
        # Verify server state
        assert server.server_state.total_requests == 0
        assert server.server_state.active_connections == 0
        
        # Verify uptime tracking works
        uptime = server.server_state.get_uptime_seconds()
        assert uptime >= 0
    
    @pytest.mark.asyncio
    async def test_multiple_tool_calls(self):
        """Test calling multiple tools in sequence."""
        server = SkeletonServer()
        
        # Call echo tool
        echo_handler = server.tool_registry.get_handler("echo")
        echo_result = await echo_handler({"text": "test"})
        assert echo_result.isError is False
        
        # Call reverse tool
        reverse_handler = server.tool_registry.get_handler("reverse")
        reverse_result = await reverse_handler({"text": "test"})
        assert reverse_result.isError is False
        
        # Both should work independently
        assert "test" in echo_result.content[0].text
        assert "tset" in reverse_result.content[0].text
    
    @pytest.mark.asyncio
    async def test_concurrent_tool_execution(self):
        """Test that tools can be executed concurrently."""
        import asyncio
        
        server = SkeletonServer()
        echo_handler = server.tool_registry.get_handler("echo")
        reverse_handler = server.tool_registry.get_handler("reverse")
        
        # Execute both tools concurrently
        results = await asyncio.gather(
            echo_handler({"text": "concurrent"}),
            reverse_handler({"text": "concurrent"})
        )
        
        assert len(results) == 2
        assert all(not result.isError for result in results)
        assert "concurrent" in results[0].content[0].text
        assert "tnerrucnoc" in results[1].content[0].text


class TestTemplateAsReference:
    """Test that template serves as good reference."""
    
    def test_server_has_comprehensive_docstrings(self):
        """Test that server class has docstrings."""
        assert SkeletonServer.__doc__ is not None
        assert len(SkeletonServer.__doc__) > 50
    
    def test_methods_have_docstrings(self):
        """Test that methods have docstrings."""
        assert SkeletonServer.register_tools.__doc__ is not None
        assert SkeletonServer.get_server_name.__doc__ is not None
        assert SkeletonServer.get_server_version.__doc__ is not None
    
    def test_server_follows_naming_convention(self):
        """Test that server follows naming conventions."""
        server = SkeletonServer()
        server_name = server.get_server_name()
        
        # Server name should be lowercase with hyphens
        assert server_name == server_name.lower()
        assert " " not in server_name
        assert "_" not in server_name
    
    def test_server_version_is_semantic(self):
        """Test that server version follows semantic versioning."""
        server = SkeletonServer()
        version = server.get_server_version()
        
        # Should match pattern X.Y.Z
        parts = version.split(".")
        assert len(parts) == 3
        assert all(part.isdigit() for part in parts)


class TestArgumentParser:
    """Test argument parser functionality."""
    
    def test_argument_parser_creation(self):
        """Test that argument parser can be created."""
        from src.templates.skeleton_server.server import main
        from src.core.mcp_server import BaseMCPServer
        
        parser = BaseMCPServer.create_argument_parser(
            description="Skeleton MCP Server - Template for creating new MCP servers"
        )
        
        # Test default arguments
        args = parser.parse_args([])
        assert args.transport == "stdio"
        assert args.host == "0.0.0.0"
        assert args.port == 8000
        assert args.dev is False
    
    def test_argument_parser_http_mode(self):
        """Test argument parser with HTTP mode."""
        from src.core.mcp_server import BaseMCPServer
        
        parser = BaseMCPServer.create_argument_parser()
        args = parser.parse_args(["--transport", "http", "--port", "9000"])
        
        assert args.transport == "http"
        assert args.port == 9000
