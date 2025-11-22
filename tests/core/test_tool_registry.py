"""Tests for ToolRegistry class."""

import pytest
from mcp.types import Tool
from src.core.tool_registry import ToolRegistry


# Sample tool and handler for testing
def create_sample_tool(name: str = "test_tool") -> Tool:
    """Create a sample tool for testing."""
    return Tool(
        name=name,
        description="A test tool",
        inputSchema={
            "type": "object",
            "properties": {
                "value": {
                    "type": "string",
                    "description": "A test value"
                }
            },
            "required": ["value"]
        }
    )


async def sample_handler(arguments: dict):
    """Sample handler function."""
    return {"result": "success"}


class TestToolRegistry:
    """Test ToolRegistry class functionality."""
    
    def test_init(self):
        """Test ToolRegistry initialization."""
        registry = ToolRegistry()
        
        assert isinstance(registry.tools, dict)
        assert isinstance(registry.handlers, dict)
        assert len(registry.tools) == 0
        assert len(registry.handlers) == 0
        assert registry.count() == 0
    
    def test_register_tool(self):
        """Test tool registration."""
        registry = ToolRegistry()
        tool = create_sample_tool()
        
        registry.register_tool(tool, sample_handler)
        
        assert registry.count() == 1
        assert tool.name in registry.tools
        assert tool.name in registry.handlers
        assert registry.get_tool(tool.name) == tool
        assert registry.get_handler(tool.name) == sample_handler
    
    def test_register_multiple_tools(self):
        """Test registering multiple tools."""
        registry = ToolRegistry()
        
        tool1 = create_sample_tool("tool1")
        tool2 = create_sample_tool("tool2")
        tool3 = create_sample_tool("tool3")
        
        registry.register_tool(tool1, sample_handler)
        registry.register_tool(tool2, sample_handler)
        registry.register_tool(tool3, sample_handler)
        
        assert registry.count() == 3
        assert registry.tool_exists("tool1")
        assert registry.tool_exists("tool2")
        assert registry.tool_exists("tool3")
    
    def test_register_duplicate_tool_raises_error(self):
        """Test that registering duplicate tool raises error."""
        registry = ToolRegistry()
        tool = create_sample_tool()
        
        registry.register_tool(tool, sample_handler)
        
        # Try to register same tool again
        with pytest.raises(ValueError, match="already registered"):
            registry.register_tool(tool, sample_handler)
    
    def test_register_non_callable_handler_raises_error(self):
        """Test that registering non-callable handler raises error."""
        registry = ToolRegistry()
        tool = create_sample_tool()
        
        with pytest.raises(TypeError, match="must be callable"):
            registry.register_tool(tool, "not_a_function")
    
    def test_get_handler(self):
        """Test getting tool handler."""
        registry = ToolRegistry()
        tool = create_sample_tool()
        
        registry.register_tool(tool, sample_handler)
        
        handler = registry.get_handler(tool.name)
        assert handler == sample_handler
    
    def test_get_handler_not_found(self):
        """Test getting handler for non-existent tool."""
        registry = ToolRegistry()
        
        handler = registry.get_handler("nonexistent")
        assert handler is None
    
    def test_get_tool(self):
        """Test getting tool definition."""
        registry = ToolRegistry()
        tool = create_sample_tool()
        
        registry.register_tool(tool, sample_handler)
        
        retrieved_tool = registry.get_tool(tool.name)
        assert retrieved_tool == tool
        assert retrieved_tool.name == tool.name
        assert retrieved_tool.description == tool.description
    
    def test_get_tool_not_found(self):
        """Test getting non-existent tool."""
        registry = ToolRegistry()
        
        tool = registry.get_tool("nonexistent")
        assert tool is None
    
    def test_list_tools(self):
        """Test listing all tools."""
        registry = ToolRegistry()
        
        # Empty registry
        assert registry.list_tools() == []
        
        # Add tools
        tool1 = create_sample_tool("tool1")
        tool2 = create_sample_tool("tool2")
        
        registry.register_tool(tool1, sample_handler)
        registry.register_tool(tool2, sample_handler)
        
        tools = registry.list_tools()
        assert len(tools) == 2
        assert tool1 in tools
        assert tool2 in tools
    
    def test_tool_exists(self):
        """Test checking if tool exists."""
        registry = ToolRegistry()
        tool = create_sample_tool()
        
        assert not registry.tool_exists(tool.name)
        
        registry.register_tool(tool, sample_handler)
        
        assert registry.tool_exists(tool.name)
        assert not registry.tool_exists("nonexistent")
    
    def test_unregister_tool(self):
        """Test unregistering a tool."""
        registry = ToolRegistry()
        tool = create_sample_tool()
        
        registry.register_tool(tool, sample_handler)
        assert registry.count() == 1
        
        result = registry.unregister_tool(tool.name)
        assert result is True
        assert registry.count() == 0
        assert not registry.tool_exists(tool.name)
    
    def test_unregister_tool_not_found(self):
        """Test unregistering non-existent tool."""
        registry = ToolRegistry()
        
        result = registry.unregister_tool("nonexistent")
        assert result is False
    
    def test_clear(self):
        """Test clearing all tools."""
        registry = ToolRegistry()
        
        # Add multiple tools
        for i in range(5):
            tool = create_sample_tool(f"tool{i}")
            registry.register_tool(tool, sample_handler)
        
        assert registry.count() == 5
        
        registry.clear()
        
        assert registry.count() == 0
        assert registry.list_tools() == []
    
    def test_count(self):
        """Test counting registered tools."""
        registry = ToolRegistry()
        
        assert registry.count() == 0
        
        tool1 = create_sample_tool("tool1")
        registry.register_tool(tool1, sample_handler)
        assert registry.count() == 1
        
        tool2 = create_sample_tool("tool2")
        registry.register_tool(tool2, sample_handler)
        assert registry.count() == 2
        
        registry.unregister_tool("tool1")
        assert registry.count() == 1
        
        registry.clear()
        assert registry.count() == 0
