# Custom MCP Servers Directory

This directory is for **your custom MCP servers**. Create new servers here based on the skeleton template.

## 🎯 Quick Start

### Creating a New Server

1. **Copy the skeleton template**:
   ```bash
   cp -r src/templates/skeleton_server src/custom/my_server
   ```

2. **Customize your server** (edit `src/custom/my_server/server.py`):
   - Rename the class (e.g., `MyServer`)
   - Update `get_server_name()` to return your server name
   - Update `get_server_version()` with your version
   - Add your custom tools in `register_tools()`

3. **Create your tools**:
   - Define Tool objects with proper schemas
   - Create async handler functions
   - Register tools with `self.tool_registry.register_tool()`

4. **Test your server**:
   ```bash
   # Stdio mode (for Claude Desktop)
   python -m src.custom.my_server.server
   
   # HTTP mode (for remote access)
   python -m src.custom.my_server.server --transport http --port 8001
   ```

5. **Add to Claude Desktop**:
   Edit your MCP configuration file (`~/.config/claude/config.json` or similar):
   ```json
   {
     "mcpServers": {
       "my-server": {
         "command": "python",
         "args": ["-m", "src.custom.my_server.server"],
         "cwd": "/absolute/path/to/MyMCP"
       }
     }
   }
   ```

## 📁 Directory Structure

```
src/custom/
├── README.md               # This file
├── __init__.py            # Package initialization
└── my_server/             # Your custom server (example)
    ├── __init__.py        # Import your server class
    ├── server.py          # Main server implementation
    ├── README.md          # Documentation for your server
    └── tools.py           # (Optional) Separate file for tools
```

## 📝 Server Checklist

When creating a new server, make sure to:

- [ ] Copy from skeleton template
- [ ] Rename server class
- [ ] Update `get_server_name()` (use lowercase-with-hyphens)
- [ ] Update `get_server_version()` (use semantic versioning)
- [ ] Define at least one tool
- [ ] Create handler function(s) for your tools
- [ ] Register all tools in `register_tools()`
- [ ] Add input validation in handlers
- [ ] Add error handling with try/except
- [ ] Test in stdio mode
- [ ] Test in HTTP mode
- [ ] Document your tools in README.md
- [ ] Add usage examples

## 🔧 Example Server Structure

Here's a minimal example of a custom server:

```python
from core import BaseMCPServer
from mcp.types import Tool, TextContent, CallToolResult

class MyServer(BaseMCPServer):
    def register_tools(self) -> None:
        # Define your tool
        my_tool = Tool(
            name="my_tool",
            description="What my tool does",
            inputSchema={
                "type": "object",
                "properties": {
                    "input": {"type": "string", "description": "Input parameter"}
                },
                "required": ["input"]
            }
        )
        
        # Create handler
        async def handle_my_tool(arguments: dict) -> CallToolResult:
            try:
                input_val = arguments.get("input", "")
                if not input_val:
                    return CallToolResult(
                        content=[TextContent(type="text", text="Error: input required")],
                        isError=True
                    )
                
                result = process(input_val)  # Your logic here
                
                return CallToolResult(
                    content=[TextContent(type="text", text=f"Result: {result}")],
                    isError=False
                )
            except Exception as e:
                self.logger.error(f"Error: {e}", exc_info=True)
                return CallToolResult(
                    content=[TextContent(type="text", text=f"Error: {e}")],
                    isError=True
                )
        
        # Register the tool
        self.tool_registry.register_tool(my_tool, handle_my_tool)
    
    def get_server_name(self) -> str:
        return "my-server"
    
    def get_server_version(self) -> str:
        return "1.0.0"

def main():
    parser = BaseMCPServer.create_argument_parser(
        description="My Custom MCP Server"
    )
    args = parser.parse_args()
    server = MyServer(config_path=args.config)
    server.run(
        transport=args.transport,
        host=args.host,
        port=args.port,
        dev_mode=args.dev
    )

if __name__ == "__main__":
    main()
```

## 📚 Available Resources

### Template
- **Skeleton Server**: `src/templates/skeleton_server/` - Minimal working template with 2 example tools

### Examples
- **Math Server**: `src/builtin/math_server/` - 25+ mathematical tools
- **Stats Server**: `src/builtin/stats_server/` - 32+ statistical tools

### Documentation
- **Skeleton README**: `src/templates/skeleton_server/README.md` - Detailed guide
- **Core README**: `src/core/README.md` - BaseMCPServer documentation
- **Main README**: Root README.md - Project overview

## 🎓 Tool Implementation Patterns

### Simple String Tool
```python
Tool(name="uppercase", ...)
async def handle_uppercase(args):
    text = args.get("text", "")
    return CallToolResult(content=[TextContent(text=text.upper())])
```

### Numeric Calculation Tool
```python
Tool(name="multiply", ...)
async def handle_multiply(args):
    a = args.get("a", 0)
    b = args.get("b", 0)
    result = a * b
    return CallToolResult(content=[TextContent(text=str(result))])
```

### List Processing Tool
```python
Tool(name="sum_list", ...)
async def handle_sum_list(args):
    numbers = args.get("numbers", [])
    total = sum(numbers)
    return CallToolResult(content=[TextContent(text=str(total))])
```

### API Integration Tool
```python
Tool(name="fetch_data", ...)
async def handle_fetch_data(args):
    url = args.get("url", "")
    async with httpx.AsyncClient() as client:
        response = await client.get(url)
        return CallToolResult(content=[TextContent(text=response.text)])
```

## 🚀 Advanced Topics

### Multiple Tools
Register multiple tools in the same server:
```python
def register_tools(self):
    self.tool_registry.register_tool(tool1, handler1)
    self.tool_registry.register_tool(tool2, handler2)
    self.tool_registry.register_tool(tool3, handler3)
```

### Shared State
Use instance variables for shared state:
```python
def __init__(self, config_path=None):
    super().__init__(config_path)
    self.cache = {}  # Shared state across tools

async def handle_tool(self, args):
    # Access shared state
    if key in self.cache:
        return cached_value
```

### External Dependencies
Add dependencies to your server:
```python
import httpx
import numpy as np
from custom_library import custom_function
```

### Configuration
Access configuration in your handlers:
```python
async def handle_tool(self, args):
    max_items = self.config.get("max_items", 100)
    # Use configuration value
```

## ⚠️ Important Notes

1. **Custom servers are NOT tracked by git** - They're in `.gitignore`
2. **This directory is for YOUR code** - Experiment freely
3. **Use the skeleton template** - It includes best practices
4. **Test thoroughly** - Test both stdio and HTTP modes
5. **Document your work** - Future you will thank you

## 🤝 Sharing Your Server

If you create something useful:
1. Add comprehensive documentation
2. Include usage examples
3. Write tests (see `tests/templates/`)
4. Consider contributing to the main repository

## 📞 Getting Help

- Review the skeleton template: `src/templates/skeleton_server/`
- Check example servers: `src/builtin/math_server/` and `src/builtin/stats_server/`
- Read the MCP protocol: https://modelcontextprotocol.io/
- Look at tests: `tests/` directory

## 📄 License

Your custom servers remain your intellectual property. The framework code follows the project license.
